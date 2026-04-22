"""
Preprocessing for Project B: News Source Classification.

Contract:
    prepare_data(csv_path: str) -> (X, y)
    X: List[str]  — headline strings fed directly to model.predict()
    y: List[int]  — 0 = Fox News, 1 = NBC News

Supported CSV formats:
  1. url, headline, label  — fully labelled
  2. url, headline         — label inferred from domain
  3. url                   — headline scraped via direct fetch, Google News RSS, slug fallback
"""

import html
import random
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote

import pandas as pd

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_SOURCE_SUFFIX = re.compile(
    r"\s*[|\-–—]\s*(fox news|nbc news|reuters|ap|associated press).*$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = text.strip()
    text = _SOURCE_SUFFIX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _label_from_url(url: str) -> int:
    return 0 if "foxnews.com" in str(url) else 1


def _headline_from_slug(url: str) -> str:
    """Derive a headline from the URL path. Uses the longest segment to handle
    live-blog card URLs (e.g. /rcrd52541) where the last segment is an ID."""
    try:
        path = urlparse(str(url)).path.rstrip("/")
        segments = [re.sub(r"\.[a-z]+$", "", s) for s in path.split("/") if s]
        slug = max(segments, key=len) if segments else ""
        return slug.replace("-", " ")
    except Exception:
        return ""


def _scrape_headline_direct(url: str) -> Optional[str]:
    """Scrape headline directly from the live article URL."""
    try:
        time.sleep(random.uniform(0.3, 0.9))
        req = urllib.request.Request(str(url), headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            page = resp.read().decode("utf-8", errors="replace")

        # Prefer <title> — fastest and most reliable across both sites.
        title_m = re.search(r"<title[^>]*>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
        if title_m:
            text = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1)).strip())
            text = re.sub(
                r"\s*[|–—]\s*(fox news|nbc news|reuters|ap).*$",
                "", text, flags=re.IGNORECASE,
            ).strip()
            if text:
                return text

        # Fall back to <h1 class="...headline..."> for pages where
        # <title> contains only the site name or is missing.
        h1 = re.search(
            r'<h1[^>]+class=["\'][^"\']*headline[^"\']*["\'][^>]*>(.*?)</h1>',
            page, re.IGNORECASE | re.DOTALL,
        )
        if h1:
            return html.unescape(re.sub(r"<[^>]+>", "", h1.group(1)).strip())

        # Last resort — any <h1> on the page.
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.IGNORECASE | re.DOTALL)
        if h1:
            return html.unescape(re.sub(r"<[^>]+>", "", h1.group(1)).strip())
    except Exception:
        pass
    return None


def _scrape_headline_gnews(url: str) -> Optional[str]:
    """Query Google News RSS using slug keywords + site filter.

    Used as Pass 2 for URLs where direct scraping failed (paywalled,
    redirected, or rate-limited). Converts the URL slug into a keyword
    query so Google News can match the cached headline without hitting
    the origin server.
    """
    try:
        slug = _headline_from_slug(url)
        if not slug:
            return None
        keywords = " ".join(slug.split()[:6])
        domain = "foxnews.com" if "foxnews.com" in url else "nbcnews.com"
        query = quote(f"{keywords} site:{domain}")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        time.sleep(random.uniform(0.5, 1.5))
        req = urllib.request.Request(rss_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8", errors="replace")
        root = ET.fromstring(xml_data)
        item = root.find(".//item/title")
        if item is not None and item.text:
            text = re.sub(
                r"\s*[|\-–—]\s*(fox news|nbc news).*$",
                "", item.text, flags=re.IGNORECASE,
            ).strip()
            if text:
                return text
    except Exception:
        pass
    return None

def _scrape_pass(
    fn: Callable[[str], Optional[str]],
    indexed_urls: List[Tuple[int, str]],
    workers: int,
    label: str = "",
    batch_size: int = 100,
    stall_sleep: int = 60,
    verbose: bool = True,
) -> Dict[int, Optional[str]]:
    """Run fn concurrently over (idx, url) pairs in batches.

    Batching serves two purposes:
      1. Progress reporting every batch_size URLs.
      2. Stall detection — if a full batch returns 0 hits the site is likely
         rate-limiting us, so we sleep stall_sleep seconds before continuing.
    """
    results: Dict[int, Optional[str]] = {}
    total = len(indexed_urls)
    batches = [indexed_urls[i:i + batch_size] for i in range(0, total, batch_size)]

    for b_num, batch in enumerate(batches):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fn, url): idx for idx, url in batch}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        batch_hits = sum(1 for idx, _ in batch if results.get(idx))
        processed = min((b_num + 1) * batch_size, total)

        if verbose:
            running_hits = sum(1 for v in results.values() if v)
            print(f"  [{label}] {processed}/{total} — hits so far: {running_hits}")

        # If every URL in the batch failed, the server is likely throttling us.
        # Sleep to let the rate-limit window reset before the next batch.
        if batch_hits == 0 and b_num < len(batches) - 1:
            if verbose:
                print(f"  [{label}] 0/{len(batch)} hits — sleeping {stall_sleep}s for rate-limit reset...")
            time.sleep(stall_sleep)

    return results


def _resolve_headlines(urls: List[str], workers: int = 5) -> List[str]:
    """Resolve a headline for every URL using a three-pass waterfall:

      Pass 1 — Direct scrape: fetch the live article page and extract
               the headline from <title> or <h1>. Fastest and most
               accurate; succeeds for most publicly accessible articles.

      Pass 2 — Google News RSS: for URLs that failed Pass 1 (blocked,
               paywalled, or redirected), query Google News with slug
               keywords. Returns Google's cached headline without hitting
               the origin server. Blocked on datacenter IPs (AWS/GCP).

      Pass 3 — Slug fallback: converts the URL path into a readable
               string (e.g. /trump-signs-bill → "trump signs bill").
               Always succeeds but produces lower-quality headlines.
    """
    indexed = list(enumerate(urls))

    # Pass 1: attempt direct scrape for all URLs.
    direct = _scrape_pass(_scrape_headline_direct, indexed, workers, label="direct", verbose=True, batch_size=1000)

    # Pass 2: retry only the URLs that came back empty from Pass 1.
    missed = [(i, urls[i]) for i in range(len(urls)) if not direct.get(i)]
    gnews: Dict[int, Optional[str]] = {}
    if missed:
        gnews = _scrape_pass(_scrape_headline_gnews, missed, workers, label="gnews", verbose=True, batch_size=1000)

    # Pass 3: slug fallback fills any remaining gaps so every URL gets a value.
    return [direct.get(i) or gnews.get(i) or _headline_from_slug(urls[i]) for i in range(len(urls))]


# def scrape_to_csv(input_csv: str, output_csv: str, workers: int = 5) -> None:
#     """
#     Scrape headlines for all URLs in input_csv and write to output_csv.
#     Order: direct scrape → Google News RSS → slug fallback.
#     Output columns: url, headline, label
#     """
#     df = pd.read_csv(input_csv)
#     if "url" not in df.columns:
#         raise ValueError("Input CSV must have a 'url' column.")
#
#     urls = df["url"].tolist()
#     total = len(urls)
#     indexed = list(enumerate(urls))
#
#     # --- Pass 1: direct scrape ---
#     print(f"Pass 1: direct scrape — {total} URLs ({workers} workers)...")
#     direct = _scrape_pass(_scrape_headline_direct, indexed, workers, label="direct", stall_sleep=30)
#     direct_hits = sum(1 for v in direct.values() if v)
#     print(f"  Direct total: {direct_hits}/{total} ({direct_hits / total * 100:.1f}%)\n")
#
#     # --- Pass 2: Google News RSS for misses ---
#     missed = [(i, urls[i]) for i in range(total) if not direct.get(i)]
#     gnews: Dict[int, Optional[str]] = {}
#     if missed:
#         print(f"Pass 2: Google News RSS — {len(missed)} remaining URLs ({workers} workers)...")
#         gnews = _scrape_pass(_scrape_headline_gnews, missed, workers, label="gnews", stall_sleep=60)
#         gnews_hits = sum(1 for v in gnews.values() if v)
#         print(f"  GNews total: {gnews_hits}/{len(missed)} ({gnews_hits / len(missed) * 100:.1f}%)\n")
#     else:
#         gnews_hits = 0
#
#     slug_hits = total - direct_hits - gnews_hits
#     print(f"Final: {direct_hits} direct ({direct_hits/total*100:.1f}%), "
#           f"{gnews_hits} gnews ({gnews_hits/total*100:.1f}%), "
#           f"{slug_hits} slug fallback ({slug_hits/total*100:.1f}%)")
#
#     df["headline"] = [direct.get(i) or gnews.get(i) or _headline_from_slug(urls[i]) for i in range(total)]
#     df["label"] = df["url"].apply(_label_from_url)
#     df["headline"] = df["headline"].astype(str).apply(_clean)
#     df = df[df["headline"].str.len() >= 10].reset_index(drop=True)
#
#     print(f"Writing {len(df)} rows to {output_csv!r}...")
#     df[["url", "headline", "label"]].to_csv(output_csv, index=False)
#     print("Done.")


def prepare_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Read a CSV and return (headlines, labels).

    Expected columns:
      - 'headline': article headline text (optional — scraped if absent)
      - 'label':    0 or 1  (optional — inferred from URL domain if absent)
      - 'url':      article URL (required when 'headline' or 'label' is missing)
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        if "url" not in df.columns:
            raise ValueError("CSV must have a 'label' or 'url' column.")
        df["label"] = df["url"].apply(_label_from_url)

    if "headline" not in df.columns:
        if "url" not in df.columns:
            raise ValueError("CSV must have a 'headline' or 'url' column.")
        df["headline"] = _resolve_headlines(df["url"].tolist())

    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].astype(str).apply(_clean)
    df = df[df["headline"].str.len() >= 10].reset_index(drop=True)

    return df["headline"].tolist(), df["label"].astype(int).tolist()


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input",   default="data/url_only_data.csv")
#     parser.add_argument("--output",  default="data/headlines_scraped.csv")
#     parser.add_argument("--workers", type=int, default=5)
#     args = parser.parse_args()
#     scrape_to_csv(args.input, args.output, args.workers)
