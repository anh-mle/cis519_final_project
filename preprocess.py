"""
Preprocessing for Project B: News Source Classification.

Contract:
    prepare_data(csv_path: str) -> (X, y)
    X: List[str]  — headline strings fed directly to model.predict()
    y: List[int]  — 0 = Fox News, 1 = NBC News

Supported CSV formats:
  1. url, headline, label  — fully labelled
  2. url, headline         — label inferred from domain
  3. url                   — headline scraped via Wayback Machine, slug fallback
"""

import html
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_SOURCE_SUFFIX = re.compile(
    r"\s*[|\-\u2013\u2014]\s*(fox news|nbc news|reuters|ap|associated press).*$",
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


def _scrape_headline_wayback(url: str) -> Optional[str]:
    """Fetch a cached copy from the Wayback Machine and extract the headline.
    Falls back through <title>, headline <h1>, and generic <h1>.
    Returns None on failure — caller should fall back to slug."""
    wayback_url = f"https://web.archive.org/web/{url}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(wayback_url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                page = resp.read().decode("utf-8", errors="replace")

            # <title> tag — most reliably present in archived pages
            title_m = re.search(r"<title[^>]*>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
            if title_m:
                text = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1)).strip())
                # Strip trailing site name: "Headline | Fox News"
                text = re.sub(
                    r"\s*[|\u2013\u2014]\s*(fox news|nbc news|reuters|ap).*$",
                    "", text, flags=re.IGNORECASE,
                ).strip()
                if text:
                    return text

            # Fox News <h1 class="... headline ...">
            h1 = re.search(
                r'<h1[^>]+class=["\'][^"\']*headline[^"\']*["\'][^>]*>(.*?)</h1>',
                page, re.IGNORECASE | re.DOTALL,
            )
            if h1:
                return html.unescape(re.sub(r"<[^>]+>", "", h1.group(1)).strip())

            # Generic <h1>
            h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.IGNORECASE | re.DOTALL)
            if h1:
                return html.unescape(re.sub(r"<[^>]+>", "", h1.group(1)).strip())

        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def _resolve_headlines(urls: List[str], workers: int = 5) -> List[str]:
    """Scrape headlines via Wayback Machine in parallel. Slug fallback on failure."""
    results: dict[int, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scrape_headline_wayback, url): i for i, url in enumerate(urls)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return [results[i] or _headline_from_slug(urls[i]) for i in range(len(urls))]


def scrape_to_csv(input_csv: str, output_csv: str, workers: int = 5) -> None:
    """
    Scrape headlines for all URLs in input_csv and write to output_csv.
    Uses Wayback Machine with slug fallback. Prints scrape vs fallback stats.
    Output columns: url, headline, label
    """
    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        raise ValueError("Input CSV must have a 'url' column.")

    urls = df["url"].tolist()
    total = len(urls)
    print(f"Scraping {total} URLs via Wayback Machine ({workers} workers)...")

    wayback: dict[int, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scrape_headline_wayback, url): i for i, url in enumerate(urls)}
        for i, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            wayback[idx] = future.result()
            if i % 100 == 0:
                ok = sum(1 for v in wayback.values() if v)
                print(f"  {i}/{total} — wayback: {ok}, fallback: {i - ok}")

    wayback_hits = sum(1 for v in wayback.values() if v)
    slug_hits = total - wayback_hits
    print(f"\nResults: {wayback_hits} wayback ({wayback_hits/total*100:.1f}%), "
          f"{slug_hits} slug fallback ({slug_hits/total*100:.1f}%)")

    df["headline"] = [wayback[i] or _headline_from_slug(urls[i]) for i in range(total)]
    df["label"] = df["url"].apply(_label_from_url)
    df["headline"] = df["headline"].astype(str).apply(_clean)
    df = df[df["headline"].str.len() >= 10].reset_index(drop=True)

    print(f"Writing {len(df)} rows to {output_csv!r}...")
    df[["url", "headline", "label"]].to_csv(output_csv, index=False)
    print("Done.")


def prepare_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Read a CSV and return (headlines, labels).

    Expected columns:
      - 'headline': article headline text (optional — scraped via Wayback if absent)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="data/url_only_data.csv")
    parser.add_argument("--output",  default="data/headlines_scraped.csv")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    scrape_to_csv(args.input, args.output, args.workers)
