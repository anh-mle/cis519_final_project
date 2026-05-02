"""Additional Fox News headlines via the publisher's sitemap
https://www.foxnews.com/sitemap.xml.
Output: data/fox_direct.csv with columns (url, headline, label=0).
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess import _clean  # noqa: E402

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA}
SITEMAP_INDEX = "https://www.foxnews.com/sitemap.xml"

_NON_ARTICLE_TOKENS = (
    "/video/", "/videos/", "/category/", "/categories/", "/sitemap",
    "/feed", "/feeds", "/topic/", "/topics/", "/photo/", "/photos/",
    "/podcast", "/tv/", "/tag/", "/about/", "/contact", "/privacy",
    "/terms", "/games/", "/weather/", "/static/", "/account",
)


def fetch_xml(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  fetch failed: {url} ({e})")
        return None


def get_sub_sitemaps() -> list[str]:
    xml = fetch_xml(SITEMAP_INDEX)
    if not xml:
        return []
    return [html.unescape(u) for u in re.findall(r"<loc>([^<]+)</loc>", xml)]


def get_article_urls(sub_url: str) -> list[str]:
    xml = fetch_xml(sub_url)
    if not xml:
        return []
    return [
        html.unescape(u)
        for u in re.findall(r"<loc>([^<]+)</loc>", xml)
        if "sitemap" not in u
    ]


def looks_like_article(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return False
    if not path or path == "/":
        return False
    if any(t in path for t in _NON_ARTICLE_TOKENS):
        return False
    segs = [s for s in path.strip("/").split("/") if s]
    if len(segs) < 2:
        return False
    last = segs[-1]
    if "-" not in last:
        return False
    if len(last) < 15:
        return False
    return True


def scrape_headline(url: str) -> Optional[str]:
    try:
        time.sleep(0.3)  # politeness; ThreadPool naturally spaces requests too
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # Spec literal: h1 class="headline speakable"
        h1 = soup.find("h1", class_="headline speakable")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        # Looser: any h1 with "headline" in its class list
        for tag in soup.find_all("h1"):
            cls = tag.get("class") or []
            if any("headline" in c.lower() for c in cls):
                t = tag.get_text(strip=True)
                if t:
                    return t

        # Last resort: <title> (matches preprocess.py's existing path)
        title = soup.find("title")
        if title and title.get_text(strip=True):
            return title.get_text(strip=True)
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10000, help="success count target")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-budget-min", type=float, default=10.0,
                    help="stop scraping after this many minutes (saves whatever we have)")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="flush CSV after every N new successes")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "fox_direct.csv")
    args = ap.parse_args()

    # Build exclusion set so we never re-scrape an eval URL.
    exclude: Set[str] = set()
    eval_csv = ROOT / "data" / "url_only_data.csv"
    if eval_csv.exists():
        exclude = set(
            pd.read_csv(eval_csv)["url"]
            .astype(str)
            .map(lambda u: u.split("?")[0].rstrip("/"))
        )
        print(f"Excluding {len(exclude)} URLs from url_only_data.csv (eval source)")

    print(f"\n=== sitemap index: {SITEMAP_INDEX} ===")
    subs = get_sub_sitemaps()
    print(f"Found {len(subs)} sub-sitemaps:")
    for s in subs:
        print(f"  {s}")

    # We can change the max target if we want to scrape more
    MAX_URLS = max(args.target * 8, 50_000)
    print(f"\n=== fetching sub-sitemaps (cap candidate pool at {MAX_URLS}) ===")
    seen: Set[str] = set()
    all_urls: list[str] = []
    for sub_url in subs:
        if len(all_urls) >= MAX_URLS:
            print(f"  reached cap of {MAX_URLS} URLs — stopping enumeration")
            break
        urls = get_article_urls(sub_url)
        new = 0
        for u in urls:
            u_clean = u.split("?")[0].rstrip("/")
            if u_clean in seen or u_clean in exclude:
                continue
            seen.add(u_clean)
            all_urls.append(u_clean)
            new += 1
            if len(all_urls) >= MAX_URLS:
                break
        print(f"  {sub_url}: +{new} new (running total {len(all_urls)})")
        time.sleep(0.3)

    article_urls = [u for u in all_urls if looks_like_article(u)]
    print(f"\nArticle candidates after filter: {len(article_urls)} / {len(all_urls)} raw")

    rows: list[dict] = []
    attempted = 0
    last_saved = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
        df.to_csv(args.out, index=False)

    print(
        f"\n=== scraping (target={args.target}, time_budget={args.time_budget_min}min, "
        f"workers={args.workers}, checkpoint every {args.checkpoint_every}) ==="
    )
    t0 = time.time()
    deadline = t0 + args.time_budget_min * 60
    stopped_reason = ""
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scrape_headline, u): u for u in article_urls}
        for f in as_completed(futures):
            url = futures[f]
            attempted += 1
            try:
                raw = f.result()
            except Exception:
                raw = None
            if raw:
                cleaned = _clean(raw)
                if len(cleaned) >= 10:
                    rows.append({"url": url, "headline": cleaned, "label": 0})
                    if len(rows) - last_saved >= args.checkpoint_every:
                        flush()
                        last_saved = len(rows)
            if attempted % 200 == 0:
                el = time.time() - t0
                print(
                    f"  attempted={attempted}/{len(article_urls)}  "
                    f"successes={len(rows)}  hit_rate={len(rows)/attempted:.1%}  "
                    f"elapsed={el/60:.1f}min"
                )
            if len(rows) >= args.target:
                stopped_reason = f"target {args.target} reached"
                break
            if time.time() >= deadline:
                stopped_reason = f"time budget ({args.time_budget_min} min) exhausted"
                break
        # Cancel any still-pending futures so the executor shuts down promptly.
        for fut in futures:
            fut.cancel()

    flush()
    print(f"  stop reason: {stopped_reason or 'queue exhausted'}")

    elapsed = (time.time() - t0) / 60
    print(f"\n=== done in {elapsed:.1f} min ===")
    print(f"  successes: {len(df)} / attempted {attempted}")
    print(f"  wrote {args.out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
