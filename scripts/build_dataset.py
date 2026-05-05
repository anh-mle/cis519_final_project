"""
Build a balanced Fox + NBC URL dataset from publisher sitemaps.

This py file produce the training/eval data under data/combined_url_balanced_8k.
Walks both publisher sitemap indexes, filters to article URLs, dedupes by
the slug text produced by preprocess._headline_from_slug, balances per source, and writes
a single-column file containing 8k url data.

Usage (default reproduces the 4000+4000 = 8000 dataset):
    python3 scripts/build_dataset.py

Common overrides:
    --per-source 6000                 # bigger/smaller dataset
    --out data/my_dataset.csv         # alternate output path / format
    --no-dedupe                       # skip slug dedup (URL-balance only)

Note: live sitemaps update constantly, so reruns produce a different *set* of
articles. The seed only governs sampling order within the candidate pool.
"""

from __future__ import annotations

import argparse
import html
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess import _headline_from_slug  # noqa: E402

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA}

FOX_INDEX = "https://www.foxnews.com/sitemap.xml"
NBC_INDEX = "https://www.nbcnews.com/sitemap/nbcnews/sitemap-index"

# NBC's index lists video/slideshow/curation sub-sitemaps too; keep articles only.
_NBC_ARTICLE_SITEMAP_RE = re.compile(r"sitemap-\d{4}-\d{2}-article\.xml$")

_NON_ARTICLE_TOKENS = (
    "/video/", "/videos/", "/category/", "/categories/", "/sitemap",
    "/feed", "/feeds", "/topic/", "/topics/", "/photo/", "/photos/",
    "/podcast", "/tv/", "/tag/", "/about/", "/contact", "/privacy",
    "/terms", "/games/", "/weather/", "/static/", "/account",
    "/select/", "/shopping/", "/think/",
)


def fetch_xml(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  fetch failed: {url} ({e})")
        return None


def get_sub_sitemaps(index_url: str, *, nbc: bool) -> list[str]:
    xml = fetch_xml(index_url)
    if not xml:
        return []
    locs = [html.unescape(u) for u in re.findall(r"<loc>([^<]+)</loc>", xml)]
    if nbc:
        return [u for u in locs if _NBC_ARTICLE_SITEMAP_RE.search(u)]
    return locs


def get_article_urls(sub_url: str) -> list[str]:
    xml = fetch_xml(sub_url)
    if not xml:
        return []
    return [
        html.unescape(u)
        for u in re.findall(r"<loc>([^<]+)</loc>", xml)
        if "sitemap" not in u
    ]


def looks_like_article(url: str, *, nbc: bool) -> bool:
    """Per-source heuristic: a URL is article-like if its last path segment
    is a hyphen-joined slug long enough to be a real headline."""
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
    # NBC commonly suffixes with `-rcna<digits>`; strip before measuring.
    slug = re.sub(r"-rcna\d+$", "", last) if nbc else last
    if "-" not in slug:
        return False
    if len(slug) < 15:
        return False
    return True


def collect(index_url: str, max_urls: int, *, nbc: bool) -> list[str]:
    name = "NBC" if nbc else "Fox"
    print(f"\n=== {name}: {index_url} ===")
    subs = get_sub_sitemaps(index_url, nbc=nbc)
    print(f"  sub-sitemaps: {len(subs)}")

    seen: set[str] = set()
    urls: list[str] = []
    for sub_url in subs:
        if len(urls) >= max_urls:
            print(f"  reached cap of {max_urls} — stopping enumeration")
            break
        for u in get_article_urls(sub_url):
            uc = u.split("?")[0].rstrip("/")
            if uc in seen:
                continue
            seen.add(uc)
            urls.append(uc)
            if len(urls) >= max_urls:
                break
        time.sleep(0.2)

    articles = [u for u in urls if looks_like_article(u, nbc=nbc)]
    print(f"  raw: {len(urls)}  article-filtered: {len(articles)}")
    return articles


def take_unique_slugs(urls: list[str], n_target: int, rng: random.Random) -> list[str]:
    pool = list(urls)
    rng.shuffle(pool)
    seen: set[str] = set()
    kept: list[str] = []
    for u in pool:
        s = _headline_from_slug(u).strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        kept.append(u)
        if len(kept) >= n_target:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=4000,
                    help="number of unique-slug URLs per source (default 4000)")
    ap.add_argument("--fox-cap", type=int, default=12000,
                    help="max Fox URLs to enumerate before filtering")
    ap.add_argument("--nbc-cap", type=int, default=35000,
                    help="max NBC URLs to enumerate before filtering "
                         "(NBC needs deeper pool — slugs collapse often)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="skip slug dedup; balance by URL count only")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "combined_url_balanced_8k.xlsx")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fox_pool = collect(FOX_INDEX, args.fox_cap, nbc=False)
    nbc_pool = collect(NBC_INDEX, args.nbc_cap, nbc=True)

    rng = random.Random(args.seed)
    if args.no_dedupe:
        rng.shuffle(fox_pool); rng.shuffle(nbc_pool)
        keep = min(args.per_source, len(fox_pool), len(nbc_pool))
        fox_pick, nbc_pick = fox_pool[:keep], nbc_pool[:keep]
    else:
        fox_pick = take_unique_slugs(fox_pool, args.per_source, rng)
        nbc_pick = take_unique_slugs(nbc_pool, args.per_source, rng)

    print(f"\nfox unique-slug picks: {len(fox_pick)} / {args.per_source}")
    print(f"nbc unique-slug picks: {len(nbc_pick)} / {args.per_source}")
    if len(fox_pick) < args.per_source or len(nbc_pick) < args.per_source:
        print("WARN: under-target — increase --fox-cap / --nbc-cap to compensate.")

    combined = fox_pick + nbc_pick
    rng.shuffle(combined)

    df = pd.DataFrame({"url": combined})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)

    print(f"\n=== done ===")
    print(f"  fox: {len(fox_pick)}  nbc: {len(nbc_pick)}  total: {len(df)}")
    try:
        rel = args.out.resolve().relative_to(ROOT)
    except ValueError:
        rel = args.out
    print(f"  wrote {rel}")


if __name__ == "__main__":
    main()
