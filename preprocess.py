"""
Slug-based preprocessing for Fox News vs NBC News classification.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from urllib.parse import urlparse, unquote

import pandas as pd


def _label_from_url(url: str) -> str:
    """Map a URL's domain to its source label.

    Returns 'foxnews' or 'nbcnews', or '' if neither domain is present so the
    caller can drop the row.
    """
    u = str(url).lower()
    if "foxnews.com" in u:
        return "foxnews"
    if "nbcnews.com" in u:
        return "nbcnews"
    parsed = urlparse(u)
    domain = parsed.netloc
    if "fox" in domain:
        return "foxnews"
    if "nbc" in domain:
        return "nbcnews"
    return ""


def _headline_from_slug(url: str) -> str:
    """Extract a readable string from the URL path.
    """
    try:
        parsed = urlparse(str(url))
        path = unquote(parsed.path)
        segments = [s for s in path.split("/") if s]

        if not segments:
            return ""

        slug = ""
        for seg in reversed(segments):
            if "rcna" in seg.lower():
                continue
            if re.match(r"^[a-z]*\d+$", seg.lower()):
                continue
            if len(seg) < 5:
                continue
            slug = seg
            break

        if not slug:
            slug = segments[-1]

        text = slug.replace("-", " ").replace("_", " ")
        text = re.sub(r"\brcna\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text
    except Exception:
        return ""


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """Main entry point called by eval_project_b.py.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    has_headline = any(c.lower() == "headline" for c in df.columns)
    has_label = any(c.lower() == "label" for c in df.columns)

    texts: List[str] = []
    labels: List[str] = []

    if has_headline and has_label:
        h_col = next(c for c in df.columns if c.lower() == "headline")
        l_col = next(c for c in df.columns if c.lower() == "label")
        for h, l in zip(df[h_col], df[l_col]):
            if pd.isna(h) or pd.isna(l):
                continue
            text = str(h).strip().lower()
            if not text:
                continue
            # Accept both int (0/1) and string ('foxnews'/'nbcnews') labels.
            try:
                lid = int(l)
                label = {0: "foxnews", 1: "nbcnews"}.get(lid, "")
            except (ValueError, TypeError):
                label = str(l).strip().lower()
            if label not in {"foxnews", "nbcnews"}:
                continue
            texts.append(text)
            labels.append(label)
        return texts, labels
    
    url_col = None
    for c in df.columns:
        if c.lower() in {"url", "urls", "link", "links"}:
            url_col = c
            break
    if url_col is None:
        url_col = df.columns[0]

    for url in df[url_col]:
        if pd.isna(url):
            continue
        url = str(url).strip()
        label = _label_from_url(url)
        if not label:
            continue
        text = _headline_from_slug(url)
        if not text:
            continue
        texts.append(text)
        labels.append(label)

    return texts, labels


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    test_urls = [
        "https://www.foxnews.com/us/oklahoma-park-shooting-injured-nearly-two-dozen-started-argument-unsanctioned-party-police-say",
        "https://www.nbcnews.com/politics/politics-news/2-killed-us-strike-alleged-drug-boat-caribbean-rcna343597",
        "https://www.foxnews.com/politics/obama-admits-genuine-tension-marriage-pressure-stay-politics",
        "https://www.nbcnews.com/video/suspect-in-shooting-near-white-house-loaded-into-ambulance-262718533814",
    ]
    print("URL → (label, text):")
    for u in test_urls:
        print(f"  {_label_from_url(u):8s}  '{_headline_from_slug(u)}'")
