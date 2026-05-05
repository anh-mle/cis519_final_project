"""
Preprocessing for Project B: News Source Classification.

Contract:
    prepare_data(csv_path: str) -> (X, y)
    X: List[str]  — URL slug strings fed directly to model.predict()
    y: List[int]  — 0 = Fox News, 1 = NBC News
"""

import re
from typing import List, Tuple
from urllib.parse import urlparse

import pandas as pd


def _label_from_url(url: str) -> int:
    return 0 if "foxnews.com" in str(url) else 1


def _slug_from_url(url: str) -> str:
    """Derive a readable string from the URL path slug.

    Uses the longest path segment to handle live-blog card URLs
    (e.g. /rcrd52541) where the last segment is an opaque ID.
    """
    try:
        path = urlparse(str(url)).path.rstrip("/")
        segments = [re.sub(r"\.[a-z]+$", "", s) for s in path.split("/") if s]
        slug = max(segments, key=len) if segments else ""
        slug = re.sub(r"-[a-z]+\d+$", "", slug)  # strip trailing article IDs (e.g. -n1273087, -rcna162987)
        return slug.replace("-", " ").strip().lower()
    except Exception:
        return ""


def prepare_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """
    Read a CSV with a 'url' column and return (X, y).

    X: slug text derived from each URL path.
    y: 0 = Fox News, 1 = NBC News (inferred from domain).
    """
    df = pd.read_csv(csv_path)

    if "url" not in df.columns:
        raise ValueError("CSV must have a 'url' column.")

    df["slug"] = df["url"].apply(_slug_from_url)
    df["label"] = df["url"].apply(_label_from_url)

    df = df[df["slug"].str.len() >= 3].reset_index(drop=True)

    return df["slug"].tolist(), df["label"].astype(int).tolist()

