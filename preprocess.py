import pandas as pd
import re
from typing import List, Tuple


def extract_label_from_url(url: str) -> int:
    url = str(url).lower()
    return 0 if "foxnews.com" in url else 1


def slug_to_text(url: str) -> str:
    url = str(url).split("?")[0].rstrip("/")
    parts = [p for p in url.split("/") if p]

    if not parts:
        return ""

    slug = parts[-1]

    # NBC sometimes ends with rcna ID
    if re.search(r"rcna\d+", slug.lower()) and len(parts) >= 2:
        slug = parts[-2]

    slug = slug.replace("-", " ").replace("_", " ")
    slug = re.sub(r"[^a-zA-Z0-9\s']", " ", slug)
    slug = re.sub(r"\s+", " ", slug).strip().lower()

    return slug


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    df = pd.read_csv(path)

    if "url" not in df.columns:
        raise ValueError("CSV must contain a 'url' column")

    df["text"] = df["url"].apply(slug_to_text)
    df["label"] = df["url"].apply(extract_label_from_url)

    df = df[df["text"].str.len() > 5].reset_index(drop=True)

    X = df["text"].tolist()
    y = df["label"].tolist()

    return X, y