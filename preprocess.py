import pandas as pd
import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    return text.strip()


def extract_label(url: str) -> int:
    url = str(url).lower()
    return 0 if "foxnews.com" in url else 1


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

    if "headline" not in df.columns:
        raise ValueError("CSV must contain 'headline' column")

    # 🔥 fix: handle missing values
    df["headline"] = df["headline"].fillna("").astype(str).apply(clean_text)

    df["label"] = df["url"].apply(extract_label)

    df = df[df["headline"].str.len() > 5].reset_index(drop=True)

    X = df["headline"].tolist()
    y = df["label"].tolist()

    return X, y