"""Phase 2: train a LogisticRegression head on frozen MiniLM embeddings.
"""

from __future__ import annotations

import base64
import hashlib
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess import _clean, prepare_data  # noqa: E402

CACHE = ROOT / "data" / "embeddings_minilm.npz"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def cache_load() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    z = np.load(CACHE, allow_pickle=False)
    keys = z["keys"][:]
    embs = z["embs"][:]
    return dict(zip(keys.tolist(), embs))


def cache_save(d: dict[str, np.ndarray]) -> None:
    keys = np.array(list(d.keys()))
    embs = np.stack(list(d.values()), axis=0)
    np.savez_compressed(CACHE, keys=keys, embs=embs)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def encode_with_cache(texts: List[str]) -> np.ndarray:
    cache = cache_load()
    print(f"  Cache: {len(cache)} pre-existing embeddings")

    keys = [sha1(t) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    print(f"  Need to encode: {len(missing_idx)} / {len(texts)}")

    if missing_idx:
        from sentence_transformers import SentenceTransformer

        print(f"  Loading {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        missing_texts = [texts[i] for i in missing_idx]
        t0 = time.time()
        new_embs = model.encode(
            missing_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"  Encoded in {time.time()-t0:.1f}s")
        for i, idx in enumerate(missing_idx):
            cache[keys[idx]] = new_embs[i].astype(np.float32)
        cache_save(cache)

    return np.stack([cache[k] for k in keys], axis=0).astype(np.float32)


def main() -> None:
    print("Loading data...")
    X_gnews, y_gnews = prepare_data(str(ROOT / "data" / "gnews_headlines_full.csv"))
    X_ft, y_ft = prepare_data(str(ROOT / "data" / "ft_train.csv"))
    X_val, y_val = prepare_data(str(ROOT / "data" / "ft_val.csv"))

    # Phase 4.C: extra Fox headlines scraped directly from publisher sitemap
    # (publisher distribution; all label=0). class_weight='balanced' on the LR
    # compensates for the resulting Fox-side imbalance.
    fox_path = ROOT / "data" / "fox_direct.csv"
    X_fox: list[str] = []
    y_fox: list[int] = []
    if fox_path.exists():
        X_fox, y_fox = prepare_data(str(fox_path))
        print(f"  fox_direct: {len(X_fox)} (publisher distribution, label=0 only)")

    # Combine. ft_train upsampled 3x; fox_direct as-is (already balanced by class_weight).
    X_train = X_gnews + X_ft * 3 + X_fox
    y_train = y_gnews + y_ft * 3 + y_fox
    print(f"  gnews: {len(X_gnews)}  ft_train×3: {len(X_ft)*3}  "
          f"fox_direct: {len(X_fox)}  total: {len(X_train)}")
    print(f"  ft_val (held out): {len(X_val)}")

    print("\nEncoding train set (with disk cache)...")
    Z_train = encode_with_cache(X_train)
    print(f"  shape: {Z_train.shape}")

    print("\nEncoding ft_val...")
    Z_val = encode_with_cache(X_val)

    # Sample weights: gnews + ft_train×3 at weight 1.0; fox_direct at 0.3.
    # The asymmetric Fox-only addition would otherwise push the boundary toward
    # NBC at the cost of ft_val accuracy
    n_main = len(X_gnews) + 3 * len(X_ft)
    sample_weight = np.ones(len(X_train), dtype=np.float32)
    if len(X_fox) > 0:
        sample_weight[n_main:] = 0.3
    print("\nFitting LogisticRegression(C=1.0, class_weight='balanced'); "
          "fox_direct sample_weight=0.3...")
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    lr.fit(Z_train, y_train, sample_weight=sample_weight)

    val_preds = lr.predict(Z_val)
    correct = sum(int(p == t) for p, t in zip(val_preds, y_val))
    val_acc = correct / len(y_val)
    print(f"\nft_val accuracy (embed-LR alone): {val_acc:.4f}")
    print(classification_report(y_val, val_preds, target_names=["Fox", "NBC"], digits=4))

    manifest = {
        "type": "embed_lr",
        "encoder_model_name": MODEL_NAME,
        "lr": lr,
        "ft_val_accuracy": val_acc,
    }
    blob = base64.b64encode(zlib.compress(pickle.dumps(manifest), level=9)).decode("ascii")
    print(f"\nEmbed-LR blob length: {len(blob)} chars (~{len(blob)//1024} KB)")

    out = ROOT / "tasks" / "embed_lr_blob.txt"
    out.write_text(blob)
    print(f"Wrote {out.relative_to(ROOT)} — paste into model.py as _EMBED_LR_B64")


if __name__ == "__main__":
    main()
