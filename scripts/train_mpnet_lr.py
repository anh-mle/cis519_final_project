"""Phase 4 (Option B): add a second frozen encoder + LR head for ensemble diversity.

Trains a LogisticRegression on top of MPNet-base embeddings (different
architecture than the existing MiniLM-L6 LR head). The two encoders are
trained on different objectives and produce decorrelated errors → ensembling
their predictions should add 1–3 points on HF, no fine-tuning needed.

Output: tasks/mpnet_lr_blob.txt — base64+zlib-compressed pickle of
        {"encoder_model_name": "...", "lr": LogisticRegression(...)}.

Embeddings are cached to data/embeddings_mpnet.npz keyed by SHA-1 of the
cleaned headline so re-runs are near-instant after the first encode.
"""

from __future__ import annotations

import base64
import hashlib
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess import _clean, prepare_data  # noqa: E402

CACHE = ROOT / "data" / "embeddings_mpnet.npz"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


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
    print(f"  cache: {len(cache)} pre-existing embeddings")

    keys = [sha1(t) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    print(f"  to encode: {len(missing_idx)} / {len(texts)}")

    if missing_idx:
        # Manual MPNet encoding via raw transformers (no sentence_transformers
        # dep — keeps us aligned with what the HF worker can run).
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"  loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()

        missing_texts = [texts[i] for i in missing_idx]
        new_embs = []
        bs = 32
        t0 = time.time()
        for start in range(0, len(missing_texts), bs):
            batch = missing_texts[start:start + bs]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            new_embs.append(pooled.cpu().numpy().astype(np.float32))
            if (start // bs) % 50 == 0:
                elapsed = time.time() - t0
                done = start + len(batch)
                eta = elapsed / max(1, done) * (len(missing_texts) - done)
                print(f"    {done}/{len(missing_texts)} ({done/len(missing_texts):.0%})  eta {eta:.0f}s")
        new_embs_arr = np.concatenate(new_embs, axis=0)
        for i, idx in enumerate(missing_idx):
            cache[keys[idx]] = new_embs_arr[i]
        cache_save(cache)
        print(f"  encoded in {time.time()-t0:.0f}s")

    return np.stack([cache[k] for k in keys], axis=0).astype(np.float32)


def main() -> None:
    print("Loading data...")
    X_gnews, y_gnews = prepare_data(str(ROOT / "data" / "gnews_headlines_full.csv"))
    X_ft, y_ft = prepare_data(str(ROOT / "data" / "ft_train.csv"))
    X_val, y_val = prepare_data(str(ROOT / "data" / "ft_val.csv"))

    fox_path = ROOT / "data" / "fox_direct.csv"
    X_fox: list[str] = []
    y_fox: list[int] = []
    if fox_path.exists():
        X_fox, y_fox = prepare_data(str(fox_path))
        print(f"  fox_direct: {len(X_fox)} (publisher distribution, label=0 only)")

    X_train = X_gnews + X_ft * 3 + X_fox
    y_train = y_gnews + y_ft * 3 + y_fox
    print(f"  gnews: {len(X_gnews)}  ft×3: {len(X_ft)*3}  "
          f"fox_direct: {len(X_fox)}  total: {len(X_train)}")
    print(f"  ft_val: {len(X_val)}")

    print("\nEncoding train...")
    Z_train = encode_with_cache(X_train)
    print(f"  shape: {Z_train.shape}")

    print("\nEncoding ft_val...")
    Z_val = encode_with_cache(X_val)

    n_main = len(X_gnews) + 3 * len(X_ft)
    sample_weight = np.ones(len(X_train), dtype=np.float32)
    if len(X_fox) > 0:
        sample_weight[n_main:] = 0.3
    print("\nFitting LogisticRegression(C=1.0, class_weight='balanced'); "
          "fox_direct sample_weight=0.3...")
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(Z_train, y_train, sample_weight=sample_weight)

    val_preds = lr.predict(Z_val)
    val_acc = sum(int(p == t) for p, t in zip(val_preds, y_val)) / len(y_val)
    print(f"\nMPNet-LR alone on ft_val: {val_acc:.4f}")
    print(classification_report(y_val, val_preds, target_names=["Fox", "NBC"], digits=4))

    manifest = {"encoder_model_name": MODEL_NAME, "lr": lr, "ft_val_accuracy": val_acc}
    blob = base64.b64encode(zlib.compress(pickle.dumps(manifest), level=9)).decode("ascii")
    out = ROOT / "tasks" / "mpnet_lr_blob.txt"
    out.write_text(blob)
    print(f"\nWrote {out.relative_to(ROOT)}  ({len(blob)//1024} KB)")
    print("Now: run scripts/_phase4_edit.py to wire the second encoder into model.py")


if __name__ == "__main__":
    main()
