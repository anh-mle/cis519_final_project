"""Phase 1 (Option B): split dev_split.csv into fine-tune train/val.

Output:
  - data/ft_train.csv  (1500 rows, publisher distribution)
  - data/ft_val.csv    ( 400 rows, publisher distribution)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "dev_split.csv"
TRAIN_OUT = ROOT / "data" / "ft_train.csv"
VAL_OUT = ROOT / "data" / "ft_val.csv"
FT_TRAIN_N = 1500
FT_VAL_N = 400


def main() -> None:
    if not SRC.exists():
        raise SystemExit(
            f"{SRC} missing — run `python scripts/diagnose.py` to regenerate."
        )

    df = pd.read_csv(SRC).dropna(subset=["headline", "url"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {SRC.relative_to(ROOT)}")
    if len(df) < FT_TRAIN_N + FT_VAL_N:
        raise SystemExit(
            f"need ≥ {FT_TRAIN_N + FT_VAL_N} rows; only have {len(df)}"
        )

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    train_idx = perm[:FT_TRAIN_N]
    val_idx = perm[FT_TRAIN_N:FT_TRAIN_N + FT_VAL_N]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    assert len(train_df) == FT_TRAIN_N
    assert len(val_df) == FT_VAL_N
    assert set(train_df["url"]).isdisjoint(set(val_df["url"]))

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)

    def show(name: str, d: pd.DataFrame) -> None:
        fox = int((d["label"] == 0).sum())
        nbc = int((d["label"] == 1).sum())
        print(f"  {name}: n={len(d)}  Fox={fox} ({fox/len(d):.1%})  "
              f"NBC={nbc} ({nbc/len(d):.1%})")

    print("\nSplits written:")
    show("ft_train", train_df)
    show("ft_val  ", val_df)
    print(f"\nWrote {TRAIN_OUT.relative_to(ROOT)} and {VAL_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
