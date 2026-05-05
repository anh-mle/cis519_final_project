# Project B — News Source Classifier: Training Report

Binary classification of news URL slugs into **Fox News** or **NBC News** using a
from-scratch Bidirectional LSTM trained on `data/url_8k_output.csv`
(8,000 total samples; 6,800 train / 1,200 validation after stratified split).

---

## Baseline Model

### Architecture
```
Embedding (vocab × 256) → Bi-LSTM (2 layers, hid=256) → last hidden state → Linear(512 → 2)
```

### Configuration
| Parameter | Value |
|---|---|
| `emb_dim` | 256 |
| `hid_dim` | 256 |
| `n_layers` | 2 |
| `seq_len` | 64 |
| `drop_p` | 0.3 |
| `n_epochs` | 10 |
| Label smoothing | none |
| Early stopping | none |

### Results
| Split | Accuracy |
|---|---|
| Local validation | ~80% |
| New (unseen) dataset | 68% |

### Issues Observed
- **Overfitting** — 12-point gap between local and unseen accuracy. Training accuracy hit 100% by epoch 5 while validation loss kept rising.
- **Wasted context** — only the last hidden state was passed to the classifier, discarding all information from the middle of the sequence.
- **Oversized embedding** — `emb_dim=256` for a ~4K vocab created ~1M embedding parameters, a large source of overfitting relative to the dataset size.
- **No regularisation** — no dropout, no label smoothing, no early stopping; nothing to prevent the model from memorising the training set.

---

## Progress Log

| # | Change | Val Accuracy | Note |
|---|---|---|---|
| 0 | Baseline | ~80% local / 68% unseen | Severe overfitting |
| 1 | Attention pooling over all hidden states | ~82% / 81% | Overfitting resolved; train/val gap closed |
| 2 | `emb_dim` 256 → 128, `drop_p` 0.3 → 0.45, label smoothing 0.1, early stopping | ~82% / 81% | Regularisation holding; bottleneck shifts to capacity |
| 3 | `SpatialDropout` on embeddings, triple pooling (attn + max + mean), MLP head, Bahdanau attention | ~82% / 81% | Richer features but no accuracy gain — data is the ceiling |
| 4 | Data: raw headlines → URL slugs (`url_8k_output.csv`), 3,804 → 8,000 samples | **90.83%** | Major jump; cleaner vocabulary and more data broke the ceiling |
| 5 | `n_layers` 2 → 3 | — | Extra LSTM layer; not yet retrained |

---

## Current Architecture

```
Input: URL slug  (tokenised · max_len = 80)
             │
             ▼
┌─────────────────────────────────────────┐
│  Embedding        vocab × 128           │
│  SpatialDropout   p = 0.40              │
└─────────────────────────────────────────┘
             │  [B, T, 128]
             ▼
┌─────────────────────────────────────────┐
│  Bidirectional LSTM                     │
│    layers = 3  ·  hidden = 256          │
│    dropout = 0.40  (between layers)     │
└─────────────────────────────────────────┘
             │  [B, T, 512]
             ▼
┌─────────────────────────────────────────┐
│  Mean Pooling  (over non-pad positions) │
└─────────────────────────────────────────┘
             │  [B, 512]
             ▼
┌─────────────────────────────────────────┐
│  Linear(512 → 2)                        │
└─────────────────────────────────────────┘
             │
             ▼
       logits  [fox, nbc]
```

### Current Configuration
| Parameter | Value |
|---|---|
| `emb_dim` | 128 |
| `hid_dim` | 256 |
| `n_layers` | 3 |
| `seq_len` | 80 |
| `drop_p` | 0.40 |
| `n_epochs` | 25 |
| Label smoothing | 0.05 |
| Early stopping | patience = 5 |
