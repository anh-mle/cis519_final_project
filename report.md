# Project B — News Source Classifier: Training Report

Binary classification of news headlines into **Fox News** or **NBC News** using a
from-scratch Bidirectional LSTM trained on `data/headlines_scraped.csv`
(3,804 total headlines; 3,233 train / 571 validation after stratified split).

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
| `seq_len` | 64 |
| `drop_p` | 0.3 |
| `n_epochs` | 10 |
| Label smoothing | none |
| Early stopping | none |

### Training Grid
Six runs covering Adam and AdamW at learning rates 5e-4, 1e-3, 2e-3.

### Results
| Split | Accuracy |
|---|---|
| Local validation | **80%** |
| New (unseen) dataset | **68%** |

### Problem Identified
A 12-point gap between local and unseen accuracy is a clear sign of **overfitting**.
The training log confirmed it — training accuracy reached 100% by epoch 5 while
validation loss continued to rise. Only the last hidden state of the LSTM was used,
discarding all contextual information from the middle of each headline.

---

## Round 1 — Tackling Overfitting

### Changes Made

| Change | Reason |
|---|---|
| Attention pooling over all hidden states | The last hidden state discards most of the sequence; attention lets the model weight every token and focus on the most informative words wherever they appear |
| Embedding dropout (`nn.Dropout`) | Prevents the model from memorising specific word-vector patterns that do not generalise |
| LayerNorm after attention output | Stabilises the attended representation; reduces sensitivity to distribution shift between training and test data |
| `drop_p` 0.3 → 0.45 | Training was reaching 100% accuracy — more aggressive dropout was needed to close the train/val gap |
| `emb_dim` 256 → 128 | The embedding table had ~500K parameters for a 4K vocabulary. Reducing it removes a large source of overfitting with minimal loss of expressiveness |
| Label smoothing 0.1 | Prevents the model from making overconfident predictions; softer decision boundaries transfer better to unseen data |
| Early stopping (patience = 5) | Saves the best-validation checkpoint and halts training before the model memorises training noise |
| `seq_len` 64 → 80 | A small number of headlines were being truncated; capturing a few extra tokens improves recall of tail context |
| `n_epochs` 10 → 25 | Gives early stopping enough budget to find the true optimum |
| GRID shifted to AdamW-focused runs | AdamW's built-in weight decay provides an additional regularisation signal on top of dropout |

### Results
| Split | Accuracy |
|---|---|
| Local validation | ~82% |
| New (unseen) dataset | **81%** |

The train/val gap collapsed — overfitting was resolved. The 13-point improvement on
unseen data confirmed the diagnosis was correct.

---

## Round 2 — Richer Representations

### Diagnosis
With overfitting resolved, training accuracy no longer reached 100% and the
train/val gap was small. The model was now limited by its **representational capacity**,
not regularisation. A single attention head over the hidden states was not capturing
everything discriminative in the headline.

### Changes Made

| Change | Reason |
|---|---|
| Tokeniser: add `?` and `!` as tokens | Fox News frequently uses rhetorical questions; NBC uses exclamation marks differently. These punctuation signals were previously invisible to the model |
| `SpatialDropout` on embeddings (replaces `nn.Dropout`) | Regular dropout masks random scalar values. Spatial dropout drops entire embedding channels across all time-steps, forcing the LSTM not to rely on any single word feature — a much stronger inductive bias for sequences |
| **Triple pooling**: attention + max + mean (concatenated) | Each pooling mode captures a different aspect: attention captures learned token importance, max-pooling captures the single most discriminative feature per channel, mean-pooling captures overall headline tone. Concatenating all three gives the classifier 3× richer input (6H instead of 2H) |
| Two-layer MLP classifier (Linear → GELU → Dropout → Linear) | With 6H features entering the classifier, a nonlinear combination layer allows the model to learn interactions between the three pooling modes that a single linear layer cannot |
| `drop_p` 0.45 → 0.40 | Spatial dropout is a stronger regulariser than regular dropout, so the scalar dropout rate could be slightly reduced to compensate |

### Results
| Split | Accuracy |
|---|---|
| Local validation | **81.79%** |
| New (unseen) dataset | **81%** |

Accuracy on unseen data remained at 81%. The model was now generalising well
(no gap between local and unseen accuracy), meaning the bottleneck had shifted
entirely to **model capacity and vocabulary coverage**.

---

## Round 3 — Ensemble and Better Attention

### Diagnosis
Two observations from the Round 2 grid results pointed to the same root cause:

1. **Plain Adam with zero weight decay kept winning** — AdamW's regularisation was no
   longer helping because the model was no longer overfitting. Adding more regularisation
   would only hurt.
2. **Vocabulary coverage was only 45.4%** — more than half of unique word types in the
   corpus were assigned `<UNK>` due to `min_freq=2`. This means the model could not
   distinguish many words that might be strong discriminators between the two sources.

With only ~3,200 training examples, a single model trained on one random seed carries
high **variance** — different initialisations produce slightly different decision boundaries,
and any one of them may be suboptimal.

### Changes Made

| Change | Reason |
|---|---|
| `label_smoothing` 0.1 → 0.05 | Adam with no weight decay was winning, indicating the model was no longer prone to overconfident predictions. Smoothing too hard was capping the confidence on correctly-learned patterns and slightly underfitting |
| GRID rebalanced toward Adam | Prior runs consistently showed Adam at lr=1e-3 outperforming AdamW; the grid now allocates more budget to exploring Adam learning rates |
| **Bahdanau attention** (tanh + 2-layer projection) | Replaces the previous single linear scorer with `v(tanh(W·h))`. The tanh non-linearity allows the attention to learn non-linear relevance patterns — e.g. scoring "crisis" highly for Fox and "declared" highly for NBC — which a linear scorer cannot express |
| **Multi-seed ensemble** (5 seeds, best config) | Trains the best grid configuration five times with different random seeds (42, 123, 456, 789, 2024) and averages their softmax probabilities at inference. Averaging cancels out errors that are idiosyncratic to individual initialisations, consistently yielding +1–3% over a single model on small datasets |

### Architecture (final)
```
Input headline (tokenised, max 80 tokens)
        │
        ▼
Embedding (vocab × 128)
SpatialDropout(p=0.40)
        │  B × T × 128
        ▼
Bidirectional LSTM — 2 layers, hidden=256, dropout=0.40
        │  B × T × 512
        ▼
Triple Pooling (concatenated)
  ├─ Bahdanau attention  →  512   (tanh(W·h) → v, learned importance)
  ├─ Max pooling         →  512   (most discriminative feature per channel)
  └─ Mean pooling        →  512   (overall headline tone)
        │  B × 1536
        ▼
MLP Classifier
  LayerNorm(1536) → Linear(1536→256) → GELU → Dropout → Linear(256→2)
        │
        ▼
  logits [foxnews, nbcnews]
```

---

## Summary of Progress

| Version | Key Change | New Dataset Accuracy |
|---|---|---|
| Baseline | Last-state BiLSTM, no regularisation | 68% |
| Round 1 | Attention pooling + dropout + label smoothing + early stopping | 81% |
| Round 2 | Spatial dropout + triple pooling + MLP head + `?`/`!` tokens | 81% |
| Round 3 | Bahdanau attention + ensemble (5 seeds) | TBD after rerun |

---

## Known Ceiling and Next Steps

The practical accuracy ceiling for a **from-scratch BiLSTM** on ~3,200 headlines is
approximately **82–85%** (single model) / **84–87%** (ensemble). The primary bottleneck
is vocabulary coverage: 45% of unique word types map to `<UNK>`, meaning the model
cannot use the majority of the lexical signal present in the headlines.

To push beyond this ceiling while staying within the BiLSTM framework, the most
impactful remaining change is **pre-trained word embeddings** (GloVe 100d or FastText).
This would initialise `self.emb` with vectors trained on billions of tokens, giving the
model meaningful representations for rare and out-of-vocabulary words rather than
collapsing them all to a single `<UNK>` vector. This is an initialisation change only —
the architecture remains a BiLSTM.
