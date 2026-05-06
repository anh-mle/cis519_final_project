# News Source Classification: Fox News vs NBC News

Binary classification of news article URLs as **Fox News** or **NBC News** using headline text extracted from the URL slug. We compare a fine-tuned DistilBERT model against recurrent architectures (Bidirectional LSTM, Bidirectional GRU) and a Logistic Regression baseline.

---

## Results

| Model | Validation Accuracy | Leaderboard Accuracy | Avg Inference Time |
|---|---|---|---|
| Logistic Regression (baseline) | 84.9% | — | — |
| Bidirectional GRU | 91.76% | 87.67% | 0.28 |
| Bidirectional LSTM | 91.03% | — | — |
| **DistilBERT** | **99.92%** | **99.33%** | **12.23 ms** |

Leaderboard: [HuggingFace Spaces](https://huggingface.co/spaces/cis4190/NewsHeadlineClassifier)


---

## Repository Structure

```
├── README.md                         # Project overview and instructions
├── report.md                         # Final project report
├── model.py                          # DistilBERT classifier (final model)
├── preprocess.py                     # URL slug preprocessing + prepare_data()
├── eval_project_b.py                 # Course evaluation script
├── model.pt                          # Trained DistilBERT weights
├── training_script.ipynb             # DistilBERT training notebook
├── acc_curves.png                    # Training accuracy curves
├── loss_curves.png                   # Training loss curves
├── confusion.png                     # Confusion matrix
├── comparison.png                    # Model comparison visualization
└── data/
    ├── combined_url_balanced_8k.csv # Final dataset (8k URLs, balanced)
    └── url_only_data.csv            # Provided dataset
```

---

## Model Architecture

### DistilBERT
```
DistilBERT encoder (6 layers, 768 hidden, 12 heads) → [CLS] token → Dropout(p=0.1) → Linear(768 → 2)
```

Fine-tuned end-to-end (all encoder layers unfrozen) with AdamW (`lr=2e-5`, `weight_decay=0.01`), batch size 32, 3 epochs, linear warmup over 10% of steps, gradient clipping at max-norm 1.0. Input is the cleaned URL slug, max length 128 tokens. Training takes ~6 minutes on a Tesla T4 GPU.

### Bidirectional LSTM
```
Embedding (vocab × 128) → SpatialDropout(p=0.40) → Bidirectional LSTM — 3 layers → Mean Pooling → Linear Classifier.
```

### Bidirectional GRU
```
Tokenized URL Slugs → Embedding (vocab × 128) → SpatialDropout → BiGRU → Mean Pooling → Linear Classifier
```

### Logistic Regression (baseline)
```
TF-IDF Unigrams + TF-IDF Bigrams/Trigrams → FeatureUnion (concat) → LogisticRegression (max_iter=1000)
```

---

## Dataset

We built our own dataset by crawling the public XML sitemaps of foxnews.com and nbcnews.com — no manual labeling needed since the source domain is the label. The full pipeline is in `scripts/build_dataset.py`.

| Property | Value |
|---|---|
| Total URLs | 8,000 |
| Fox News | 4,000 |
| NBC News | 4,000 |
| Train / Val split | 85% / 15% (stratified) |
| Label source | URL domain |

**Collection steps:**
1. Walk both sitemap indexes, filtering NBC to article-month sub-sitemaps only
2. Drop non-article URLs (`/video/`, `/topic/`, `/photo/`, etc.)
3. Deduplicate by extracted slug text to prevent train/val leakage
4. Sample 4,000 per source and shuffle with a fixed seed

**Text extraction** — URL path segments are walked right-to-left; the first segment that is not an NBC article ID (`rcna…`), not purely numeric, and longer than 4 characters becomes the slug:

```
https://www.foxnews.com/world/israel-defense-forces-confirm-targeted-strike-beirut
  → "israel defense forces confirm targeted strike beirut"

https://www.nbcnews.com/tech/tech-news/esim-card-gaza-palestine-israel-war-hamas-rcna134498
  → "tech news"   (last segment skipped — contains rcna, falls back to parent)
```

This slug-only design is deterministic and requires no network calls at eval time, guaranteeing consistent behavior on the grader's infrastructure.

---

## Quick Start

```bash
pip install torch transformers pandas openpyxl scikit-learn
```

**Inference**
```python
from model import Model
import torch

model = Model()
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

print(model.predict(["senate passes major legislation", "world"]))
# ['foxnews', 'nbcnews']
```

**Rebuild dataset**
```bash
python3 scripts/build_dataset.py                   # 4k+4k = 8k URLs
python3 scripts/build_dataset.py --per-source 6000 # larger run
```

**Train** — upload `combined_url_balanced_8k.csv` to Google Colab, open `News_Classifier_Training.ipynb`, set `Config.DATA_PATH`, and run all cells. Download `model.pt` when done.

**Evaluate**
```bash
python3 eval_project_b.py \
    --model model.py --preprocess preprocess.py \
    --csv data/url_only_data.csv --weights model.pt
```

---

## References

- Sanh, V., et al. (2019). [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108). arXiv:1910.01108.
- Zhou, P., et al. (2016). [Text Classification with Bidirectional LSTM and Two-dimensional Max Pooling](https://arxiv.org/abs/1611.06639). arXiv:1611.06639.
- Fox News Sitemap: https://www.foxnews.com/sitemap.xml
- NBC News Sitemap: https://www.nbcnews.com/sitemap/nbcnews/sitemap-index
