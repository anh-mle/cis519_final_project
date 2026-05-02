# Keeping notes for writing report 

## Current Best result

- HF leaderboard accuracy: **0.7830** 
- Final architecture: 3-way probability-averaging ensemble of 
    (a) TF-IDF stacking classifier with hand-crafted features, 
    (b) frozen MiniLM-L6 + LR head
    (c) frozen MPNet-base + LR head, weighted **0.30 / 0.25 / 0.45**.


## Data Scraping

Assembled training data from **3 pipelines**:

### 1. Staff-provided URLs -> publisher-direct headlines (`headlines_scraped.csv`, 3,804 rows)

- **Source:** `data/url_only_data.csv`
- **Method:** `preprocess.py`'s URL-resolution waterfall:
- **Output:** 3,804 of 3,815 URLs successfully resolved (99.7% hit rate).

### 2. Google News RSS scrape → broad-coverage headlines (`gnews_headlines_full.csv`, 19,518 rows)

- **Source:** Google News RSS feed. Added during Anh's data collection + aggregation
- **Method (in `data/collect_urls.py`):** for each of ~150 topic keywords (politics, sports, climate, tech, health, etc.) and each of {`foxnews.com`, `nbcnews.com`}:
- **Output:** 19,518 unique headlines, distributed roughly **9,101 Fox / 10,417 NBC**. URL column is empty (Google News RSS does not expose direct article URLs — only Google's own redirect IDs).
- **Distribution caveat:** these headlines are **Google-News-rephrased**. Google News sometimes rewrites publisher headlines for clarity, so this corpus has a slight phrasing mismatch with the leaderboard's hidden test set (which is direct from publisher pages).

### 3. Fox News sitemap scrape → publisher-direct supplement (`fox_direct.csv`, 2,828 rows)

This source was added in Soojin's iteration of the project.

- **Source:** Fox News's public sitemap at `https://www.foxnews.com/sitemap.xml`.
- **Method (in `scripts/scrape_fox_sitemap.py`)**
- **Output:** 2,828 publisher-direct Fox headlines in 10 minutes. All label=0 (Fox).
- **Asymmetry caveat:** NBC News exposes no public sitemap (`/sitemap.xml`, `/sitemap_index.xml`, `/sitemap-news.xml` all return 404). NBC publisher-direct supplementation is left for future iterations.


## Data Processing (in steps)

All cleaning happens via `preprocess.prepare_data(csv_path) → (X: List[str], y: List[int])`. 

**Step 1 — Schema detection.** Accept three CSV variants:
- `(url, headline, label)` — fully labeled (e.g. `headlines_scraped.csv`, `fox_direct.csv`).
- `(url, headline)` — label inferred from URL domain (`"foxnews.com" in url → 0` else `1`).
- `(url)` only — headlines scraped via the URL-resolution waterfall described above.

**Step 2 — Drop NaN headlines.** `df.dropna(subset=["headline"])`.

**Step 3 — Strip whitespace.** Leading/trailing whitespace removed with `str.strip()`.

**Step 4 — Strip publisher branding suffix.** Regex `\s*[|\-–—]\s*(fox news|nbc news|reuters|ap|associated press).*$` (case-insensitive). This removes patterns like `"... | Fox News"` or `"... — NBC News"` that would otherwise be a trivial leak. **Important:** the regex is the most load-bearing line in `preprocess.py` — without it, a model would learn "ends with `| fox news`" instead of headline content.

**Step 5 — Collapse multiple spaces.** `re.sub(r"\s+", " ")`.

**Step 6 — Lowercase.** `text.lower()`.

**Step 7 — Length filter.** Drop rows where cleaned headline length < 10 characters. This removes slug-fallback artifacts like `"live blog 52541"` and pages where the publisher returned only a category name.

**Step 8 — Label encoding.** 0 = Fox News, 1 = NBC News.

### Validation diagnostics ran during cleaning

- **Average headline length per source:** train (gnews) 71.7 chars, eval (headlines_scraped) 80.6 chars — confirmed a real distribution shift between training and evaluation distributions (driver of fine-tuning failures, see below).
- **Class balance:** gnews 47/53 (Fox/NBC), headlines_scraped 53/47, ft_train 51.7/48.3, ft_val 50.5/49.5 — well-balanced across all sources.
- **Train/eval overlap (after `_clean()`):** 78 headlines (~0.4%) appeared in both gnews and headlines_scraped after cleaning. Likely the same news event covered in both pipelines. Removing them dropped accuracy by 0.0035 — minor leakage, not a model issue.
- **Source-suffix leak audit:** ran the publisher-suffix regex against all 3,804 raw eval headlines — matched 0/3,804. The eval set's headlines come from `<title>` tags that the publisher already strips, so the regex isn't the source of our model's discriminative signal. **The model genuinely learns content, not a brand-suffix shortcut.** 77 cleaned eval headlines still contain "fox news" / "nbc news" as content tokens (e.g. *"Fox News Power Rankings"*)


## Splits

`headlines_scraped.csv` (3,804) was split deterministically (NumPy seed=42):

```
headlines_scraped.csv (3,804 publisher-distribution)
  ├── 50/50 split (random shuffle, seed=42)
  │
  ├── dev_split.csv  (1,902)
  │     ├── ft_train.csv (1,500)  → LR-head training (×3 upsampled)
  │     └── ft_val.csv   (  400)  → held-out validation
  │
  └── final_split.csv (1,902)     → reserved untouched test
```

For training the final model,
- gnews_headlines_full.csv (19,518, sample_weight=1.0)
- ft_train.csv × 3 (4,500 effective, sample_weight=1.0)
- fox_direct.csv (2,828, sample_weight=0.3 to mitigate Fox-only asymmetry)


## Records of Model Improvements

| Phase | Architecture / Change | HF Leaderboard | Local ft_val | Δ vs prior | Why this helped |
|---|---|---|---|---|---|
| **Baseline** | TF-IDF (max_features=100) + `LogisticRegression(max_iter=100)`, 80/20 split | 0.6649 | — | — | Spec-provided starting point |
| **Step 2** | Stacking ensemble: FeatureUnion of word TF-IDF (max_features=20k, ngram=(1,3), sublinear_tf) + char TF-IDF (analyzer=char_wb, ngram=(2,5), sublinear_tf) → StackingClassifier with Bagging(LinearSVC) + ComplementNB base learners, LogisticRegression meta-learner. Plus 7 hand-crafted scalar features (length, word count, digit count, punct density, has-colon, has-question, has-em-dash). RandomizedSearchCV-tuned over 11 hyperparameters. | 0.6742 | 0.6533 (full eval) | +0.0093 | More expressive features but still bag-of-words; minor lift over baseline |
| **Phase 3.A** | + frozen `sentence-transformers/all-MiniLM-L6-v2` (384-dim) → mean-pool + L2-norm → `LogisticRegression(C=1.0, class_weight='balanced')`. Probability-averaged with stack at 0.20 / 0.80. | **0.7755** | 0.7125 | **+0.1013** | Pretrained semantic embeddings replace bag-of-words; biggest single jump in the project |
| **Phase 4** | + frozen `sentence-transformers/all-mpnet-base-v2` (768-dim) → second LR head. 3-way ensemble at 0.25 / 0.35 / 0.40 (stack / MiniLM-LR / MPNet-LR). | 0.7814 | 0.7575 | +0.0059 | Two encoders with different architectures and pretraining recipes produce decorrelated errors (157/400 ft_val examples disagreed); ensembling reduces variance |
| **Phase 4.C** *(final)* | + `fox_direct.csv` (2,828 publisher-direct Fox headlines) added to LR training with sample_weight=0.3. Re-tuned ensemble weights to **0.30 / 0.25 / 0.45**. | **0.7830** | 0.7600 | +0.0016 | Fox-side publisher data shifts the LR boundary slightly toward publisher distribution; modest because the 2,828 rows are diluted by 19k gnews + 4.5k upsampled ft_train |


## Experiments that Didn't go through

Tested **four fine-tuning recipes** to see if updating encoder weights would beat the frozen-encoder baseline. **None of them did** 

| Recipe | Configuration | Local ft_val | HF Leaderboard | Δ vs Phase 3.A frozen | Why it failed |
|---|---|---|---|---|---|
| **Phase 3.B aggressive** | Full FT on `MiniLM-L6` for 4 epochs, lr=2e-5, weight_decay=0.01, **`ft_train` upsampled 3×** (so 4,500 effective rows of publisher-distribution data competing with 19k gnews), `metric_for_best_model='accuracy'` | **0.7925** (highest local) | 0.7353 | **−4.0 pts** | The 3× upsampling overweighted ft_train's 1,500 specific examples in the loss. Model memorized those phrasings → very high local accuracy on ft_val (which is from the same source) but generalization broke on HF's hidden test (different URLs, different time period). Classic overfitting signature. |
| **Phase 3.B conservative** *(not submitted to HF)* | Full FT, 2 epochs, lr=1e-5 (gentler), weight_decay=0.05 (5× stronger regularization), no upsampling, `metric_for_best_model='eval_loss'` | 0.7275 | — | — | Underfit. Too many knobs dialed gentler at once; the model couldn't even reach Phase 3.A's frozen-encoder accuracy. |
| **Phase 3.B middle** | Full FT, 4 epochs, lr=2e-5, weight_decay=0.01, **no upsampling** (only difference from aggressive). `metric_for_best_model='eval_loss'`. | 0.7650 | **0.6574** *(below the spec baseline of 0.6649!)* | **−12.0 pts** | Without upsampling, gnews dominates training (19k of ~21k rows). Encoder weights drifted to fit Google-News-rephrased patterns, **losing generalization in the opposite direction** from aggressive. The 12-point loss is the largest single regression we observed. |
| **Phase 4.D LoRA** | LoRA adapter on Q/V projections, rank=8, alpha=16, dropout=0.1 → ~70K trainable parameters (~0.3% of MiniLM's 22M). lr=2e-4, 3 epochs. After training, `model.merge_and_unload()` to fold LoRA back into base weights → standard `BertForSequenceClassification` checkpoint pushed to HF Hub. | 0.7125 | 0.7772 | **−0.58 pts** | The gentlest possible adaptation — only 0.3% of parameters trainable, structurally limiting how much the encoder can specialize. **Still hurt HF.** This is the strongest evidence that the issue isn't overfitting magnitude but the small absolute volume of publisher-distribution training data: even a tiny gradient signal is enough to bias an encoder away from the broad pretraining the test set rewards. |


## Open items for future iteration

1. **Symmetrize publisher data on NBC side.** Maybe helpful but not sure if it's worth the effort because adding extra FOX news didn't add too much to accuracy
2. **LLM data augmentation?:** I am wondering if this is achieved by groups that scored above 0.9 but not sure if this is allowed?
