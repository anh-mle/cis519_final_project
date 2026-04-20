# Project B — News Classifier

## 1. Project Structure

```
project-b-news-classifier/
├── model.py              # NewsClassifier class — TF-IDF + Logistic Regression pipeline (TODO: super vanilla model right now)
├── preprocess.py         # Data preprocessing module.
├── eval_project_b.py     
├── model.joblib          # Serialised trained pipeline (TF-IDF vectoriser + classifier).
├── requirements.txt      # Python dependencies (scikit-learn, torch, pandas, etc.)
│
└── data/
    ├── headlines_sample.csv   # Dummy data for training
    ├── headlines_scraped.csv  # Headlines scraped from Fox/NBC URLs.
    ├── val_sample.csv         # Held-out validation split - need to update this to run against the valuation script
from a list of URLs (TODO: Update this script)
```

**Label convention:** `0` = Fox News, `1` = NBC News.

## 2. Setup

```bash
cd /home/ubuntu
source venv/bin/activate
```

## 3. Training the Model

Run from the `project-b-news-classifier/` directory:

```bash
cd /home/ubuntu/project-b-news-classifier
python model.py --data data/headlines_scraped.csv
```

- Reads the CSV, splits 80/20 train/val, fits the TF-IDF + Logistic Regression pipeline.
- Prints validation accuracy and a full classification report.
- Saves the trained pipeline to `model.joblib` in the same directory.

To train on the dummy sample dataset instead:

```bash
python model.py --data data/headlines_sample.csv
```

## 4. Running the Eval Script

```bash
cd /home/ubuntu/project-b-news-classifier

python eval_project_b.py \
    --model      model.py \
    --preprocess preprocess.py \
    --csv        data/val_sample.csv
```

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--weights` | *(none)* | Path to a `.pt` checkpoint to load into a PyTorch model |
| `--batch-size` | `32` | Batch size for inference |
