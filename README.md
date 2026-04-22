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
    ├── headlines_scraped.csv           # Headlines scraped from Fox/NBC URLs provided in        url_only_data.csv. This dataset is used to run the evaluation scrip
    ├── gnews_headlines_full.csv        # Headlines scraped from Google News RSS Feed with Fox News and NBC News. This file is used to train the data
    ├── collect_urls.py                 # Script to collect url from Google News RSS Feed with Fox News and NBC News. Output is gnews_headlines_full.csv
    └── urls_data_only.csv              # Provided dataset

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
python model.py --data data/gnews_headlines_full.csv
```

## 4. Running the Eval Script

```bash
cd /home/ubuntu/project-b-news-classifier

python eval_project_b.py \
    --model      model.py \
    --preprocess preprocess.py \
    --csv        data/headlines_scraped.csv
```

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--weights` | *(none)* | Path to a `.pt` checkpoint to load into a PyTorch model |
| `--batch-size` | `32` | Batch size for inference |
