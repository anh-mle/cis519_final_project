import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

class NewsClassifier:
    @staticmethod
    def _build_pipeline() -> Pipeline:
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    max_features=10000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.7,
                )),
                ("char", TfidfVectorizer(
                    max_features=10000,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=5,
                )),
            ])),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])

    def __init__(self, weights_path: str | None = None) -> None:
        self.pipeline = self._build_pipeline()
        self._trained = False

        sentinel = "__no_weights__.pth"
        load_path = None
        if weights_path and weights_path != sentinel:
            load_path = weights_path
        elif os.path.exists(_DEFAULT_WEIGHTS):
            load_path = _DEFAULT_WEIGHTS

        if load_path:
            self._load(load_path)

    def _load(self, path: str) -> None:
        try:
            self.pipeline = joblib.load(path)
            self._trained = True
        except Exception as exc:
            print(f"[NewsClassifier] Warning: could not load weights from {path!r}: {exc}")

    def eval(self) -> None:
        pass

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = list(batch)
        if not self._trained:
            return [0] * len(texts)
        return [int(p) for p in self.pipeline.predict(texts)]

    def train(self, data_path: str) -> None:
        sys.path.insert(0, str(Path(__file__).parent))
        from preprocess import prepare_data

        print(f"Loading data from {data_path!r}...")
        X, y = prepare_data(data_path)
        print(f"  Total examples : {len(X)}")
        print(f"  Fox News (0)   : {y.count(0)}")
        print(f"  NBC News (1)   : {y.count(1)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"  Train: {len(X_train)}   Val: {len(X_val)}")

        self.pipeline = self._build_pipeline()
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
        print("\nRunning GridSearchCV (3-fold, 4 combinations)...")
        grid = GridSearchCV(
            self.pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1, refit=True
        )
        grid.fit(X_train, y_train)
        print(f"Best params : {grid.best_params_}")
        print(f"Best CV acc : {grid.best_score_:.4f}")
        self.pipeline = grid.best_estimator_
        self._trained = True

        y_pred = self.predict(X_val)
        val_acc = sum(p == t for p, t in zip(y_pred, y_val)) / len(y_val)
        print(f"\nVal accuracy: {val_acc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_val, y_pred, target_names=["Fox News", "NBC News"]))

        output_path = Path(__file__).parent / "model.joblib"
        joblib.dump(self.pipeline, output_path)
        print(f"Model saved to {output_path!r}")


def get_model() -> NewsClassifier:
    return NewsClassifier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gnews_headlines_full.csv")
    args = parser.parse_args()

    classifier = NewsClassifier(weights_path="__no_weights__.pth")
    classifier.train(args.data)
