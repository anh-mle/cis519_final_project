"""
model.py
DistilBERT model for Project B evaluator.

The evaluator will:
1. import this file
2. call get_model()
3. optionally load weights using --weights model.pth
4. call model.predict(batch)
"""

import os
import re
from typing import Any, Iterable, List

import torch
import torch.nn as nn


_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_HERE, "model.pt")


class Model(nn.Module):
    ID_TO_LABEL = {0: "foxnews", 1: "nbcnews"}

    def __init__(self, weights_path: str = None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(_CKPT, map_location=self.device)

        self.vocab = ckpt["vocab"]
        config = ckpt["config"]

        self.max_len = config["max_len"]

        self.embedding = nn.Embedding(
            len(self.vocab),
            config["embed_dim"],
            padding_idx=0
        )

        self.gru = nn.GRU(
            input_size=config["embed_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0
        )

        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(
            config["hidden_dim"] * 2,
            config["num_classes"]
        )

        self.load_state_dict(ckpt["model"])
        self.to(self.device)
        self.eval()

    def tokenize(self, text: str):
        text = str(text).lower()
        return re.findall(r"[a-z0-9']+", text)

    def encode(self, text: str):
        tokens = self.tokenize(text)

        ids = [
            self.vocab.get(token, self.vocab.get("<UNK>", 1))
            for token in tokens
        ]

        ids = ids[:self.max_len]

        while len(ids) < self.max_len:
            ids.append(self.vocab.get("<PAD>", 0))

        return ids

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, hidden = self.gru(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        combined = self.dropout(combined)

        return self.classifier(combined)

    def predict(self, batch: Iterable[Any]) -> List[str]:
        texts = [str(x) for x in batch]

        if not texts:
            return []

        encoded = [self.encode(text) for text in texts]
        input_ids = torch.tensor(encoded, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.forward(input_ids)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        return [self.ID_TO_LABEL[p] for p in preds]


def get_model():
    return Model()