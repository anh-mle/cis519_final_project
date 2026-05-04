import os
import re
from collections import Counter
from typing import Any, Iterable, List

import torch
import torch.nn as nn

# ── Hyper-params ────────────────────────────
_SEQ_LEN  = 80
_EMB_DIM  = 128
_HID_DIM  = 256
_N_LAYERS = 2
_DROP_P   = 0.40
_N_CLS    = 2

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_HERE, 'model.pt')


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, corpus, cap: int, min_freq: int):
        freq = Counter(tok for sent in corpus for tok in self._lex(sent))
        self._w2i = {'<pad>': self.PAD, '<unk>': self.UNK}
        for w, c in freq.most_common(cap - 2):
            if c < min_freq:
                break
            self._w2i[w] = len(self._w2i)
        self.freq = freq

    @staticmethod
    def _lex(text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+|[?!]", text.lower())

    def encode(self, text: str, maxlen: int):
        toks = self._lex(text)[:maxlen]
        ids  = [self._w2i.get(t, self.UNK) for t in toks]
        n    = max(len(ids), 1)
        ids += [self.PAD] * (maxlen - len(ids))
        return ids, n

    def __len__(self) -> int:
        return len(self._w2i)


# ── Model components ───────────────────────────────────────────────────────────

class _SpatialDropout(nn.Module):
    """Drops entire embedding channels (same mask across all time-steps)."""
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class _Attention(nn.Module):
    """Bahdanau (tanh) attention over BiLSTM hidden states."""
    def __init__(self, hid: int):
        super().__init__()
        self.W = nn.Linear(hid * 2, hid, bias=False)
        self.v = nn.Linear(hid, 1,       bias=False)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)
        scores = scores.masked_fill(x == Vocab.PAD, -1e9)
        w = torch.softmax(scores, dim=-1)
        return (w.unsqueeze(-1) * h).sum(1)


# ── BiLSTM ─────────────────────────────────────────────────────────────────────

class _BiLSTM(nn.Module):
    """Embedding → SpatialDrop → Bi-LSTM → (attn + max + mean) pool → MLP → logits."""

    def __init__(self, vsz: int, emb: int, hid: int, layers: int, drop: float, n_cls: int):
        super().__init__()
        self.emb      = nn.Embedding(vsz, emb, padding_idx=Vocab.PAD)
        self.emb_drop = _SpatialDropout(drop)
        self.rnn      = nn.LSTM(emb, hid, layers,
                                batch_first=True, bidirectional=True,
                                dropout=drop if layers > 1 else 0.)
        self.attn     = _Attention(hid)
        feat = hid * 6
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(drop),
            nn.Linear(feat, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, n_cls),
        )

    def forward(self, x: torch.Tensor, sl: torch.Tensor) -> torch.Tensor:
        e      = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, sl.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))

        pad_mask = (x == Vocab.PAD)
        z_attn = self.attn(out, x)
        z_max  = out.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(1).values
        lens   = sl.float().clamp(min=1).unsqueeze(-1).to(out.device)
        z_mean = out.masked_fill(pad_mask.unsqueeze(-1), 0.).sum(1) / lens

        return self.classifier(torch.cat([z_attn, z_max, z_mean], dim=-1))


# ── Vocab builder ──────────────────────────────────────────────────────────────

def _load_vocab() -> Vocab:
    ckpt = torch.load(_CKPT, map_location='cpu')
    v = Vocab.__new__(Vocab)
    v._w2i = ckpt['vocab']
    v.freq = Counter()
    return v


# ── Model ───────────────────────────────────────────────────────────────

class Model(nn.Module):

    ID_TO_LABEL = {0: 'foxnews', 1: 'nbcnews'}

    def __init__(self, weights_path: str = None) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab  = _load_vocab()
        self.model  = _BiLSTM(len(self.vocab), _EMB_DIM, _HID_DIM, _N_LAYERS, _DROP_P, _N_CLS)
        self.to(self.device)
        ckpt = torch.load(_CKPT, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])

    def forward(self, x: torch.Tensor, sl: torch.Tensor) -> torch.Tensor:
        return self.model(x, sl)

    def predict(self, batch: Iterable[Any]) -> List[str]:
        texts = list(batch)
        if not texts:
            return []

        ids_list, lens = [], []
        for text in texts:
            ids, n = self.vocab.encode(str(text), _SEQ_LEN)
            ids_list.append(ids)
            lens.append(n)

        x  = torch.tensor(ids_list, dtype=torch.long).to(self.device)
        sl = torch.tensor(lens,     dtype=torch.long)

        with torch.no_grad():
            logits = self.model(x, sl)
        preds = logits.argmax(-1).cpu().tolist()
        return [self.ID_TO_LABEL[p] for p in preds]


def get_model() -> Model:
    return Model()

