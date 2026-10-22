"""Provence (Chirkova, Formal, Nikoulina & Clinchant, "Provence: efficient
and robust context pruning for retrieval-augmented generation", ICLR 2025)
as a compressor for this pipeline.

Provence is a DeBERTa-v3-large cross-encoder with a per-token keep/drop
head. It reads "[CLS] question [SEP] context [SEP]" (at most 512 tokens)
and a sentence is kept when the share of its tokens predicted "keep"
(keep probability > threshold) is itself above the threshold. The
released checkpoint is naver/provence-reranker-debertav3-v1.

The weights are loaded here with plain transformers instead of the
checkpoint's trust_remote_code file, which targets transformers 4.x and
splits sentences with nltk; this pipeline already has its own units.

How the conversation is fed to it (the paper is about retrieved passages):
- units are packed in order into 512-token windows, never across a
  session boundary, each after the normalised question exactly as
  Provence's own preprocessing does;
- each turn's first unit is preceded by its role label ("User:"), which is
  context for the model but not part of any unit's score; a window that
  starts mid-turn repeats the label;
- a unit too long for one window is split, and its pieces' tokens pooled.

Two ways to use it:
- budgeted (threshold=None): a unit's score is the mean keep probability
  of its tokens, and the shared fusion/selection fills the token budget,
  exactly as for CPC;
- native (a threshold): Provence's own decision, so it keeps as much as
  it sees fit and there is no token budget, as in the paper.
"""

import json
import string
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from core.units import Unit, role_label

DEFAULT_MODEL = "naver/provence-reranker-debertav3-v1"
MAX_LEN = 512
CLS_ID, SEP_ID, PAD_ID = 1, 2, 0  # DeBERTa-v3 ids, hard-coded in Provence too


def normalize_query(text: str) -> str:
    """Provence's training questions were lower-cased, punctuation-free."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return " ".join(text.split())


@dataclass
class Window:
    ids: List[int]
    spans: List[Tuple[int, int, int]] = field(default_factory=list)  # (unit index, start, end)


def pack_windows(units: Sequence[Unit], query_ids: Sequence[int], encode: Callable[[str], List[int]],
                 max_len: int = MAX_LEN) -> List[Window]:
    head = [CLS_ID, *query_ids, SEP_ID]
    room = max_len - len(head) - 1  # -1 for the closing [SEP]
    windows: List[Window] = []
    body: List[int] = []
    spans: List[Tuple[int, int, int]] = []

    def flush():
        nonlocal body, spans
        if body:
            windows.append(Window([*head, *body, SEP_ID], spans))
        body, spans = [], []

    session = object()
    previous_entry = None
    for index, unit in enumerate(units):
        if unit.session != session:
            flush()
            session = unit.session
        new_turn = unit.entry_index != previous_entry
        previous_entry = unit.entry_index
        label = encode(" " + role_label(unit))
        tokens = encode(" " + unit.text)
        size = max(1, room - len(label))
        for start in range(0, len(tokens), size):
            piece = tokens[start:start + size]
            if body and len(body) + len(label) + len(piece) > room:
                flush()
            if new_turn or not body:
                body += label
            new_turn = False
            offset = len(head) + len(body)
            body += piece
            spans.append((index, offset, offset + len(piece)))
    flush()
    return windows


def unit_scores(windows: Sequence[Window], probs: Sequence[Sequence[float]], n_units: int,
                threshold: Optional[float] = None) -> List[float]:
    """Per unit: mean keep probability of its tokens, or, with a threshold,
    the share of its tokens whose keep probability exceeds it (Provence's
    sentence rounding)."""
    sums = [0.0] * n_units
    counts = [0] * n_units
    for window, window_probs in zip(windows, probs):
        for index, start, end in window.spans:
            values = window_probs[start:end]
            sums[index] += sum(1.0 for p in values if p > threshold) if threshold is not None else sum(values)
            counts[index] += end - start
    return [total / count if count else 0.0 for total, count in zip(sums, counts)]


class ProvenceModel:
    """DeBERTa-v3 encoder + Provence's token keep/drop head."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None, batch_size: int = 32) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        with open(hf_hub_download(model_name, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        for key in ("model_type", "architectures", "auto_map", "_name_or_path", "transformers_version", "torch_dtype"):
            config.pop(key, None)
        config = DebertaV2Config(**config)

        state = load_file(hf_hub_download(model_name, "model.safetensors"))
        self.encoder = DebertaV2Model(config)
        self.encoder.load_state_dict({key[len("deberta."):]: value for key, value in state.items()
                                      if key.startswith("deberta.")})
        self.head = torch.nn.Linear(config.hidden_size, 2)
        self.head.load_state_dict({"weight": state["token_classifier.weight"], "bias": state["token_classifier.bias"]})
        self.encoder.eval().to(self.device)
        self.head.eval().to(self.device)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def keep_probs(self, sequences: Sequence[Sequence[int]]) -> List[List[float]]:
        torch = self._torch
        probs: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                width = max(len(ids) for ids in batch)
                ids = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
                for row, sequence in enumerate(batch):
                    ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                ids = ids.to(self.device)
                hidden = self.encoder(input_ids=ids, attention_mask=(ids != PAD_ID).long()).last_hidden_state
                keep = torch.softmax(self.head(hidden).float(), dim=-1)[:, :, 1].cpu().tolist()
                probs.extend(row[:len(sequence)] for row, sequence in zip(keep, batch))
        return probs


class ProvenceCompressor:
    """Scores units with Provence's keep probabilities; TokenWise fills the
    token budget with them like any other `score_units` scorer."""

    def __init__(self, model: str = DEFAULT_MODEL, threshold: Optional[float] = None,
                 model_factory: Optional[Callable[[], object]] = None) -> None:
        self.model_name = model
        self.threshold = threshold
        self._model_factory = model_factory or (lambda: ProvenceModel(model))
        self._model = None

    def _scores(self, units: Sequence[Unit], query: str, threshold: Optional[float]) -> List[float]:
        if self._model is None:
            self._model = self._model_factory()
        query_ids = self._model.encode(normalize_query(query))[:MAX_LEN // 2]
        windows = pack_windows(units, query_ids, self._model.encode)
        probs = self._model.keep_probs([window.ids for window in windows]) if windows else []
        return unit_scores(windows, probs, len(units), threshold)

    def score_units(self, units: Sequence[Unit], queries: Sequence[str]) -> List[List[float]]:
        return [self._scores(units, query, None) for query in queries]


class ProvenceSelector(ProvenceCompressor):
    """Provence's native decision at `threshold`: keeps every unit whose
    share of kept tokens exceeds it. TokenWise calls `select_units`, so
    there is no token budget; the result may be empty, as in the paper."""

    def select_units(self, units: Sequence[Unit], query: str) -> List[int]:
        shares = self._scores(units, query, self.threshold)
        return [index for index, share in enumerate(shares) if share > self.threshold]
