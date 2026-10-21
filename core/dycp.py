"""DyCP (Choi, Zhang & Choi, "DyCP: Dynamic Context Pruning for Long-Form
Dialogue with LLMs", arXiv 2601.07994) as a compressor for this pipeline.

DyCP keeps whole turns, not sentences: each user message plus the reply
that follows it is one turn, embedded once with an off-the-shelf
bi-encoder (facebook/contriever-msmarco in the paper's main results).
Turns are scored by dot product with the query embedding, the scores are
z-normalised, and KadaneDial picks contiguous spans of turns whose summed
gain (z - tau) stays high, stopping once the best remaining span gains
less than theta. The amount kept is therefore decided per query; there is
no token budget.

Details the paper leaves open, and what is done here:
- turn text is the turn's units joined with spaces (after this pipeline's
  sentence split, fragment merging and de-duplication);
- embeddings use Contriever's mean pooling, unnormalised, truncated at
  512 tokens;
- a span is kept only if its gain reaches theta, except that the first
  (best) span is always kept, so the answer model always gets something.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

from core.units import Unit

DEFAULT_RETRIEVER = "facebook/contriever-msmarco"


def kadane_dial(scores: Sequence[float], tau: float = 0.6, theta: float = 1.0) -> List[Tuple[int, int]]:
    """Contiguous (start, end) index spans, inclusive, best first."""
    n = len(scores)
    if n == 0:
        return []
    mean = sum(scores) / n
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
    gains = [((s - mean) / std if std > 1e-12 else 0.0) - tau for s in scores]

    spans: List[Tuple[int, int]] = []
    while True:
        best, best_span = -math.inf, None
        running, start = -math.inf, 0
        for j, gain in enumerate(gains):
            if gain == -math.inf:
                running = -math.inf
                continue
            if running > 0:
                running += gain
            else:
                running, start = gain, j
            if running > best:
                best, best_span = running, (start, j)
        if best_span is None or (spans and best < theta):
            break
        spans.append(best_span)
        for k in range(best_span[0], best_span[1] + 1):
            gains[k] = -math.inf
        if best < theta:
            break
    return spans


def group_turns(units: Sequence[Unit]) -> List[List[int]]:
    """Unit ids grouped into DyCP turns: a user message plus every entry
    after it (assistant reply, tool results) until the next user message."""
    turns: List[List[int]] = []
    current_entry = None
    for unit in sorted(units, key=lambda unit: unit.id):
        if unit.entry_index != current_entry:
            current_entry = unit.entry_index
            if unit.role == "user" or not turns:
                turns.append([])
        turns[-1].append(unit.id)
    return turns


class ContrieverEncoder:
    """Contriever-style bi-encoder: mean-pooled last hidden states."""

    def __init__(self, model_name: str = DEFAULT_RETRIEVER, device: Optional[str] = None,
                 max_length: int = 512, batch_size: int = 64) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        torch = self._torch
        vectors: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(list(texts[start:start + self.batch_size]), padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vectors.extend(pooled.float().cpu().tolist())
        return vectors


class DyCPCompressor:
    """Selects whole turns with KadaneDial. TokenWise calls `select_units`
    instead of scoring sentences and filling a budget."""

    def __init__(self, retriever: str = DEFAULT_RETRIEVER, tau: float = 0.6, theta: float = 1.0,
                 encoder_factory: Optional[Callable[[], object]] = None) -> None:
        self.retriever = retriever
        self.tau = tau
        self.theta = theta
        self._encoder_factory = encoder_factory or (lambda: ContrieverEncoder(retriever))
        self._encoder = None

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self._encoder is None:
            self._encoder = self._encoder_factory()
        return self._encoder.encode(texts)

    def select_units(self, units: Sequence[Unit], query: str) -> List[int]:
        turns = group_turns(units)
        if not turns:
            return []
        by_id = {unit.id: unit for unit in units}
        texts = [" ".join(by_id[index].text for index in turn) for turn in turns]
        vectors = self._encode([query, *texts])
        query_vector, turn_vectors = vectors[0], vectors[1:]
        scores = [sum(a * b for a, b in zip(query_vector, vector)) for vector in turn_vectors]
        selected = [index for start, end in kadane_dial(scores, self.tau, self.theta)
                    for turn in turns[start:end + 1] for index in turn]
        return sorted(selected)
