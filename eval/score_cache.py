import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from core.units import Unit


class CachedScorer:
    """Disk cache in front of an expensive `score_units` scorer (CPC on a
    GPU). Keyed by the exact candidate units -- text plus the turn/role/
    session layout the encoder sees -- so a cache entry can only ever be
    reused for identical input. The scorer itself is built lazily, on the
    first miss: re-running selection ablations over already-scored
    questions never loads the model at all."""

    def __init__(self, factory: Callable[[], object], cache_dir: str, key: str, embedding_top_k: int = 600) -> None:
        self._factory = factory
        self._scorer = None
        self.directory = Path(cache_dir) / key
        self.directory.mkdir(parents=True, exist_ok=True)
        # Full per-sentence embeddings would be ~40 MB per question for
        # Mistral; only the best-scoring candidates (which is where MMR's
        # re-ranking pool comes from) are kept on disk.
        self.embedding_top_k = embedding_top_k
        self.hits = 0
        self.misses = 0

    def _scorer_instance(self):
        if self._scorer is None:
            self._scorer = self._factory()
        return self._scorer

    @staticmethod
    def _digest(units: Sequence[Unit]) -> str:
        payload = "\x1e".join(f"{u.role}\x1f{u.session}\x1f{u.entry_index}\x1f{u.text}" for u in units)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def score_units(self, units: Sequence[Unit], queries: Sequence[str]) -> List[List[float]]:
        path = self.directory / f"{self._digest(units)}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        missing = [query for query in dict.fromkeys(queries) if query not in data]
        if missing:
            self.misses += 1
            for query, scores in zip(missing, self._scorer_instance().score_units(units, missing)):
                data[query] = [round(score, 5) for score in scores]
            path.write_text(json.dumps(data), encoding="utf-8")
        else:
            self.hits += 1
        return [data[query] for query in queries]

    def unit_embeddings(self, units: Sequence[Unit], ids: Sequence[int]) -> Dict[int, np.ndarray]:
        """Embeddings for `ids`, from disk when stored. On a miss, embeds
        once and stores the requested ids plus the top-k units by best
        cached score, so later selection configs usually hit."""
        digest = self._digest(units)
        path = self.directory / f"{digest}.emb.npz"
        stored: Dict[int, np.ndarray] = {}
        if path.exists():
            with np.load(path) as archive:
                stored = dict(zip(archive["ids"].tolist(), archive["vectors"].astype(np.float32)))
        if all(index in stored for index in ids):
            return {index: stored[index] for index in ids}

        keep = set(ids) | set(stored)
        scores_path = self.directory / f"{digest}.json"
        if scores_path.exists():
            columns = list(json.loads(scores_path.read_text(encoding="utf-8")).values())
            best = [max(column) for column in zip(*columns)]
            keep.update(sorted(range(len(best)), key=best.__getitem__, reverse=True)[: self.embedding_top_k])
        fresh = self._scorer_instance().unit_embeddings(units, sorted(keep - set(stored)))
        stored.update(fresh)
        ordered = sorted(stored)
        np.savez(path, ids=np.array(ordered, dtype=np.int32),
                 vectors=np.stack([stored[index] for index in ordered]).astype(np.float16))
        return {index: stored[index] for index in ids}
