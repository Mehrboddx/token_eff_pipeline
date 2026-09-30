import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from core.units import Unit, has_sessions, role_label, session_header

_WORD = re.compile(r"\w+")


@dataclass
class SelectionConfig:
    """How ranked units become an excerpt. Every field defaults to off, so
    the default is plain greedy top-k under the budget."""

    lexical_weight: float = 0.0  # hybrid: weight of BM25 in the fused score
    recency_weight: float = 0.0  # additive bonus (in z units) for later units
    mmr_lambda: float = 0.0  # penalty (in z units) per unit of similarity
    mmr_pool: int = 400  # MMR only re-ranks the top of the list
    mmr_similarity: str = "lexical"  # "lexical" (word Jaccard) or "semantic" (scorer embeddings)
    top_sessions: int = 0  # two-level: restrict to the best K sessions
    window: int = 0  # also take +-N neighbouring sentences in the same turn
    pair_turns: bool = False  # also take the best sentence of the partner turn


def zscore(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / n)
    if std < 1e-12:
        return [0.0] * n
    return [(value - mean) / std for value in values]


def fuse_scores(
    primary: Sequence[Sequence[float]],
    config: SelectionConfig,
    lexical: Optional[Sequence[Sequence[float]]] = None,
) -> List[float]:
    """One score per unit. Each query's scores are z-normalized (so
    sub-queries with different score ranges are comparable), optionally
    blended with that query's BM25 scores, then max-pooled across queries:
    a unit only needs to answer one sub-question to rank highly, which is
    what gives multi-hop questions coverage of every fact they need."""
    per_query = []
    for index, scores in enumerate(primary):
        fused = zscore(scores)
        if lexical is not None and config.lexical_weight > 0:
            weight = config.lexical_weight
            fused = [(1 - weight) * a + weight * b for a, b in zip(fused, zscore(lexical[index]))]
        per_query.append(fused)

    final = [max(column) for column in zip(*per_query)]
    if config.recency_weight and len(final) > 1:
        last = len(final) - 1
        final = [score + config.recency_weight * index / last for index, score in enumerate(final)]
    return final


def _words(text: str) -> set:
    return {word.lower() for word in _WORD.findall(text)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ranked_candidates(units: Sequence[Unit], scores: Sequence[float], config: SelectionConfig) -> List[int]:
    """Candidate ids best-first, restricted to the top sessions when
    two-level selection is on."""
    candidate_ids = list(range(len(units)))
    if config.top_sessions > 0 and has_sessions(units):
        best: Dict = {}
        for unit in units:
            if unit.session is not None:
                best[unit.session] = max(best.get(unit.session, -math.inf), scores[unit.id])
        keep = set(sorted(best, key=best.get, reverse=True)[: config.top_sessions])
        candidate_ids = [unit.id for unit in units if unit.session is None or unit.session in keep]
    return sorted(candidate_ids, key=lambda index: (scores[index], index), reverse=True)


def mmr_pool_ids(units: Sequence[Unit], scores: Sequence[float], config: SelectionConfig) -> List[int]:
    """Exactly the candidates MMR will re-rank, so a caller can fetch their
    embeddings ahead of selection."""
    return _ranked_candidates(units, scores, config)[: config.mmr_pool]


def _centered(embeddings: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Center on the pool mean, then re-normalize. Raw transformer sentence
    embeddings are anisotropic -- even unrelated sentences can sit at high
    cosine -- which would make every candidate look redundant; centering
    turns cosine into similarity relative to this haystack."""
    if not embeddings:
        return {}
    ids = list(embeddings)
    matrix = np.stack([embeddings[index] for index in ids]).astype(np.float32)
    matrix -= matrix.mean(axis=0, keepdims=True)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return dict(zip(ids, matrix))


def select_units(
    units: Sequence[Unit],
    scores: Sequence[float],
    token_budget: int,
    count_tokens: Callable[[str], int],
    config: Optional[SelectionConfig] = None,
    embeddings: Optional[Dict[int, np.ndarray]] = None,
) -> List[int]:
    """Greedy selection by score under `token_budget`, where the budget
    covers everything the renderer will emit for the selection: sentence
    text plus each new session header and role label.

    `embeddings` (unit id -> vector) backs `mmr_similarity="semantic"`;
    a pair where either side has no vector falls back to word overlap."""
    config = config or SelectionConfig()
    if not units:
        return []
    if config.mmr_lambda > 0 and config.mmr_similarity == "semantic" and not embeddings:
        raise ValueError("semantic MMR needs unit embeddings from the scorer (use --compressor cpc)")

    grouped = has_sessions(units)
    ranked = _ranked_candidates(units, scores, config)

    by_entry: Dict[int, List[int]] = {}
    for unit in units:
        by_entry.setdefault(unit.entry_index, []).append(unit.id)
    entry_order = sorted(by_entry)
    entry_position = {entry: position for position, entry in enumerate(entry_order)}

    header_cost: Dict = {}
    label_cost: Dict[str, int] = {}

    def overhead(unit: Unit, sessions: set, entries: set) -> int:
        cost = 0
        if grouped and unit.session not in sessions:
            if unit.session not in header_cost:
                header_cost[unit.session] = count_tokens(session_header(unit))
            cost += header_cost[unit.session]
        if unit.entry_index not in entries:
            if unit.role not in label_cost:
                label_cost[unit.role] = count_tokens(role_label(unit))
            cost += label_cost[unit.role]
        return cost

    selected: set = set()
    sessions_present: set = set()
    entries_present: set = set()
    used = 0

    def partner_best(unit: Unit) -> Optional[int]:
        position = entry_position[unit.entry_index]
        step = 1 if unit.role == "user" else -1
        neighbour_position = position + step
        if not 0 <= neighbour_position < len(entry_order):
            return None
        partner_ids = by_entry[entry_order[neighbour_position]]
        partner = units[partner_ids[0]]
        if partner.role == unit.role or partner.session != unit.session:
            return None
        return max(partner_ids, key=lambda index: (scores[index], index))

    def bundle(index: int) -> List[int]:
        unit = units[index]
        members = [index]
        if config.window > 0:
            siblings = by_entry[unit.entry_index]
            position = siblings.index(index)
            low = max(0, position - config.window)
            members.extend(siblings[low:position] + siblings[position + 1 : position + 1 + config.window])
        if config.pair_turns and unit.role in ("user", "model"):
            partner = partner_best(unit)
            if partner is not None:
                members.append(partner)
        return [member for member in dict.fromkeys(members) if member not in selected]

    def cost_of(members: List[int]) -> int:
        sessions = set(sessions_present)
        entries = set(entries_present)
        cost = 0
        for member in members:
            unit = units[member]
            cost += unit.tokens + overhead(unit, sessions, entries)
            sessions.add(unit.session)
            entries.add(unit.entry_index)
        return cost

    def try_add(index: int) -> List[int]:
        nonlocal used
        members = bundle(index)
        if not members:
            return []
        cost = cost_of(members)
        if used + cost > token_budget:
            members = [index] if index not in selected else []
            if not members:
                return []
            cost = cost_of(members)
            if used + cost > token_budget:
                return []
        for member in members:
            selected.add(member)
            sessions_present.add(units[member].session)
            entries_present.add(units[member].entry_index)
        used += cost
        return members

    if config.mmr_lambda > 0:
        pool = ranked[: config.mmr_pool]
        rest = ranked[config.mmr_pool :]
        words = {index: _words(units[index].text) for index in pool}
        vectors = _centered(embeddings) if config.mmr_similarity == "semantic" else {}

        def similarity(a: int, b: int, b_words: set) -> float:
            if a in vectors and b in vectors:
                return max(0.0, float(vectors[a] @ vectors[b]))
            return _jaccard(words[a], b_words)

        max_similarity = {index: 0.0 for index in pool}
        remaining = list(pool)
        while remaining and used < token_budget:
            choice = max(
                remaining,
                key=lambda index: (scores[index] - config.mmr_lambda * max_similarity[index], index),
            )
            remaining.remove(choice)
            if choice in selected:
                continue
            for added in try_add(choice):
                added_words = words.get(added) or _words(units[added].text)
                for index in remaining:
                    value = similarity(index, added, added_words)
                    if value > max_similarity[index]:
                        max_similarity[index] = value
        ranked = rest

    for index in ranked:
        if used >= token_budget:
            break
        if index not in selected:
            try_add(index)

    return sorted(selected)
