from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from core.scorers import BM25Scorer, LexicalScorer
from core.selection import (AGGREGATE, ASSISTANT_RECALL, SelectionConfig, fuse_scores, mmr_pool_ids,
                            replay_turn_scores, select_units, session_cover_scores, time_focus_scores,
                            zscore)
from core.units import Unit, render_flat, render_units


@dataclass
class CompressionResult:
    text: str
    selected: Optional[List[Unit]]  # None for abstractive compressors
    tokens: int


class TokenWise:
    """Compresses history units down to a token budget for a query.

    `model` decides how units are scored:
      - has `score_units(units, queries)` (e.g. CPCCompressor): extractive,
        its scores feed the shared fusion/selection/rendering below;
      - has only `compress(query, token_budget, context)` (e.g.
        GeminiCompressor): abstractive, it receives the rendered units;
      - None: lexical word-overlap scoring.

    Budgets are always counted in cl100k (tiktoken) tokens, whatever the
    scorer's own tokenizer, so every compressor is held to the same unit."""

    PRETRAINED_PRESETS: Dict[str, Dict[str, str]] = {
        "mistral": {
            "tokenizer_name_or_path": "deadcode99/cpc-1.0-mistral-7b-tokenizer",
            "lora_name_or_path": "deadcode99/cpc-1.0-mistral-7b-ds-v5-iter66-lora-bidirectional-attn",
        },
        "llama": {
            "tokenizer_name_or_path": "deadcode99/cpc-1.0-llama-1b-tokenizer",
            "lora_name_or_path": "deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn",
        },
    }

    def __init__(self, model: Any = None, selection: Optional[SelectionConfig] = None) -> None:
        import tiktoken

        self.model = model
        self.selection = selection or SelectionConfig()
        self.secondary = None  # optional second scorer fused in (SelectionConfig.provence_weight)
        self.entities = None  # optional core.entities.EntityIndex (SelectionConfig.entity_weight)
        self._encoding = tiktoken.encoding_for_model("gpt-4")
        self._lexical = LexicalScorer()
        self._bm25 = BM25Scorer()

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def compress_units(
        self,
        query: str,
        token_budget: int,
        units: Sequence[Unit],
        extra_queries: Sequence[str] = (),
    ) -> CompressionResult:
        units = list(units)
        if not units:
            return CompressionResult(text="", selected=[], tokens=0)

        if self.model is not None and hasattr(self.model, "select_units"):
            # Selectors such as DyCP decide what (and how much) to keep
            # themselves: no scoring, fusion or token budget here.
            selected_ids = self.model.select_units(units, query)
            selected = [units[index] for index in selected_ids]
            text = render_units(selected)
            return CompressionResult(text=text, selected=selected, tokens=self.count_tokens(text))

        if self.model is not None and not hasattr(self.model, "score_units"):
            text = self.model.compress(query=query, token_budget=token_budget, context=render_units(units))
            return CompressionResult(text=text, selected=None, tokens=self.count_tokens(text))

        queries = [query, *dict.fromkeys(q for q in extra_queries if q and q != query)]
        scorer = self.model if self.model is not None else self._lexical
        primary = scorer.score_units(units, queries)
        lexical = self._bm25.score_units(units, queries) if self.selection.lexical_weight > 0 else None
        secondary = (self.secondary.score_units(units, queries)
                     if self.secondary is not None and self.selection.provence_weight > 0 else None)

        scores = fuse_scores(primary, self.selection, lexical, secondary)
        if self.selection.feedback > 0:
            # Pseudo-relevance feedback: a bridge sentence ("improving my old area,
            # West County") shares no words with the question ("what area was hit by
            # a flood?") but does with the best first-pass sentence ("my old area
            # was hit by a flood"). Re-score with both as the query, max-pooled.
            top = sorted(range(len(units)), key=lambda index: (scores[index], index),
                         reverse=True)[: self.selection.feedback]
            expanded = "\n".join([query, *(units[index].text for index in top)])
            second = zscore(scorer.score_units(units, [expanded])[0])
            scores = [max(first, again) for first, again in zip(scores, second)]
        if self.entities is not None and self.selection.entity_weight > 0:
            weight = self.selection.entity_weight
            keyed = zscore(self.entities.scores(units, query, scores))
            scores = [(1 - weight) * a + weight * b for a, b in zip(scores, keyed)]
        if self.selection.time_focus > 0:
            scores = time_focus_scores(units, scores, query, self.selection.time_focus)
        if self.selection.session_cover > 0 and AGGREGATE.search(query):
            scores = session_cover_scores(units, scores, self.selection.session_cover)
        if self.selection.replay_turn and ASSISTANT_RECALL.search(query):
            scores = replay_turn_scores(units, scores)
        embeddings = None
        if self.selection.mmr_lambda > 0 and self.selection.mmr_similarity == "semantic":
            if not hasattr(scorer, "unit_embeddings"):
                raise ValueError("semantic MMR needs a scorer that provides unit embeddings (use --compressor cpc)")
            embeddings = scorer.unit_embeddings(units, mmr_pool_ids(units, scores, self.selection))
        selected_ids = select_units(units, scores, token_budget, self.count_tokens, self.selection, embeddings)
        selected = [units[index] for index in selected_ids]
        text = (render_flat if self.selection.flat_output else render_units)(selected)
        return CompressionResult(text=text, selected=selected, tokens=self.count_tokens(text))
