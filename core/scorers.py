import math
import re
from collections import Counter
from typing import List, Sequence

from core.units import Unit

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class LexicalScorer:
    """Query word overlap normalized by sentence length -- the original
    TokenWise fallback scorer, now behind the shared score_units API."""

    def score_units(self, units: Sequence[Unit], queries: Sequence[str]) -> List[List[float]]:
        unit_tokens = [set(_tokens(unit.text)) for unit in units]
        scores = []
        for query in queries:
            query_tokens = set(_tokens(query))
            scores.append([len(tokens & query_tokens) / len(tokens) if tokens else 0.0 for tokens in unit_tokens])
        return scores


class BM25Scorer:
    """Okapi BM25 over the candidate units themselves (each unit is a
    "document"). Complements dense scoring on exact names, numbers and
    rare entities, which embeddings tend to blur."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_units(self, units: Sequence[Unit], queries: Sequence[str]) -> List[List[float]]:
        docs = [Counter(_tokens(unit.text)) for unit in units]
        lengths = [sum(doc.values()) for doc in docs]
        average_length = (sum(lengths) / len(lengths)) if lengths else 0.0
        document_frequency: Counter = Counter()
        for doc in docs:
            document_frequency.update(doc.keys())
        total = len(docs)

        scores = []
        for query in queries:
            terms = set(_tokens(query))
            idf = {
                term: math.log(1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                for term in terms
                if document_frequency[term]
            }
            query_scores = []
            for doc, length in zip(docs, lengths):
                score = 0.0
                norm = self.k1 * (1 - self.b + self.b * length / average_length) if average_length else self.k1
                for term, weight in idf.items():
                    frequency = doc.get(term, 0)
                    if frequency:
                        score += weight * frequency * (self.k1 + 1) / (frequency + norm)
                query_scores.append(score)
            scores.append(query_scores)
        return scores
