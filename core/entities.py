"""Entity keys: a second, structured way to find a sentence besides its
meaning. Evidence that CPC ranks low is usually a fact about an entity
mentioned in passing ("…for my daughter Sara's birthday"), a status
("tough as a single parent") or an alias that bridges two sentences
("my old area, West County"). Each unit gets a set of keys, written once
when the history is stored, by one of two sources (eval/entity_keys.py):

  - ner: spaCy -- named entities, noun phrases (with appositive aliases),
    content lemmas, and the speaker for first-person sentences;
  - llm: a small local instruct model reads the sentence (with the one
    before it and the speaker) and writes facts "entity | category | value"
    with general categories (child, occupation, instrument, ...), so a
    question about "children" can find "my daughter Sara".

At question time units are scored by the rare keys they share with the
question, plus a bridge term: rare noun phrases shared with the best
first-pass units (pseudo-relevance feedback over keys, after HippoRAG and
entity-memory graph retrieval). The score is z-normalized and blended
into the primary score with SelectionConfig.entity_weight.

Keys are stored per unit as {"t": [...terms], "c": [...chunks], "e": [...entities]},
all lower-cased lemmas, keyed by unit_key(unit)."""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from core.units import Unit, role_label

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
ENTITY_LABELS = {"PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LANGUAGE"}
BRIDGE_WEIGHT = 0.5  # bridge keys count half as much as the question's own keys
BRIDGE_TOP = 3  # first-pass units whose noun phrases seed the bridge


def unit_key(unit: Unit) -> str:
    return hashlib.sha1(f"{role_label(unit)}\x00{unit.text}".encode("utf-8")).hexdigest()[:20]


_NLP = None


def nlp():
    """spaCy's small English pipeline, loaded once (CPU)."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _chunk_key(span) -> Optional[str]:
    words = [token.lemma_.lower() for token in span
             if not token.is_stop and not token.is_punct and token.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")]
    return " ".join(words) if words else None


def doc_keys(doc, speaker: Optional[str] = None) -> Dict[str, List[str]]:
    """NER keys of one parsed sentence (or question)."""
    terms = {token.lemma_.lower() for token in doc
             if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "NUM") and not token.is_stop and token.is_alpha}
    entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ENTITY_LABELS}
    chunks = set()
    for chunk in doc.noun_chunks:
        key = _chunk_key(chunk)
        if key:
            chunks.add(key)
    # Apposition links an alias to its name: "my old area, West County".
    for token in doc:
        if token.dep_ == "appos":
            head = _chunk_key(doc[token.head.left_edge.i: token.head.i + 1])
            name = _chunk_key(doc[token.left_edge.i: token.right_edge.i + 1])
            for key in (head, name):
                if key:
                    chunks.add(key)
                    entities.add(key)
    if speaker and any(token.lower_ in FIRST_PERSON for token in doc):
        entities.add(speaker.lower())
        terms.add(speaker.lower())
    chunks |= entities
    return {"t": sorted(terms), "c": sorted(chunks), "e": sorted(entities)}


def question_keys(question: str) -> Dict[str, List[str]]:
    return doc_keys(nlp()(question))


class EntityIndex:
    """Keys for every unit (loaded from one or more JSON files written by
    eval/entity_keys.py) and the question-time scorer."""

    def __init__(self, paths: Iterable[str]) -> None:
        # Several files (e.g. ner + llm) are merged per unit as the union of keys.
        self.keys: Dict[str, Dict[str, List[str]]] = {}
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                for key, found in json.load(handle).items():
                    if key not in self.keys:
                        self.keys[key] = found
                    else:
                        merged = self.keys[key]
                        self.keys[key] = {part: sorted(set(merged.get(part, [])) | set(found.get(part, [])))
                                          for part in ("t", "c", "e")}

    def unit_keys(self, unit: Unit) -> Dict[str, List[str]]:
        found = self.keys.get(unit_key(unit))
        if found is None:
            raise KeyError(f"no entity keys for unit {unit.text[:60]!r} -- run eval/entity_keys.py first")
        return found

    def scores(self, units: Sequence[Unit], question: str, primary: Sequence[float]) -> List[float]:
        """Per unit: idf-weighted overlap of its keys with the question's
        keys, plus BRIDGE_WEIGHT x overlap of its noun phrases with those of
        the BRIDGE_TOP best first-pass units (excluding those units)."""
        keys = [self.unit_keys(unit) for unit in units]
        bags = [set(k["t"]) | set(k["c"]) | set(k["e"]) for k in keys]
        n = len(units)
        df: Dict[str, int] = {}
        for bag in bags:
            for key in bag:
                df[key] = df.get(key, 0) + 1

        def idf(key: str) -> float:
            return math.log(1 + n / df[key]) if key in df else 0.0

        asked = question_keys(question)
        wanted = set(asked["t"]) | set(asked["c"]) | set(asked["e"])
        top = sorted(range(n), key=lambda index: (primary[index], index), reverse=True)[:BRIDGE_TOP]
        bridge = set()
        for index in top:
            bridge |= set(keys[index]["c"])
        bridge -= wanted
        top_set = set(top)

        out = []
        for index, bag in enumerate(bags):
            score = sum(idf(key) for key in bag & wanted)
            if index not in top_set:
                score += BRIDGE_WEIGHT * sum(idf(key) for key in bag & bridge)
            out.append(score)
        return out


def load_index(paths: Optional[Sequence[str]]) -> Optional[EntityIndex]:
    return EntityIndex(paths) if paths else None


def existing_keys(path: str) -> Dict[str, Dict[str, List[str]]]:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
