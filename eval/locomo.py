"""LoCoMo (Maharana et al., "Evaluating Very Long-Term Conversational
Memory of LLM Agents", ACL 2024) as items in LongMemEval's shape, so the
whole eval (compression, evidence recall, answering, judging) runs on it
unchanged.

LoCoMo has 10 conversations between two people, each up to 35 dated
sessions, with QA pairs whose evidence is given as dialogue ids ("D3:12" =
session 3, turn 12). Differences from LongMemEval, and how they are kept:
- both sides are people, named in every question ("When did Caroline...?"):
  each turn carries its speaker, which labels it instead of User/Assistant;
  speaker_a is mapped to the "user" role, speaker_b to "assistant";
- sessions have absolute dates and questions have none, so each session is
  labelled with its own date ("1:56 pm on 8 May, 2023") rather than "N days
  before this question";
- shared photos are kept as their BLIP caption, the usual practice;
- category 5 (adversarial: the asked-about fact never happened) is left
  out by default, as in most LoCoMo evaluations; requested explicitly, its
  reference is that the information is not in the conversation.
"""

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

LOCOMO_COMMIT = "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc"
LOCOMO_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}/data/locomo10.json"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
CACHE_PATH = Path.home() / ".cache" / "locomo" / f"locomo10-{LOCOMO_COMMIT[:8]}.json"

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DEFAULT_QUESTION_TYPES = ["multi-hop", "temporal", "open-domain", "single-hop"]
ADVERSARIAL_ANSWER = "Not mentioned in the conversation."

_DIA_ID = re.compile(r"D(\d+):(\d+)")


def download_locomo(path=CACHE_PATH):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(LOCOMO_URL, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != LOCOMO_SHA256:
            raise SystemExit(f"LoCoMo download from {LOCOMO_URL} does not match the pinned checksum")
        path.write_bytes(data)
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_ids(evidence):
    """{(session, turn)} from LoCoMo's evidence strings, a few of which hold
    several ids ("D8:6; D9:17") or are malformed ("D:11:26", skipped)."""
    return {(int(session), int(turn)) for entry in evidence for session, turn in _DIA_ID.findall(entry)}


def _turn_text(turn):
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        text = f"{text} [shares a photo: {caption}]".strip()
    return text


def conversation_items(sample):
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    numbers = sorted(int(key.split("_")[1]) for key in conversation
                     if re.fullmatch(r"session_\d+", key) and conversation[key])
    sample_id = sample["sample_id"]
    session_ids = [f"{sample_id}_S{number}" for number in numbers]
    time_labels = [conversation.get(f"session_{number}_date_time") for number in numbers]

    # Turns are shared between a conversation's questions; only evidence
    # turns are copied, to flag them.
    base_sessions = []
    for number in numbers:
        turns = []
        for turn in conversation[f"session_{number}"]:
            match = _DIA_ID.fullmatch(turn.get("dia_id", ""))
            turns.append(((int(match.group(1)), int(match.group(2))) if match else None, {
                "role": "user" if turn["speaker"] == speaker_a else "assistant",
                "speaker": turn["speaker"],
                "content": _turn_text(turn),
                "has_answer": False,
            }))
        base_sessions.append(turns)

    items = []
    for index, qa in enumerate(sample["qa"]):
        question_type = CATEGORIES[qa["category"]]
        wanted = evidence_ids(qa.get("evidence") or [])
        sessions = [[dict(turn, has_answer=True) if key in wanted else turn for key, turn in session]
                    for session in base_sessions]
        answer = ADVERSARIAL_ANSWER if question_type == "adversarial" else str(qa["answer"])
        items.append({
            "question_id": f"{sample_id}_q{index:03d}",
            "question_type": question_type,
            "question": qa["question"],
            "answer": answer,
            "question_date": None,
            "haystack_sessions": sessions,
            "haystack_session_ids": session_ids,
            "session_time_labels": time_labels,
            "answer_session_ids": [f"{sample_id}_S{session}" for session in sorted({s for s, _ in wanted})],
        })
    return items


def load_locomo(limit=None, seed=42, question_types=None):
    items = [item for sample in download_locomo() for item in conversation_items(sample)]
    question_types = question_types or DEFAULT_QUESTION_TYPES
    items = [item for item in items if item["question_type"] in question_types]
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items
