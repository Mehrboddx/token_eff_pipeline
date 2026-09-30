import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

ROLE_LABELS = {"user": "User", "model": "Assistant", "tool": "Tool"}

_WORD = re.compile(r"\w+")


@dataclass
class Unit:
    """One scorable sentence of history. Metadata travels beside the text
    so no scorer ever sees it; only the renderer puts it back."""

    id: int  # position among the candidates, chronological
    text: str
    role: str
    entry_index: int  # the Memory history entry (one turn) it came from
    session: Any = None  # never rendered verbatim: ids can leak labels
    time: Optional[str] = None  # human-readable time label for the header
    tokens: int = 0
    meta: dict = field(default_factory=dict)


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def prepare_units(
    entries: Iterable[dict],
    count_tokens: Callable[[str], int],
    min_words: int = 4,
) -> List[Unit]:
    """Memory sentence entries -> scorable units.

    Fragments under `min_words` words (markdown debris like "**", "Thanks!",
    list bullets) are merged into a neighbour from the same turn: on their
    own they are nearly free under a token budget, so greedy selection
    would favour them over real content. Exact repeats (role + text) keep
    only their first occurrence."""
    by_entry: dict[int, list[dict]] = {}
    order: list[int] = []
    for entry in entries:
        if not _WORD.search(entry["text"]):
            continue
        index = entry["entry_index"]
        if index not in by_entry:
            by_entry[index] = []
            order.append(index)
        by_entry[index].append(entry)

    units: List[Unit] = []
    seen: set[tuple[str, str]] = set()
    for index in order:
        group = by_entry[index]
        texts = [entry["text"] for entry in group]

        merged: list[str] = []
        pending = ""
        for text in texts:
            if word_count(text) < min_words:
                if merged:
                    merged[-1] = f"{merged[-1]} {text}"
                else:
                    pending = f"{pending} {text}".strip()
                continue
            merged.append(f"{pending} {text}".strip() if pending else text)
            pending = ""
        if pending:
            merged.append(pending)

        first = group[0]
        meta = first.get("meta") or {}
        for text in merged:
            key = (first["role"], text)
            if key in seen:
                continue
            seen.add(key)
            units.append(
                Unit(
                    id=len(units),
                    text=text,
                    role=first["role"],
                    entry_index=index,
                    session=meta.get("session"),
                    time=meta.get("time"),
                    tokens=count_tokens(text),
                    meta=meta,
                )
            )
    return units


def session_header(unit: Unit) -> str:
    return f"[Earlier session, {unit.time}]" if unit.time else "[Earlier session]"


def role_label(unit: Unit) -> str:
    return f"{ROLE_LABELS.get(unit.role, unit.role)}:"


def has_sessions(units: List[Unit]) -> bool:
    return any(unit.session is not None for unit in units)


def render_units(units: List[Unit]) -> str:
    """Selected units, chronologically, grouped under one header per
    session and one role label per turn. " … " marks sentences skipped
    inside a turn so the reader knows the excerpt isn't contiguous."""
    units = sorted(units, key=lambda unit: unit.id)
    grouped = has_sessions(units)
    lines: list[str] = []
    current_session = object()
    current_entry = None
    previous_id = None
    for unit in units:
        if grouped and unit.session != current_session:
            current_session = unit.session
            current_entry = None
            if lines:
                lines.append("")
            lines.append(session_header(unit))

        if unit.entry_index != current_entry:
            current_entry = unit.entry_index
            lines.append(f"{role_label(unit)} {unit.text}")
        else:
            separator = " " if previous_id is not None and unit.id == previous_id + 1 else " … "
            lines[-1] += separator + unit.text
        previous_id = unit.id
    return "\n".join(lines)
