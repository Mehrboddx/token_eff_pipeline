"""Rendering a compressed excerpt of an agent transcript.

Kept free of model imports so it can be unit-tested anywhere. Unlike
core.units.render_units (made for chat sentences), lines are joined with
newlines, so code and tool output keep their layout; a line with "…"
marks lines dropped inside one entry, and each entry keeps its position
in the transcript so the agent can tell what came first."""

import json
from collections.abc import Sequence

ROLE_NAMES = {"assistant": "you", "tool": "tool output", "user": "user", "colleague": "colleague"}


def render_excerpt(selected: Sequence, entries: Sequence[dict]) -> str:
    if len(entries) == 1:
        return _render_lines(sorted(selected, key=lambda unit: unit.id))
    blocks = []
    by_entry: dict[int, list] = {}
    for unit in sorted(selected, key=lambda unit: unit.id):
        by_entry.setdefault(unit.entry_index, []).append(unit)
    for index, units in by_entry.items():
        role = entries[index].get("role", "user")
        label = entries[index].get("label") or ROLE_NAMES.get(role, role)
        blocks.append(f"[step {index + 1}, {label}]\n{_render_lines(units)}")
    return "\n\n".join(blocks)


def _render_lines(units: Sequence) -> str:
    lines: list[str] = []
    previous = None
    for unit in units:
        if previous is not None and unit.id != previous + 1:
            lines.append("…")
        lines.append(unit.text)
        previous = unit.id
    return "\n".join(lines)


def message_text(message: dict) -> str:
    """What a transcript message says, tool calls included."""
    text = message.get("content") or ""
    if isinstance(text, list):  # content blocks
        text = "\n".join(str(block.get("text", "")) for block in text if isinstance(block, dict))
    calls = []
    for call in message.get("tool_calls") or []:
        arguments = (call.get("function") or {}).get("arguments", "")
        try:
            arguments = json.loads(arguments).get("command", arguments)
        except (TypeError, ValueError, AttributeError):
            pass
        calls.append(f"$ {arguments}")
    return "\n".join(part for part in [text, *calls] if part)
