"""Recorded coding-agent runs -> one list of action-observation pairs.

CodeTraceBench stores runs in three formats; each becomes
[{"action": str, "observation": str}, ...] in the order they happened.
A pair is complete when the action produced an observation. Kept free of
third-party imports so it can be unit-tested anywhere."""

import json
from typing import Any


def _text(content: Any) -> str:
    """Message content as plain text (a string, or a list of content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content)
    return str(content)


def _call_text(call: dict) -> str:
    """A tool call as a line the receiver can read: `$ command` for shell calls, `name(arguments)` otherwise."""
    function = call.get("function") or {}
    name = function.get("name", "tool")
    arguments = function.get("arguments", "")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (TypeError, ValueError):
        return f"{name}({arguments})"
    if set(parsed) <= {"command", "is_input", "timeout", "security_risk"} and isinstance(parsed.get("command"), str) and name in ("execute_bash", "bash"):
        return f"$ {parsed['command']}"
    shown = {key: value for key, value in parsed.items() if key != "security_risk"}
    return f"{name}({json.dumps(shown, ensure_ascii=False)})"


def from_miniswe(trajectory: dict) -> list[dict]:
    """mini-SWE-agent: system, task, then assistant / user(observation) turns."""
    messages = trajectory.get("messages") or []
    pairs = []
    for index in range(2, len(messages) - 1):
        if messages[index].get("role") == "assistant" and messages[index + 1].get("role") == "user":
            pairs.append({"action": _text(messages[index].get("content")), "observation": _text(messages[index + 1].get("content"))})
    return pairs


def from_sweagent(trajectory: dict) -> list[dict]:
    """SWE-agent: a list of steps that each hold the response and its observation."""
    pairs = []
    for step in trajectory.get("trajectory") or []:
        action = step.get("response") or "\n".join(part for part in (step.get("thought"), step.get("action")) if part)
        observation = step.get("observation")
        if action and observation is not None:
            pairs.append({"action": action, "observation": str(observation)})
    return pairs


def from_openhands(llm_log: dict) -> list[dict]:
    """OpenHands: the last LLM-call log holds the whole conversation so far: an
    assistant message with tool calls, followed by one tool message per call."""
    messages = llm_log.get("messages") or []
    pairs = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            action = "\n".join(part for part in [_text(message.get("content")), *(_call_text(call) for call in calls)] if part)
            outputs = []
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                outputs.append(_text(messages[index].get("content")))
                index += 1
            if outputs:
                pairs.append({"action": action, "observation": "\n".join(outputs)})
            continue
        index += 1
    return pairs


MAX_OBSERVATION = 1_000_000  # characters


def cap_observation(text: str, limit: int = MAX_OBSERVATION) -> str:
    """A tool output longer than `limit` keeps its head and tail, as agent harnesses
    truncate long outputs. Some recorded runs dumped whole files of several million
    characters, more than any receiver context holds."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"


def pair_text(pair: dict) -> str:
    return f"{pair['action'].strip()}\n--- output ---\n{pair['observation'].strip()}"


def history_text(pairs: list[dict]) -> str:
    """The pre-handoff history as the receiver reads it (full-history condition)."""
    return "\n\n".join(f"[step {number}]\n{pair_text(pair)}" for number, pair in enumerate(pairs, 1))
