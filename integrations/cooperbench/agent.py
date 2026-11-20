"""CooperBench's mini-swe-agent v2 with the CPC-Mem compressor in its loop.

Imported inside CooperBench's runner process (see launch.py), never by the
GPU image: it needs only the standard library and CooperBench itself, and
reaches the scorers over HTTP (compress_server.py).

Three arms, picked per run with `agent.cpc_arm` in the --agent-config YAML:

  none      the stock agent (its own LLM summary compaction at 28k prompt
            tokens), plus the token ledger; the baseline.
  messages  every message from a colleague is compressed before the
            receiving agent sees it, scored against the receiver's task
            and its latest step. History is handled as in `none`.
  full      messages as above, and the history compaction is replaced:
            instead of an LLM call that summarises the older turns, the
            compressor keeps the verbatim lines most relevant to the task
            and the current step, under a fixed token budget.

Every LLM call (agent steps and the stock summary calls) and every
compression is written to <log_dir>/<agent_id>_ledger.jsonl, so all arms are
measured the same way.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal

from cooperbench.agents.mini_swe_agent_v2 import adapter as msa_adapter
from cooperbench.agents.mini_swe_agent_v2.agents.default import AgentConfig, DefaultAgent
from cooperbench.agents.mini_swe_agent_v2.environments import docker as msa_docker

from integrations.cooperbench.render import message_text

_context = threading.local()  # the adapter's log_dir, for the agent built on this thread


class CpcAgentConfig(AgentConfig):
    cpc_arm: Literal["none", "messages", "full"] = "none"
    cpc_url: str = "http://127.0.0.1:8765"
    cpc_timeout: float = 300.0
    message_ratio: float = 0.5
    """Share of a colleague message's tokens kept."""
    message_min_tokens: int = 150
    """Messages at or under this many tokens are delivered verbatim (nothing worth dropping)."""
    history_budget: int = 6000
    """Tokens the compressed older history may take (replaces the LLM summary)."""
    query_task_chars: int = 3000
    """How much of the task text is used as a scoring query."""


def _usage(response: dict) -> dict:
    usage = (response or {}).get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "cached_tokens": prompt_details.get("cached_tokens") or 0,
        "reasoning_tokens": completion_details.get("reasoning_tokens") or 0,
    }


class Ledger:
    def __init__(self, path: Path | None):
        self.path = path
        self.lock = threading.Lock()
        if path is not None:  # the runner creates the run's log dir only after the agents finish
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row: Any) -> None:
        if self.path is None:
            return
        row["t"] = round(time.time(), 3)
        with self.lock, self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")


class CompressingInbox:
    """The messaging connector, with every delivered message compressed.
    Covers both delivery paths: the inbox poll before each step and the
    replies returned by `send_message --wait`."""

    def __init__(self, comm: Any, agent: CpcAgent):
        self._comm = comm
        self._agent = agent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._comm, name)

    def receive(self) -> list[dict]:
        return [self._agent.compress_message(message) for message in self._comm.receive()]

    def send_and_wait(self, recipient: str, content: str, timeout: int = 60):
        delivered, replies = self._comm.send_and_wait(recipient, content, timeout=timeout)
        return delivered, [self._agent.compress_message(message) for message in replies]


class CpcAgent(DefaultAgent):
    def __init__(self, *args: Any, config_class: type = CpcAgentConfig, **kwargs: Any):
        super().__init__(*args, config_class=config_class, **kwargs)
        log_dir = getattr(_context, "log_dir", None)
        self.ledger = Ledger(Path(log_dir) / f"{self.agent_id}_ledger.jsonl" if log_dir else None)
        self.ledger.write(event="start", arm=self.config.cpc_arm, model=getattr(self.model.config, "model_name", ""))
        self._task = ""
        self._wrap_model()
        if self.comm is not None and self.config.cpc_arm in ("messages", "full"):
            self.comm = CompressingInbox(self.comm, self)

    # ---- token ledger -------------------------------------------------
    def _wrap_model(self) -> None:
        model, ledger = self.model, self.ledger
        query, summarize = model.query, getattr(model, "summarize_context", None)

        def logged_query(messages, **kwargs):
            message = query(messages, **kwargs)
            ledger.write(event="llm", kind="step", **_usage(message.get("extra", {}).get("response")))
            return message

        model.query = logged_query
        if callable(summarize):

            def logged_summarize(messages, **kwargs):
                message = summarize(messages, **kwargs)
                ledger.write(event="llm", kind="summary", **_usage(message.get("extra", {}).get("response")))
                return message

            model.summarize_context = logged_summarize

    def run(self, task: str = "", **kwargs: Any) -> dict:
        self._task = task
        try:
            return super().run(task=task, **kwargs)
        finally:
            self.ledger.write(event="end", steps=self.n_calls, compactions=self._compaction_count)

    # ---- compressor calls ---------------------------------------------
    def _task_query(self) -> str:
        return self._task[: self.config.query_task_chars]

    def _current_step(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return message_text(message)[: self.config.query_task_chars]
        return ""

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.config.cpc_url.rstrip("/") + "/compress",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.config.cpc_timeout) as response:
            return json.loads(response.read())

    def compress_message(self, message: dict) -> dict:
        content = message.get("content") or ""
        try:
            result = self._post(
                {
                    "entries": [{"role": "colleague", "text": content}],
                    "query": self._task_query(),
                    "extra_queries": [self._current_step()],
                    "ratio": self.config.message_ratio,
                    "min_tokens": self.config.message_min_tokens,
                }
            )
        except Exception as error:  # deliver it untouched rather than lose it
            self.ledger.write(event="compress_error", kind="message", error=str(error)[:300])
            return message
        self.ledger.write(event="compress", kind="message", sender=message.get("from"), **_stats(result))
        if result["units_kept"] == result["units_in"]:
            return message
        share = round(100 * result["tokens_out"] / max(result["tokens_in"], 1))
        return {**message, "content": f"(excerpt, {share}% of the original message)\n{result['text']}"}

    def _compact_messages(self) -> None:
        if self.config.cpc_arm != "full":
            super()._compact_messages()
            return
        prefix, conversation = self.messages[:2], self.messages[2:]
        boundary = self._find_turn_boundary(conversation, self.config.compaction_keep_recent_turns)
        old_turns, recent_turns = conversation[:boundary], conversation[boundary:]
        if not old_turns:
            return
        entries = [{"role": message.get("role", "user"), "text": message_text(message)} for message in old_turns]
        try:
            result = self._post(
                {
                    "entries": entries,
                    "query": self._current_step(),
                    "extra_queries": [self._task_query()],
                    "budget": self.config.history_budget,
                }
            )
        except Exception as error:  # keep the run alive with the stock compaction, and count it
            self.ledger.write(event="compress_error", kind="history", error=str(error)[:300])
            super()._compact_messages()
            return

        self._close_current_segment("solver")
        excerpt = self.model.format_message(
            role="user",
            content=(
                "[Your earlier steps were compressed to save context. Below are verbatim lines from them, "
                "kept for relevance to your task and your current step; '…' marks omitted lines. "
                "Re-read a file if you need more of it.]\n\n" + result["text"]
            ),
        )
        self._segments.append({"kind": "cpc_excerpt", "messages": [excerpt]})
        self.messages = prefix + [excerpt] + recent_turns
        self._compaction_count += 1
        self.ledger.write(
            event="compress",
            kind="history",
            prompt_tokens_before=self._last_prompt_tokens,
            messages=len(old_turns),
            **_stats(result),
        )


def _stats(result: dict) -> dict:
    return {key: result.get(key) for key in ("tokens_in", "tokens_out", "units_in", "units_kept", "budget", "ms")}


def install() -> None:
    """Make CooperBench's mini_swe_agent_v2 adapter build CpcAgent, and hand
    it the run's log_dir. Runs are threads in one process, so patching the
    module once covers every agent in the run."""
    runner = msa_adapter.MiniSweAgentV2Runner
    if getattr(runner.run, "_cpc_installed", False):
        return
    original = runner.run

    def run(self, *args: Any, log_dir: str | None = None, **kwargs: Any):
        _context.log_dir = log_dir
        try:
            return original(self, *args, log_dir=log_dir, **kwargs)
        finally:
            _context.log_dir = None

    run._cpc_installed = True  # type: ignore[attr-defined]
    runner.run = run
    msa_adapter.DefaultAgent = CpcAgent

    # On a shared machine, task containers get the resource limits in CB_TASK_LIMITS
    # (e.g. "--cpus=2 --memory=8g --cpu-shares=256"), the same for every arm.
    limits = shlex.split(os.environ.get("CB_TASK_LIMITS", ""))
    if limits:
        start = msa_docker.DockerEnvironment._start_container

        def limited_start(self: Any) -> Any:
            self.config.run_args = [*self.config.run_args, *(arg for arg in limits if arg not in self.config.run_args)]
            return start(self)

        msa_docker.DockerEnvironment._start_container = limited_start
