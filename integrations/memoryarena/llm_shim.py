"""An OpenAI-compatible chat endpoint that forwards to Vertex Gemini through LiteLLM.

MemoryArena's agents and judges call `OpenAI(base_url=...).chat.completions.create`.
Pointing OPENAI_BASE_URL at this shim lets their code run unchanged; only the
transport differs. The request's `model` names a role and a tag, "agent|cpc" or
"judge|bm25": the role picks the real model and its reasoning level, and the tag
(the memory condition) goes into the usage log, so cost is known per condition.

Sampling parameters are not forwarded: Gemini 3.5 Flash-Lite and 3.6 Flash
ignore temperature and top-p (LiteLLM model map: supports_sampling_params false).

Gemini 3 returns a thought signature with each function call and expects it back
on the next turn. The OpenAI client rebuilds the assistant message from the id,
name and arguments only, so the shim remembers each call's provider fields by
call id and puts them back on the way in.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Shim:
    def __init__(self, roles: dict, usage_path: str | None = None, default_max_tokens: int = 8192, retries: int = 4):
        self.roles = roles  # role -> {"model": ..., "reasoning_effort": ... or None}
        self.usage_path = usage_path
        self.default_max_tokens = default_max_tokens
        self.retries = retries
        self.provider_fields: dict[str, dict] = {}  # tool call id -> provider fields (thought signature)
        self.lock = threading.Lock()

    def resolve(self, name: str) -> tuple[str, str, dict]:
        role, _, tag = (name or "").partition("|")
        if role not in self.roles:
            raise ValueError(f"unknown role {role!r} (model {name!r})")
        return role, tag, self.roles[role]

    def restore_provider_fields(self, messages: list[dict]) -> list[dict]:
        for message in messages:
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and "provider_specific_fields" not in call and call.get("id") in self.provider_fields:
                    call["provider_specific_fields"] = self.provider_fields[call["id"]]
        return messages

    def remember_provider_fields(self, body: dict) -> None:
        for choice in body.get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                fields = call.get("provider_specific_fields")
                if fields and call.get("id"):
                    self.provider_fields[call["id"]] = fields

    def complete(self, request: dict) -> dict:
        import litellm

        role, tag, spec = self.resolve(request.get("model", ""))
        kwargs = {
            "model": spec["model"],
            "messages": self.restore_provider_fields(request.get("messages") or []),
            "max_tokens": request.get("max_completion_tokens") or request.get("max_tokens") or self.default_max_tokens,
            "drop_params": True,
        }
        for key in ("tools", "tool_choice"):
            if request.get(key) is not None:
                kwargs[key] = request[key]
        if spec.get("reasoning_effort"):
            kwargs["reasoning_effort"] = spec["reasoning_effort"]
        start = time.perf_counter()
        error = ""
        for attempt in range(self.retries):
            try:
                body = litellm.completion(**kwargs).model_dump()
                self.remember_provider_fields(body)
                usage = body.get("usage") or {}
                self.log(role, tag, spec["model"], usage.get("prompt_tokens") or 0,
                         usage.get("completion_tokens") or 0, "", start)
                return body
            except Exception as caught:  # rate limits and transient API errors: back off and retry
                error = f"{type(caught).__name__}: {caught}"[:500]
                time.sleep(min(60, 5 * 2 ** attempt))
        self.log(role, tag, spec["model"], 0, 0, error, start)
        raise RuntimeError(error)

    def log(self, role, tag, model, prompt_tokens, completion_tokens, error, start) -> None:
        if not self.usage_path:
            return
        row = {"role": role, "tag": tag, "model": model, "prompt_tokens": prompt_tokens,
               "completion_tokens": completion_tokens, "error": error,
               "ms": round((time.perf_counter() - start) * 1000, 1), "t": time.time()}
        with self.lock, open(self.usage_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def serve(shim: Shim, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the shim on 127.0.0.1 in a daemon thread -> (server, base_url ending in /v1)."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._reply(404, {"error": {"message": f"no route {self.path}"}})
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self._reply(200, shim.complete(request))
            except Exception as caught:
                self._reply(500, {"error": {"message": f"{type(caught).__name__}: {caught}"[:500]}})

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"
