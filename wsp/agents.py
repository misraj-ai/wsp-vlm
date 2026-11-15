"""Agent backends. Each exposes reply(system, messages) -> (text, meta).

  anthropic      Claude through the Anthropic Messages API (ANTHROPIC_API_KEY). Streaming, so long
                 replies do not time out. agent.extra is passed through to the API call unchanged:
                 put any reasoning / thinking / effort settings for your model there.
  openrouter     any model on OpenRouter (OPENROUTER_API_KEY), streamed over its OpenAI-compatible chat API
                 (standard library only). agent.model is the OpenRouter model id; agent.extra is merged into the
                 request body, e.g. {"reasoning": {"effort": "high"}} or {"provider": {...}}.
  openai_compat  any OpenAI-compatible chat endpoint (for GPT models, Astra, or a local server):
                 agent.base_url, agent.api_key_env.
  scripted       a Python file defining reply(turn, last_message, state) -> str. For tests and for
                 hand-written baselines driven through the same harness.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time


class AnthropicAgent:
    def __init__(self, cfg):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.cfg = cfg

    def reply(self, system, messages):
        delay = 10
        for attempt in range(8):
            try:
                with self.client.messages.stream(model=self.cfg["model"], max_tokens=self.cfg["max_tokens"],
                                                 system=system, messages=messages, **(self.cfg.get("extra") or {})) as s:
                    msg = s.get_final_message()
                text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
                usage = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage)
                return text, {"usage": usage, "stop_reason": msg.stop_reason, "model": msg.model, "attempts": attempt + 1}
            except (self.anthropic.APIConnectionError, self.anthropic.RateLimitError,
                    self.anthropic.InternalServerError, self.anthropic.APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 600)
        raise RuntimeError("the model API kept failing; the run can be resumed later")


class OpenAICompatAgent:
    def __init__(self, cfg):
        import openai

        self.client = openai.OpenAI(base_url=cfg.get("base_url"), api_key=os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY")))
        self.cfg = cfg

    def reply(self, system, messages):
        delay = 10
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(model=self.cfg["model"],
                                                        messages=[{"role": "system", "content": system}] + messages,
                                                        **(self.cfg.get("extra") or {}))
                return r.choices[0].message.content or "", {"usage": r.usage.model_dump() if r.usage else None,
                                                            "attempts": attempt + 1}
            except Exception:  # noqa: BLE001 - provider-specific errors
                if attempt == 7:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 600)


class AgentAPIError(RuntimeError):
    """A non-retryable API error (bad request, auth, no credit). The run stops; it can be resumed later."""


class OpenRouterAgent:
    """OpenRouter chat completions over its OpenAI-compatible API, streamed (server-sent events) by default, using
    only the standard library. Retries rate limits, server errors, dropped connections and mid-stream provider
    errors; stops on auth / credit / bad-request errors. Reasoning tokens count against max_tokens on OpenRouter,
    so leave room for the visible reply when you set a reasoning effort."""

    RETRY = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
    NETWORK = 599  # our code for a dropped connection / unreadable stream

    def __init__(self, cfg):
        self.cfg = cfg
        env = cfg.get("api_key_env") or "OPENROUTER_API_KEY"
        key = os.environ.get(env)
        if not key:
            raise AgentAPIError(f"set {env} to use the openrouter backend")
        self.stream = bool(cfg.get("stream", True))
        self.url = (cfg.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                        "Accept": "text/event-stream" if self.stream else "application/json",
                        **(cfg.get("headers") or {})}
        self.timeout = float(cfg.get("timeout_s", 1800))

    def _body(self, system, messages):
        body = {"model": self.cfg["model"], "max_tokens": self.cfg["max_tokens"],
                "messages": [{"role": "system", "content": system}] + messages}
        body.update(self.cfg.get("extra") or {})
        body["stream"] = self.stream
        return body

    def _once(self, body):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=self.headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            return None, e.code, e.read().decode(errors="replace")[:2000]
        meta = {"finish_reason": None, "usage": None, "model": None, "provider": None, "id": None}
        with resp:
            if not self.stream:
                d = json.loads(resp.read().decode())
                if d.get("error"):
                    return None, int(d["error"].get("code") or 502), json.dumps(d["error"])[:2000]
                ch = d["choices"][0]
                return dict(meta, text=ch["message"].get("content") or "", finish_reason=ch.get("finish_reason"),
                            usage=d.get("usage"), model=d.get("model"), provider=d.get("provider"), id=d.get("id")), 200, ""
            parts = []
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):   # keep-alive comments
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):                                                     # mid-stream provider error
                    return None, int(d["error"].get("code") or 502), json.dumps(d["error"])[:2000]
                for k in ("model", "provider", "id"):
                    meta[k] = d.get(k) or meta[k]
                if d.get("usage"):
                    meta["usage"] = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
                    if ch.get("finish_reason"):
                        meta["finish_reason"] = ch["finish_reason"]
        return dict(meta, text="".join(parts)), 200, ""

    def reply(self, system, messages):
        import urllib.error

        body = self._body(system, messages)
        delay, last = float(self.cfg.get("retry_delay_s", 10)), ""
        for attempt in range(int(self.cfg.get("max_attempts", 8))):
            try:
                res, status, err = self._once(body)
            except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as e:   # dropped connection, bad chunk
                res, status, err = None, self.NETWORK, f"{type(e).__name__}: {e}"
            if res is not None:
                if not res["text"].strip() and res["finish_reason"] == "length":
                    raise AgentAPIError("the reply was empty because max_tokens ran out (reasoning tokens count against "
                                        "max_tokens on OpenRouter): raise agent.max_tokens or lower the reasoning effort")
                text = res.pop("text")
                return text, dict(res, attempts=attempt + 1)
            last = f"HTTP {status}: {err}"
            if status not in self.RETRY and status != self.NETWORK:
                raise AgentAPIError(f"OpenRouter refused the request ({last})")
            time.sleep(delay)
            delay = min(delay * 2, 600)
        raise RuntimeError(f"OpenRouter kept failing ({last}); the run can be resumed later")


class ScriptedAgent:
    def __init__(self, cfg):
        spec = importlib.util.spec_from_file_location("wsp_script", cfg["script"])
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.turn = 0
        self.state = {}

    def reply(self, system, messages):
        self.turn += 1
        return self.mod.reply(self.turn, messages[-1]["content"], self.state), {"scripted": True}


def make(cfg):
    b = cfg["backend"]
    if b == "anthropic":
        return AnthropicAgent(cfg)
    if b == "openrouter":
        return OpenRouterAgent(cfg)
    if b == "openai_compat":
        return OpenAICompatAgent(cfg)
    if b == "scripted":
        return ScriptedAgent(cfg)
    raise ValueError(f"unknown agent backend {b}")
