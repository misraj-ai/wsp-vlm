"""Agent backends. Each exposes reply(system, messages) -> (text, meta).

  anthropic      Claude through the Anthropic Messages API (ANTHROPIC_API_KEY). Streaming, so long
                 replies do not time out. agent.extra is passed through to the API call unchanged:
                 put any reasoning / thinking / effort settings for your model there.
  openai_compat  any OpenAI-compatible chat endpoint (for GPT models, Astra, or a local server):
                 agent.base_url, agent.api_key_env.
  scripted       a Python file defining reply(turn, last_message, state) -> str. For tests and for
                 hand-written baselines driven through the same harness.
"""
from __future__ import annotations

import importlib.util
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
    if b == "openai_compat":
        return OpenAICompatAgent(cfg)
    if b == "scripted":
        return ScriptedAgent(cfg)
    raise ValueError(f"unknown agent backend {b}")
