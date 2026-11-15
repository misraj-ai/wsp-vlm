"""The openrouter backend against a local fake of OpenRouter's chat-completions API (no network, no key):
request shape, streaming with keep-alive comments, retries, non-retryable errors, and a full agent run
whose replies come over HTTP."""
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from wsp import agents

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Fake:
    """Scripted responses, one per request: ("stream", text) | ("json", text) | ("status", code) |
    ("midstream_error",) | ("length",) | ("mock_agent",) -> reply produced by tests/mock_agent.py"""

    def __init__(self):
        self.queue, self.requests = [], []
        spec = importlib.util.spec_from_file_location("mock_agent", os.path.join(ROOT, "tests", "mock_agent.py"))
        self.mock = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mock)
        self.turn, self.state = 0, {}


def make_handler(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _sse(self, chunks):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": OPENROUTER PROCESSING\n\n")
            for c in chunks:
                self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            kind = fake.queue.pop(0) if fake.queue else ("mock_agent",)
            if kind[0] == "mock_agent":
                fake.turn += 1
                kind = ("stream", fake.mock.reply(fake.turn, body["messages"][-1]["content"], fake.state))
            if kind[0] == "status":
                self.send_response(kind[1])
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"code": kind[1], "message": "fake"}}).encode())
            elif kind[0] == "json":
                out = {"id": "gen-1", "model": body["model"], "provider": "Fake",
                       "choices": [{"message": {"role": "assistant", "content": kind[1]}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif kind[0] == "midstream_error":
                self._sse([{"id": "gen-2", "choices": [{"delta": {"content": "par"}}]},
                           {"error": {"code": 502, "message": "provider disconnected"}}])
            elif kind[0] == "length":
                self._sse([{"id": "gen-3", "choices": [{"delta": {"content": ""}, "finish_reason": "length"}]}])
            else:
                text = kind[1]
                pieces = [text[i:i + 17] for i in range(0, len(text), 17)] or [""]
                chunks = [{"id": "gen-4", "model": body["model"], "provider": "Fake",
                           "choices": [{"delta": {"content": p}}]} for p in pieces]
                chunks.append({"id": "gen-4", "choices": [{"delta": {}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": len(text)}})
                self._sse(chunks)
    return H


@pytest.fixture
def server(monkeypatch):
    fake = Fake()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield fake, f"http://127.0.0.1:{srv.server_address[1]}/api/v1"
    srv.shutdown()


def cfg(url, **kw):
    c = {"backend": "openrouter", "model": "some-lab/some-model", "max_tokens": 4000, "base_url": url,
         "extra": {"reasoning": {"effort": "high"}, "provider": {"sort": "throughput"}},
         "headers": {"X-Title": "wsp-vlm"}, "retry_delay_s": 0.01, "timeout_s": 30}
    c.update(kw)
    return c


def test_request_shape_and_streaming(server):
    fake, url = server
    fake.queue = [("stream", "Hello from the stream, with Arabic: مرحبا")]
    a = agents.make(cfg(url))
    text, meta = a.reply("Answer the user's request.", [{"role": "user", "content": "hi"}])
    assert text == "Hello from the stream, with Arabic: مرحبا"
    assert meta["finish_reason"] == "stop" and meta["usage"]["prompt_tokens"] == 100 and meta["attempts"] == 1
    r = fake.requests[0]
    assert r["path"] == "/api/v1/chat/completions"
    assert r["headers"]["Authorization"] == "Bearer sk-or-test" and r["headers"]["X-Title"] == "wsp-vlm"
    b = r["body"]
    assert b["model"] == "some-lab/some-model" and b["max_tokens"] == 4000 and b["stream"] is True
    assert b["messages"][0] == {"role": "system", "content": "Answer the user's request."}
    assert b["reasoning"] == {"effort": "high"} and b["provider"] == {"sort": "throughput"}


def test_retries_then_succeeds(server):
    fake, url = server
    fake.queue = [("status", 429), ("midstream_error",), ("status", 503), ("stream", "ok")]
    text, meta = agents.make(cfg(url)).reply("s", [{"role": "user", "content": "x"}])
    assert text == "ok" and meta["attempts"] == 4


def test_non_retryable_errors_stop(server):
    fake, url = server
    for code in (400, 401, 402):
        fake.queue = [("status", code)]
        with pytest.raises(agents.AgentAPIError, match=f"HTTP {code}"):
            agents.make(cfg(url)).reply("s", [{"role": "user", "content": "x"}])
    fake.queue = [("length",)]
    with pytest.raises(agents.AgentAPIError, match="max_tokens"):
        agents.make(cfg(url)).reply("s", [{"role": "user", "content": "x"}])


def test_non_streaming_and_missing_key(server, monkeypatch):
    fake, url = server
    fake.queue = [("json", "plain reply")]
    text, meta = agents.make(cfg(url, stream=False)).reply("s", [{"role": "user", "content": "x"}])
    assert text == "plain reply" and meta["provider"] == "Fake" and fake.requests[-1]["body"]["stream"] is False
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(agents.AgentAPIError, match="OPENROUTER_API_KEY"):
        agents.make(cfg(url))


def test_full_agent_run_over_openrouter(server, toy_cfg):
    """The mock agent's replies served over HTTP: the whole driver loop through the openrouter backend."""
    from wsp import config as config_mod
    from wsp.driver import Run

    fake, url = server
    c = config_mod.load(toy_cfg, {"agent": cfg(url)})
    r = Run(c)
    r.loop()
    assert r.state["done"] and r.state["passes"] == 6 and r.state["failed"] == 1
    assert len(fake.requests) == 13
    # the conversation sent each turn grows and alternates user/assistant; handoffs restart it
    sizes = [len(q["body"]["messages"]) for q in fake.requests]
    assert sizes[:3] == [2, 4, 6] and sizes[6] == 2
    meta = json.load(open(os.path.join(r.dir, "turns", "t001", "meta.json")))
    assert meta["usage"]["prompt_tokens"] == 100 and meta["provider"] == "Fake"
