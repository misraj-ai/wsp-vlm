"""Client for the agent's persistent Python kernel (a separate CPU-only process)."""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_boot.py")


class Kernel:
    def __init__(self, run_dir: str, python: str | None = None, sandbox_cmd: list[str] | None = None):
        self.run_dir = os.path.abspath(run_dir)
        self.python = python or sys.executable
        self.sandbox = list(sandbox_cmd or [])
        self.proc = None
        self.n = 0

    def start(self):
        r, w = os.pipe()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC", "OPENAI", "HF_TOKEN", "HUGGING"))}
        env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8")})
        cmd = self.sandbox + [self.python, "-I", BOOT, "--pkg", PKG, "--run", self.run_dir, "--reply-fd", str(w)]
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        self.errlog = os.path.join(self.run_dir, "logs", "kernel_stderr.log")
        self._err = open(self.errlog, "a")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err,
                                     text=True, pass_fds=(w,), env=env)
        os.close(w)
        self.reply = os.fdopen(r, "r")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def run(self, code: str, timeout: float) -> dict:
        if not self.alive():
            self.start()
        self.n += 1
        t0 = time.time()
        try:
            self.proc.stdin.write(json.dumps({"id": self.n, "code": code}) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            return self._died(t0)
        ready, _, _ = select.select([self.reply], [], [], timeout)
        if not ready:
            self.stop()
            return {"output": f"TIMEOUT after {timeout:.0f} s: the kernel was restarted. Variables are lost; files in your "
                              "working folder and saved edits remain.", "seconds": round(time.time() - t0, 1), "restarted": True}
        line = self.reply.readline()
        if not line:
            return self._died(t0)
        return json.loads(line)

    def _died(self, t0):
        err = ""
        try:
            self._err.flush()
            err = open(self.errlog).read()[-3000:]
        except Exception:
            pass
        self.stop()
        return {"output": "The kernel process died (often out of memory) and was restarted. Variables are lost; files "
                          f"and saved edits remain.\n{err}", "seconds": round(time.time() - t0, 1), "restarted": True}
