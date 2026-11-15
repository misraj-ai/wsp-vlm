"""Client for the agent's persistent Python kernel: a separate CPU-only process, sandboxed with
bubblewrap by default (see wsp/sandbox.py). A run refuses to start without the sandbox unless the
config explicitly allows it (kernel.sandbox: none + kernel.allow_unsandboxed: true, for tests only)."""
from __future__ import annotations

import json
import os
import resource
import select
import subprocess
import sys
import time

from . import sandbox

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_boot.py")


class SandboxError(RuntimeError):
    pass


class Kernel:
    def __init__(self, run_dir: str, kcfg: dict | None = None, practice_dirs: list[str] = ()):
        k = kcfg or {}
        self.run_dir = os.path.realpath(run_dir)
        self.python = k.get("python") or sys.executable
        self.mode = k.get("sandbox", "bwrap")
        self.mem = int(float(k.get("mem_gb", 96)) * (1 << 30))
        self.extra_ro = list(k.get("extra_ro_binds") or [])
        self.threads = int(k.get("threads", 8))
        self.practice_dirs = list(practice_dirs)
        if self.mode == "bwrap":
            if not sandbox.available():
                raise SandboxError("bubblewrap (bwrap) is not installed: install it (apt-get install bubblewrap), or for tests "
                                   "only set kernel.sandbox: none and kernel.allow_unsandboxed: true")
        elif self.mode == "none":
            if not k.get("allow_unsandboxed"):
                raise SandboxError("kernel.sandbox: none needs kernel.allow_unsandboxed: true; without the sandbox the agent's "
                                   "code can read every file this user can read")
        else:
            raise SandboxError(f"unknown kernel.sandbox '{self.mode}' (bwrap | none)")
        self.proc = None
        self.n = 0

    def _limits(self):
        if self.mem > 0:
            resource.setrlimit(resource.RLIMIT_AS, (self.mem, self.mem))

    def start(self):
        r, w = os.pipe()
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        self.errlog = os.path.join(self.run_dir, "logs", "kernel_stderr.log")
        self._err = open(self.errlog, "a")
        if self.mode == "bwrap":
            cmd = sandbox.command(self.python, BOOT, PKG, self.run_dir, self.practice_dirs, w, self.extra_ro, self.threads)
            env = {}
        else:
            cmd = [self.python, "-I", BOOT, "--pkg", PKG, "--meta", os.path.join(self.run_dir, "meta"),
                   "--work", os.path.join(self.run_dir, "work"), "--weights", os.path.join(self.run_dir, "weights"),
                   "--reply-fd", str(w)]
            env = {k: v for k, v in os.environ.items()
                   if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "ANTHROPIC", "OPENAI", "HF_", "HUGGING"))}
            env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(self.threads)})
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err, text=True,
                                     pass_fds=(w,), env=env, preexec_fn=self._limits)
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
