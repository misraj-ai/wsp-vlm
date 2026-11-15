"""Bubblewrap sandbox for the agent's kernel, following the original repo's scripts/mathedit/sandbox.py:
no network, a clean environment, a read-only system, a memory cap, and only these paths visible:

  /work            read-write   the agent's folder (edit/, notes.md, pass results)
  /run/meta        read-only    model description, config, practice index (practice.json)
  /run/weights     read-only    the exported base weights
  /opt/wsp_pkg     read-only    this package (the kernel's helpers)
  <practice dirs>  read-only    the folders holding practice images, at their own absolute paths
  /usr, the Python installation (read-only), /proc, /dev, a private /tmp

Everything else on the machine (home folders, other runs, API configuration, any private data) does not
exist inside the sandbox. Unlike the original, the kernel process is persistent: variables survive
between runs (files in /work survive restarts too).
"""
from __future__ import annotations

import os
import shutil
import site
import sys

WORK, META, WEIGHTS, PKG = "/work", "/run/meta", "/run/weights", "/opt/wsp_pkg"


def available() -> bool:
    return shutil.which("bwrap") is not None


def _python_roots() -> list[str]:
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                           os.path.dirname(os.path.dirname(os.path.realpath(sys.executable))))}
    try:
        roots |= {os.path.realpath(p) for p in site.getsitepackages()}
    except AttributeError:
        pass
    roots = sorted(r for r in roots if not r.startswith("/usr/") and r != "/usr" and os.path.isdir(r))
    return [r for r in roots if not any(r != o and r.startswith(o + "/") for o in roots)]


def _minimal_dirs(dirs) -> list[str]:
    ds = sorted({os.path.realpath(d) for d in dirs})
    return [d for d in ds if not any(d != o and d.startswith(o + "/") for o in ds)]


def command(python: str, boot: str, pkg_root: str, run_dir: str, practice_dirs: list[str], reply_fd: int,
            extra_ro: list[str] = (), threads: int = 8) -> list[str]:
    run_dir = os.path.realpath(run_dir)
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for r in _python_roots() + [os.path.realpath(p) for p in extra_ro]:
        cmd += ["--ro-bind", r, r]
    cmd += ["--ro-bind", pkg_root, PKG,
            "--ro-bind", os.path.join(run_dir, "meta"), META,
            "--ro-bind", os.path.realpath(os.path.join(run_dir, "weights")), WEIGHTS]
    for d in _minimal_dirs(practice_dirs):
        cmd += ["--ro-bind", d, d]
    cmd += ["--bind", os.path.join(run_dir, "work"), WORK, "--chdir", WORK,
            "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin", "--setenv", "OMP_NUM_THREADS", str(threads),
            "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "PYTHONUNBUFFERED", "1"]
    rel_boot = os.path.relpath(boot, pkg_root)
    cmd += [python, "-I", os.path.join(PKG, rel_boot), "--pkg", PKG, "--meta", META, "--work", WORK,
            "--weights", WEIGHTS, "--reply-fd", str(reply_fd)]
    return cmd
