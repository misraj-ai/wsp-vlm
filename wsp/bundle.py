"""Submission bundles: the ONLY thing that moves from the agent machine to the evaluation machine.

A bundle (a .tar.gz) holds:
  spec.json       everything the evaluation needs, frozen at export time: model id + revision + dtype +
                  loading options, the evaluation protocol, ALL task definitions (prompts, metrics),
                  grading options, frozen-tensor patterns, and which tasks the run was about
  submission/     manifest.json + tensors/ (the edit; empty for the unedited base model)
  run_info.json   run name, experiment, condition, seed, budget used, agent backend/model, edit digest,
                  a hash of the harness code

It holds no practice data, no outputs and no private data. The evaluation machine reads the bundle and
the private test set, and nothing from the agent machine's run directories. Results stay on the
evaluation machine; only the aggregate summaries (no texts, no references) are meant to leave it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile

SPEC_KEYS_MODEL = ("path", "revision", "dtype", "trust_remote_code", "attn_implementation", "auto_class",
                   "chat_template_kwargs", "discovery")


def code_hash() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith((".py", ".md")):
                with open(os.path.join(dp, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()[:16]


def make_spec(cfg: dict) -> dict:
    return {"model": {k: cfg["model"].get(k) for k in SPEC_KEYS_MODEL},
            "protocol": cfg["protocol"], "tasks": cfg["tasks"], "grading": cfg["grading"],
            "editing": {"frozen_patterns": cfg["editing"].get("frozen_patterns", [])},
            "experiment": cfg["experiment"]}


def _add_bytes(tar, name, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def export(out_path: str, cfg: dict, submission_dir: str | None, run_info: dict) -> str:
    """Write a bundle. submission_dir=None exports an empty edit (the unedited model, condition A0)."""
    if not out_path.endswith(".tar.gz"):
        out_path += ".tar.gz"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    spec = make_spec(cfg)
    info = dict(run_info, code_hash=code_hash())
    with tarfile.open(out_path, "w:gz") as tar:
        _add_bytes(tar, "spec.json", json.dumps(spec, indent=1, ensure_ascii=False).encode())
        if submission_dir:
            man = os.path.join(submission_dir, "manifest.json")
            _add_bytes(tar, "submission/manifest.json", open(man, "rb").read())
            for name in json.load(open(man)):
                fn = name.replace("/", "_") + ".safetensors"
                tar.add(os.path.join(submission_dir, "tensors", fn), arcname=f"submission/tensors/{fn}")
        else:
            _add_bytes(tar, "submission/manifest.json", b"{}")
        _add_bytes(tar, "run_info.json", json.dumps(info, indent=1, ensure_ascii=False).encode())
    return out_path


def open_bundle(path: str) -> tuple[dict, str, dict, str]:
    """Unpack a bundle (or use an unpacked folder). Returns (spec, submission_dir, run_info, tmpdir or '')."""
    tmp = ""
    if os.path.isdir(path):
        root = path
    else:
        tmp = tempfile.mkdtemp(prefix="wsp_bundle_")
        with tarfile.open(path, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                    raise ValueError(f"unsafe member in bundle: {m.name}")
            tar.extractall(tmp, filter="data")
        root = tmp
    spec = json.load(open(os.path.join(root, "spec.json")))
    info = json.load(open(os.path.join(root, "run_info.json")))
    return spec, os.path.join(root, "submission"), info, tmp


def cleanup(tmp: str):
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
