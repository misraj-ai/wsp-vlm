"""Weight export, the edit store (with the refusal rules), and applying an edit to a loaded model.

An "edit" is a set of whole tensors that replace the originals. The agent writes them with
save_edit() in its kernel; the harness re-validates them before every forward pass and before
the private evaluation. Edits are stored in float32 and cast to the model dtype when applied, in
the forward tool and in the evaluation alike, so the agent tests exactly what is evaluated.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DT = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16, "torch.float32": torch.float32}


def export_weights(model, out_dir: str, shard_bytes: int = 2 * 1024 ** 3) -> dict:
    """Write every parameter under its model name (the names the agent and the edits use)."""
    os.makedirs(out_dir, exist_ok=True)
    index, shard, size, k = {}, {}, 0, 0

    def flush():
        nonlocal shard, size, k
        if shard:
            fn = f"weights-{k:05d}.safetensors"
            save_file(shard, os.path.join(out_dir, fn))
            for n in shard:
                index[n]["file"] = fn
            shard, size, k = {}, 0, k + 1

    for n, p in model.named_parameters():
        t = p.detach().cpu().contiguous()
        index[n] = {"shape": list(t.shape), "dtype": str(t.dtype)}
        shard[n] = t
        size += t.numel() * t.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=0)
    return index


class WeightsReader:
    def __init__(self, d: str):
        self.dir = d
        with open(os.path.join(d, "index.json")) as f:
            self.index = json.load(f)
        self._handles = {}

    def get(self, name: str) -> torch.Tensor:
        if name not in self.index:
            raise KeyError(f"no tensor named {name}")
        fn = self.index[name]["file"]
        if fn not in self._handles:
            self._handles[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self._handles[fn].get_tensor(name)

    def dtype(self, name: str) -> torch.dtype:
        return DT[self.index[name]["dtype"]]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16
                          else t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


class EditStore:
    def __init__(self, d: str, weights: WeightsReader, frozen: list[str], frozen_patterns: list[str] = (),
                 refuse_rescale: bool = True, vision_prefix: str | None = None, connector: list[str] = (),
                 allow_vision: bool = True, allow_connector: bool = True):
        self.dir = d
        self.w = weights
        self.frozen = set(frozen)
        self.frozen_patterns = [re.compile(p) for p in frozen_patterns]
        self.refuse_rescale = refuse_rescale
        self.vision_prefix, self.connector = vision_prefix, list(connector)
        self.allow_vision, self.allow_connector = allow_vision, allow_connector
        os.makedirs(os.path.join(d, "tensors"), exist_ok=True)
        self.mpath = os.path.join(d, "manifest.json")
        if not os.path.exists(self.mpath):
            self._write({})

    # -- manifest ---------------------------------------------------------------------------
    def manifest(self) -> dict:
        with open(self.mpath) as f:
            return json.load(f)

    def _write(self, m):
        tmp = self.mpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1)
        os.replace(tmp, self.mpath)

    def digest(self) -> str:
        m = self.manifest()
        return hashlib.sha256(json.dumps(sorted((k, v["sha256"]) for k, v in m.items())).encode()).hexdigest()[:16]

    # -- rules ------------------------------------------------------------------------------
    def check_name(self, name: str):
        if name not in self.w.index:
            raise ValueError(f"no tensor named {name}")
        if name in self.frozen or any(p.search(name) for p in self.frozen_patterns):
            raise ValueError(f"{name} is frozen (embeddings, output layer and final norm are out of scope)")
        if not self.allow_vision and self.vision_prefix and name.startswith(self.vision_prefix + "."):
            raise ValueError(f"{name}: vision-tower edits are disabled in this condition")
        if not self.allow_connector and any(name.startswith(c + ".") for c in self.connector):
            raise ValueError(f"{name}: connector edits are disabled in this condition")

    def validate(self, name: str, t: torch.Tensor) -> torch.Tensor:
        self.check_name(name)
        if not isinstance(t, torch.Tensor):
            raise ValueError("edit must be a torch.Tensor")
        t = t.detach().to("cpu", torch.float32)
        shape = tuple(self.w.index[name]["shape"])
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)} != {shape}")
        if not torch.isfinite(t).all():
            raise ValueError(f"{name}: contains NaN or inf")
        if self.refuse_rescale and t.numel() > 1:
            o = self.w.get(name).float()
            oo = float((o * o).sum())
            if oo > 0:
                c = float((t * o).sum()) / oo
                resid = float(((t - c * o) ** 2).sum()) ** 0.5
                if resid <= 1e-6 * max(1e-12, float((t * t).sum()) ** 0.5) and abs(c - 1) > 1e-6:
                    raise ValueError(f"{name}: an edit that only multiplies the whole tensor by a constant ({c:.4g}) is refused")
        return t

    # -- write / read -------------------------------------------------------------------------
    def _file(self, name):
        return os.path.join(self.dir, "tensors", name.replace("/", "_") + ".safetensors")

    def save(self, name: str, t: torch.Tensor) -> dict:
        t = self.validate(name, t)
        dt = self.w.dtype(name)
        o = self.w.get(name)
        cast = t.to(dt)
        changed = int((cast != o.to(dt)).sum())
        m = self.manifest()
        if changed == 0:
            self.remove(name)
            return {"name": name, "changed_entries": 0, "note": "identical to the original after casting; edit removed"}
        save_file({name: t.contiguous()}, self._file(name))
        of = o.float()
        info = {"shape": list(t.shape), "model_dtype": str(dt), "changed_entries": changed, "numel": t.numel(),
                "rel_change_fro": round(float((cast.float() - of).norm() / of.norm().clamp_min(1e-12)), 6),
                "sha256": _sha(cast), "base_sha256": _sha(o.to(dt))}
        m[name] = info
        self._write(m)
        return dict(info, name=name)

    def remove(self, name: str):
        m = self.manifest()
        m.pop(name, None)
        self._write(m)
        if os.path.exists(self._file(name)):
            os.remove(self._file(name))

    def clear(self):
        for n in list(self.manifest()):
            self.remove(n)

    def load_all(self) -> dict[str, torch.Tensor]:
        """Re-validate and load every edited tensor (float32). Raises if any file is inconsistent."""
        out = {}
        for name, info in self.manifest().items():
            self.check_name(name)
            with safe_open(self._file(name), framework="pt") as f:
                t = f.get_tensor(name)
            t = self.validate(name, t)
            if _sha(t.to(self.w.dtype(name))) != info["sha256"]:
                raise ValueError(f"{name}: tensor file does not match its manifest hash")
            out[name] = t
        return out

    def summary(self) -> dict:
        m = self.manifest()
        return {"tensors": len(m), "changed_entries": sum(v["changed_entries"] for v in m.values()),
                "digest": self.digest(), "by_tensor": {k: {"changed": v["changed_entries"], "rel_change": v["rel_change_fro"]}
                                                       for k, v in m.items()}}

    def snapshot(self, dest: str):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(self.dir, dest)


def load_submission(sub_dir: str, model, disc: dict, cfg: dict) -> tuple[dict, dict]:
    """Validate a submitted edit against a freshly loaded model (independent of the run's files):
    names exist and are not frozen, shapes match, the base tensors hash to what the edit was made
    against (same base revision), and the tensors match their manifest. Returns (edits, report)."""
    with open(os.path.join(sub_dir, "manifest.json")) as f:
        man = json.load(f)
    params = dict(model.named_parameters())
    frozen = set(disc["frozen"])
    pats = [re.compile(p) for p in cfg["editing"].get("frozen_patterns", [])]
    out, report = {}, {"tensors": len(man), "changed_entries": 0}
    for name, info in man.items():
        if name not in params:
            raise ValueError(f"submission edits unknown tensor {name}")
        if name in frozen or any(p.search(name) for p in pats):
            raise ValueError(f"submission edits frozen tensor {name}")
        p = params[name]
        if _sha(p.detach().cpu()) != info["base_sha256"]:
            raise ValueError(f"{name}: the loaded base weights differ from the ones the edit was made against")
        fn = os.path.join(sub_dir, "tensors", name.replace("/", "_") + ".safetensors")
        with safe_open(fn, framework="pt") as f:
            t = f.get_tensor(name).to(torch.float32)
        if tuple(t.shape) != tuple(p.shape) or not torch.isfinite(t).all():
            raise ValueError(f"{name}: bad shape or non-finite values")
        if _sha(t.to(p.dtype)) != info["sha256"]:
            raise ValueError(f"{name}: tensor does not match its manifest hash")
        out[name] = t
        report["changed_entries"] += info["changed_entries"]
    report["digest"] = hashlib.sha256(json.dumps(sorted((k, v["sha256"]) for k, v in man.items())).encode()).hexdigest()[:16]
    return out, report


def apply_permanently(model, edits: dict):
    params = dict(model.named_parameters())
    for n, t in edits.items():
        params[n].data.copy_(t.to(device=params[n].device, dtype=params[n].dtype))


def store_for(edit_dir: str, weights_dir: str, disc: dict, cfg: dict) -> EditStore:
    e = cfg["editing"]
    return EditStore(edit_dir, WeightsReader(weights_dir), disc["frozen"], e.get("frozen_patterns", []),
                     e.get("refuse_rescale", True), disc.get("vision_root"),
                     disc.get("connector", []), e.get("allow_vision", True), e.get("allow_connector", True))
