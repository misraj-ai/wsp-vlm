"""Runs INSIDE the agent's kernel process (CPU only). Executes code blocks in one persistent
namespace and replies on a dedicated file descriptor. Started by wsp.kernel.Kernel; not imported
by the harness itself.

Namespace given to the agent (documented in the task prompt):
  np, torch, json, os, math, re, Image (PIL)
  MODEL                       discovery info: blocks, writer matrices, frozen tensors, widths
  list_params(pattern=None)   -> [(name, shape, dtype)]
  W(name)                     original tensor (float32 copy)
  W_current(name)             edited tensor if one is saved, else the original
  save_edit(name, tensor)     replace a tensor (validated; refusals raise ValueError)
  remove_edit(name), clear_edits(), edits()
  practice(task=None, split=None) -> list of {id, task, split, image, reference}
  load_image(id)              PIL image of a practice item
  load_gen(k, model)          the per-item results of generation pass k ("original" | "edited")
  load_fwd(k, model)          the arrays of teacher-forced pass k
  WORK                        your working folder (persists across kernel restarts)
"""
import argparse
import contextlib
import io
import json
import math  # noqa: F401  (given to the agent)
import os
import re  # noqa: F401
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--reply-fd", type=int, required=True)
args = ap.parse_args()

if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)
sys.path.insert(0, args.pkg)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from wsp.edits import WeightsReader, store_for  # noqa: E402

torch.set_grad_enabled(False)


def _forbidden(*a, **k):
    raise RuntimeError("gradient computation and optimizers are not allowed in this experiment")


torch.Tensor.backward = _forbidden
torch.autograd.grad = _forbidden
torch.autograd.backward = _forbidden
torch.Tensor.requires_grad_ = lambda self, *a, **k: _forbidden() if (a and a[0]) or k.get("requires_grad", False) else self
import torch.optim as _optim  # noqa: E402

_Opt = _optim.Optimizer
for _n in dir(_optim):
    _c = getattr(_optim, _n)
    if isinstance(_c, type) and issubclass(_c, _Opt) and _c is not _Opt:
        setattr(_optim, _n, _forbidden)
_Opt.__init__ = _forbidden

RUN = args.run
META = os.path.join(RUN, "meta")
WORK = os.path.join(RUN, "work")
with open(os.path.join(META, "model.json")) as f:
    MODEL = json.load(f)
with open(os.path.join(META, "config.json")) as f:
    _CFG = json.load(f)
with open(os.path.join(META, "practice.json"), encoding="utf-8") as f:
    _PRACTICE = json.load(f)
_W = WeightsReader(os.path.join(RUN, "weights"))
_STORE = store_for(os.path.join(WORK, "edit"), os.path.join(RUN, "weights"), MODEL, _CFG)
_BY_ID = {r["id"]: r for r in _PRACTICE}


def list_params(pattern=None):
    rx = re.compile(pattern) if pattern else None
    return [(n, tuple(v["shape"]), v["dtype"]) for n, v in _W.index.items() if not rx or rx.search(n)]


def W(name):
    return _W.get(name).float()


def W_current(name):
    if name in _STORE.manifest():
        from safetensors import safe_open
        with safe_open(_STORE._file(name), framework="pt") as f:
            return f.get_tensor(name).float()
    return W(name)


def save_edit(name, tensor):
    r = _STORE.save(name, tensor)
    print(f"[save_edit] {name}: {r.get('changed_entries')} entries changed"
          + (f", relative change {r['rel_change_fro']:.4%}" if "rel_change_fro" in r else "") + (f" ({r['note']})" if "note" in r else ""))
    return r


def remove_edit(name):
    _STORE.remove(name)
    print(f"[remove_edit] {name}")


def clear_edits():
    _STORE.clear()
    print("[clear_edits] all edits removed")


def edits():
    return _STORE.summary()


def practice(task=None, split=None):
    return [dict(r) for r in _PRACTICE if (task is None or r["task"] == task) and (split is None or r["split"] == split)]


def load_image(item_id):
    return Image.open(_BY_ID[item_id]["image"]).convert("RGB")


def load_gen(k, model="original"):
    p = os.path.join(WORK, f"gen_{k}_{model}.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def load_fwd(k, model="original"):
    d = os.path.join(WORK, f"fwd_{k}", model)
    out = {"meta": json.load(open(os.path.join(d, "meta.json")))}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npy"):
            out[fn[:-4]] = np.load(os.path.join(d, fn), allow_pickle=False)
    return out


NS = {k: v for k, v in globals().items() if k in (
    "np", "torch", "json", "os", "math", "re", "Image", "MODEL", "WORK", "list_params", "W", "W_current",
    "save_edit", "remove_edit", "clear_edits", "edits", "practice", "load_image", "load_gen", "load_fwd")}
NS["__name__"] = "__agent__"
os.chdir(WORK)

reply = os.fdopen(args.reply_fd, "w", buffering=1)
for line in sys.stdin:
    msg = json.loads(line)
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(msg["code"], f"<run {msg['id']}>", "exec"), NS)
        except SystemExit:
            print("SystemExit ignored")
        except BaseException:
            traceback.print_exc()
    reply.write(json.dumps({"id": msg["id"], "output": buf.getvalue(), "seconds": round(time.time() - t0, 2)}) + "\n")
    reply.flush()
