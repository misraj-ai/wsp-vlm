"""Shared evaluation code (private test sets, decode sweeps, pilots): generate + grade + summarize."""
from __future__ import annotations

import json
import os
import time

from . import model_io
from .forward import summarize
from .grading import grade


def run_items(model, processor, cfg, items, protocol=None, progress=True) -> list[dict]:
    proto = protocol or cfg["protocol"]
    rows, t0 = [], time.time()
    chunk = max(1, int(proto.get("batch_size", 8))) * 8
    for b in range(0, len(items), chunk):
        part = items[b:b + chunk]
        outs = model_io.generate(model, processor, part, cfg["tasks"], proto, cfg["model"].get("chat_template_kwargs"))
        for it, o in zip(part, outs):
            g = grade(cfg["tasks"][it["task"]], o["text"], it["reference"], cfg["grading"], o["hit_max"])
            rows.append(dict({"id": it["id"], "task": it["task"], "text": o["text"], "n_tokens": o["n_tokens"]}, **g))
        if progress:
            print(f"  {len(rows)}/{len(items)} items, {time.time() - t0:.0f} s", flush=True)
    return rows


def write_results(path: str, rows: list[dict], tasks: dict, meta: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summarize(rows, tasks), "items": rows}, f, ensure_ascii=False, indent=0)
