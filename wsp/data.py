"""Task data. One JSONL file per task, one object per line:

    {"id": "table_000123", "image": "images/t123.png", "reference": "<table>...</table>", "meta": {...}}

`image` is relative to the JSONL file's directory (or absolute). `reference` is the gold transcription
(plain text for cer tasks, HTML or markdown table for table tasks). Optional "prompt" overrides the
task prompt for that item. The SAME format is used for public practice data and the private set.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

from PIL import Image


def load_task_file(path: str, task: str) -> list[dict]:
    base = os.path.dirname(os.path.abspath(path))
    items, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in ("id", "image", "reference"):
                if k not in r:
                    raise ValueError(f"{path}:{ln}: missing '{k}'")
            if r["id"] in seen:
                raise ValueError(f"{path}:{ln}: duplicate id {r['id']}")
            seen.add(r["id"])
            img = r["image"] if os.path.isabs(r["image"]) else os.path.join(base, r["image"])
            items.append({"id": str(r["id"]), "task": task, "image": img, "reference": r["reference"],
                          "prompt": r.get("prompt"), "meta": r.get("meta", {})})
    return items


def load_dir(d: str, tasks: list[str], max_per_task: int | None = None) -> list[dict]:
    out = []
    for t in tasks:
        p = os.path.join(d, f"{t}.jsonl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no data file for task '{t}': {p}")
        rows = load_task_file(p, t)
        out += rows[:max_per_task] if max_per_task else rows
    ids = [r["id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("item ids must be unique across tasks (prefix them with the task name)")
    return out


def split_dev_holdout(items: list[dict], dev_fraction: float, seed: int) -> dict:
    """Deterministic per-task split, fixed before the run starts (the agent cannot re-draw it)."""
    out = {}
    by_task = {}
    for r in items:
        by_task.setdefault(r["task"], []).append(r["id"])
    for t, ids in by_task.items():
        ids = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        k = int(round(len(ids) * dev_fraction))
        for i in ids[:k]:
            out[i] = "dev"
        for i in ids[k:]:
            out[i] = "holdout"
    return out


def open_image(path: str, max_side: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    return img


def select(items_by_id: dict, req: dict, split_of: dict, max_items: int) -> list[dict]:
    """Pick practice items for a request: explicit "ids", or {"task", "split", "n", "seed"}."""
    if req.get("ids"):
        unknown = [i for i in req["ids"] if i not in items_by_id]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        ids = list(dict.fromkeys(req["ids"]))
    else:
        tasks = req.get("task") or req.get("tasks")
        tasks = [tasks] if isinstance(tasks, str) else tasks
        split = req.get("split")
        pool = [i for i, r in items_by_id.items()
                if (not tasks or r["task"] in tasks) and (not split or split_of.get(i) == split)]
        pool.sort()
        rng = random.Random(req.get("seed", 0))
        rng.shuffle(pool)
        ids = pool[: int(req.get("n", min(len(pool), max_items)))]
    if len(ids) > max_items:
        raise ValueError(f"too many items: {len(ids)} > {max_items} per pass")
    if not ids:
        raise ValueError("no items selected")
    return [items_by_id[i] for i in ids]
