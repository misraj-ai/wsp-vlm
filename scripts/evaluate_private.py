"""Evaluate the base model or a submitted edit on the PRIVATE test sets. Run this on the machine
that holds the private data, never from inside an agent run.

    python scripts/evaluate_private.py --config configs/my_run.yaml --data /secure/private --out results/base.json
    python scripts/evaluate_private.py --config configs/my_run.yaml --data /secure/private \
        --submission runs/claude_b25_s1/submission --out results/claude_b25_s1.json
"""
import argparse
import hashlib
import json
import time

from _common import parse_sets
from wsp import config as config_mod
from wsp import model_io
from wsp.data import load_dir
from wsp.edits import apply_permanently, load_submission
from wsp.evaluation import run_items, write_results

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True, help="directory with the private <task>.jsonl files")
ap.add_argument("--submission", help="a run's submission/ folder (omit for the unedited model)")
ap.add_argument("--out", required=True)
ap.add_argument("--tasks", nargs="*")
ap.add_argument("--max-per-task", type=int)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()

cfg = config_mod.load(a.config, parse_sets(a.set))
tasks = a.tasks or list(cfg["tasks"])
items = load_dir(a.data, tasks, a.max_per_task)
model, proc = model_io.load_model(cfg["model"])
disc = model_io.discover(model, cfg["model"].get("discovery"))
report = {"tensors": 0, "changed_entries": 0, "digest": None}
if a.submission:
    edits, report = load_submission(a.submission, model, disc, cfg)
    apply_permanently(model, edits)
print(f"evaluating {len(items)} items, edit {report}")
t0 = time.time()
rows = run_items(model, proc, cfg, items)
meta = {"model": cfg["model"]["path"], "revision": cfg["model"].get("revision"), "dtype": cfg["model"]["dtype"],
        "protocol": cfg["protocol"], "tasks": {t: cfg["tasks"][t] for t in tasks}, "grading": cfg["grading"],
        "submission": a.submission, "edit": report, "n_items": len(items), "seconds": round(time.time() - t0, 1),
        "data_hash": hashlib.sha256(json.dumps(sorted((r["id"], r["reference"]) for r in items)).encode()).hexdigest()[:16]}
write_results(a.out, rows, cfg["tasks"], meta)
print(json.dumps({"summary": json.load(open(a.out))["summary"]}, indent=1, ensure_ascii=False))
