"""EVALUATION MACHINE. Evaluate one bundle on the private test sets.

    python scripts/eval/evaluate.py --bundle bundles/base.tar.gz --data /secure/private --out results/base
    python scripts/eval/evaluate.py --bundle bundles/table/claude_b25_s1.tar.gz --data /secure/private --out results/table/claude_b25_s1

Reads only the bundle (edit + frozen spec) and the private folder, which must contain a PRIVATE marker
file. Every task defined in the spec that has a <task>.jsonl in the private folder is evaluated, so
single-task runs are also checked for side effects on the other tasks.

Writes into --out:
  items.jsonl    per-item outputs and scores: PRIVATE, keep it on this machine
  summary.json   aggregates only (no texts, no references): safe to share
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wsp import bundle, model_io  # noqa: E402
from wsp.data import PRIVATE_MARKER, load_dir, private_root  # noqa: E402
from wsp.edits import apply_permanently, load_submission  # noqa: E402
from wsp.evaluation import run_items  # noqa: E402
from wsp.forward import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--data", required=True, help="private folder with <task>.jsonl files and a PRIVATE marker")
ap.add_argument("--out", required=True)
ap.add_argument("--tasks", nargs="*", help="default: every spec task with a file in --data")
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch-size", type=int, help="override the protocol batch size (only if outputs do not depend on it)")
ap.add_argument("--max-per-task", type=int)
a = ap.parse_args()

if not private_root(a.data):
    raise SystemExit(f"{a.data} has no {PRIVATE_MARKER} marker: create an empty file named {PRIVATE_MARKER} in the private "
                     "folder, so that agent-side tools refuse to read it")
spec, sub, info, tmp = bundle.open_bundle(a.bundle)
try:
    cfg = {"model": dict(spec["model"], device=a.device), "protocol": dict(spec["protocol"]), "tasks": spec["tasks"],
           "grading": spec["grading"], "editing": spec["editing"]}
    if a.batch_size:
        cfg["protocol"]["batch_size"] = a.batch_size
    tasks = a.tasks or [t for t in spec["tasks"] if os.path.exists(os.path.join(a.data, f"{t}.jsonl"))]
    items = load_dir(a.data, tasks, a.max_per_task, allow_private=True)
    model, proc = model_io.load_model(cfg["model"])
    disc = model_io.discover(model, cfg["model"].get("discovery"))
    edits, rep = load_submission(sub, model, disc, cfg)
    apply_permanently(model, edits)
    print(f"{info.get('run')}: {len(items)} private items over {tasks}; edit {rep}")
    t0 = time.time()
    rows = run_items(model, proc, cfg, items)
finally:
    bundle.cleanup(tmp)
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "items.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
data_hash = hashlib.sha256(json.dumps(sorted((r["id"], r["reference"]) for r in items)).encode()).hexdigest()[:16]
summary = {"run_info": info, "edit": rep, "tasks_evaluated": tasks, "trained_on": spec["experiment"]["tasks"],
           "n_items": len(rows), "data_hash": data_hash, "seconds": round(time.time() - t0, 1),
           "model": spec["model"], "protocol": cfg["protocol"], "summary": summarize(rows, spec["tasks"])}
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary["summary"], indent=1, ensure_ascii=False))
