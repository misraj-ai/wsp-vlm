"""Choose the evaluation protocol BEFORE any agent run: run the unedited model under several decoding /
prompt / resolution settings on the DEV practice items and report per-task scores. The best setting
becomes `protocol` (and possibly the task prompts) in the run config, fixed for every condition.
This is the "best decoding-only baseline" the edits must beat.

    python scripts/decode_sweep.py --config configs/my_run.yaml --grid configs/sweep.example.yaml --n 100 --out sweep/
"""
import argparse
import copy
import json
import os

import yaml

from _common import parse_sets
from wsp import config as config_mod
from wsp import model_io
from wsp.data import load_dir, split_dev_holdout
from wsp.evaluation import run_items
from wsp.forward import summarize

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--grid", required=True, help="YAML list of {name, protocol: {...}, tasks: {task: {prompt}}} overrides")
ap.add_argument("--n", type=int, default=100, help="DEV items per task")
ap.add_argument("--out", required=True)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = config_mod.load(a.config, parse_sets(a.set))
grid = yaml.safe_load(open(a.grid))
os.makedirs(a.out, exist_ok=True)
items = load_dir(cfg["practice"]["dir"], list(cfg["tasks"]))
split = split_dev_holdout(items, cfg["practice"]["dev_fraction"], cfg["seed"])
dev = []
for t in cfg["tasks"]:
    dev += [r for r in items if r["task"] == t and split[r["id"]] == "dev"][: a.n]
model, proc = model_io.load_model(cfg["model"])
table = []
for g in grid:
    c = copy.deepcopy(cfg)
    c["protocol"].update(g.get("protocol", {}))
    for t, tv in (g.get("tasks") or {}).items():
        c["tasks"][t].update(tv)
    print(f"== {g['name']}")
    rows = run_items(model, proc, c, dev)
    s = summarize(rows, c["tasks"])
    table.append({"name": g["name"], "overrides": g, "score": s["score"],
                  "by_task": {t: v["score"] for t, v in s["by_task"].items()},
                  "loop_rate": s["loop_rate"], "cut_rate": s["cut_rate"], "mean_tokens": s["mean_tokens"]})
    json.dump(table, open(os.path.join(a.out, "sweep.json"), "w"), indent=1, ensure_ascii=False)
table.sort(key=lambda r: -(sum(r["by_task"].values()) / len(r["by_task"])))
print("\nsetting | task-mean score | " + " | ".join(cfg["tasks"]) + " | loop | cut")
for r in table:
    print(f"{r['name']} | {sum(r['by_task'].values()) / len(r['by_task']):.4f} | "
          + " | ".join(f"{r['by_task'].get(t, float('nan')):.4f}" for t in cfg["tasks"]) + f" | {r['loop_rate']} | {r['cut_rate']}")
print(f"\nbest: {table[0]['name']} -> copy its overrides into the run config")
