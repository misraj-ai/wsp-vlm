"""AGENT MACHINE. Package a finished run (or the unedited base model) as a bundle for the evaluation machine.

    python scripts/agent/export_bundle.py --config configs/generated/table/claude_b25_s1.yaml --out bundles/table/claude_b25_s1
    python scripts/agent/export_bundle.py --config configs/base.yaml --base --out bundles/base     # condition A0

The bundle carries the edit and a frozen evaluation spec (model + revision, protocol, every task definition,
grading). Copy the .tar.gz to the evaluation machine; nothing else needs to go there.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wsp import bundle  # noqa: E402
from wsp import config as config_mod  # noqa: E402
from wsp.cli import parse_sets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", action="store_true", help="export the unedited model (empty edit)")
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = config_mod.load(a.config, parse_sets(a.set))
if not cfg["model"].get("revision"):
    print("WARNING: model.revision is not pinned; the evaluation machine may load different weights "
          "(it will refuse an edit whose base hashes differ)")
if a.base:
    info = {"run": "base", "experiment": None, "condition": "base", "seed": None}
    sub = None
else:
    run = os.path.abspath(cfg["run_dir"])
    sub = os.path.join(run, "submission")
    st = json.load(open(os.path.join(run, "state.json")))
    if not st.get("done") or not os.path.exists(os.path.join(sub, "manifest.json")):
        raise SystemExit(f"{run} has not finished (no submission yet)")
    info = {"run": cfg["run_name"], "experiment": cfg["experiment"], "condition": cfg.get("condition"),
            "seed": cfg["seed"], "passes_used": st["passes"], "passes_budget": cfg["budget"]["passes"],
            "turns": st["turn"], "python_runs": st["python_runs"], "failed_requests": st["failed"],
            "agent": {k: cfg["agent"].get(k) for k in ("backend", "model")},
            "sandbox": cfg["kernel"].get("sandbox"),
            "submission": json.load(open(os.path.join(sub, "submission.json")))}
p = bundle.export(a.out, cfg, sub, info)
print(f"bundle written: {p}")
