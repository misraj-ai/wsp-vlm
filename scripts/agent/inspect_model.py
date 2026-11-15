"""AGENT MACHINE. Look at a model through the harness before spending any budget:
  - what discovery found (language-model blocks, vision tower, connector, the residual-writing matrices,
    frozen tensors, image token), i.e. what the agent will be told and allowed to edit;
  - optionally, the model's answers on a few practice items with the configured prompt and protocol, graded.

    python scripts/agent/inspect_model.py --config configs/examples/smolvlm_handwriting/experiments/handwriting.yaml --n 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wsp import config as config_mod  # noqa: E402
from wsp import model_io  # noqa: E402
from wsp.cli import parse_sets  # noqa: E402
from wsp.data import load_dir  # noqa: E402
from wsp.evaluation import run_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n", type=int, default=0, help="practice items per task to generate (0 = discovery only)")
ap.add_argument("--json", action="store_true", help="also print the full discovery JSON")
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = config_mod.load(a.config, parse_sets(a.set))
model, proc = model_io.load_model(cfg["model"])
disc = model_io.discover(model, cfg["model"].get("discovery"))
print(model_io.describe(disc, model))
problems = []
for key in ("lm", "vision"):
    for b in disc[key]:
        if not b["mlp_writer_weight"] or not b["attn_writer_weight"]:
            problems.append(f"{b['name']}: writer not found (mlp={b['mlp_writer_weight']}, attn={b['attn_writer_weight']})")
if not disc["vision"]:
    problems.append("no vision tower found: set model.discovery.vision_layers")
if not disc["connector"]:
    problems.append("no connector found: set model.discovery.connector")
if disc["image_token_id"] is None:
    problems.append("no image token id in the config: positions='image' captures will not work")
print("\nCHECKS: " + ("all good" if not problems else "\n  " + "\n  ".join(problems)))
if a.json:
    print(json.dumps(disc, indent=1))
if a.n:
    tasks = cfg["experiment"]["tasks"]
    items = load_dir(cfg["practice"]["dir"], tasks, a.n)
    rows = run_items(model, proc, cfg, items, progress=False)
    for it, r in zip(items, rows):
        print(f"\n--- {r['id']} ({r['task']}) score {r['score']:.3f} tokens {r['n_tokens']} loop {r['loop']} cut {r['hit_max']}")
        print("reference:", it["reference"][:300])
        print("output:   ", r["text"][:300])
