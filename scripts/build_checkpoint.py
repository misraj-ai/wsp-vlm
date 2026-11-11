"""Write a full Hugging Face checkpoint with a submission applied (to share or serve the edited model).

    python scripts/build_checkpoint.py --config configs/my_run.yaml --submission runs/x/submission --out ckpt/x
"""
import argparse
import json
import os

from _common import parse_sets
from wsp import config as config_mod
from wsp import model_io
from wsp.edits import apply_permanently, load_submission

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--submission", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = config_mod.load(a.config, parse_sets(a.set))
model, proc = model_io.load_model(cfg["model"])
disc = model_io.discover(model, cfg["model"].get("discovery"))
edits, rep = load_submission(a.submission, model, disc, cfg)
apply_permanently(model, edits)
model.save_pretrained(a.out)
proc.save_pretrained(a.out)
json.dump({"base": cfg["model"]["path"], "revision": cfg["model"].get("revision"), "edit": rep},
          open(os.path.join(a.out, "wsp_edit.json"), "w"), indent=1)
print(f"saved {a.out}: {rep}")
