"""Write a full Hugging Face checkpoint with a bundle's edit applied (to share or serve the edited model).
Needs no private data; runs on either machine.

    python scripts/eval/build_checkpoint.py --bundle bundles/table/claude_b25_s1.tar.gz --out ckpt/table_claude_b25_s1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wsp import bundle, model_io  # noqa: E402
from wsp.edits import apply_permanently, load_submission  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
spec, sub, info, tmp = bundle.open_bundle(a.bundle)
try:
    cfg = {"model": dict(spec["model"], device=a.device), "editing": spec["editing"]}
    model, proc = model_io.load_model(cfg["model"])
    disc = model_io.discover(model, cfg["model"].get("discovery"))
    edits, rep = load_submission(sub, model, disc, cfg)
    apply_permanently(model, edits)
finally:
    bundle.cleanup(tmp)
model.save_pretrained(a.out)
proc.save_pretrained(a.out)
json.dump({"base": spec["model"], "edit": rep, "run_info": info}, open(os.path.join(a.out, "wsp_edit.json"), "w"), indent=1)
print(f"saved {a.out}: {rep}")
