"""Pilot: a failure taxonomy of the UNEDITED model on the public practice data (no budget used).
Run it before any agent run, to decide the protocol and to write down predictions.

It answers, per task:
  - how much of the total error comes from "switch-like" failures (loops, cut-offs, empty output,
    wrong script) vs ordinary misreadings: weight editing is expected to help mostly with the former;
  - for tables: does a wrong header predict a wrong body? (the header -> body hypothesis)

    python scripts/pilot.py --config configs/my_run.yaml --out pilot/ [--n 200]
"""
import argparse
import json
import math
import os

from _common import parse_sets
from wsp import config as config_mod
from wsp import model_io
from wsp.data import load_dir
from wsp.evaluation import run_items, write_results


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def taxonomy(rows, task_cfg):
    err = [1 - r["score"] for r in rows]
    tot = sum(err) or 1e-9
    switch = [r["loop"] or r["hit_max"] or r["empty"] or (task_cfg.get("arabic") and r.get("latin_share", 0) > 0.5) for r in rows]
    t = {"n": len(rows), "score": mean(r["score"] for r in rows),
         "loop_rate": mean(float(r["loop"]) for r in rows), "cut_rate": mean(float(r["hit_max"]) for r in rows),
         "empty_rate": mean(float(r["empty"]) for r in rows),
         "share_of_error_from_switch_like_failures": sum(e for e, s in zip(err, switch) if s) / tot}
    if task_cfg.get("arabic"):
        t["mean_latin_share"] = mean(r.get("latin_share", 0) for r in rows)
    if task_cfg["metric"] == "table":
        ok = [r for r in rows if r["header_exact"]]
        bad = [r for r in rows if not r["header_exact"]]
        t.update({
            "parsed_rate": mean(float(r["parsed"]) for r in rows),
            "header_exact_rate": len(ok) / max(1, len(rows)),
            "body_cell_f1_if_header_exact": mean(r["body_cell_f1"] for r in ok),
            "body_cell_f1_if_header_wrong": mean(r["body_cell_f1"] for r in bad),
            "body_row_f1_if_header_exact": mean(r["body_row_f1"] for r in ok),
            "body_row_f1_if_header_wrong": mean(r["body_row_f1"] for r in bad),
            "shape_ok_if_header_exact": mean(float(r["shape_ok"]) for r in ok),
            "shape_ok_if_header_wrong": mean(float(r["shape_ok"]) for r in bad),
            "corr_header_f1_body_f1": pearson([r["header_cell_f1"] for r in rows], [r["body_cell_f1"] for r in rows]),
            "share_of_error_from_header_wrong_tables": sum(1 - r["score"] for r in bad) / tot,
            "header_f1_bins": {b: {"n": len(rs), "body_cell_f1": mean(r["body_cell_f1"] for r in rs)} for b, rs in (
                ("<0.5", [r for r in rows if r["header_cell_f1"] < 0.5]),
                ("0.5-0.9", [r for r in rows if 0.5 <= r["header_cell_f1"] < 0.9]),
                (">=0.9 not exact", [r for r in rows if r["header_cell_f1"] >= 0.9 and not r["header_exact"]]),
                ("exact", ok))},
        })
    else:
        cs = sorted(r["cer"] for r in rows)
        t.update({"cer_mean": mean(cs), "cer_p50": cs[len(cs) // 2], "cer_p90": cs[int(0.9 * (len(cs) - 1))]})
    return t


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, help="items per task (default: all practice items)")
ap.add_argument("--check-batching", type=int, default=0, metavar="N",
                help="also generate N items one at a time and report how often the batched output differs")
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()
cfg = config_mod.load(a.config, parse_sets(a.set))
os.makedirs(a.out, exist_ok=True)
items = load_dir(cfg["practice"]["dir"], list(cfg["tasks"]), a.n)
model, proc = model_io.load_model(cfg["model"])
rows = run_items(model, proc, cfg, items)
write_results(os.path.join(a.out, "pilot_items.json"), rows, cfg["tasks"], {"protocol": cfg["protocol"]})
rep = {t: taxonomy([r for r in rows if r["task"] == t], cfg["tasks"][t]) for t in cfg["tasks"]}
if a.check_batching:
    import copy
    sub = items[: a.check_batching]
    p1 = copy.deepcopy(cfg["protocol"])
    p1["batch_size"] = 1
    single = run_items(model, proc, cfg, sub, p1, progress=False)
    same = sum(x["text"] == y["text"] for x, y in zip(single, rows[: len(sub)]))
    rep["_batching_check"] = {"items": len(sub), "identical_outputs": same,
                              "note": "if these differ, outputs depend on batch composition; set protocol.batch_size: 1"}
json.dump(rep, open(os.path.join(a.out, "pilot_report.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
