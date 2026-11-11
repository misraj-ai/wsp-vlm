"""Non-LLM baselines that use the SAME harness, rules and forward-pass budget as an agent run:

  --direction diffmeans   the automated version of the paper's Qwen edit: a difference-of-means
                          direction between failing and good outputs, removed from the residual-writing
                          MLP matrices of the most separating layers, strength chosen on DEV,
                          confirmed on HOLDOUT
  --direction random      matched control: same layers, columns and strengths, random direction

If an agent cannot beat diffmeans at equal budget, its "reasoning" adds nothing over this script;
if diffmeans cannot beat random, the direction itself carries no information.

    python scripts/scripted_baseline.py --config configs/my_run.yaml --direction diffmeans --set run_dir=runs/diffmeans_b25_s1
"""
import argparse
import json
import math

import torch

from _common import parse_sets
from wsp import config as config_mod
from wsp.driver import Run

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--direction", choices=["diffmeans", "random"], default="diffmeans")
ap.add_argument("--label", choices=["auto", "loop", "header", "score"], default="auto",
                help="what counts as a failure: loops/cut-offs, wrong table header, low score, or all of them")
ap.add_argument("--tasks", nargs="*", help="tasks used to find the direction (default: all); candidates are scored on all")
ap.add_argument("--where", choices=["lm", "vision"], default="lm")
ap.add_argument("--layers-k", type=int, default=2)
ap.add_argument("--alphas", default="0.5,1,2,4")
ap.add_argument("--topk-cols", type=int, default=0, help="edit only the k columns that write the direction most (0 = all)")
ap.add_argument("--confirm-passes", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--set", action="append", default=[])
a = ap.parse_args()

cfg = config_mod.load(a.config, parse_sets(a.set))
run = Run(cfg, with_agent=False)
B = cfg["budget"]["passes"]
if B < 3 + a.confirm_passes:
    raise SystemExit("budget too small: needs 2 discovery passes + at least 1 candidate + confirmation")
tasks = a.tasks or list(cfg["tasks"])
log = {"args": vars(a), "steps": []}


def fwd(req):
    k, res = run.forward(req)
    if not res.get("ok"):
        raise SystemExit(f"forward request failed: {res.get('error')}")
    log["steps"].append({"request": k, "req": req, "paired": res.get("paired"),
                         "summary": {m: {"score": s["score"]} for m, s in res.get("summary", {}).items()}})
    return k, res


# 1. the unedited model on DEV
k1, _ = fwd({"type": "generate", "model": "original", "task": tasks, "split": "dev",
             "n": cfg["limits"]["gen_max_items"], "seed": a.seed})
rows = [json.loads(x) for x in open(f"{run.work}/gen_{k1}_original.jsonl", encoding="utf-8")]
scores = sorted(r["score"] for r in rows)
q25, q50 = scores[len(scores) // 4], scores[len(scores) // 2]


def is_bad(r):
    loop = r["loop"] or r["hit_max"]
    header = r.get("header_exact") is False
    low = r["score"] <= q25
    return {"loop": loop, "header": header, "score": low, "auto": loop or header}[a.label]


bad = [r for r in rows if is_bad(r)]
if a.label == "auto" and len(bad) < 8:
    bad = [r for r in rows if r["score"] <= q25]
good = [r for r in rows if not is_bad(r) and r["score"] >= q50 and not r["loop"]]
if len(bad) < 4 or len(good) < 4:  # fall back to a rank split (score, then CER as tie-breaker)
    ranked = sorted(rows, key=lambda r: (r["score"], -r.get("cer", 0.0)))
    third = max(4, len(ranked) // 3)
    bad, good = ranked[:third], ranked[-third:]
    log["label_fallback"] = "rank split: bottom vs top third"
half = cfg["limits"]["tf_max_items"] // 2
bad, good = bad[:half], good[:half]
log["labels"] = {"bad": len(bad), "good": len(good), "q25": q25, "q50": q50}
if len(bad) < 4 or len(good) < 4:
    raise SystemExit(f"not enough contrast: {len(bad)} failing vs {len(good)} good outputs")

# 2. activations of the model's own outputs (failing vs good)
key = "lm_resid" if a.where == "lm" else "vision_resid"
k2, _ = fwd({"type": "teacher_forced", "model": "original",
             "items": [{"id": r["id"], "target_from": f"gen_{k1}_original"} for r in bad + good],
             "capture": {key: "all", "positions": "target", "reduce": "mean"}})
import numpy as np  # noqa: E402

d = f"{run.work}/fwd_{k2}/original"
meta = json.load(open(f"{d}/meta.json"))
blocks = run.disc["lm"] if a.where == "lm" else run.disc["vision"]
nb = len(bad)
sep = []
for i in range(len(blocks)):
    X = np.load(f"{d}/act.{'lm' if a.where == 'lm' else 'vision'}.resid.{i}.npy").astype(np.float32)
    assert X.shape[0] == len(meta["items"]), "reduce=mean gives one row per item"
    u = X[:nb].mean(0) - X[nb:].mean(0)
    un = u / (np.linalg.norm(u) + 1e-12)
    pb, pg = X[:nb] @ un, X[nb:] @ un
    pooled = math.sqrt((pb.var() + pg.var()) / 2) + 1e-12
    sep.append({"layer": i, "dprime": float((pb.mean() - pg.mean()) / pooled), "u": un})
order = sorted(sep, key=lambda s: -s["dprime"])[: a.layers_k]
log["layers"] = [{"layer": s["layer"], "dprime": round(s["dprime"], 3)} for s in order]
gen = torch.Generator().manual_seed(a.seed)
for s_ in order:  # one fixed random direction per chosen layer (the matched control)
    s_["u_rand"] = torch.nn.functional.normalize(torch.randn(len(s_["u"]), generator=gen), dim=0)

# 3. candidates, scored on DEV against the original (paired), all tasks
alphas = [float(x) for x in a.alphas.split(",")]
cands = [(s, al) for s in order for al in alphas]
room = B - 2 - a.confirm_passes
best = None
for s, al in cands[:room]:
    blk = blocks[s["layer"]]
    name = blk["mlp_writer_weight"]
    Wt = run.store.w.get(name).float()                       # [d_model, d_hidden]
    u = torch.tensor(s["u"]) if a.direction == "diffmeans" else s["u_rand"]
    proj = u @ Wt                                            # what each hidden unit writes along u
    if a.topk_cols:
        mask = torch.zeros_like(proj)
        mask[proj.abs().topk(min(a.topk_cols, proj.numel())).indices] = 1
        proj = proj * mask
    run.store.clear()
    run.store.save(name, Wt - al * torch.outer(u, proj))
    k, res = fwd({"type": "generate", "model": "both", "split": "dev", "n": cfg["limits"]["gen_both_max_items"],
                  "seed": a.seed + 1})
    p = res["paired"]["all"]
    log["steps"][-1]["candidate"] = {"layer": s["layer"], "alpha": al, "tensor": name, "diff": p["diff"], "se": p["se"]}
    if best is None or p["diff"] > best[2]:
        best = (s, al, p["diff"], name, u)

# 4. freeze the best (if it helps on DEV) and confirm on HOLDOUT
run.store.clear()
if best and best[2] > 0:
    s, al, _, name, u = best
    Wt = run.store.w.get(name).float()
    proj = u @ Wt
    if a.topk_cols:
        mask = torch.zeros_like(proj)
        mask[proj.abs().topk(min(a.topk_cols, proj.numel())).indices] = 1
        proj = proj * mask
    run.store.save(name, Wt - al * torch.outer(u, proj))
    log["chosen"] = {"layer": s["layer"], "alpha": al, "tensor": name, "dev_diff": best[2]}
    for c in range(B - run.state["passes"]):
        k, res = fwd({"type": "generate", "model": "both", "split": "holdout", "n": cfg["limits"]["gen_both_max_items"],
                      "seed": a.seed + 100 + c})
        log.setdefault("holdout", []).append(res["paired"])
else:
    log["chosen"] = None
log["unused_passes"] = B - run.state["passes"]
run.finish("SCRIPTED BASELINE\n" + json.dumps({k: v for k, v in log.items() if k != "steps"}, indent=1, default=str))
json.dump(log, open(f"{run.dir}/baseline_log.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in log.items() if k != "steps"}, indent=1, default=str))
