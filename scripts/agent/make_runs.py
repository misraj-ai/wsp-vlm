"""AGENT MACHINE. Expand a study matrix (experiments x conditions x seeds) into one config per run and two
shell scripts:

  agent_runs.sh   for the agent machine: every run, then its bundle export (and the base bundle)
  eval_runs.sh    for the evaluation machine: evaluate every bundle, then one comparison report per
                  experiment and condition group

    python scripts/agent/make_runs.py --matrix configs/matrix.example.yaml
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", required=True)
ap.add_argument("--private", default="/secure/private", help="private data path on the evaluation machine (for eval_runs.sh)")
a = ap.parse_args()
M = yaml.safe_load(open(a.matrix))
mdir = os.path.dirname(os.path.abspath(a.matrix))
exp_dir = os.path.join(mdir, "experiments")
out = M.get("configs_out", "configs/generated")
runs_root, bundles_root = M.get("runs_root", "runs"), M.get("bundles_root", "bundles")
results_root, reports_root = M.get("results_root", "results"), M.get("reports_root", "reports")
agent_sh = ["#!/bin/bash", "# agent machine: runs every condition, then exports one bundle per run", "set -e",
            f"python scripts/agent/export_bundle.py --config {os.path.relpath(os.path.join(mdir, 'base.yaml'))} --base "
            f"--out {bundles_root}/base"]
eval_sh = ["#!/bin/bash", "# evaluation machine: copy the bundles/ folder here first; the private data never leaves this machine",
           "set -e", f"PRIVATE={a.private}",
           f"python scripts/eval/evaluate.py --bundle {bundles_root}/base.tar.gz --data $PRIVATE --out {results_root}/base"]
n = 0
for exp in M["experiments"]:
    if not os.path.exists(os.path.join(exp_dir, f"{exp}.yaml")):
        raise SystemExit(f"no experiment config {exp_dir}/{exp}.yaml")
    per_exp = []
    for cond in M["conditions"]:
        for seed in M["seeds"]:
            name = f"{cond['name']}_s{seed}"
            cdir = os.path.join(out, exp)
            os.makedirs(cdir, exist_ok=True)
            cfg = {"base": os.path.relpath(os.path.join(exp_dir, f"{exp}.yaml"), cdir), "run_name": f"{exp}_{name}",
                   "condition": cond["name"], "run_dir": f"{runs_root}/{exp}/{name}", "seed": seed,
                   "budget": {"passes": cond["passes"]}}
            for k, v in (cond.get("set") or {}).items():
                cfg[k] = v
            path = os.path.join(cdir, f"{name}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            if cond["kind"] == "agent":
                agent_sh.append(f"python scripts/agent/run_agent.py --config {path}")
            elif cond["kind"] in ("diffmeans", "random"):
                extra = " ".join(cond.get("args", []))
                agent_sh.append(f"python scripts/agent/scripted_baseline.py --config {path} --direction {cond['kind']} {extra}".rstrip())
            else:
                raise SystemExit(f"unknown condition kind {cond['kind']}")
            agent_sh.append(f"python scripts/agent/export_bundle.py --config {path} --out {bundles_root}/{exp}/{name}")
            eval_sh.append(f"python scripts/eval/evaluate.py --bundle {bundles_root}/{exp}/{name}.tar.gz --data $PRIVATE "
                           f"--out {results_root}/{exp}/{name}")
            per_exp.append(f"{results_root}/{exp}/{name}")
            n += 1
    eval_sh.append(f"python scripts/eval/compare.py --base {results_root}/base --edited {' '.join(per_exp)} "
                   f"--out {reports_root}/{exp}.md")
sh_dir = M.get("scripts_out", ".")
os.makedirs(sh_dir, exist_ok=True)
for fn, lines in (("agent_runs.sh", agent_sh), ("eval_runs.sh", eval_sh)):
    p = os.path.join(sh_dir, fn)
    open(p, "w").write("\n".join(lines) + "\n")
    os.chmod(p, 0o755)
print(f"{n} run configs in {out}/; {os.path.join(sh_dir, 'agent_runs.sh')} and {os.path.join(sh_dir, 'eval_runs.sh')} written")
