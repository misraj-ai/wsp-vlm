"""End to end on the toy VLM, as two machines:

  agent machine:      a full mock-agent run (sandboxed kernel when bwrap exists), both scripted baselines,
                      bundle export (runs + base). It never sees the private set.
  evaluation machine: only the bundles + a private set marked PRIVATE: evaluate, then compare.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(*args, ok=True):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def test_two_machine_pipeline(toy, toy_cfg, tmp_path, private_copy):
    # ---- agent machine -------------------------------------------------------------------------
    out = sh("scripts/agent/run_agent.py", "--config", toy_cfg).stdout
    assert "done=True" in out and "passes=6" in out
    run = yaml.safe_load(open(toy_cfg))["run_dir"]
    st = json.load(open(os.path.join(run, "state.json")))
    assert st["failed"] == 1 and st["handoffs"] == 2 and st["python_runs"] == 2
    calls = [json.loads(x) for x in open(os.path.join(run, "tool_calls.jsonl"))]
    assert any(c.get("refused") for c in calls)
    assert sum(1 for c in calls if c["kind"] == "forward" and c["ok"]) == 6
    assert len(json.load(open(os.path.join(run, "submission", "manifest.json")))) == 1

    bundles = tmp_path / "agent_machine_bundles"
    sh("scripts/agent/export_bundle.py", "--config", toy_cfg, "--out", str(bundles / "mock"))
    sh("scripts/agent/export_bundle.py", "--config", toy_cfg, "--base", "--out", str(bundles / "base"))
    for d in ("diffmeans", "random"):
        rd = str(tmp_path / f"b_{d}")
        o = sh("scripts/agent/scripted_baseline.py", "--config", toy_cfg, "--direction", d, "--alphas", "1,4",
               "--set", f"run_dir={rd}").stdout
        assert '"layers"' in o
        s = json.load(open(os.path.join(rd, "state.json")))
        assert s["done"] and s["passes"] >= 5 and s["failed"] == 0
        sh("scripts/agent/export_bundle.py", "--config", toy_cfg, "--set", f"run_dir={rd}", "--out", str(bundles / d))

    # ---- evaluation machine: gets the bundles only ---------------------------------------------
    eval_dir = tmp_path / "eval_machine"
    shutil.copytree(bundles, eval_dir / "bundles")
    names = ["base", "mock", "diffmeans", "random"]
    for n in names:
        sh("scripts/eval/evaluate.py", "--bundle", str(eval_dir / "bundles" / f"{n}.tar.gz"), "--data", str(private_copy),
           "--out", str(eval_dir / "results" / n), "--device", "cpu")
        summ = json.load(open(eval_dir / "results" / n / "summary.json"))
        assert set(summ["tasks_evaluated"]) == {"ocr", "arabic", "table", "handwriting"}
        assert "text" not in json.dumps(summ["summary"])                      # shareable: aggregates only
    rep = sh("scripts/eval/compare.py", "--base", str(eval_dir / "results" / "base"), "--edited",
             *[str(eval_dir / "results" / n) for n in names[1:]], "--out", str(eval_dir / "report.md")).stdout
    assert "| table |" in rep and "header exact" in rep

    # the evaluation refuses an unmarked folder
    unmarked = tmp_path / "unmarked"
    shutil.copytree(toy / "data" / "private", unmarked)
    r = sh("scripts/eval/evaluate.py", "--bundle", str(eval_dir / "bundles" / "base.tar.gz"), "--data", str(unmarked),
           "--out", str(tmp_path / "x"), "--device", "cpu", ok=False)
    assert r.returncode != 0 and "PRIVATE" in (r.stdout + r.stderr)


def test_agent_side_refuses_private_data(toy_cfg, private_copy):
    from wsp import config as config_mod
    from wsp.data import PrivateDataError, load_dir
    from wsp.driver import Run

    with pytest.raises(PrivateDataError):
        load_dir(str(private_copy), ["ocr"])
    cfg = config_mod.load(toy_cfg, {"practice": {"dir": str(private_copy)}})
    with pytest.raises(PrivateDataError):
        Run(cfg, with_agent=False)


def test_single_task_experiment_is_scoped(toy_cfg):
    from wsp import config as config_mod
    from wsp.driver import Run

    cfg = config_mod.load(toy_cfg, {"experiment": {"tasks": ["table"]}})
    assert cfg["experiment"]["name"] == "table"
    r = Run(cfg, with_agent=False)
    assert {it["task"] for it in r.items} == {"table"}
    p = r.task_prompt()
    assert "better at extracting" in p or "better at Extract" in p or "(the task 'table')" in p
    assert "- table (" in p and "- ocr (" not in p and "Text tasks:" not in p and "Table tasks:" in p
    k, res = r.forward({"type": "generate", "model": "original", "task": "ocr", "n": 2})
    assert not res["ok"] and "no items" in res["error"]                         # other tasks are not available
    k, res = r.forward({"type": "generate", "model": "original", "n": 4})
    assert res["ok"] and set(res["summary"]["original"]["by_task"]) == {"table"}
