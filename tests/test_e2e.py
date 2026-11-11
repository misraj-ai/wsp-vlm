"""End to end on the toy VLM: a full mock-agent run, the private evaluation of its submission, the
paired comparison, and both scripted baselines under the same budget."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_full_pipeline(toy, toy_cfg, tmp_path):
    import yaml

    out = sh("scripts/run_agent.py", "--config", toy_cfg)
    assert "done=True" in out and "passes=6" in out
    run = yaml.safe_load(open(toy_cfg))["run_dir"]
    st = json.load(open(os.path.join(run, "state.json")))
    assert st["failed"] == 1 and st["handoffs"] == 2 and st["python_runs"] == 2
    calls = [json.loads(x) for x in open(os.path.join(run, "tool_calls.jsonl"))]
    assert any(c.get("refused") for c in calls)                                    # policy refusal logged
    assert sum(1 for c in calls if c["kind"] == "forward" and c["ok"]) == 6
    man = json.load(open(os.path.join(run, "submission", "manifest.json")))
    assert len(man) == 1
    assert os.path.exists(os.path.join(run, "final_answer.md"))

    priv = str(toy / "data" / "private")
    sh("scripts/evaluate_private.py", "--config", toy_cfg, "--data", priv, "--out", str(tmp_path / "base.json"))
    sh("scripts/evaluate_private.py", "--config", toy_cfg, "--data", priv, "--submission", os.path.join(run, "submission"),
       "--out", str(tmp_path / "edit.json"))
    rep = sh("scripts/compare.py", "--base", str(tmp_path / "base.json"), "--edited", str(tmp_path / "edit.json"))
    assert "| table |" in rep and "header exact" in rep

    for d in ("diffmeans", "random"):
        rd = str(tmp_path / f"b_{d}")
        o = sh("scripts/scripted_baseline.py", "--config", toy_cfg, "--direction", d, "--alphas", "1,4",
               "--set", f"run_dir={rd}")
        assert '"layers"' in o
        s = json.load(open(os.path.join(rd, "state.json")))
        assert s["done"] and s["passes"] >= 5 and s["failed"] == 0
