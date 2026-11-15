"""The shipped configs load, inherit correctly, and the example matrices expand into loadable run configs."""
import os
import subprocess
import sys

import pytest

from wsp import config as config_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("path", ["configs/experiments/ocr.yaml", "configs/experiments/all.yaml",
                                  "configs/examples/smolvlm_handwriting/experiments/handwriting.yaml"])
def test_configs_load(path):
    c = config_mod.load(os.path.join(ROOT, path))
    assert c["kernel"]["sandbox"] == "bwrap" and c["experiment"]["tasks"]


def test_smolvlm_example_matrix(tmp_path):
    import shutil
    shutil.copytree(os.path.join(ROOT, "configs"), tmp_path / "configs")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "agent", "make_runs.py"), "--matrix",
                        "configs/examples/smolvlm_handwriting/matrix.yaml", "--private", "/secure/hw"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    gen = tmp_path / "configs/examples/smolvlm_handwriting/generated/handwriting"
    assert len(list(gen.glob("*.yaml"))) == 21
    c = config_mod.load(str(gen / "claude_b20_s2.yaml"))
    assert c["experiment"]["tasks"] == ["handwriting"] and list(c["tasks"]) == ["handwriting"]
    assert c["budget"]["passes"] == 20 and c["seed"] == 2 and c["agent"]["backend"] == "openrouter"
    assert c["model"]["processor_kwargs"]["size"]["longest_edge"] == 1920
    agent_sh = (tmp_path / "configs/examples/smolvlm_handwriting/agent_runs.sh").read_text()
    eval_sh = (tmp_path / "configs/examples/smolvlm_handwriting/eval_runs.sh").read_text()
    assert "--direction random" in agent_sh and "claude_b50_s3" in agent_sh
    assert "results/smolvlm_handwriting/base" in eval_sh and "reports/smolvlm_handwriting/handwriting.md" in eval_sh
