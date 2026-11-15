import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Tiny offline VLM + synthetic data (built once per test session)."""
    d = tmp_path_factory.mktemp("toy")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "agent", "make_toy.py"), "--out", str(d)], check=True,
                   capture_output=True)
    return d


@pytest.fixture
def toy_cfg(toy, tmp_path):
    import yaml

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "toy.yaml")))
    cfg["model"]["path"] = str(toy / "model")
    cfg["practice"]["dir"] = str(toy / "data" / "public")
    cfg["run_dir"] = str(tmp_path / "run")
    cfg["agent"]["script"] = os.path.join(ROOT, "tests", "mock_agent.py")
    if shutil.which("bwrap") is None:  # the harness refuses to run unsandboxed unless told to; tests say so explicitly
        cfg["kernel"] = {"sandbox": "none", "allow_unsandboxed": True}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.fixture
def private_copy(toy, tmp_path):
    """The toy private set as it would sit on the evaluation machine: marked PRIVATE."""
    d = tmp_path / "secure" / "private"
    shutil.copytree(toy / "data" / "private", d)
    (d / "PRIVATE").write_text("")
    return d
