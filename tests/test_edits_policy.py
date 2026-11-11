import json
import os

import pytest
import torch

from wsp import config as config_mod
from wsp import model_io
from wsp.edits import WeightsReader, export_weights, load_submission, store_for
from wsp.kernel import Kernel
from wsp.policy import check


@pytest.fixture
def setup(toy_cfg, tmp_path):
    cfg = config_mod.load(toy_cfg)
    model, _ = model_io.load_model(cfg["model"])
    disc = model_io.discover(model)
    export_weights(model, str(tmp_path / "weights"))
    store = store_for(str(tmp_path / "edit"), str(tmp_path / "weights"), disc, cfg)
    return cfg, model, disc, store, tmp_path


def test_edit_rules(setup):
    cfg, model, disc, store, _ = setup
    name = disc["lm"][1]["mlp_writer_weight"]
    W = store.w.get(name).float()
    r = store.save(name, W + 0.01)
    assert r["changed_entries"] > 0 and store.summary()["tensors"] == 1
    with pytest.raises(ValueError, match="frozen"):
        store.save(disc["frozen"][0], store.w.get(disc["frozen"][0]).float() + 1)
    with pytest.raises(ValueError, match="multiplies"):
        store.save(name, W * 2)
    with pytest.raises(ValueError, match="shape"):
        store.save(name, W[:1])
    with pytest.raises(ValueError, match="NaN"):
        store.save(name, W * float("nan"))
    assert store.save(name, W)["changed_entries"] == 0 and store.summary()["tensors"] == 0  # identical -> removed


def test_vision_and_connector_switches(setup):
    cfg, model, disc, _, tmp = setup
    cfg["editing"]["allow_vision"] = False
    cfg["editing"]["allow_connector"] = False
    s = store_for(str(tmp / "edit2"), str(tmp / "weights"), disc, cfg)
    v = disc["vision"][0]["mlp_writer_weight"]
    with pytest.raises(ValueError, match="vision"):
        s.save(v, s.w.get(v).float() + 1)
    c = [n for n, _ in model.named_parameters() if n.startswith(disc["connector"][0] + ".")][0]
    with pytest.raises(ValueError, match="connector"):
        s.save(c, s.w.get(c).float() + 1)


def test_submission_integrity(setup):
    cfg, model, disc, store, tmp = setup
    name = disc["lm"][2]["mlp_writer_weight"]
    store.save(name, store.w.get(name).float() + 0.05)
    sub = str(tmp / "sub")
    store.snapshot(sub)
    edits, rep = load_submission(sub, model, disc, cfg)
    assert name in edits and rep["tensors"] == 1
    # tampered tensor file -> refused
    from safetensors.torch import save_file
    fn = os.path.join(sub, "tensors", name + ".safetensors")
    save_file({name: store.w.get(name).float() + 0.06}, fn)
    with pytest.raises(ValueError, match="manifest hash"):
        load_submission(sub, model, disc, cfg)
    # different base weights -> refused
    store.snapshot(sub)
    with torch.no_grad():
        dict(model.named_parameters())[name].add_(1.0)
    with pytest.raises(ValueError, match="base weights differ"):
        load_submission(sub, model, disc, cfg)


def test_policy():
    bad = ["loss.backward()", "import transformers", "from torch import optim", "torch.optim.Adam(p)",
           "x.requires_grad_(True)", "import subprocess", "eval('1')", "__import__('os')", "import requests",
           "scipy.optimize.minimize(f, x)", "x.cuda()", "torch.autograd.grad(y, x)"]
    for b in bad:
        assert check(b)[0], b
    ok = ["u = X.mean(0) - Y.mean(0)\nWt = W(name) - 2 * torch.outer(u, u @ W(name))\nimport re; re.compile('a')",
          "# import transformers (comment only)\nprint(1)", "s = 'model.eval() is mentioned in a string'"]
    for o in ok:
        assert not check(o)[0], (o, check(o))
    assert check("torch.linalg.lstsq(A, B)", warn=[r"lstsq"])[1]


def test_kernel_blocks_gradients_and_persists(setup, tmp_path):
    cfg, model, disc, store, tmp = setup
    run = tmp_path / "krun"
    for d in ("meta", "work"):
        os.makedirs(run / d)
    os.symlink(tmp / "weights", run / "weights")
    json.dump(disc, open(run / "meta" / "model.json", "w"))
    json.dump(cfg, open(run / "meta" / "config.json", "w"), default=str)
    json.dump([], open(run / "meta" / "practice.json", "w"))
    k = Kernel(str(run))
    try:
        assert "3" in k.run("x = 1 + 2\nprint(x)", 60)["output"]
        assert "3" in k.run("print(x)", 60)["output"]                     # state persists between runs
        out = k.run("t = torch.ones(2)\nt.requires_grad_(True)", 60)["output"]
        assert "not allowed" in out
        out = k.run("import torch.optim as o\no.SGD([torch.zeros(1)], lr=1)", 60)["output"]
        assert "not allowed" in out
        assert "False" in k.run("print(torch.cuda.is_available())", 60)["output"]
        assert "TIMEOUT" in k.run("import time\ntime.sleep(5)", 1)["output"]   # restarted
        assert "NameError" in k.run("print(x)", 60)["output"]                  # variables lost after restart
    finally:
        k.stop()
