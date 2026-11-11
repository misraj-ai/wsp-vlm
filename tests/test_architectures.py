"""Module discovery on tiny random versions of candidate VLM families, and a full generation +
teacher-forced read on a tiny Qwen3.5 (the family used in the original paper)."""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch
import transformers as tf

from wsp import model_io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHS = [("qwen3_5", "Qwen3_5ForConditionalGeneration"), ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
         ("lighton_ocr", "LightOnOcrForConditionalGeneration"), ("paddleocr_vl", "PaddleOCRVLForConditionalGeneration"),
         ("smolvlm", "SmolVLMForConditionalGeneration"), ("lfm2_vl", "Lfm2VlForConditionalGeneration"),
         ("internvl", "InternVLForConditionalGeneration"), ("idefics3", "Idefics3ForConditionalGeneration")]


def _shrink(c):
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=32, depth=2, embed_dim=64, num_heads=2, out_hidden_size=64, vocab_size=512,
                     linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                     projection_dim=64, text_hidden_size=64).items():
        if hasattr(c, k):
            setattr(c, k, v)
    if getattr(c, "layer_types", None):
        c.layer_types = (list(c.layer_types) * 2)[:2]


@pytest.mark.parametrize("mt,cls", ARCHS)
def test_discovery(mt, cls):
    if getattr(tf, cls, None) is None:
        pytest.skip(f"{cls} not in this transformers version")
    cfg = tf.AutoConfig.for_model(mt)
    for sub in ("text_config", "vision_config"):
        if hasattr(cfg, sub):
            _shrink(getattr(cfg, sub))
    if mt == "qwen3_5":
        cfg.text_config.layer_types = ["linear_attention", "full_attention"]
    m = getattr(tf, cls)(cfg)
    d = model_io.discover(m)
    params = dict(m.named_parameters())
    assert len(d["lm"]) == 2 and d["vision"] and d["connector"] and d["vision_root"]
    for blk in d["lm"] + d["vision"]:
        for key in ("mlp_writer_weight", "attn_writer_weight"):
            name = blk[key]
            assert name in params, (mt, key, name)
            assert params[name].shape[0] == blk["d_model"], (mt, name, params[name].shape, blk["d_model"])
            assert not any(x in name for x in ("q_proj", "k_proj", "v_proj")), (mt, name)
    assert any("embed" in f for f in d["frozen"])
    assert any("norm" in f for f in d["frozen"]), (mt, d["frozen"])


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="Qwen processors need torchvision")
def test_qwen35_generate_and_capture(tmp_path, toy):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_toy.py"), "--out", str(tmp_path), "--arch", "qwen35"],
                   check=True, capture_output=True)
    m, p = model_io.load_model({"path": str(tmp_path / "model"), "dtype": "float32", "device": "cpu"})
    d = model_io.discover(m)
    img = str(next((toy / "data" / "public" / "images").glob("table_*.png")))
    items = [{"id": "a", "task": "t", "image": img}, {"id": "b", "task": "t", "image": img}]
    outs = model_io.generate(m, p, items, {"t": {"prompt": "Extract the table."}}, {"max_new_tokens": 6, "batch_size": 2})
    assert len(outs) == 2 and outs[0]["text"] == outs[1]["text"]          # same image, greedy -> same output in a batch
    r = model_io.teacher_forced(m, p, d, items[0], "Extract the table.", "<table><tr><th>A</th></tr></table>", {},
                                {"lm_resid": "all", "lm_mlp_act": [0], "lm_attn_out": [0, 1], "vision_resid": [0],
                                 "connector": True, "topk": 2})
    assert r["target_logprobs"].numel() == r["input_ids"].numel() - r["prompt_len"]
    assert {"lm.resid.0", "lm.resid.1", "lm.attn_out.0", "lm.attn_out.1", "vision.resid.0"} <= set(r["acts"])
    r2 = model_io.teacher_forced(m, p, d, items[0], "Extract the table.", "abc", {}, {"lm_resid": [1], "positions": "image"})
    assert r2["positions"].numel() > 0 and r2["acts"]["lm.resid.1"].shape[0] == r2["positions"].numel()
