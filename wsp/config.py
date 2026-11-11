"""Experiment configuration: one YAML file per run. Defaults below; the YAML overrides them."""
from __future__ import annotations

import copy
import json
import os

import yaml

DEFAULTS = {
    "run_name": "run",
    "run_dir": "runs/run",
    "seed": 0,
    "weights_cache": None,            # e.g. cache/<model>-<revision>: exported base weights shared by all runs
    "model": {
        "path": None,                 # HF hub id or local directory
        "revision": None,
        "dtype": "bfloat16",          # bfloat16 | float16 | float32
        "device": "cuda",
        "trust_remote_code": False,
        "attn_implementation": None,  # "eager" enables attention-pattern reads
        "auto_class": None,           # force e.g. "AutoModelForImageTextToText"
        "chat_template_kwargs": {},   # passed to apply_chat_template (e.g. {"enable_thinking": false})
        "discovery": {},              # regex overrides: lm_layers, vision_layers, connector, writer, attn_out
    },
    # The evaluation protocol. Fix it (scripts/decode_sweep.py) BEFORE any agent run; it is the same for
    # practice feedback, baselines and the private evaluation.
    "protocol": {
        "max_new_tokens": 2048,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "image_max_side": None,       # resize so the longest edge is at most this (None = as is)
        "batch_size": 8,
        "seed": 0,
        "system_prompt": None,
    },
    "tasks": {},                      # name -> {prompt, metric: cer|table, arabic: bool}
    "grading": {"arabic_norm": {}, "loop": {}},
    "practice": {
        "dir": None,                  # directory holding <task>.jsonl (public data only)
        "dev_fraction": 0.5,          # DEV (choose edits) / HOLDOUT (confirm a frozen edit), fixed before the run
        "max_items_per_task": None,
    },
    "budget": {
        "passes": 25,
        "max_failed_requests": 40,
        "python_runs": 300,
        "max_turns": 400,
        "must_use_all_passes": True,
        "explain_every_step": True,
    },
    "limits": {
        "gen_max_items": 256,         # per generation pass (one model)
        "gen_both_max_items": 128,    # per model in a paired ("both") pass
        "tf_max_items": 256,
        "tf_max_tokens_total": 400_000,
        "tf_max_attn_items": 16,
        "kernel_timeout_s": 900,
        "output_max_chars": 15000,
    },
    "editing": {
        "frozen_patterns": [],        # extra regexes; input/output embeddings and the final LM norm are always frozen
        "refuse_rescale": True,       # refuse an edit that only multiplies a whole tensor by a constant
        "allow_vision": True,
        "allow_connector": True,
    },
    "policy": {
        "forbidden_extra": [],        # extra regexes refused in kernel code
        "warn": [r"lstsq", r"pinv", r"linalg\.solve"],
    },
    "kernel": {
        "python": None,               # interpreter for the kernel (default: this one)
        "sandbox_cmd": [],            # e.g. ["bwrap", "--unshare-net", ...] (see README)
    },
    "agent": {
        "backend": "anthropic",       # anthropic | openai_compat | scripted
        "model": "claude-opus-5-5",
        "max_tokens": 32000,
        "extra": {},                  # passed through to the API call (e.g. a thinking / effort setting)
        "handoff_every_turns": 40,
        "prompt_template": None,      # default: wsp/prompts/task_v1.md
        "script": None,               # scripted backend: path to a .py exposing replies(state) or a .json list
    },
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str, overrides: dict | None = None) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    # relative paths are resolved against the current working directory
    if not cfg["tasks"]:
        raise ValueError("config: at least one task is required under 'tasks'")
    for name, t in cfg["tasks"].items():
        t.setdefault("metric", "cer")
        t.setdefault("arabic", False)
        if "prompt" not in t:
            raise ValueError(f"config: task {name} needs a prompt")
    return cfg


def dump(cfg: dict, path: str):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False, default=str)
