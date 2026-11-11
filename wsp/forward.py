"""The forward-pass tool: the ONLY way the agent can run the model. Each successful request is one
pass of the budget; a failed request is not charged (the driver does the accounting).

Request types (JSON in a ```forward block):

  {"type": "generate", "model": "original" | "edited" | "both",
   "ids": [...]  or  "task": "table" | ["ocr", "arabic"], "split": "dev" | "holdout", "n": 64, "seed": 0,
   "max_new_tokens": <= protocol}
      -> the model(s) transcribe the selected practice items with the evaluation protocol; every
         output is graded; per-item results go to WORK/gen_<k>_<model>.jsonl

  {"type": "teacher_forced", "model": "original" | "edited" | "both",
   "items": [{"id": ..., "target": "text"} | {"id": ..., "target_from": "gen_<k>_<model>"} | {"id": ...}],
   "capture": {"lm_resid": [layers] | "all", "lm_mlp_out": [...], "lm_mlp_act": [...], "lm_attn_out": [...],
               "vision_resid": [...], "vision_mlp_act": [...], "connector": true, "lm_attn": [layers],
               "positions": "target" | "prompt_last" | "image" | "all", "reduce": "none" | "mean", "topk": 10}}
      -> scores the given targets (default: the reference) and saves log-probs and activations to
         WORK/fwd_<k>/<model>/
"""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import torch

from . import model_io
from .data import select
from .grading import grade
from .stats import paired


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(rows: list[dict], tasks: dict) -> dict:
    out = {"n": len(rows), "score": _mean([r["score"] for r in rows]),
           "loop_rate": _mean([float(r["loop"]) for r in rows]), "cut_rate": _mean([float(r["hit_max"]) for r in rows]),
           "empty_rate": _mean([float(r["empty"]) for r in rows]), "mean_tokens": _mean([r["n_tokens"] for r in rows]),
           "by_task": {}}
    for t in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == t]
        s = {"n": len(rs), "score": _mean([r["score"] for r in rs]), "loop_rate": _mean([float(r["loop"]) for r in rs]),
             "cut_rate": _mean([float(r["hit_max"]) for r in rs])}
        if tasks[t].get("metric") == "table":
            ok = [r for r in rs if r.get("header_exact")]
            bad = [r for r in rs if not r.get("header_exact")]
            s.update({"parsed_rate": _mean([float(r["parsed"]) for r in rs]),
                      "header_exact_rate": _mean([float(r["header_exact"]) for r in rs]),
                      "header_cell_f1": _mean([r["header_cell_f1"] for r in rs]),
                      "body_cell_f1": _mean([r["body_cell_f1"] for r in rs]),
                      "body_cell_f1_if_header_exact": _mean([r["body_cell_f1"] for r in ok]),
                      "body_cell_f1_if_header_wrong": _mean([r["body_cell_f1"] for r in bad]),
                      "shape_ok_rate": _mean([float(r["shape_ok"]) for r in rs])})
        else:
            s.update({"cer": _mean([r["cer"] for r in rs]), "wer": _mean([r["wer"] for r in rs])})
        if tasks[t].get("arabic"):
            s["latin_share"] = _mean([r.get("latin_share") for r in rs])
        out["by_task"][t] = s
    return out


def _f(x, nd=4):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def format_result(res: dict) -> str:
    """Compact, readable rendering of a forward result for the agent (full data stays on disk)."""
    if not res.get("ok"):
        return json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=1, ensure_ascii=False) + \
            ("\n" + res["trace"][-800:] if res.get("trace") else "")
    lines = []
    if res["type"] == "generate":
        lines.append(f"generate, model={res['model']}, {res['items']} items, {res['seconds']} s"
                     + (f", edit digest {res['edit_digest']}" if res.get("edit_digest") else ""))
        for name, s in res["summary"].items():
            lines.append(f"[{name}] score {_f(s['score'])} | loop {_f(s['loop_rate'], 3)} | cut at max tokens "
                         f"{_f(s['cut_rate'], 3)} | empty {_f(s['empty_rate'], 3)} | mean tokens {_f(s['mean_tokens'], 1)}")
            for t, v in s["by_task"].items():
                extra = (f"parsed {_f(v['parsed_rate'], 3)} header_exact {_f(v['header_exact_rate'], 3)} header_f1 "
                         f"{_f(v['header_cell_f1'], 3)} body_f1 {_f(v['body_cell_f1'], 3)} (if header exact "
                         f"{_f(v['body_cell_f1_if_header_exact'], 3)}, if wrong {_f(v['body_cell_f1_if_header_wrong'], 3)}) "
                         f"shape_ok {_f(v['shape_ok_rate'], 3)}" if "header_exact_rate" in v else
                         f"CER {_f(v['cer'])} WER {_f(v['wer'])}")
                if "latin_share" in v:
                    extra += f" latin_share {_f(v['latin_share'], 3)}"
                lines.append(f"   {t:<12} n={v['n']:<4} score {_f(v['score'])} loop {_f(v['loop_rate'], 3)} cut "
                             f"{_f(v['cut_rate'], 3)} | {extra}")
        if res.get("paired"):
            lines.append("[paired, edited - original] task | n | original | edited | diff (SE) | improved/worsened/same")
            for t, p in list(res["paired"]["by_task"].items()) + [("ALL", res["paired"]["all"])]:
                lines.append(f"   {t:<12} {p['n']:<4} {_f(p['base'])} {_f(p['edited'])} {p['diff']:+.4f} ({_f(p['se'])}) "
                             f"{p['improved']}/{p['worsened']}/{p['same']}")
        lines.append("per-item outputs: " + ", ".join(f"load_gen({res['files'][m].rsplit('_', 2)[-2]}, '{m}')"
                                                       for m in res["files"]))
    else:
        lines.append(f"teacher_forced, model={res['model']}, {res['items']} items, {res['seconds']} s; mean target "
                     f"log-prob {res.get('mean_target_logprob')}" + (f", edit digest {res['edit_digest']}" if res.get("edit_digest") else ""))
        for name, sh in res["shapes"].items():
            lines.append(f"[{name}] {res['dirs'][name]}: " + ", ".join(f"{k} {tuple(v)}" for k, v in sh.items()))
    return "\n".join(lines)


class ForwardServer:
    def __init__(self, cfg, model, processor, disc, items: list[dict], split_of: dict, store, work_dir: str):
        self.cfg, self.model, self.proc, self.disc = cfg, model, processor, disc
        self.items = {r["id"]: dict(r, split=split_of.get(r["id"])) for r in items}
        self.split_of, self.store, self.work = split_of, store, work_dir
        self.swap = model_io.WeightSwapper(model)
        self.tasks, self.protocol, self.lim = cfg["tasks"], cfg["protocol"], cfg["limits"]
        self.chat = cfg["model"].get("chat_template_kwargs") or {}

    def handle(self, req: dict, k: int) -> dict:
        t0 = time.time()
        try:
            if not isinstance(req, dict):
                raise ValueError("the forward block must hold one JSON object")
            typ = req.get("type")
            if typ == "generate":
                res = self._generate(req, k)
            elif typ == "teacher_forced":
                res = self._teacher_forced(req, k)
            else:
                raise ValueError("type must be 'generate' or 'teacher_forced'")
            res.update({"ok": True, "type": typ, "seconds": round(time.time() - t0, 1)})
            return res
        except Exception as e:  # not charged
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:],
                    "seconds": round(time.time() - t0, 1)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _models(self, req):
        m = req.get("model", "original")
        if m not in ("original", "edited", "both"):
            raise ValueError("model must be original, edited or both")
        names = ["original", "edited"] if m == "both" else [m]
        edits = None
        if "edited" in names:
            edits = self.store.load_all()
            if not edits:
                raise ValueError("no edits are saved yet: save_edit() something before asking for the edited model")
        return m, names, edits

    # -- generation ---------------------------------------------------------------------------
    def _generate(self, req, k):
        m, names, edits = self._models(req)
        lim = self.lim["gen_both_max_items"] if m == "both" else self.lim["gen_max_items"]
        items = select(self.items, req, self.split_of, lim)
        proto = dict(self.protocol)
        if req.get("max_new_tokens"):
            if int(req["max_new_tokens"]) > proto["max_new_tokens"]:
                raise ValueError(f"max_new_tokens may not exceed the protocol's {proto['max_new_tokens']}")
            proto["max_new_tokens"] = int(req["max_new_tokens"])
        res = {"model": m, "items": len(items), "files": {}, "summary": {}}
        scores = {}
        for name in names:
            with self.swap.applied(edits if name == "edited" else None):
                outs = model_io.generate(self.model, self.proc, items, self.tasks, proto, self.chat)
            rows = []
            for it, o in zip(items, outs):
                g = grade(self.tasks[it["task"]], o["text"], it["reference"], self.cfg["grading"], o["hit_max"])
                rows.append(dict({"id": it["id"], "task": it["task"], "split": it["split"], "text": o["text"],
                                  "n_tokens": o["n_tokens"], "reference": it["reference"]}, **g))
            fn = os.path.join(self.work, f"gen_{k}_{name}.jsonl")
            with open(fn, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            res["files"][name] = fn
            res["summary"][name] = summarize(rows, self.tasks)
            scores[name] = {r["id"]: r["score"] for r in rows}
        if m == "both":
            by_task = {}
            for t in sorted({it["task"] for it in items}):
                ids = {it["id"] for it in items if it["task"] == t}
                by_task[t] = paired({i: scores["original"][i] for i in ids}, {i: scores["edited"][i] for i in ids})
            res["paired"] = {"all": paired(scores["original"], scores["edited"]), "by_task": by_task}
        if edits:
            res["edit_digest"] = self.store.digest()
        return res

    # -- teacher forcing ----------------------------------------------------------------------
    def _target(self, spec):
        it = self.items.get(spec.get("id"))
        if it is None:
            raise ValueError(f"unknown practice id {spec.get('id')}")
        if "target" in spec:
            return it, str(spec["target"])
        if "target_from" in spec:
            fn = os.path.join(self.work, spec["target_from"] + ".jsonl")
            with open(fn, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["id"] == it["id"]:
                        return it, r["text"]
            raise ValueError(f"{spec['target_from']} has no output for {it['id']}")
        return it, it["reference"]

    def _teacher_forced(self, req, k):
        m, names, edits = self._models(req)
        specs = req.get("items")
        if not specs:
            specs = [{"id": it["id"]} for it in select(self.items, req, self.split_of, self.lim["tf_max_items"])]
        if len(specs) > self.lim["tf_max_items"]:
            raise ValueError(f"at most {self.lim['tf_max_items']} items per teacher-forced pass")
        cap = dict(req.get("capture") or {})
        if cap.get("lm_attn") and len(specs) > self.lim["tf_max_attn_items"]:
            raise ValueError(f"attention patterns: at most {self.lim['tf_max_attn_items']} items")
        pairs = [self._target(s) for s in specs]
        out = {"model": m, "items": len(pairs), "dirs": {}, "shapes": {}}
        for name in names:
            d = os.path.join(self.work, f"fwd_{k}", name)
            os.makedirs(d, exist_ok=True)
            buf, offs, meta_items, total = {}, {}, [], 0
            with self.swap.applied(edits if name == "edited" else None):
                for (it, target), s in zip(pairs, specs):
                    prompt = s.get("prompt") or it.get("prompt") or self.tasks[it["task"]]["prompt"]
                    r = model_io.teacher_forced(self.model, self.proc, self.disc, it, prompt, target, self.protocol,
                                                cap, self.chat, append_eos=s.get("append_eos", True))
                    total += int(r["input_ids"].shape[0])
                    if total > self.lim["tf_max_tokens_total"]:
                        raise ValueError(f"over {self.lim['tf_max_tokens_total']} tokens in this pass")
                    tok = getattr(self.proc, "tokenizer", self.proc)
                    tgt_ids = r["input_ids"][r["prompt_len"]:]
                    arrays = {"target_logprobs": r["target_logprobs"], "target_ids": tgt_ids, "positions": r["positions"]}
                    if "topk_ids" in r:
                        arrays["topk_ids"], arrays["topk_logprobs"] = r["topk_ids"], r["topk_logprobs"]
                    for key, t in r["acts"].items():
                        arrays["act." + key] = t
                    for key, t in arrays.items():
                        buf.setdefault(key, []).append(t.numpy().astype(np.float16 if t.is_floating_point() and key.startswith("act.") else
                                                                         (np.float32 if t.is_floating_point() else np.int64)))
                        offs.setdefault(key, [0]).append(offs.get(key, [0])[-1] + int(t.shape[0]))
                    if r.get("attn"):
                        for L, a in r["attn"].items():
                            np.save(os.path.join(d, f"attn_L{L}_item{len(meta_items)}.npy"), a.numpy())
                    meta_items.append({"id": it["id"], "task": it["task"], "prompt_len": r["prompt_len"],
                                       "n_tokens": int(r["input_ids"].shape[0]), "n_target": int(tgt_ids.shape[0]),
                                       "mean_target_logprob": round(float(r["target_logprobs"].mean()), 4) if r["target_logprobs"].numel() else None,
                                       "target_tokens": [tok.decode([i]) for i in tgt_ids.tolist()[:400]]})
            shapes = {}
            for key, parts in buf.items():
                arr = np.concatenate(parts, 0)
                np.save(os.path.join(d, key + ".npy"), arr)
                shapes[key] = list(arr.shape)
            meta = {"items": meta_items, "offsets": offs, "capture": cap,
                    "note": "row i of each array belongs to item j for offsets[key][j] <= i < offsets[key][j+1]"}
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            out["dirs"][name] = d
            out["shapes"][name] = shapes
            out.setdefault("mean_target_logprob", {})[name] = _mean([x["mean_target_logprob"] for x in meta_items])
        if edits:
            out["edit_digest"] = self.store.digest()
        return out
