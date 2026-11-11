"""The agent loop. One conversation per run (restarted with a handoff every N turns). Each turn the
agent replies with an explanation and exactly one ```python or ```forward block; the harness runs it,
logs everything, and sends back the output. The run ends after the final answer that follows the
last forward pass; the edit as it stands then is the submission (RUN/submission/).

Layout of a run directory:
  meta/         config.json, model.json (discovery), practice.json (items + fixed splits), prompt.txt
  weights/      the base weights under model parameter names (what the kernel reads)
  work/         the agent's folder: edit/, notes.md, gen_<k>_<model>.jsonl, fwd_<k>/
  turns/tNNN/   prompt.txt (what the agent received), answer.md (its reply), meta.json (usage, timing)
  requests/     fwd_<k>.json (request) and fwd_<k>.done.json (what was returned)
  tool_calls.jsonl   every python run and forward request, in order, with edit digests
  state.json, conversation.json, submission/, final_answer.md, logs/
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time

from . import agents as agents_mod
from . import config as config_mod
from . import model_io, policy
from .data import load_dir, split_dev_holdout
from .edits import export_weights, store_for
from .forward import ForwardServer, format_result
from .grading import DEFAULT_NORM
from .kernel import Kernel

SYSTEM = "Answer the user's request."
BLOCK = re.compile(r"```(python|forward)[ \t]*\n(.*?)```", re.S)
PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    h = n * 2 // 3
    return s[:h] + f"\n... [{len(s) - n} characters cut] ...\n" + s[-(n - h):]


def _jdump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


class Run:
    def __init__(self, cfg: dict, with_agent: bool = True):
        self.cfg = cfg
        self.dir = os.path.abspath(cfg["run_dir"])
        for d in ("meta", "work", "turns", "requests", "logs"):
            os.makedirs(os.path.join(self.dir, d), exist_ok=True)
        self.work = os.path.join(self.dir, "work")
        self.model, self.processor = model_io.load_model(cfg["model"])
        self.disc = model_io.discover(self.model, cfg["model"].get("discovery"))
        tasks = list(cfg["tasks"])
        self.items = load_dir(cfg["practice"]["dir"], tasks, cfg["practice"].get("max_items_per_task"))
        sp = os.path.join(self.dir, "meta", "splits.json")
        if os.path.exists(sp):
            self.split_of = json.load(open(sp))
        else:
            self.split_of = split_dev_holdout(self.items, cfg["practice"]["dev_fraction"], cfg["seed"])
            _jdump(sp, self.split_of)
        wdir = os.path.join(self.dir, "weights")
        if not os.path.exists(os.path.join(wdir, "index.json")):
            cache = cfg.get("weights_cache")
            if cache:  # export once per base model and share it read-only between runs
                if not os.path.exists(os.path.join(cache, "index.json")):
                    export_weights(self.model, cache)
                if not os.path.exists(wdir):
                    os.symlink(os.path.abspath(cache), wdir)
            else:
                export_weights(self.model, wdir)
        _jdump(os.path.join(self.dir, "meta", "config.json"), cfg)
        _jdump(os.path.join(self.dir, "meta", "model.json"), self.disc)
        _jdump(os.path.join(self.dir, "meta", "practice.json"),
               [{"id": r["id"], "task": r["task"], "split": self.split_of[r["id"]], "image": r["image"],
                 "reference": r["reference"]} for r in self.items])
        self.store = store_for(os.path.join(self.work, "edit"), os.path.join(self.dir, "weights"), self.disc, cfg)
        self.server = ForwardServer(cfg, self.model, self.processor, self.disc, self.items, self.split_of, self.store, self.work)
        self.kernel = Kernel(self.dir, cfg["kernel"].get("python"), cfg["kernel"].get("sandbox_cmd"))
        self.agent = agents_mod.make(cfg["agent"]) if with_agent else None
        self.state_path = os.path.join(self.dir, "state.json")
        self.conv_path = os.path.join(self.dir, "conversation.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {
            "turn": 0, "passes": 0, "failed": 0, "python_runs": 0, "requests": 0, "done": False, "handoffs": 0,
            "since_handoff": 0, "explain_streak": 0, "next": None, "final_requested": False}
        self.conv = json.load(open(self.conv_path)) if os.path.exists(self.conv_path) else []

    # -- prompt ---------------------------------------------------------------------------------
    def task_prompt(self) -> str:
        c, b, lim = self.cfg, self.cfg["budget"], self.cfg["limits"]
        tpl = open(c["agent"].get("prompt_template") or os.path.join(PROMPTS, "task_v1.md"), encoding="utf-8").read()
        tasks = "\n".join(f"- {n}: prompt {json.dumps(t['prompt'], ensure_ascii=False)}; metric: "
                          + ("table (header and body cells)" if t["metric"] == "table" else "character error rate")
                          + ("; Arabic" if t.get("arabic") else "") for n, t in c["tasks"].items())
        norm = dict(DEFAULT_NORM, **(c["grading"].get("arabic_norm") or {}))
        grading = ("- Text tasks: score = 1 - CER (character error rate, clipped at 0) after normalizing both the output and the "
                   f"reference ({', '.join(k + '=' + str(v) for k, v in norm.items())}). Reasoning between <think> tags and a single "
                   "code fence around the answer are removed first.\n"
                   "- Table tasks: the first table in the output (HTML or markdown) is parsed. Header = the leading rows of <th> cells "
                   "(or the first row if none, or the row above |---| in markdown). score = 0.5 * header-cell F1 + 0.5 * body-cell F1 "
                   "over normalized cell texts. No table in the output scores 0.\n"
                   "- Also reported, not scored: loops (a repeating tail), outputs cut at the token limit, empty outputs, Latin share in Arabic tasks.")
        p = c["protocol"]
        proto = (f"chat template of the model, the image then the task prompt in one user message"
                 + (f", system prompt {json.dumps(p['system_prompt'], ensure_ascii=False)}" if p.get("system_prompt") else "")
                 + f"; " + ("greedy decoding" if not p["do_sample"] else f"sampling (temperature {p['temperature']}, top_p {p['top_p']}, top_k {p['top_k']})")
                 + f", repetition_penalty {p['repetition_penalty']}, no_repeat_ngram_size {p['no_repeat_ngram_size']}, "
                 f"at most {p['max_new_tokens']} new tokens, images resized to longest edge {p['image_max_side'] or 'unchanged'}; "
                 f"edits are cast to the model dtype ({c['model']['dtype']}) when applied.")
        counts = {}
        for r in self.items:
            counts.setdefault(r["task"], {"dev": 0, "holdout": 0})[self.split_of[r["id"]]] += 1
        stats = "; ".join(f"{t}: {v['dev']} dev + {v['holdout']} holdout" for t, v in counts.items())
        e = c["editing"]
        scope = "" if e.get("allow_vision", True) and e.get("allow_connector", True) else \
            " (in this condition: " + ", ".join(x for x, ok in (("no vision-tower edits", e.get("allow_vision", True)),
                                                                ("no connector edits", e.get("allow_connector", True))) if not ok) + ")"
        explain = ("Begin every reply by explaining in plain words what you will do and why (a block without an explanation is "
                   "sent back unrun). " if b.get("explain_every_step", True) else "")
        rep = {"MODEL_DESCRIPTION": model_io.describe(self.disc, self.model), "TASKS": tasks, "GRADING": grading,
               "PROTOCOL": proto, "PYTHON_RUNS": b["python_runs"], "KERNEL_TIMEOUT": lim["kernel_timeout_s"],
               "PRACTICE_STATS": stats, "PASSES": b["passes"], "GEN_MAX": lim["gen_max_items"],
               "GEN_BOTH_MAX": lim["gen_both_max_items"], "TF_MAX": lim["tf_max_items"], "TF_TOKENS": lim["tf_max_tokens_total"],
               "TF_ATTN_MAX": lim["tf_max_attn_items"], "EDIT_SCOPE_NOTE": scope, "EXPLAIN_RULE": explain,
               "LIMITS_NOTE": f"   Requests that fail are not counted, but at most {b['max_failed_requests']} failures are allowed in a run."}
        for k, v in rep.items():
            tpl = tpl.replace("{{" + k + "}}", str(v))
        return tpl

    def status(self) -> str:
        b = self.cfg["budget"]
        return (f"Forward passes used: {self.state['passes']} of {b['passes']}. Python runs used: {self.state['python_runs']} of "
                f"{b['python_runs']}. Edit: {self.store.summary()['tensors']} tensors, digest {self.store.digest()}.")

    def handoff_message(self, last: str) -> str:
        notes_p = os.path.join(self.work, "notes.md")
        notes = open(notes_p, encoding="utf-8").read() if os.path.exists(notes_p) else "(no notes.md was written)"
        return (self.task_prompt() + "\n\nHANDOFF. This is a new conversation continuing your earlier work on this task. "
                "The kernel, your files, your saved edit and the budget carry over; your earlier messages do not.\n\n"
                f"YOUR NOTEBOOK (WORK/notes.md, last part):\n{_clip(notes, 12000)}\n\nYOUR CURRENT EDIT:\n"
                f"{json.dumps(self.store.summary(), indent=1)}\n\nSTATUS: {self.status()}\n\nTHE LAST MESSAGE OF THE EARLIER "
                f"CONVERSATION (the output of your last block):\n{_clip(last, 6000)}")

    # -- one block --------------------------------------------------------------------------------
    def _log_call(self, rec):
        rec["t"] = time.time()
        with open(os.path.join(self.dir, "tool_calls.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def run_python(self, code: str) -> str:
        b, lim = self.cfg["budget"], self.cfg["limits"]
        if self.state["python_runs"] >= b["python_runs"]:
            return "REFUSED: no Python runs left.\n" + self.status()
        refusals, warns = policy.check(code, self.cfg["policy"].get("forbidden_extra"), self.cfg["policy"].get("warn"))
        if refusals:
            self._log_call({"kind": "python", "refused": refusals, "code": code})
            return "REFUSED (not run, not counted):\n- " + "\n- ".join(refusals) + "\n" + self.status()
        self.state["python_runs"] += 1
        r = self.kernel.run(code, lim["kernel_timeout_s"])
        self._log_call({"kind": "python", "run": self.state["python_runs"], "code": code, "out": r.get("output"),
                        "s": r.get("seconds"), "warnings": warns, "edit_digest": self.store.digest()})
        w = ("WARNING: " + "; ".join(warns) + "\n") if warns else ""
        return (f"{w}OUTPUT of Python run {self.state['python_runs']} of at most {b['python_runs']} ({r.get('seconds')} s):\n```\n"
                f"{_clip(r.get('output') or '(no output)', lim['output_max_chars'])}\n```\n{self.status()}")

    def run_forward(self, body: str) -> str:
        b = self.cfg["budget"]
        if self.state["passes"] >= b["passes"]:
            return "REFUSED: all forward passes are used.\n" + self.status()
        if self.state["failed"] >= b["max_failed_requests"]:
            return "REFUSED: the limit of failed requests is reached; no more forward requests.\n" + self.status()
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            req = {"_unparsed": body, "_error": str(e)}
        k, res = self.forward(req)
        if res.get("ok"):
            head = f"RESULT of forward request {k} (counted: pass {self.state['passes']} of {b['passes']})"
        else:
            head = f"RESULT of forward request {k} (FAILED, not counted; {self.state['failed']} of {b['max_failed_requests']} failures)"
        return f"{head}:\n```\n{_clip(format_result(res), self.cfg['limits']['output_max_chars'])}\n```\n{self.status()}"

    def forward(self, req: dict) -> tuple[int, dict]:
        """Run one forward request with budget accounting and logging (used by the agent loop and the baselines)."""
        self.state["requests"] += 1
        k = self.state["requests"]
        _jdump(os.path.join(self.dir, "requests", f"fwd_{k}.json"), req)
        if "_unparsed" in req:
            res = {"ok": False, "error": f"the forward block is not valid JSON: {req['_error']}"}
        else:
            res = self.server.handle(req, k)
        _jdump(os.path.join(self.dir, "requests", f"fwd_{k}.done.json"), res)
        if res.get("ok"):
            self.state["passes"] += 1
        else:
            self.state["failed"] += 1
        self._log_call({"kind": "forward", "request": k, "ok": res.get("ok"), "pass": self.state["passes"],
                        "edit_digest": self.store.digest(), "summary": res.get("paired") or res.get("summary")})
        self.save()
        return k, res

    # -- loop -------------------------------------------------------------------------------------
    def save(self):
        _jdump(self.state_path, self.state)
        _jdump(self.conv_path, self.conv)

    def final_instruction(self) -> str:
        return ("\n\nAll forward passes are used. Make your final choice now: the edit as it stands in your edit folder when you "
                "reply with FINAL ANSWER is submitted (you may still use Python runs to change or remove edits first). Then reply "
                "with FINAL ANSWER, a short report, and no code block.")

    def loop(self, max_turns: int | None = None):
        b = self.cfg["budget"]
        max_turns = max_turns or b["max_turns"]
        if self.state["next"] is None and not self.conv:
            first = self.task_prompt()
            open(os.path.join(self.dir, "meta", "prompt.txt"), "w", encoding="utf-8").write(first)
            self.state["next"] = first
        while not self.state["done"]:
            if self.state["turn"] >= max_turns:
                self.finish("(the run stopped at the turn limit before a final answer)")
                break
            if self.state["since_handoff"] >= self.cfg["agent"]["handoff_every_turns"]:
                self.state["handoffs"] += 1
                self.state["since_handoff"] = 0
                self.conv = []
                self.state["next"] = self.handoff_message(self.state["next"])
            self.state["turn"] += 1
            self.state["since_handoff"] += 1
            t = self.state["turn"]
            td = os.path.join(self.dir, "turns", f"t{t:03d}")
            os.makedirs(td, exist_ok=True)
            msg = self.state["next"]
            open(os.path.join(td, "prompt.txt"), "w", encoding="utf-8").write(msg)
            self.conv.append({"role": "user", "content": msg})
            t0 = time.time()
            text, meta = self.agent.reply(SYSTEM, self.conv)
            meta["seconds"] = round(time.time() - t0, 1)
            open(os.path.join(td, "answer.md"), "w", encoding="utf-8").write(text)
            _jdump(os.path.join(td, "meta.json"), meta)
            self.conv.append({"role": "assistant", "content": text})
            self.state["next"] = self.respond(text)
            self.save()
        self.kernel.stop()

    def respond(self, text: str) -> str:
        b = self.cfg["budget"]
        blocks = BLOCK.findall(text)
        done_passes = self.state["passes"] >= b["passes"] or self.state["failed"] >= b["max_failed_requests"]
        if not blocks and "FINAL ANSWER" in text:
            if b.get("must_use_all_passes", True) and not done_passes:
                return (f"Not yet: {b['passes'] - self.state['passes']} forward passes remain and they must be used before the "
                        "final answer. Continue with one ```python or ```forward block.\n" + self.status())
            self.finish(text)
            return ""
        if len(blocks) != 1:
            return ("Your reply must contain exactly one ```python or ```forward block (it had "
                    f"{len(blocks)}). Nothing was run.\n" + self.status() + (self.final_instruction() if done_passes else ""))
        kind, body = blocks[0]
        before = text.split("```" + kind, 1)[0].strip()
        if b.get("explain_every_step", True) and len(before) < 20 and self.state["explain_streak"] < 2:
            self.state["explain_streak"] += 1
            return "EXPLAIN EVERY STEP: begin with a plain explanation of what this block does and why. Nothing was run; send it again with the explanation."
        self.state["explain_streak"] = 0
        out = self.run_python(body) if kind == "python" else self.run_forward(body)
        if self.state["passes"] >= b["passes"] or self.state["failed"] >= b["max_failed_requests"]:
            self.state["final_requested"] = True
            out += self.final_instruction()
        return out

    def finish(self, text: str):
        open(os.path.join(self.dir, "final_answer.md"), "w", encoding="utf-8").write(text)
        sub = os.path.join(self.dir, "submission")
        self.store.load_all()  # re-validate before submitting
        self.store.snapshot(sub)
        _jdump(os.path.join(sub, "submission.json"), {"run": self.cfg["run_name"], "summary": self.store.summary(),
                                                      "passes_used": self.state["passes"], "turns": self.state["turn"],
                                                      "python_runs": self.state["python_runs"]})
        self.state["done"] = True
        self.save()


def main(config_path: str, overrides: dict | None = None, max_turns: int | None = None):
    cfg = config_mod.load(config_path, overrides)
    r = Run(cfg)
    r.loop(max_turns)
    return r
