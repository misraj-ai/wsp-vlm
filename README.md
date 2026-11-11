# wsp-vlm: a Weight-Space Programming harness for small vision-language models

This harness tests whether an LLM agent can improve a ~1B vision-language model on OCR, Arabic documents,
table extraction and handwriting by editing its weights directly: no gradients, no training. It follows the
rules of *Sparks of Weight-Space Programming* (Takehi et al., 2026), Experiment 8, and adds private test sets,
scripted baselines and multi-seed runs. The experiment design is in [PLAN.md](PLAN.md).

```
agent (Claude / any OpenAI-compatible model / scripted)
   │  one ```python or ```forward block per turn
   ▼
driver ──► python kernel (CPU, separate process): reads weight files, save_edit(), practice data; cannot run the model
   │
   └─────► forward-pass tool (GPU, budgeted): generate + grade practice items, or teacher-forced reads with activations
                                               from the vision tower, connector and language model
submission/ (edit tensors + manifest) ──► evaluate_private.py on the PRIVATE test sets ──► compare.py
```

## Install and test

```bash
pip install -r requirements.txt
pytest tests          # 23 tests, ~1 min on CPU: builds tiny Llava- and Qwen3.5-style VLMs offline
```

The tests cover:
- graders and Arabic normalization;
- the edit rules (frozen tensors, rescale refusal, shapes, NaN) and tamper / base-mismatch detection;
- the code policy, plus the kernel's runtime blocks on gradients and optimizers;
- kernel persistence and timeouts;
- a full mock-agent run with every harness path, the private evaluation, the paired report, and both scripted baselines;
- module discovery on Qwen3.5, Qwen3-VL, LightOnOCR, PaddleOCR-VL, SmolVLM, LFM2-VL, InternVL and Idefics3.

## Data format

One JSONL file per task in a directory (`<task>.jsonl`), the same format for public practice data and the private set:

```json
{"id": "table_000123", "image": "images/t123.png", "reference": "<table><tr><th>الاسم</th>...</table>"}
```

- `id`: unique across all tasks (prefix with the task name).
- `image`: relative to the JSONL file, or absolute.
- `reference`: plain text for `cer` tasks; an HTML or markdown table for `table` tasks. Header = leading rows of
  `<th>` cells (or the first row if none; or the row above `|---|` in markdown).
- Optional: `prompt` (per-item override), `meta`.

## Workflow

```bash
# 0. leakage check: remove every public item it flags
python scripts/check_overlap.py --public data/public --private /secure/private --tasks ocr arabic table handwriting

# 1. pilot on the unedited model (no budget): failure taxonomy, header -> body analysis, batching check
python scripts/pilot.py --config configs/my.yaml --out pilot/ --n 300 --check-batching 32

# 2. fix the protocol: decoding / image size / prompt variants on DEV practice items
python scripts/decode_sweep.py --config configs/my.yaml --grid configs/sweep.example.yaml --n 100 --out sweep/

# 3. base model on the private set (once)
python scripts/evaluate_private.py --config configs/my.yaml --data /secure/private --out results/base.json

# 4. baselines at the same budget (diffmeans = automated version of the paper's edit; random = matched control)
python scripts/scripted_baseline.py --config configs/my.yaml --direction diffmeans --set run_dir=runs/diffmeans_b25_s1 --set seed=1
python scripts/scripted_baseline.py --config configs/my.yaml --direction random    --set run_dir=runs/random_b25_s1 --set seed=1

# 5. agent runs (export ANTHROPIC_API_KEY); the same command resumes after a crash
python scripts/run_agent.py --config configs/my.yaml --set run_dir=runs/claude_b25_s1 --set budget.passes=25 --set seed=1

# 6. evaluate each submission on the private set, then compare against the base
python scripts/evaluate_private.py --config configs/my.yaml --data /secure/private \
    --submission runs/claude_b25_s1/submission --out results/claude_b25_s1.json
python scripts/compare.py --base results/base.json --edited results/claude_b25_s1.json results/diffmeans_b25_s1.json --out results/report.md

# optional: a full HF checkpoint with an edit applied
python scripts/build_checkpoint.py --config configs/my.yaml --submission runs/claude_b25_s1/submission --out ckpt/claude_b25_s1
```

Start from [configs/experiment.example.yaml](configs/experiment.example.yaml). Keep the protocol, tasks, grading and
practice split seed identical across every condition you compare.

## What the agent gets (and the rules the harness enforces)

The first message is generated from [wsp/prompts/task_v1.md](wsp/prompts/task_v1.md). Read `runs/<run>/meta/prompt.txt` for what your agent actually saw.

**Python kernel** (CPU-only separate process, persistent namespace):
- `MODEL` (discovery info: blocks, the MLP and attention matrices that write the residual stream, widths, frozen tensors);
- `W(name)`, `W_current(name)`, `list_params()`;
- `save_edit()`, `remove_edit()`, `clear_edits()`, `edits()`;
- `practice()`, `load_image()`, `load_gen()`, `load_fwd()`, and `WORK` (keep `notes.md` there).

**Forward-pass tool**, budgeted. A failed request is not charged; failures are capped.
- `generate`: original, edited, or both (paired). Every output is graded and summarized per task, with loop, cut-off
  and empty rates, Latin share for Arabic, and for tables header-exact rate and body F1 split by header right/wrong.
- `teacher_forced`: log-probs, top-k, and activations from the vision tower, connector and language model (residual, MLP
  hidden units, MLP and attention outputs, optional attention maps), at target, prompt-last, image or all positions.

**Rules:**

| Rule | Enforced by |
|---|---|
| no gradients, optimizers, training or iterative fitting | static policy (`wsp/policy.py`) refuses the code; the kernel disables autograd, `backward`, `requires_grad_(True)` and every optimizer at runtime; every run is logged for audit |
| the model runs only through the tool | the kernel has no GPU, no `transformers` import, and only exported weight files |
| frozen: input embeddings, output layer, final LM norm (plus `editing.frozen_patterns`) | `save_edit`, re-checked before every pass and at evaluation |
| no whole-tensor rescaling | refused in `save_edit` |
| optional: no vision-tower or connector edits | `editing.allow_vision` / `allow_connector` |
| the submission matches the base model | `load_submission` checks base-weight hashes, tensor hashes, names and shapes |

**End of a run.** When the passes are used up, the agent must make its final choice and reply `FINAL ANSWER`. The edit at
that moment is copied to `runs/<run>/submission/`. An early final answer is refused while passes remain
(`budget.must_use_all_passes`).

## Run directory

```
runs/<run>/
  meta/        config.json, model.json (discovery), practice.json (items + fixed split), splits.json, prompt.txt
  weights/     base weights under model parameter names (what the kernel reads)
  work/        the agent's folder: edit/, notes.md, gen_<k>_<model>.jsonl, fwd_<k>/<model>/*.npy + meta.json
  turns/tNNN/  prompt.txt (what the agent received), answer.md (its reply), meta.json (token usage, timing)
  requests/    fwd_<k>.json and fwd_<k>.done.json
  tool_calls.jsonl  every Python run (code + output) and forward request, with the edit digest at that moment
  state.json, conversation.json, final_answer.md, submission/ (manifest.json + tensors/)
```

## Agents

| Backend | Use | Settings |
|---|---|---|
| `anthropic` | Claude (default `claude-opus-5-5`) | `ANTHROPIC_API_KEY`; streaming. `agent.extra` is passed to the API call unchanged: put your reasoning/effort setting there, per the API docs for that model |
| `openai_compat` | any OpenAI-compatible endpoint (another frontier model, or a local server) | `agent.base_url`, `agent.api_key_env`, `agent.extra` |
| `scripted` | tests, or hand-written strategies through the same harness | `agent.script`: a .py file with `reply(turn, last_message, state)` |

The system message is one neutral line, as in the paper. Conversations restart every `handoff_every_turns` turns with
the task, the agent's `notes.md`, its current edit and the budget status.

## A new model

Discovery finds the language-model blocks, the vision blocks, the connector, the writer matrices and the frozen
tensors automatically. Check `runs/<run>/meta/model.json`, or the first section of `meta/prompt.txt`, before the
first real run. If it picks the wrong modules, set regex overrides in `model.discovery`:

```yaml
model:
  discovery:
    lm_layers: "model\\.language_model\\.layers$"
    vision_layers: "model\\.visual\\.blocks$"
    connector: "model\\.visual\\.merger$"
    writer: "mlp\\.down_proj$"
    attn_out: "(o_proj|out_proj)$"
```

Other model notes:
- **Thinking templates.** If the model's template has a thinking switch, set it in `model.chat_template_kwargs`.
- **Qwen3.5 on GPU.** Install `causal-conv1d` and `flash-linear-attention`; without them the linear-attention layers
  use a slow reference path.
- **Batching.** Run `pilot.py --check-batching`. If batched outputs differ from one-at-a-time outputs, set
  `protocol.batch_size: 1`.

## Sandboxing the kernel

The static policy is fair-play enforcement, not a security boundary. For real isolation, wrap the kernel with
`kernel.sandbox_cmd`. Bubblewrap example (no network; read-only system; only the run directory and the practice data visible):

```yaml
kernel:
  sandbox_cmd: ["bwrap", "--unshare-net", "--die-with-parent", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                "--ro-bind", "/lib64", "/lib64", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--bind", "/abs/path/runs/claude_b25_s1", "/abs/path/runs/claude_b25_s1",
                "--ro-bind", "/abs/path/data/public", "/abs/path/data/public",
                "--ro-bind", "/abs/path/wsp_vlm", "/abs/path/wsp_vlm"]
```

Add read-only binds for your Python environment if it lives outside `/usr`. Keep the private set on another machine,
or at least outside every bound path.

## Known limits

- Generation uses HF `generate`. That is simple and model-agnostic but slower than vLLM. A vLLM backend would mean
  applying edits by reloading weights into the engine; it is not included.
- Graders are deliberately simple and fully specified: CER/WER, multiset cell F1 for tables. If you want TEDS as
  well, add it to `wsp/grading.py` and keep it as a secondary metric, so the agent's feedback and the reported score
  stay identical.
- The tests use tiny random models, so they check the plumbing, not the science. Before the first real agent run,
  run the pilot and one scripted baseline on the real model.
