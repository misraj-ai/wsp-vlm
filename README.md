# wsp-vlm: a Weight-Space Programming harness for small vision-language models

This harness tests whether an LLM agent can improve a ~1B vision-language model on OCR, Arabic documents,
table extraction and handwriting by editing its weights directly: no gradients, no training. It follows the
rules of *Sparks of Weight-Space Programming* (Takehi et al., 2026), Experiment 8, and adds:
- private test sets on a separate machine;
- one experiment per task, plus one on all tasks;
- scripted baselines;
- multi-seed runs.

The study design is in [PLAN.md](PLAN.md).

## Two machines

```
AGENT MACHINE (public data only)                              EVALUATION MACHINE (private data only)
┌───────────────────────────────────────────────┐             ┌──────────────────────────────────────────┐
│ agent ⇄ driver ─► kernel (bubblewrap sandbox: │  bundles/   │ evaluate.py: bundle + private set        │
│                   CPU, no network, sees only  │ ──────────► │   -> items.jsonl  (stays here)           │
│                   its folder, the weights,    │  edit +     │   -> summary.json (aggregates, shareable)│
│                   the practice data)          │  frozen     │ compare.py -> report.md (aggregates)     │
│          └──────► forward-pass tool (GPU,     │  eval spec  │ check_overlap.py (with a copy of public) │
│                   budgeted, practice data)    │             │                                          │
│ scripted baselines, pilot, decode sweep       │             │                                          │
└───────────────────────────────────────────────┘             └──────────────────────────────────────────┘
```

- **Nothing private reaches the agent machine.** The private set, its outputs and its per-item results all stay
  on the evaluation machine. Only aggregate reports leave it.
- **Bundles are the only thing that travels.** A bundle (`.tar.gz`) holds the edit tensors and a frozen evaluation
  spec: model id, revision, dtype, protocol, every task definition and the grading options. The evaluation machine
  reads nothing else from the agent machine.
- **Private folders are marked.** Put an empty file named `PRIVATE` in the private folder. Every agent-side loader
  refuses a folder under such a marker, including image paths that point into one. `evaluate.py` refuses to run on
  an unmarked folder.
- **The kernel is sandboxed by default** with bubblewrap. The flags follow the original repo's
  `scripts/mathedit/sandbox.py`: `--unshare-all --clearenv`, read-only `/usr`, and a memory cap. Only these paths are visible:
  - the run's work folder (read-write);
  - its metadata, the exported weights and this package (read-only);
  - the folders holding practice images (read-only).

  Home folders, other runs, API configuration and any other data do not exist inside it. A run refuses to start
  without bubblewrap unless the config says `kernel.sandbox: none` and `kernel.allow_unsandboxed: true` (tests only).

## Install and test

```bash
pip install -r requirements.txt
apt-get install bubblewrap     # agent machine
pytest tests                   # 39 tests, ~1.5 min on CPU; builds tiny Llava- and Qwen3.5-style VLMs offline
```

The tests cover:
- the graders and Arabic normalization;
- the edit rules (frozen tensors, rescale refusal, shapes, NaN) and tamper / base-mismatch detection;
- the code policy;
- the kernel's runtime blocks on gradients and optimizers;
- sandbox isolation: an outside file, the private set, home folders, API keys, the network and writes to the
  metadata are all unreachable, while the work folder and practice images are reachable;
- the PRIVATE-marker guards;
- a single-task experiment's scoping;
- the full two-machine pipeline: mock agent run, both baselines, bundles, private evaluation, comparison;
- the OpenRouter backend against a local fake server: request shape, streaming, retries, errors, and a full agent run over HTTP;
- the Hugging Face upload script against a fake Hub: batch-by-batch uploads, skipping unchanged files, the index, the private default, refusals;
- module discovery on Qwen3.5, Qwen3-VL, LightOnOCR, PaddleOCR-VL, SmolVLM, LFM2-VL, InternVL and Idefics3;
- a tiny SmolVLM with its real processor: per-sample image batching, `processor_kwargs` reaching the processor, activation capture;
- the shipped configs, including the SmolVLM example matrix.

## Experiments

The base config holds everything shared: model, protocol, all four task definitions, grading, budgets.
Each experiment picks its tasks:

| Config | The agent gets practice data and feedback for | Evaluated on (private) |
|---|---|---|
| `configs/experiments/ocr.yaml` | ocr | all four tasks (ocr = target, others = side effects) |
| `configs/experiments/arabic.yaml` | arabic | all four |
| `configs/experiments/table.yaml` | table | all four |
| `configs/experiments/handwriting.yaml` | handwriting | all four |
| `configs/experiments/all.yaml` | all four | all four |

A single-task agent sees only its task: the prompt, the practice items and the forward tool are restricted to
it. It is told that the model's other tasks are also tested.

[configs/matrix.example.yaml](configs/matrix.example.yaml) lists experiments × conditions × seeds.
`make_runs.py` expands it into one config per run, plus one shell script for each machine.

## Worked example: SmolVLM on Arabic handwriting, Claude via OpenRouter

`configs/examples/smolvlm_handwriting/` is a complete single-task study you can copy:
- `base.yaml`: SmolVLM, `processor_kwargs` for its resolution, the handwriting task only, the `openrouter` agent;
- `experiments/handwriting.yaml`;
- `matrix.yaml`: budgets 0 / 10 / 20 / 50, plus diffmeans and random baselines, 3 seeds;
- `sweep.yaml`: resolution, decoding and prompt variants;
- `prompts/task_v1.md`: an editable copy of the agent's task text.

Before spending any budget, `scripts/agent/inspect_model.py --config <experiment config> --n 4` prints what the harness
found in the model and how it answers a few practice images.

## Data format

One JSONL file per task (`<task>.jsonl`), the same format for public and private data:

```json
{"id": "table_000123", "image": "images/t123.png", "reference": "<table><tr><th>الاسم</th>...</table>"}
```

- `id`: unique across all tasks (prefix with the task name).
- `image`: relative to the JSONL file, or absolute.
- `reference`: plain text for `cer` tasks; an HTML or markdown table for `table` tasks. Header = leading rows of
  `<th>` cells (or the first row if none; or the row above `|---|` in markdown).
- Optional: `prompt` (per-item override), `meta`.

## Workflow

**Evaluation machine, once, before anything else:**
```bash
touch /secure/private/PRIVATE
python scripts/eval/check_overlap.py --public /copy/of/public --private /secure/private --tasks ocr arabic table handwriting
# remove every flagged item from the public set on the agent machine
```

**Agent machine:**
```bash
# 1. pilot on the unedited model (no budget): failure taxonomy, header -> body analysis, batching check
python scripts/agent/pilot.py --config configs/experiments/all.yaml --out pilot/ --n 300 --check-batching 32

# 2. fix the protocol on DEV practice items, then write it into configs/base.yaml
python scripts/agent/decode_sweep.py --config configs/experiments/all.yaml --grid configs/sweep.example.yaml --n 100 --out sweep/

# 3. generate every run's config and the two run scripts
python scripts/agent/make_runs.py --matrix configs/matrix.example.yaml --private /secure/private

# 4. run everything (export ANTHROPIC_API_KEY); each line can also be run alone, and re-running resumes
./agent_runs.sh            # agent runs + scripted baselines + one bundle per run + the base bundle
```

Single run by hand:
```bash
python scripts/agent/run_agent.py --config configs/generated/table/claude_b25_s1.yaml
python scripts/agent/export_bundle.py --config configs/generated/table/claude_b25_s1.yaml --out bundles/table/claude_b25_s1
```

**Evaluation machine:**
```bash
# copy bundles/ (and the repository) to this machine; nothing else
./eval_runs.sh             # evaluate every bundle on the private set, then one report per experiment
# or by hand:
python scripts/eval/evaluate.py --bundle bundles/table/claude_b25_s1.tar.gz --data /secure/private --out results/table/claude_b25_s1
python scripts/eval/compare.py --base results/base --edited results/table/claude_b25_s1 results/table/diffmeans_b25_s1 --out reports/table.md
```

`reports/*.md|json` and `results/*/summary.json` hold aggregates only. `results/*/items.jsonl` holds model outputs
on private items, so keep it on the evaluation machine.

## Uploading bundles to Hugging Face (manual)

`scripts/tools/upload_hf.py` is run by hand after a batch of runs; nothing in the run loop calls it. Typical use:
upload the 10-pass bundles, then run the 25-pass condition, then upload again (only the new bundles go up).

```bash
export HF_TOKEN=hf_...      # or --hf-token (ends up in shell history), or a saved `hf auth login`
python scripts/tools/upload_hf.py --input bundles --repo-id <user>/wsp-vlm-bundles --path-in-repo bundles --num-workers 8 --dry-run
python scripts/tools/upload_hf.py --input bundles --repo-id <user>/wsp-vlm-bundles --path-in-repo bundles --num-workers 8
```

| Option | Meaning |
|---|---|
| `--input` | a bundle, or a folder searched recursively; the layout under it is kept in the repo |
| `--repo-id`, `--repo-type` | target repo (`model` by default, or `dataset`) |
| `--path-in-repo` | folder inside the repo |
| `--num-workers` (alias `--num-process`) | parallel uploads (and parallel hashing) |
| `--private` / `--public` | visibility when the repo is created (private by default; an existing repo keeps its setting) |
| `--revision`, `--create-pr`, `--commit-message` | branch, open a PR instead of committing, message |
| `--overwrite` | replace a file whose content changed at the same path (otherwise it is reported and skipped) |
| `--include-reports` | also upload `summary.json` files and `compare.py` reports (aggregates only) |
| `--dry-run` | check everything and list what would be uploaded |

Each upload is one commit. A file already on the Hub with the same sha256 is skipped, so re-running is safe.
`<path-in-repo>/index.jsonl` is merged and re-committed each time, with one line per bundle: run, experiment,
condition, seed, passes used, edit digest, sha256 and upload time. A README is added if the repo has none.

The script refuses:
- any path under a `PRIVATE` marker;
- every `*.jsonl` (per-item results hold outputs on private items);
- any bundle whose archive holds anything besides `spec.json`, `run_info.json` and `submission/`;
- any JSON report with per-item entries.

## What the agent gets (and the rules the harness enforces)

The first message is generated from [wsp/prompts/task_v1.md](wsp/prompts/task_v1.md). Read `runs/<run>/meta/prompt.txt`
for what your agent actually saw.

**Python kernel** (sandboxed, CPU only, persistent namespace):
- `MODEL` (blocks, the MLP and attention matrices that write the residual stream, widths, frozen tensors);
- `W(name)`, `W_current(name)`, `list_params()`;
- `save_edit()`, `remove_edit()`, `clear_edits()`, `edits()`;
- `practice()`, `load_image()`, `load_gen()`, `load_fwd()`, and `WORK` (keep `notes.md` there).

**Forward-pass tool**, budgeted. A failed request is not charged; failures are capped.
- `generate`: original, edited, or both (paired). Every output is graded and summarized per task, with loop, cut-off
  and empty rates, Latin share for Arabic, and for tables header-exact rate and body F1 split by header right/wrong.
- `teacher_forced`: log-probs, top-k, and activations from the vision tower, connector and language model (residual,
  MLP hidden units, MLP and attention outputs, optional attention maps), at target, prompt-last, image or all positions.

**Rules:**

| Rule | Enforced by |
|---|---|
| no gradients, optimizers, training or iterative fitting | static policy (`wsp/policy.py`) refuses the code; the kernel disables autograd, `backward`, `requires_grad_(True)` and every optimizer at runtime; every run is logged in `tool_calls.jsonl` |
| the model runs only through the tool | the kernel has no GPU, no `transformers` import, and only exported weight files |
| the kernel sees nothing else on the machine | bubblewrap sandbox (`wsp/sandbox.py`), on by default |
| no private data on the agent side | PRIVATE-marker guard in every loader; the evaluation is a separate pipeline on another machine |
| frozen: input embeddings, output layer, final LM norm (plus `editing.frozen_patterns`) | `save_edit`, re-checked before every pass and at evaluation |
| no whole-tensor rescaling | refused in `save_edit` |
| optional: no vision-tower or connector edits | `editing.allow_vision` / `allow_connector` |
| the evaluated edit is the one submitted, on the same base | the evaluation checks base-weight hashes, tensor hashes, names and shapes |

**End of a run.** When the passes are used up, the agent must make its final choice and reply `FINAL ANSWER`. The edit
at that moment is copied to `runs/<run>/submission/`. An early final answer is refused while passes remain.

## Run directory (agent machine)

```
runs/<experiment>/<condition>_s<seed>/
  meta/        config.json, model.json (discovery), practice.json (items + fixed split), splits.json, prompt.txt
  weights/     -> the shared exported base weights (read-only in the sandbox)
  work/        the agent's folder (the only writable path in the sandbox): edit/, notes.md, gen_<k>_<model>.jsonl, fwd_<k>/
  turns/tNNN/  prompt.txt (what the agent received), answer.md (its reply), meta.json (token usage, timing)
  requests/    fwd_<k>.json and fwd_<k>.done.json
  tool_calls.jsonl, state.json, conversation.json, final_answer.md, submission/, logs/kernel_stderr.log
```

## Agents

| Backend | Use | Settings |
|---|---|---|
| `anthropic` | Claude (default `claude-opus-5-5`) | `ANTHROPIC_API_KEY`; streaming. `agent.extra` is passed to the API call unchanged: put your reasoning/effort setting there, per the API docs for that model |
| `openrouter` | any model on OpenRouter, through one key | `OPENROUTER_API_KEY`; `agent.model` = the OpenRouter model id; `agent.extra` is merged into the request body (e.g. `reasoning`, `provider`); `agent.headers`, `agent.timeout_s`, `agent.stream` (default on). Standard library only |
| `openai_compat` | any OpenAI-compatible endpoint (another frontier model, or a local server) | `agent.base_url`, `agent.api_key_env`, `agent.extra` |
| `scripted` | tests, or hand-written strategies through the same harness | `agent.script`: a .py file with `reply(turn, last_message, state)` |

OpenRouter example (in a run config or as a matrix condition's `set:`):

```yaml
agent:
  backend: openrouter
  model: <provider>/<model-id>      # as listed on OpenRouter
  max_tokens: 64000                 # reasoning tokens count against this on OpenRouter: leave room for the reply
  extra:
    reasoning: {effort: high}       # or {max_tokens: 32000}; not both
    # provider: {order: [...], allow_fallbacks: false}   # pin the provider so every seed runs on the same backend
```

The backend streams the reply and retries rate limits, server errors, dropped connections and mid-stream provider
errors. Bad requests, auth errors and missing credit stop the run with the API's message; re-running the same
command resumes it. If a reply comes back empty because reasoning used up `max_tokens`, the run stops and says so,
instead of sending the agent an empty turn. Each turn's `meta.json` records the usage, the provider that served it,
and the generation id.

The system message is one neutral line, as in the paper. Conversations restart every `handoff_every_turns` turns with
the task, the agent's `notes.md`, its current edit and the budget status. API keys (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, ...) are used only by the driver; the
sandbox starts with an empty environment.

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
- **Processor options** (e.g. SmolVLM's `size: {longest_edge: N*384}` or `do_image_splitting`) go in `model.processor_kwargs`. They are frozen into every bundle; `decode_sweep.py` can compare them.
- **Qwen3.5 on GPU.** Install `causal-conv1d` and `flash-linear-attention`.
- **Batching.** Run `pilot.py --check-batching`. If batched outputs differ from one-at-a-time outputs, set
  `protocol.batch_size: 1`; the evaluation uses the bundle's protocol, so it follows automatically.
- **Python outside `/usr`.** If the kernel's Python (conda, a venv) lives outside `/usr`, the sandbox binds the active
  interpreter's prefixes automatically. Add anything else it needs to `kernel.extra_ro_binds`.

## Known limits

- Generation uses HF `generate`. That is simple and model-agnostic but slower than vLLM; a vLLM backend is not included.
- Graders are deliberately simple and fully specified: CER/WER, multiset cell F1 for tables. Add TEDS in
  `wsp/grading.py` only as a secondary metric, so the agent's feedback and the reported score stay identical.
- The tests use tiny random models, so they check the plumbing, not the science. Before the first real agent run,
  run the pilot and one scripted baseline on the real model.
- The sandbox keeps the kernel away from the private data. It does not protect the agent machine from a malicious
  user, and the driver process (which holds the API key) is not sandboxed.
