# Experiment plan: the limits of Weight-Space Programming on a small vision-language model

## Question

Can a frontier LLM, used as it is, improve a ~1B vision-language model on document tasks by editing its
weights directly: no gradients, no training, only reasoning, weight analysis and a fixed budget of
forward passes? We use the original paper's conditions, but on a harder setting:

- **Bigger edit space.** The model has a vision tower, a connector and a language model.
- **Mostly capability errors.** Weak Arabic, misread handwriting, wrong table structure, plus occasional loops.
- **Private test sets.** Contamination is impossible.

The aim is to map where the method works and where it stops. Reproducing the paper's headline numbers is not the aim.

## Research questions

| # | Question | Measured by |
|---|---|---|
| RQ1 | Does an agent's weight edit improve each task on the private set? | paired per-item score difference vs the unedited model, per task |
| RQ2 | Which failures does it fix: switch-like ones (loops, cut-offs, wrong script, empty output) or real misreadings? | the pilot's failure taxonomy; private-set changes in loop/cut/latin-share vs CER on finished outputs |
| RQ3 | Is a wrong table header a switchable failure? When the header is fixed, does the body follow? | header_exact, header F1, body F1 split by header right/wrong; tables whose header was fixed by the edit |
| RQ4 | Does the agent's reasoning add anything beyond feedback and a scripted method at the same budget? | agent vs diffmeans vs random direction, at equal passes; 0-pass agent runs |
| RQ5 | Where do useful edits live: vision tower, connector or language model? | edit manifests; a scope ablation (LM only vs vision + connector only) |
| RQ6 | Is the edited model better overall, or better on one task at another's cost? | all four tasks + a small general regression suite |

## Predictions (write these down before the agent runs; update after the pilot)

1. **Gains follow switch-like failures.** A task's gain scales with the pilot's
   `share_of_error_from_switch_like_failures`. Tasks whose errors are mostly misreadings move by ≤ 1–2 points.
   That is the Gemma result in the paper.
2. **Headers.** If header failures are structural (missing `<th>` row, merged header cells, wrong column count),
   an edit can fix part of them, and the body improves on those tables. If they are misread header words, no.
   The pilot's `header_f1_bins` tells which case you are in.
3. **Feedback matters.** Agent runs at 0 passes leave the model unchanged or make it worse. Gains grow with budget.
4. **Agent vs script.** At equal budget the agent ≥ diffmeans > random. If agent ≈ diffmeans, the reasoning adds little
   beyond automated steering.

## Model selection (after the pilot, before any agent run)

Choose a ~1B VLM that:
- knows Arabic and tables but makes visible errors, with baseline task scores roughly 0.3–0.8: neither near zero nor near ceiling;
- loads with `AutoModelForImageTextToText`, runs in bf16 on one GPU, and has a pinned revision;
- shows at least one switch-like failure in the pilot (loops, cut-offs, script drift, broken table markup).
  Without one, RQ1 is likely to come out null; that is still a result, but plan for it.

Discovery and capture were tested on tiny versions of Qwen3.5, Qwen3-VL, LightOnOCR, PaddleOCR-VL,
SmolVLM, LFM2-VL, InternVL and Idefics3 (`tests/test_architectures.py`).

## Data (two machines)

| Set | Where it lives | Use | Size per task (suggested) |
|---|---|---|---|
| Public practice | agent machine | agent feedback (DEV) and confirmation (HOLDOUT), a fixed 50/50 split per seed | 500–2,000 |
| Private test | evaluation machine only, folder marked with a `PRIVATE` file | the only reported numbers | ≥ 300 (≥ 500 for tables) |
| General regression | evaluation machine | side effects (RQ6): e.g. English OCR, a VQA set, a text-only set | 200–500 |

The agent machine never holds private data, private outputs or per-item private results. The evaluation machine
receives only bundles: the edit plus a frozen evaluation spec. Only aggregate reports leave it. The overlap check
runs on the evaluation machine with a copy of the public set; flagged public items are removed on the agent machine.

Fix the Arabic normalization and the table prompt (HTML with `<th>`) before the pilot. Both change the scores.

## Experiments and conditions

Each task is its own experiment, and one more experiment covers all four:

| Experiment | The agent works on (practice data + feedback) | Question it answers |
|---|---|---|
| E-OCR | printed OCR | can it fix plain transcription errors / loops? |
| E-AR | Arabic documents | script drift, Arabic-specific misreadings |
| E-TAB | tables | the header problem (RQ3): structure vs text |
| E-HW | handwriting | the hardest capability limit; expected null |
| E-ALL | all four at once | does one edit improve the model overall (RQ6), and does it beat the best single-task edit on its own task? |

Every experiment's submissions are evaluated on **all four** private test sets. The other three tasks measure the
side effects of a single-task edit.

Conditions, run inside every experiment with the same model, protocol, practice data and seeds (seed = the DEV/HOLDOUT split):

| ID | Editor | Passes | Edit scope | Seeds | Why |
|---|---|---|---|---|---|
| A0 | none (unedited, best decoding setting) | – | – | – | the baseline every edit must beat (one bundle, shared by all experiments) |
| A1 | Claude agent | 0 / 25 / 50 | all | 3 each | budget curve; 0 passes = weights + practice data only |
| A2 | second agent (Astra or another; `openrouter` or `openai_compat` backend) | 25 | all | 3 | is it the agent or the method? Pin the OpenRouter provider so all seeds run on the same backend |
| B1 | diffmeans script | 25 | LM writers | 3 | automated version of the paper's edit |
| B2 | random direction (matched) | 25 | LM writers | 3 | does the direction carry information? |
| B3 | diffmeans script, `--where vision` | 25 | vision writers | 3 | can the vision tower be steered the same way? |
| C1 | Claude agent | 25 | LM + connector only (`allow_vision: false`) | 2 | scope ablation |
| C2 | Claude agent | 25 | vision + connector only (LM frozen via `frozen_patterns`) | 2 | scope ablation |

[configs/matrix.example.yaml](configs/matrix.example.yaml) generates this as one config per run. The default matrix
is 5 experiments × 5 conditions (A1 at 0/25/50, B1, B2) × 3 seeds = 75 runs, plus A0.

**If compute is tight**, in this order:
1. E-TAB and E-ALL with A1 (0, 25), B1 and B2, 3 seeds each. That is 24 runs and answers RQ1, RQ3, RQ4 and RQ6.
2. Then the other single-task experiments.
3. Then A2 and the scope ablations.

## Metrics

- **Score per item** in [0, 1]. Text tasks: 1 − CER after normalization. Tables: 0.5 · header-cell F1 + 0.5 · body-cell F1.
- **Primary result:** mean over the four tasks of the paired private-set score difference (edited − base), with 95% CI.
- **Secondary:**
  - per-task differences (Holm-corrected);
  - CER on outputs that were not loops or cut-offs (genuine reading);
  - loop / cut / empty / latin-share rates;
  - header_exact, body F1 when the header is right vs wrong, and how many tables were "header fixed" vs "header broken";
  - regression suite.
- **Cost:** generated tokens and GPU-hours per run; agent tokens per run (from `turns/*/meta.json`).

## Statistics

- Paired per-item differences with bootstrap CIs (`scripts/eval/compare.py`); Holm correction across the 4 tasks, per experiment.
- Report each condition as mean ± SD across seeds, plus every individual run. The paper had one run per budget.
- A gain counts only if it exceeds 2 SE on the private set and no task drops significantly.
- Agent vs script: compare the per-seed private gains (3 vs 3); report all values, not only means.

## Procedure

1. **Setup.**
   - Install on both machines and run `pytest tests` (27 tests).
   - Install bubblewrap on the agent machine.
   - Prepare `data/public` (agent machine) and the private set (evaluation machine, `touch PRIVATE`) in the JSONL format.
   - On the evaluation machine, run `scripts/eval/check_overlap.py` against a copy of the public set, then remove flagged public items.
2. **Pilot** on the agent machine, no budget, on 2–3 candidate models:
   `scripts/agent/pilot.py --config configs/experiments/all.yaml --check-batching 32`.
   Read `pilot_report.json`: the switch-like error share per task, the header analysis, loop rates. Pick the model and
   write down the predictions above with numbers.
3. **Protocol.** Run `scripts/agent/decode_sweep.py` over decoding, image size and prompt variants on DEV. Freeze the
   best into `configs/base.yaml`, pin the model revision, and export the A0 bundle.
4. **Runs.** `make_runs.py`, then `agent_runs.sh`: baselines and agent runs for every experiment, one fresh run
   directory each. Re-running a command resumes after a crash.
5. **Transfer.** Copy `bundles/` to the evaluation machine.
6. **Evaluation.** On the evaluation machine, `eval_runs.sh`: every bundle on all four private sets, then one
   `compare.py` report per experiment.
7. **Analysis.**
   - Read each run's edit manifest: which tensors, how large (`rel_change_fro`).
   - Read the agent's explanations: what did it diagnose?
   - Test RQ3 on the tables whose header changed.
   - Build the cross-task matrix: each single-task experiment's effect on all four tasks.
   - Run the regression suite on the best submissions.

## Compute (rough)

- One generation pass = items × mean output tokens.
- A 50-pass agent run with 256-item passes and ~800-token outputs generates ~10M tokens. That is about 1–2
  GPU-hours on a 1B model with HF `generate` at batch 16, plus teacher-forced passes and the agent's API cost.
- The paper's 50-pass run used ~145M tokens because of 30k-token reasoning traces. OCR outputs are far shorter.
- The 24-run priority set fits in a few GPU-days; the full 75-run matrix takes roughly 2–3 GPU-weeks on one GPU. The agent API cost depends on the model and on turn counts (100–200 turns per 50-pass run in the paper).

## Risks and how the design handles them

| Risk | Mitigation |
|---|---|
| Baseline set badly, so the "gain" is a decoding fix | decode sweep first; A0 = best decoding setting |
| The agent tunes to the grader's quirks (normalization, markup) | the same grader everywhere, fixed before the runs; inspect outputs of the best runs for format tricks |
| Selection on DEV noise | fixed DEV/HOLDOUT split; the agent is told to confirm on HOLDOUT; only private numbers are reported |
| Leakage between public and private | overlap check; the private set only on the evaluation machine (PRIVATE marker, refused by every agent-side loader); bundles carry no data |
| The agent's code reads files it should not | bubblewrap sandbox: only its folder, the weights and the practice images exist inside it; no network; empty environment |
| One lucky run | 3 seeds per condition; report every run |
| Outputs depend on batch composition | `pilot.py --check-batching`; set `batch_size: 1` if they differ |
| The agent trains covertly | static policy + runtime blocks in the kernel + every run logged in `tool_calls.jsonl` for audit |
| The agent stalls or loops | turn and failure limits, handoffs with its notebook, resumable runs |

## What gets released (as the paper did)

Bundles go to a Hugging Face repo with `scripts/tools/upload_hf.py`, batch by batch as conditions finish (private until release).

Every run directory (prompts, replies, code and outputs, forward requests and results, edit manifests and tensors),
the evaluation JSONs, the compare reports, the configs and this repository. The private set itself stays private;
release its per-item scores and hashes only.
