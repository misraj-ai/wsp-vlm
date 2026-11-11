YOUR TASK. Make the vision-language model described below better at reading documents: printed-text OCR, Arabic documents, table extraction and handwriting. You do this ONLY by editing its weights directly. You may not train it: no gradient descent, no backward passes, no autograd, no optimizers, and no iterative fitting of weights to a loss. Closed-form computations on weights and recorded activations (means, differences, projections, SVD, solving a linear system once) are allowed. Your edit is then evaluated on PRIVATE test sets, one per task, that you will never see. They come from the same kinds of documents as your practice data, and they are scored exactly as your practice answers are scored. The goal is a model that is better overall: improve the tasks you can without making the others worse.

THE MODEL
{{MODEL_DESCRIPTION}}

THE TASKS (each item is one image plus the task prompt; the model's reply is graded against a reference)
{{TASKS}}

HOW EACH ANSWER IS SCORED
{{GRADING}}

HOW THE MODEL IS RUN (the evaluation protocol; the forward-pass tool uses exactly the same)
{{PROTOCOL}}

WHAT YOU HAVE
1. A persistent Python kernel over the weights (CPU only, no GPU, no internet). Reply with one ```python block to run code. The kernel cannot run the model. Its namespace provides:
   - `np`, `torch`, `json`, `os`, `math`, `re`, `Image` (PIL)
   - `MODEL`: a dict describing the network (block lists, the matrices that write the residual stream in each block, frozen tensors, widths)
   - `list_params(pattern=None)` returns [(name, shape, dtype)]; `W(name)` returns the original tensor (float32); `W_current(name)` returns your edited version if you saved one
   - `save_edit(name, tensor)` replaces a whole tensor in your edit. Shape must match. Refused: the frozen tensors listed above, NaN/inf, and an edit that only multiplies a whole tensor by one constant. `remove_edit(name)`, `clear_edits()`, `edits()` (a summary)
   - `practice(task=None, split=None)` returns the practice items (id, task, split, image path, reference); `load_image(id)` returns the image
   - `load_gen(k, model)` returns the graded outputs of generation pass k; `load_fwd(k, model)` returns the arrays of teacher-forced pass k
   - `WORK`: your working folder, which persists. Keep a notebook in `WORK/notes.md`: if this conversation is restarted, you get that file back, plus your current edit and the budget status.
   At most {{PYTHON_RUNS}} Python runs, each at most {{KERNEL_TIMEOUT}} seconds.
2. Practice data (public, never in the private test sets): {{PRACTICE_STATS}}. Every item has a fixed split, made before you started: "dev" for choosing edits, "holdout" for confirming an edit you have already frozen. Using the holdout only for confirmation is your best protection against fooling yourself.
3. A forward-pass tool with a budget of {{PASSES}} passes. Reply with one ```forward block holding one JSON object. A request that fails is not counted. Two kinds:
   (a) GENERATION: the model transcribes practice items with the evaluation protocol, and every output is graded.
       {"type": "generate", "model": "original" | "edited" | "both", "task": "table", "split": "dev", "n": 64, "seed": 0}
       or {"type": "generate", "model": "both", "ids": ["...", "..."]}
       At most {{GEN_MAX}} items for one model, or {{GEN_BOTH_MAX}} per model for "both", which runs the original and your edited weights on the same items and returns paired differences. Optional "max_new_tokens" (up to the protocol's limit). The returned summary has per-task scores, error rates, loop and cut-off rates and, for tables, header accuracy and body accuracy split by whether the header was right. Per-item outputs: `load_gen(k, model)`.
   (b) TEACHER-FORCED READ: the model reads given texts as its answers (default: the reference) and returns per-token log-probabilities and the activations you ask for.
       {"type": "teacher_forced", "model": "original" | "edited" | "both",
        "items": [{"id": "..."}, {"id": "...", "target": "any text"}, {"id": "...", "target_from": "gen_3_original"}],
        "capture": {"lm_resid": [0, 6, 12] or "all", "lm_mlp_out": [...], "lm_mlp_act": [...], "lm_attn_out": [...],
                    "vision_resid": [...], "vision_mlp_act": [...], "connector": true, "lm_attn": [layers],
                    "positions": "target" | "prompt_last" | "image" | "all", "reduce": "none" | "mean", "topk": 10}}
       "lm_resid" = the block's output (residual stream), "lm_mlp_act" = the MLP hidden units feeding its output matrix, "*_out" = what the MLP or attention adds to the residual stream, "vision_*" = the vision tower (all patches), "connector" = the projector output. "lm_attn" (attention patterns) works only with eager attention, for at most {{TF_ATTN_MAX}} items. At most {{TF_MAX}} items and {{TF_TOKENS}} tokens per pass. Arrays: `load_fwd(k, model)`; rows of item j are offsets[key][j]:offsets[key][j+1] in meta.
{{LIMITS_NOTE}}

RULES
- Edit weights only through save_edit. Run the model only through the forward-pass tool. No training of any kind (see above). No internet.
- The model's input embeddings, output layer and final norm are frozen. Everything else may be edited: vision tower, connector and language model{{EDIT_SCOPE_NOTE}}.

EACH REPLY
{{EXPLAIN_RULE}}Then give exactly one ```python or ```forward block. Once all forward passes are used you will be asked for your final choice: whatever is in your edit at that moment is submitted (an empty edit means the original model). Then reply with FINAL ANSWER followed by a short report and no code block: what you changed, why, and the evidence, including your holdout result.
