"""Static checks on the agent's kernel code. This is the first line of the no-training rule; the
kernel also disables autograd and optimizers at runtime, and every run is logged for audit.
The goal is fair play, not a security boundary: run the kernel under bubblewrap or a container
(kernel.sandbox_cmd) to cut network and file access for real."""
from __future__ import annotations

import io
import re
import tokenize


def _strip_comments(code: str) -> str:
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code


FORBIDDEN = [
    (r"\.backward\s*\(", "backward passes are not allowed"),
    (r"\bautograd\b", "autograd is not allowed"),
    (r"requires_grad_?\s*(\(\s*True|=\s*True)", "gradients are not allowed"),
    (r"\btorch\.optim\b|\bfrom\s+torch\s+import\s+optim\b|\boptim\.(SGD|Adam|AdamW|LBFGS|RMSprop|Adagrad)\b", "optimizers are not allowed"),
    (r"\btorch\.func\b|\bfunctorch\b|\bjacrev\b|\bjacfwd\b|\bvmap\s*\(\s*grad", "functional gradients are not allowed"),
    (r"\bscipy\.optimize\b|\bfrom\s+scipy\s+import\s+optimize\b|\bminimize\s*\(", "iterative optimization is not allowed"),
    (r"\b(import|from)\s+(transformers|vllm|accelerate|peft|trl|diffusers|jax|flax|tensorflow|keras|optax)\b",
     "the model may only be run through the forward-pass tool"),
    (r"\b(import|from)\s+(requests|urllib|httpx|socket|aiohttp|openai|anthropic|http)\b", "no network access"),
    (r"\b(subprocess|multiprocessing|pty)\b|\bos\.(system|popen|exec\w*|spawn\w*|fork)\b", "no subprocesses"),
    (r"\b__import__\s*\(|\bimportlib\b|(?<![\w.])exec\s*\(|(?<![\w.])eval\s*\(|(?<![\w.])compile\s*\(", "no dynamic code execution"),
    (r"\.cuda\s*\(|device\s*=\s*['\"]cuda", "the kernel is CPU-only"),
    (r"\bwsp\.(model_io|forward|driver)\b|\bWeightSwapper\b", "harness internals are off limits"),
]


def check(code: str, extra: list[str] | None = None, warn: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Returns (refusals, warnings). Comments are ignored (strings are not)."""
    body = _strip_comments(code)
    refusals = [msg + f" (matched /{pat}/)" for pat, msg in FORBIDDEN if re.search(pat, body)]
    refusals += [f"matched a forbidden pattern /{p}/" for p in (extra or []) if re.search(p, body)]
    warnings = [f"uses /{p}/: closed-form fitting is allowed, iterative fitting to a loss is not" for p in (warn or []) if re.search(p, body)]
    return refusals, warnings
