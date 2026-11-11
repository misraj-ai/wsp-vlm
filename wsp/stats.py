"""Paired statistics over per-item scores."""
from __future__ import annotations

import math
import random


def paired(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """a, b: id -> score (b - a is the effect). Items present in both only."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        return {"n": 0}
    d = [b[i] - a[i] for i in ids]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)) if n > 1 else [m]
    lo, hi = boots[int(0.025 * len(boots))], boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return {"n": n, "base": round(sum(a[i] for i in ids) / n, 4), "edited": round(sum(b[i] for i in ids) / n, 4),
            "diff": round(m, 4), "se": round(se, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "z": round(m / se, 2) if se and se == se and se > 0 else None,
            "improved": sum(x > 1e-9 for x in d), "worsened": sum(x < -1e-9 for x in d), "same": sum(abs(x) <= 1e-9 for x in d)}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values. pvals: name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 6)
    return out


def p_from_z(z: float | None) -> float:
    if z is None:
        return 1.0
    return math.erfc(abs(z) / math.sqrt(2))
