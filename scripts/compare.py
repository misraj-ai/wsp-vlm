"""Paired comparison of evaluation results (same private data): base vs one or more edited runs.

    python scripts/compare.py --base results/base.json --edited results/claude_b25_s1.json results/scripted_b25.json \
        --out results/report.md
"""
import argparse
import json

from _common import ROOT  # noqa: F401
from wsp.stats import holm, p_from_z, paired


def load(p):
    with open(p) as f:
        return json.load(f)


def rate(rows, key):
    return sum(float(bool(r.get(key))) for r in rows) / max(1, len(rows))


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--edited", nargs="+", required=True)
ap.add_argument("--out")
a = ap.parse_args()

B = load(a.base)
lines = [f"# Paired comparison against {a.base}", ""]
bi = {r["id"]: r for r in B["items"]}
out_json = {}
for path in a.edited:
    E = load(path)
    if E["meta"].get("data_hash") != B["meta"].get("data_hash"):
        raise SystemExit(f"{path}: evaluated on different data than the base")
    ei = {r["id"]: r for r in E["items"]}
    lines += [f"## {path}", f"edit: {E['meta'].get('edit')}", "",
              "| task | n | base | edited | diff | SE | 95% CI | improved / worsened | Holm p |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    res, ps = {}, {}
    for t in sorted({r["task"] for r in B["items"]}):
        ids = [i for i, r in bi.items() if r["task"] == t and i in ei]
        res[t] = paired({i: bi[i]["score"] for i in ids}, {i: ei[i]["score"] for i in ids})
        ps[t] = p_from_z(res[t].get("z"))
    adj = holm(ps)
    for t, s in res.items():
        lines.append(f"| {t} | {s['n']} | {s['base']:.4f} | {s['edited']:.4f} | {s['diff']:+.4f} | {s['se']:.4f} | "
                     f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['improved']} / {s['worsened']} | {adj[t]:.4g} |")
    allp = paired({i: r["score"] for i, r in bi.items() if i in ei}, {i: ei[i]["score"] for i in bi if i in ei})
    lines += [f"| ALL (item mean) | {allp['n']} | {allp['base']:.4f} | {allp['edited']:.4f} | {allp['diff']:+.4f} | {allp['se']:.4f} | "
              f"[{allp['ci95'][0]:+.4f}, {allp['ci95'][1]:+.4f}] | {allp['improved']} / {allp['worsened']} | |", ""]
    # behaviour: loops, cut-offs, tables' header analysis
    lines += ["| task | loop rate base -> edited | cut rate base -> edited | extra |", "| --- | --- | --- | --- |"]
    for t in res:
        b_rows = [r for r in B["items"] if r["task"] == t]
        e_rows = [ei[r["id"]] for r in b_rows if r["id"] in ei]
        extra = ""
        if "header_exact" in b_rows[0]:
            hb, he = rate(b_rows, "header_exact"), rate(e_rows, "header_exact")
            fixed = sum(1 for r in b_rows if not r["header_exact"] and ei[r["id"]]["header_exact"])
            broke = sum(1 for r in b_rows if r["header_exact"] and not ei[r["id"]]["header_exact"])
            fixed_body = [ei[r["id"]]["body_cell_f1"] - r["body_cell_f1"] for r in b_rows
                          if not r["header_exact"] and ei[r["id"]]["header_exact"]]
            extra = (f"header exact {hb:.3f} -> {he:.3f} (fixed {fixed}, broken {broke}); body F1 change on tables whose header "
                     f"was fixed: {sum(fixed_body) / len(fixed_body):+.4f}" if fixed_body else
                     f"header exact {hb:.3f} -> {he:.3f} (fixed {fixed}, broken {broke})")
        elif "cer" in b_rows[0]:
            extra = f"CER {sum(r['cer'] for r in b_rows) / len(b_rows):.4f} -> {sum(r['cer'] for r in e_rows) / len(e_rows):.4f}"
        lines.append(f"| {t} | {rate(b_rows, 'loop'):.3f} -> {rate(e_rows, 'loop'):.3f} | {rate(b_rows, 'hit_max'):.3f} -> "
                     f"{rate(e_rows, 'hit_max'):.3f} | {extra} |")
    lines.append("")
    out_json[path] = {"by_task": res, "all": allp, "holm": adj}
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
    json.dump(out_json, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
