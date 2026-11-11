"""Leakage check between the public practice data and the private test sets, run before any agent run:
near-identical images (exact + content-area gradient hash) and shared reference text (8-word windows of
normalized content words; markup removed).
Every flagged practice item should be removed from the public set.

    python scripts/check_overlap.py --public data/public --private /secure/private --tasks ocr arabic table handwriting
"""
import argparse
import hashlib
import json
import re

from PIL import Image

from _common import ROOT  # noqa: F401
from wsp.data import load_dir
from wsp.grading import clean_output, normalize


def image_keys(path, n=16):
    """(exact hash, gradient hash). Documents are mostly white page, so the gradient hash is taken on
    the content area only (cropped to the non-white pixels): it compares layout, not paper."""
    im = Image.open(path).convert("L")
    exact = hashlib.sha256(im.resize((64, 64), Image.BILINEAR).tobytes()).hexdigest()
    box = Image.eval(im, lambda v: 255 if v < 200 else 0).getbbox()
    if box:
        im = im.crop(box)
    g = list(im.resize((n + 1, n), Image.BILINEAR).getdata())
    bits = 0
    for r in range(n):
        for c in range(n):
            if g[r * (n + 1) + c] > g[r * (n + 1) + c + 1]:
                bits |= 1 << (r * n + c)
    return exact, bits


def grams(text, n=8):
    """Content words only (markup, pipes and rules removed); short texts are one whole-text key."""
    t = re.sub(r"<[^>]+>", " ", clean_output(text))
    t = re.sub(r"[|\-:]+", " ", t)
    w = normalize(t).split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


ap = argparse.ArgumentParser()
ap.add_argument("--public", required=True)
ap.add_argument("--private", required=True)
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--max-hamming", type=int, default=20, help="of 256 gradient-hash bits (content area)")
ap.add_argument("--min-shared", type=float, default=0.5, help="flag if this share of a private item's 8-word windows is in one public item")
ap.add_argument("--out", default="overlap_report.json")
a = ap.parse_args()
pub, priv = load_dir(a.public, a.tasks), load_dir(a.private, a.tasks)
ph = {r["id"]: image_keys(r["image"]) for r in pub}
pg = {}
for r in pub:
    for g in grams(r["reference"]):
        pg.setdefault(g, set()).add(r["id"])
flags = []
for r in priv:
    ex, h = image_keys(r["image"])
    near = [i for i, (pe, x) in ph.items() if pe == ex or bin(h ^ x).count("1") <= a.max_hamming]
    mine = grams(r["reference"])
    hits = {}
    for g in mine:
        for i in pg.get(g, ()):
            hits[i] = hits.get(i, 0) + 1
    shared = sorted(((i, c / len(mine)) for i, c in hits.items() if mine and c / len(mine) >= a.min_shared), key=lambda x: -x[1])
    if near or shared:
        flags.append({"private_id": r["id"], "near_duplicate_images": near[:10],
                      "shared_text_with": [{"id": i, "share": round(f, 3)} for i, f in shared[:10]]})
json.dump(flags, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"{len(priv)} private items checked against {len(pub)} public items: {len(flags)} flagged -> {a.out}")
