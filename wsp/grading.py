"""Graders: Arabic-aware text normalization, CER/WER, table extraction metrics (with header analysis),
loop / degeneration detection, and one per-item score in [0, 1] per task.

Everything here is deterministic and model-free. The SAME functions grade the agent's practice
feedback and the private evaluation, so the agent is optimizing exactly the metric reported.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser

try:  # fast C implementation if present
    from rapidfuzz.distance import Levenshtein as _RFLev

    def _lev(a, b) -> int:
        return _RFLev.distance(a, b)
except Exception:  # pragma: no cover - pure-python fallback
    def _lev(a, b) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


# --------------------------------------------------------------------------------------------
# Arabic normalization
# --------------------------------------------------------------------------------------------
DEFAULT_NORM = {
    "remove_diacritics": True,   # tashkeel: fatha, damma, kasra, sukun, shadda, tanween, dagger alef, ...
    "remove_tatweel": True,      # kashida U+0640
    "unify_alef": True,          # أ إ آ ٱ -> ا
    "ya_maqsura": True,          # ى -> ي
    "ta_marbuta": False,         # ة -> ه (off by default: it changes meaning)
    "hamza_carriers": False,     # ؤ -> و, ئ -> ي
    "digits": "western",         # "western" | "arabic_indic" | "keep"
    "strip_bidi": True,          # LRM/RLM/ALM/embeddings/isolates
    "collapse_space": True,
    "lowercase": True,           # Latin only (Arabic has no case)
    "nfkc": True,                # also maps Arabic presentation forms to base letters
    "unify_punct": True,         # Arabic comma/semicolon/question mark -> Latin
}

_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ࣓-ࣿ]")
_BIDI = re.compile("[‎‏؜‪-‮⁦-⁩﻿]")
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ","})


def normalize(text: str, opts: dict | None = None) -> str:
    o = dict(DEFAULT_NORM)
    if opts:
        o.update(opts)
    t = text or ""
    if o["nfkc"]:
        t = unicodedata.normalize("NFKC", t)
    if o["strip_bidi"]:
        t = _BIDI.sub("", t)
    if o["remove_diacritics"]:
        t = _DIACRITICS.sub("", t)
    if o["remove_tatweel"]:
        t = t.replace("ـ", "")
    if o["unify_alef"]:
        t = re.sub("[أإآٱ]", "ا", t)
    if o["ya_maqsura"]:
        t = t.replace("ى", "ي")
    if o["ta_marbuta"]:
        t = t.replace("ة", "ه")
    if o["hamza_carriers"]:
        t = t.replace("ؤ", "و").replace("ئ", "ي")
    if o["digits"] == "western":
        t = t.translate(_ARABIC_INDIC)
    elif o["digits"] == "arabic_indic":
        t = t.translate(_ARABIC_INDIC).translate(_TO_ARABIC_INDIC)
    if o["unify_punct"]:
        t = t.translate(_PUNCT)
    if o["lowercase"]:
        t = t.lower()
    if o["collapse_space"]:
        t = re.sub(r"\s+", " ", t).strip()
    return t


def script_shares(text: str) -> dict:
    """Share of letters that are Arabic vs Latin (language-drift diagnostic)."""
    ar = sum(1 for c in text if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" or "ﭐ" <= c <= "﷿" or "ﹰ" <= c <= "﻿")
    la = sum(1 for c in text if c.isascii() and c.isalpha())
    tot = max(1, ar + la)
    return {"arabic": ar / tot, "latin": la / tot}


# --------------------------------------------------------------------------------------------
# Text metrics
# --------------------------------------------------------------------------------------------
def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return _lev(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    r, p = ref.split(), pred.split()
    if not r:
        return 0.0 if not p else 1.0
    return _lev(p, r) / len(r)


# --------------------------------------------------------------------------------------------
# Output cleanup
# --------------------------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)


def clean_output(text: str) -> str:
    """Strip reasoning blocks and a single surrounding code fence; keep everything else."""
    t = _THINK.sub("", text or "")
    if "</think>" in t:  # unterminated opening tag already removed; drop reasoning before a lone close tag
        t = t.split("</think>")[-1]
    m = _FENCE.findall(t)
    if len(m) == 1:
        outside = _FENCE.sub("", t).strip()
        if len(outside) <= max(40, 0.2 * len(m[0])):  # the fence holds the answer; drop chatter around it
            t = m[0]
    return t.strip()


# --------------------------------------------------------------------------------------------
# Loops / degeneration
# --------------------------------------------------------------------------------------------
DEFAULT_LOOP = {"tail_words": 200, "ngram": 6, "dup_threshold": 0.5, "min_words": 40}


def loop_stats(text: str, opts: dict | None = None) -> dict:
    """Share of n-grams in the output tail that repeat inside the tail. A looping OCR model
    re-emits the same line or table row; a healthy transcription rarely repeats 6-word spans."""
    o = dict(DEFAULT_LOOP)
    if opts:
        o.update(opts)
    words = (text or "").split()
    tail = words[-o["tail_words"]:]
    n = o["ngram"]
    grams = [tuple(tail[i:i + n]) for i in range(max(0, len(tail) - n + 1))]
    if not grams:
        return {"dup_frac": 0.0, "loop": False, "n_words": len(words)}
    c = Counter(grams)
    dup = sum(v for v in c.values() if v > 1) / len(grams)
    return {"dup_frac": round(dup, 4), "loop": bool(len(words) >= o["min_words"] and dup >= o["dup_threshold"]),
            "n_words": len(words)}


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------
class _TableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self._rows, self._row, self._cell, self._in_table = [], None, None, None, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._in_table += 1
            if self._in_table == 1:
                self._rows = []
        elif tag == "tr" and self._in_table:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = {"text": "", "th": tag == "th",
                          "colspan": _int(a.get("colspan")), "rowspan": _int(a.get("rowspan"))}
        elif tag == "br" and self._cell is not None:
            self._cell["text"] += " "

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._cell is not None:
                self._row.append(self._cell)
                self._cell = None
            self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._in_table:
            self._in_table -= 1
            if self._in_table == 0:
                if self._row:
                    self._rows.append(self._row)
                self.tables.append(self._rows)
                self._rows, self._row = None, None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data

    def close_all(self):
        if self._rows is not None:  # unterminated table (e.g. truncated output)
            if self._cell is not None and self._row is not None:
                self._row.append(self._cell)
            if self._row:
                self._rows.append(self._row)
            self.tables.append(self._rows)
            self._rows = None


def _int(v):
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return 1


def _expand_spans(rows):
    """HTML rows with colspan/rowspan -> rectangular grid of (text, is_th)."""
    grid, pending = [], {}
    for r, row in enumerate(rows):
        out, col = [], 0
        cells = list(row)
        while cells or (r, col) in pending:
            if (r, col) in pending:
                out.append(pending.pop((r, col)))
                col += 1
                continue
            c = cells.pop(0)
            for dc in range(c["colspan"]):
                v = (c["text"], c["th"])
                out.append(v)
                for dr in range(1, c["rowspan"]):
                    pending[(r + dr, col)] = v
                col += 1
        grid.append(out)
    return grid


def parse_table(text: str) -> dict | None:
    """Return {"header": [[...]], "body": [[...]], "format": "html"|"markdown"} for the first table
    in the text, or None. Header = leading rows made of <th> cells (HTML) or the row above the
    |---| separator (markdown). If an HTML table has no <th>, its first row is taken as header."""
    t = clean_output(text)
    if "<table" in t.lower() or "<tr" in t.lower():
        p = _TableHTML()
        src = t if "<table" in t.lower() else "<table>" + t + "</table>"
        try:
            p.feed(src)
            p.close_all()
        except Exception:
            pass
        if p.tables and p.tables[0]:
            grid = _expand_spans(p.tables[0])
            n_head = 0
            for row in grid:
                if row and all(th for _, th in row):
                    n_head += 1
                else:
                    break
            if n_head == 0:
                n_head = 1
            cells = [[html.unescape(x).strip() for x, _ in row] for row in grid]
            return {"header": cells[:n_head], "body": cells[n_head:], "format": "html"}
    lines = [ln.strip() for ln in t.splitlines() if ln.strip().startswith("|") or ln.count("|") >= 2]
    if len(lines) >= 1:
        rows, sep_at = [], None
        for ln in lines:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) and any(cells):
                sep_at = len(rows)
                continue
            rows.append(cells)
        if rows:
            n_head = sep_at if sep_at else 1
            return {"header": rows[:n_head], "body": rows[n_head:], "format": "markdown"}
    return None


def _f1(pred: list, ref: list) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def table_metrics(pred_text: str, ref_text: str, norm_opts: dict | None = None) -> dict:
    """Table metrics, built to test the 'header decides the rest' hypothesis:
      header_exact     every header cell matches, same number of header cells (after normalization)
      header_cell_f1   multiset F1 over header cells
      body_cell_f1     multiset F1 over body cells
      body_row_f1      F1 over whole body rows (a row counts only if every cell matches)
      col_count_ok     predicted column count == reference column count
      shape_ok         same number of rows and columns
      parsed           a table was found in the output
    score = 0.5 * header_cell_f1 + 0.5 * body_cell_f1."""
    ref = parse_table(ref_text)
    if ref is None:
        raise ValueError("reference contains no parseable table")
    pred = parse_table(pred_text)
    nz = lambda s: normalize(s, norm_opts)
    rh = [nz(c) for row in ref["header"] for c in row]
    rb_rows = [tuple(nz(c) for c in row) for row in ref["body"]]
    rb = [c for row in rb_rows for c in row]
    ncols_ref = max((len(r) for r in ref["header"] + ref["body"]), default=0)
    if pred is None:
        return {"parsed": False, "header_exact": False, "header_cell_f1": 0.0, "body_cell_f1": 0.0,
                "body_row_f1": 0.0, "col_count_ok": False, "shape_ok": False, "score": 0.0,
                "n_rows_pred": 0, "n_rows_ref": len(ref["body"])}
    ph = [nz(c) for row in pred["header"] for c in row]
    pb_rows = [tuple(nz(c) for c in row) for row in pred["body"]]
    pb = [c for row in pb_rows for c in row]
    ncols_pred = max((len(r) for r in pred["header"] + pred["body"]), default=0)
    hf1, bf1 = _f1(ph, rh), _f1(pb, rb)
    return {
        "parsed": True,
        "header_exact": ph == rh,
        "header_cell_f1": round(hf1, 4),
        "body_cell_f1": round(bf1, 4),
        "body_row_f1": round(_f1(pb_rows, rb_rows), 4),
        "col_count_ok": ncols_pred == ncols_ref,
        "shape_ok": ncols_pred == ncols_ref and len(pred["body"]) == len(ref["body"]),
        "n_rows_pred": len(pred["body"]), "n_rows_ref": len(ref["body"]),
        "score": round(0.5 * hf1 + 0.5 * bf1, 4),
    }


# --------------------------------------------------------------------------------------------
# One entry point used by every caller
# --------------------------------------------------------------------------------------------
def grade(task_cfg: dict, pred: str, ref: str, grading_cfg: dict | None = None, hit_max: bool = False) -> dict:
    """Grade one output. task_cfg["metric"] is "cer" or "table". Returns a flat dict with "score"."""
    g = grading_cfg or {}
    norm_opts = g.get("arabic_norm")
    loop_opts = g.get("loop")
    raw = pred or ""
    out = clean_output(raw)
    res = {"hit_max": bool(hit_max), "empty": not out.strip()}
    res.update(loop_stats(raw, loop_opts))
    if task_cfg.get("metric", "cer") == "table":
        res.update(table_metrics(out, ref, norm_opts))
    else:
        p, r = normalize(out, norm_opts), normalize(ref, norm_opts)
        c = cer(p, r)
        res.update({"cer": round(c, 4), "wer": round(wer(p, r), 4), "score": round(max(0.0, 1.0 - c), 4)})
    if task_cfg.get("arabic"):
        res["latin_share"] = round(script_shares(out)["latin"], 4)
    return res
