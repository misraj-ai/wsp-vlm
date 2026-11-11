from wsp.grading import normalize, cer, wer, parse_table, table_metrics, loop_stats, grade, clean_output


def test_arabic_normalization():
    assert normalize("أَحْمَدُ") == "احمد"
    assert normalize("مـــرحبا") == "مرحبا"
    assert normalize("على") == "علي"
    assert normalize("٢٠٢٦") == "2026"
    assert normalize("‏نص‎") == "نص"
    assert normalize("مدرسة") == "مدرسة"                       # ta marbuta kept by default
    assert normalize("مدرسة", {"ta_marbuta": True}) == "مدرسه"
    assert normalize("ﻻ") == "لا"                               # presentation form -> base letters


def test_cer_wer():
    assert cer("abc", "abc") == 0
    assert abs(cer("abd", "abc") - 1 / 3) < 1e-9
    assert wer("a b c", "a x c") == 1 / 3
    assert cer("", "") == 0 and cer("x", "") == 1


def test_html_table_with_header_and_spans():
    t = "<table><tr><th>الاسم</th><th colspan='2'>القيمة</th></tr><tr><td>a</td><td>1</td><td>2</td></tr></table>"
    p = parse_table(t)
    assert p["header"] == [["الاسم", "القيمة", "القيمة"]]
    assert p["body"] == [["a", "1", "2"]]


def test_markdown_table():
    t = "```markdown\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n```"
    p = parse_table(t)
    assert p["header"] == [["A", "B"]] and p["body"] == [["1", "2"], ["3", "4"]]


def test_table_metrics_header_wrong():
    ref = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Ali</td><td>30</td></tr></table>"
    good = table_metrics(ref, ref)
    assert good["header_exact"] and good["score"] == 1.0 and good["shape_ok"]
    bad_head = "<table><tr><th>Nme</th><th>Age</th></tr><tr><td>Ali</td><td>30</td></tr></table>"
    m = table_metrics(bad_head, ref)
    assert not m["header_exact"] and m["body_cell_f1"] == 1.0 and 0 < m["score"] < 1
    none = table_metrics("no table here", ref)
    assert none["parsed"] is False and none["score"] == 0


def test_truncated_html_table_still_parses():
    t = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2"
    p = parse_table(t)
    assert p["header"] == [["A", "B"]] and p["body"] == [["1", "2"]]


def test_loop_detection():
    healthy = " ".join(f"word{i}" for i in range(300))
    looping = "start " + "the same line repeated again and again " * 60
    assert not loop_stats(healthy)["loop"]
    assert loop_stats(looping)["loop"]


def test_grade_entry_points():
    r = grade({"metric": "cer", "arabic": True}, "<think>x</think>مرحبا hello", "مرحبا")
    assert r["cer"] > 0 and 0 < r["latin_share"] < 1
    r2 = grade({"metric": "table"}, "| a | b |\n|--|--|\n| 1 | 2 |", "| a | b |\n|--|--|\n| 1 | 2 |")
    assert r2["score"] == 1.0
    assert clean_output("```\nhello\n```") == "hello"
