"""A scripted 'agent' that walks through every harness path. Used by tests/test_e2e.py."""
import re

E = "I will do the next step of the plan and explain it here in plain words. "


def _k(msg):
    m = re.search(r"forward request (\d+)", msg)
    return int(m.group(1)) if m else None


def reply(turn, last, st):
    st.setdefault("log", []).append(turn)
    if turn == 1:
        return E + "First I look at the model description and the practice data.\n```python\n" \
                   "print(MODEL['lm'][0]['mlp_writer_weight'], len(MODEL['lm']), len(practice()), len(practice(split='dev')))\n" \
                   "open(os.path.join(WORK, 'notes.md'), 'w').write('notes: started\\n')\n```"
    if turn == 2:
        assert "kernel process died" not in last and "Traceback" not in last, last
        return "```python\nprint('no explanation')\n```"                       # sent back unrun
    if turn == 3:
        assert "EXPLAIN EVERY STEP" in last
        return E + "Now ask for the edited model before any edit exists; this must fail and not count.\n```forward\n" \
                   '{"type": "generate", "model": "both", "n": 8}\n```'
    if turn == 4:
        assert "FAILED, not counted" in last, last[:500]
        return E + "Generate with the original model on DEV.\n```forward\n" \
                   '{"type": "generate", "model": "original", "split": "dev", "n": 16, "seed": 1}\n```'
    if turn == 5:
        st["gen"] = _k(last)
        assert "pass 1 of" in last
        return E + "This block breaks the rules on purpose.\n```python\nimport transformers\n```"
    if turn == 6:
        assert "REFUSED (not run" in last
        return E + "Read activations for a few practice references.\n```forward\n" \
                   '{"type": "teacher_forced", "model": "original", "split": "dev", "n": 8, ' \
                   '"capture": {"lm_resid": "all", "lm_mlp_act": [1], "vision_resid": [0], "connector": true, ' \
                   '"positions": "target", "reduce": "mean", "topk": 3}}\n```'
    if turn == 7:
        assert "HANDOFF" in last and "notes: started" in last, last[:300]
        return E + "Build a direction from the activations and edit one writer matrix; also test refusals.\n```python\n" \
                   "d = load_fwd(3); L = len(MODEL['lm']) - 1\n" \
                   "X = torch.tensor(d[f'act.lm.resid.{L}'].astype('float32'))\n" \
                   "u = X[:4].mean(0) - X[4:].mean(0); u = u / u.norm()\n" \
                   "name = MODEL['lm'][L]['mlp_writer_weight']; Wt = W(name)\n" \
                   "save_edit(name, Wt - 2.0 * torch.outer(u, u @ Wt))\n" \
                   "for bad in [(MODEL['frozen'][0], None), (name, 'rescale')]:\n" \
                   "    try:\n" \
                   "        save_edit(bad[0], W(bad[0]) * 1.5 if bad[1] else W(bad[0]) + 0.01)\n" \
                   "    except ValueError as e:\n" \
                   "        print('refused:', e)\n" \
                   "print(edits()['tensors'], d['meta']['offsets'][f'act.lm.resid.{L}'][:3], d['topk_ids'].shape)\n```"
    if turn == 8:
        assert "refused:" in last and "frozen" in last and "multiplies the whole tensor" in last, last
        return E + "Compare original and edited on DEV.\n```forward\n" \
                   '{"type": "generate", "model": "both", "split": "dev", "n": 8, "seed": 2}\n```'
    if turn == 9:
        assert "[paired, edited - original]" in last, last
        return "FINAL ANSWER too early on purpose."
    if turn == 10:
        assert "Not yet" in last
        return E + "Confirm on HOLDOUT.\n```forward\n" \
                   '{"type": "generate", "model": "both", "split": "holdout", "n": 8, "seed": 3}\n```'
    if turn == 11:
        return E + "Edited model on tables only.\n```forward\n" \
                   '{"type": "generate", "model": "edited", "task": "table", "n": 4}\n```'
    if turn == 12:
        return E + "Teacher-forced read of both models at image positions.\n```forward\n" \
                   '{"type": "teacher_forced", "model": "both", "task": "ocr", "n": 2, ' \
                   '"capture": {"lm_resid": [0], "positions": "image"}}\n```'
    if turn == 13:
        assert "All forward passes are used" in last, last[-500:]
        return "FINAL ANSWER\nI edited one MLP writer matrix along a difference-of-means direction (mock run)."
    raise AssertionError(f"unexpected turn {turn}")
