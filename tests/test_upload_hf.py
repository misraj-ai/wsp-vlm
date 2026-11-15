"""scripts/tools/upload_hf.py against a fake Hugging Face API (no network): batch-by-batch uploads (e.g. the
10-pass runs, then the 25-pass runs), skipping unchanged files, the index, the private default and the refusals."""
import hashlib
import importlib.util
import json
import os
import tarfile
import types

import pytest
import torch
from safetensors.torch import save_file

from wsp import bundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("upload_hf", os.path.join(ROOT, "scripts", "tools", "upload_hf.py"))
up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)


class FakeHub:
    def __init__(self, tmp):
        self.files, self.commits, self.repos, self.tmp = {}, [], [], tmp

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.repos.append({"repo_id": repo_id, "repo_type": repo_type, "private": private})

    def list_repo_tree(self, repo_id, recursive=False, repo_type=None, revision=None):
        for path, data in self.files.items():
            lfs = types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if path.endswith(".tar.gz") else None
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            yield types.SimpleNamespace(path=path, blob_id=blob, lfs=lfs, size=len(data))

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
        p = os.path.join(self.tmp, "dl_" + filename.replace("/", "_"))
        open(p, "wb").write(self.files[filename])
        return p

    def create_commit(self, repo_id, operations, commit_message, commit_description=None, repo_type=None,
                      revision=None, create_pr=None, num_threads=5):
        ops = list(operations)
        for op in ops:
            src = op.path_or_fileobj
            self.files[op.path_in_repo] = src if isinstance(src, bytes) else open(src, "rb").read()
        self.commits.append({"message": commit_message, "description": commit_description,
                             "paths": [op.path_in_repo for op in ops], "num_threads": num_threads})
        return types.SimpleNamespace(commit_url=f"https://hub/{repo_id}/commit/{len(self.commits)}", pr_url=None)


def make_bundle(out, run, exp, passes, value=1.0):
    sub = os.path.join(os.path.dirname(out), f"_sub_{run}")
    os.makedirs(os.path.join(sub, "tensors"), exist_ok=True)
    save_file({"lm.0.mlp.down.weight": torch.full((4, 4), value)}, os.path.join(sub, "tensors", "lm.0.mlp.down.weight.safetensors"))
    json.dump({"lm.0.mlp.down.weight": {"changed_entries": 16, "sha256": "x"}}, open(os.path.join(sub, "manifest.json"), "w"))
    json.dump({"summary": {"digest": f"d_{run}"}}, open(os.path.join(sub, "submission.json"), "w"))
    cfg = {"model": {"path": "org/vlm-1b", "revision": "abc"}, "protocol": {"max_new_tokens": 8}, "tasks": {"table": {}},
           "grading": {}, "editing": {}, "experiment": {"name": exp, "tasks": [exp]}}
    info = {"run": run, "experiment": cfg["experiment"], "condition": f"claude_b{passes}", "seed": 1,
            "passes_used": passes, "passes_budget": passes, "submission": json.load(open(os.path.join(sub, "submission.json")))}
    return bundle.export(out, cfg, sub, info)


def run(args, hub):
    return up.main(args, api=hub)


def test_batches_skip_and_index(tmp_path, capsys):
    hub, b = FakeHub(str(tmp_path)), tmp_path / "bundles"
    make_bundle(str(b / "table" / "claude_b10_s1"), "table_claude_b10_s1", "table", 10)
    make_bundle(str(b / "all" / "claude_b10_s1"), "all_claude_b10_s1", "all", 10)
    (b / "results").mkdir()
    (b / "results" / "items.jsonl").write_text('{"text": "private output"}\n')       # must never be uploaded
    base = ["--input", str(b), "--repo-id", "me/wsp", "--path-in-repo", "bundles", "--num-workers", "4"]

    assert run(base, hub) == 0                                                       # batch 1: the 10-pass runs
    assert hub.repos[0]["private"] is True and len(hub.commits) == 1
    paths = set(hub.commits[0]["paths"])
    assert paths == {"bundles/table/claude_b10_s1.tar.gz", "bundles/all/claude_b10_s1.tar.gz", "bundles/index.jsonl", "README.md"}
    assert not any("items.jsonl" in p for p in hub.files) and hub.commits[0]["num_threads"] == 4
    idx = [json.loads(x) for x in hub.files["bundles/index.jsonl"].decode().splitlines()]
    assert {r["run"] for r in idx} == {"table_claude_b10_s1", "all_claude_b10_s1"} and idx[0]["passes_used"] == 10

    assert run(base, hub) == 0 and len(hub.commits) == 1                              # nothing new: no commit
    assert "already on the Hub" in capsys.readouterr().out

    make_bundle(str(b / "table" / "claude_b25_s1"), "table_claude_b25_s1", "table", 25)   # batch 2: the 25-pass runs
    assert run(base, hub) == 0 and len(hub.commits) == 2
    assert set(hub.commits[1]["paths"]) == {"bundles/table/claude_b25_s1.tar.gz", "bundles/index.jsonl"}
    idx = [json.loads(x) for x in hub.files["bundles/index.jsonl"].decode().splitlines()]
    assert len(idx) == 3 and "table_claude_b25_s1" in hub.commits[1]["description"]

    make_bundle(str(b / "table" / "claude_b25_s1"), "table_claude_b25_s1", "table", 25, value=2.0)  # changed content
    assert run(base, hub) == 0 and len(hub.commits) == 2                              # not replaced without --overwrite
    assert "DIFFERENT" in capsys.readouterr().out
    assert run(base + ["--overwrite"], hub) == 0 and len(hub.commits) == 3
    assert hub.commits[2]["paths"][0] == "bundles/table/claude_b25_s1.tar.gz"

    assert run(base + ["--public", "--dry-run"], hub) == 0 and len(hub.commits) == 3  # dry run uploads nothing


def test_refusals(tmp_path):
    hub = FakeHub(str(tmp_path))
    good = make_bundle(str(tmp_path / "ok" / "x"), "x", "table", 10)
    # a bundle with an extra file inside is refused, and nothing is uploaded
    badp = tmp_path / "bad" / "y.tar.gz"
    badp.parent.mkdir()
    with tarfile.open(good) as src, tarfile.open(badp, "w:gz") as dst:
        for m in src.getmembers():
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
        extra = tmp_path / "items.jsonl"
        extra.write_text("{}")
        dst.add(extra, arcname="results/items.jsonl")
    assert run(["--input", str(badp.parent), "--repo-id", "me/wsp"], hub) == 2 and not hub.commits
    # a folder under a PRIVATE marker is refused outright
    priv = tmp_path / "secure"
    (priv / "b").mkdir(parents=True)
    (priv / "PRIVATE").write_text("")
    with pytest.raises(SystemExit, match="PRIVATE"):
        run(["--input", str(priv / "b"), "--repo-id", "me/wsp"], hub)
    # reports: aggregate summaries pass, anything with per-item entries does not
    rep = tmp_path / "reports"
    rep.mkdir()
    json.dump({"summary": {"score": 0.5}, "run_info": {}}, open(rep / "summary.json", "w"))
    (rep / "table.md").write_text("| task | diff |")
    assert run(["--input", str(rep), "--repo-id", "me/wsp", "--include-reports"], hub) == 0
    assert set(hub.commits[-1]["paths"]) == {"summary.json", "table.md", "README.md"}
    json.dump({"items": [{"id": "a", "text": "private"}]}, open(rep / "leak.json", "w"))
    assert run(["--input", str(rep), "--repo-id", "me/wsp", "--include-reports"], hub) == 2
