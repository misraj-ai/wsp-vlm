"""Upload finished bundles (and, optionally, aggregate evaluation reports) to a Hugging Face repo.
Run it by hand whenever a batch of runs is done (for example after the 10-pass runs, then again after the
25-pass runs). It is not part of any run loop.

    export HF_TOKEN=hf_...              # or --hf-token (visible in shell history), or a saved `hf auth login`
    python scripts/tools/upload_hf.py --input bundles --repo-id <user>/wsp-vlm-bundles --path-in-repo bundles --num-workers 8

What it does:
  - finds every bundle (*.tar.gz) under --input (a folder or a single file) and checks each one: it must hold only
    spec.json, run_info.json and submission/ (manifest + tensors), and every tensor in the manifest must be there;
  - skips files already on the Hub with the same content (sha256), so re-running after the next batch uploads only
    the new bundles; --overwrite re-uploads changed files at the same path;
  - uploads the rest in ONE commit, with --num-workers parallel uploads (and parallel hashing);
  - keeps <path-in-repo>/index.jsonl on the Hub up to date: one line per bundle with run, experiment, condition, seed,
    passes used, edit digest, sha256 and upload time;
  - creates the repo PRIVATE unless --public is given, and adds a short README if the repo has none.

It refuses anything that could carry private data: a path under a PRIVATE marker, any *.jsonl
(results/*/items.jsonl holds model outputs on private items), and with --include-reports only summary.json files
and compare.py reports without per-item entries are allowed.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wsp.data import PRIVATE_MARKER, private_root  # noqa: E402

BUNDLE_FIXED = {"spec.json", "run_info.json", "submission", "submission/manifest.json", "submission/tensors"}

CARD = """---
tags: [weight-space-programming, wsp-vlm]
---
# Weight-space edits (wsp-vlm bundles)

Each `*.tar.gz` is one submission from the wsp-vlm harness: an edit to a vision-language model's weights made by an
agent (or a scripted baseline) without gradients, plus the frozen evaluation spec it was made under.

- `spec.json`: base model id + revision + dtype, evaluation protocol, task definitions, grading options
- `run_info.json`: run name, experiment, condition, seed, forward passes used, edit digest, harness code hash
- `submission/`: `manifest.json` + `tensors/*.safetensors` (the replaced tensors, float32; cast to the model dtype when applied)

`index.jsonl` lists every bundle. Apply one with `scripts/eval/build_checkpoint.py --bundle <file>` from the wsp-vlm repo.
No practice data, model outputs or private test data are included.
"""


# ------------------------------------------------------------------------------------------------
# hashing and checks
# ------------------------------------------------------------------------------------------------
def file_hashes(path: str) -> tuple[str, str, int]:
    """(sha256, git blob sha1, size): the Hub reports sha256 for LFS files and the git blob id for small ones."""
    size = os.path.getsize(path)
    s256, s1 = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            s256.update(chunk)
            s1.update(chunk)
    return s256.hexdigest(), s1.hexdigest(), size


def check_bundle(path: str) -> dict:
    """Validate a bundle's members; return its run_info (+ spec summary) for the index."""
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        names = {m.name.rstrip("/") for m in members}
        for m in members:
            n = m.name.rstrip("/")
            if m.name.startswith("/") or ".." in n.split("/") or not (m.isfile() or m.isdir()):
                raise ValueError(f"unsafe member {m.name}")
            if n not in BUNDLE_FIXED and not (n.startswith("submission/tensors/") and n.endswith(".safetensors")
                                              and n.count("/") == 2):
                raise ValueError(f"unexpected member {m.name} (a bundle holds only spec.json, run_info.json, submission/)")
        for need in ("spec.json", "run_info.json", "submission/manifest.json"):
            if need not in names:
                raise ValueError(f"missing {need}")
        spec = json.load(tar.extractfile("spec.json"))
        info = json.load(tar.extractfile("run_info.json"))
        manifest = json.load(tar.extractfile("submission/manifest.json"))
    for tname in manifest:
        if f"submission/tensors/{tname.replace('/', '_')}.safetensors" not in names:
            raise ValueError(f"manifest lists {tname} but its tensor file is missing")
    for key in ("model", "protocol", "tasks", "grading"):
        if key not in spec:
            raise ValueError(f"spec.json has no '{key}'")
    sub = info.get("submission") or {}
    return {"kind": "bundle", "run": info.get("run"), "experiment": (info.get("experiment") or {}).get("name")
            if isinstance(info.get("experiment"), dict) else info.get("experiment"),
            "condition": info.get("condition"), "seed": info.get("seed"), "passes_used": info.get("passes_used"),
            "passes_budget": info.get("passes_budget"), "agent": info.get("agent"),
            "edit_tensors": len(manifest), "edit_digest": (sub.get("summary") or {}).get("digest"),
            "changed_entries": sum(v.get("changed_entries", 0) for v in manifest.values()),
            "code_hash": info.get("code_hash"), "model": spec["model"].get("path"),
            "model_revision": spec["model"].get("revision")}


def check_report(path: str) -> dict:
    """Only aggregate files: evaluate.py summary.json or compare.py reports (.md / .json), no per-item entries."""
    name = os.path.basename(path)
    if path.endswith(".md"):
        return {"kind": "report"}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    def has_items(x):
        if isinstance(x, dict):
            return "items" in x or any(has_items(v) for v in x.values())
        if isinstance(x, list):
            return any(has_items(v) for v in x)
        return False
    if has_items(d):
        raise ValueError(f"{name} has per-item entries; only aggregate reports may be uploaded")
    return {"kind": "summary" if name == "summary.json" else "report"}


def collect(inp: str, include_reports: bool) -> list[tuple[str, str]]:
    """[(local path, path relative to --input)] of every file to consider; refuses unsafe inputs."""
    if private_root(inp):
        raise SystemExit(f"refused: {inp} is under a {PRIVATE_MARKER} marker ({private_root(inp)})")
    if os.path.isfile(inp):
        files = [(inp, os.path.basename(inp))]
    else:
        files = []
        for dp, dns, fns in os.walk(inp):
            dns[:] = sorted(d for d in dns if not d.startswith("."))
            for fn in sorted(fns):
                files.append((os.path.join(dp, fn), os.path.relpath(os.path.join(dp, fn), inp)))
    out, refused = [], []
    for p, rel in files:
        if p.endswith(".tar.gz"):
            out.append((p, rel))
        elif p.endswith(".jsonl"):
            refused.append(f"{rel} (per-item files are never uploaded)")
        elif include_reports and (p.endswith(".md") or p.endswith(".json")):
            out.append((p, rel))
        elif not os.path.basename(p).startswith("."):
            refused.append(f"{rel} (not a bundle{'' if include_reports else '; reports need --include-reports'})")
    for r in refused:
        print(f"  skipped: {r}")
    return out


def _hash_job(p):
    return p, file_hashes(p)


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", "--input-path", dest="input", required=True, help="a bundle, or a folder of bundles (searched recursively)")
    ap.add_argument("--repo-id", required=True, help="e.g. <user>/wsp-vlm-bundles")
    ap.add_argument("--hf-token", default=None, help="default: $HF_TOKEN, then a saved login")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--path-in-repo", default="", help="folder in the repo (the layout under --input is kept below it)")
    ap.add_argument("--num-workers", "--num-process", dest="num_workers", type=int, default=8,
                    help="parallel uploads and parallel hashing")
    ap.add_argument("--revision", default=None, help="branch (default: main)")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--private", dest="private", action="store_true", default=True, help="create the repo private (default)")
    vis.add_argument("--public", dest="private", action="store_false", help="create the repo public")
    ap.add_argument("--create-pr", action="store_true", help="open a pull request instead of committing to the branch")
    ap.add_argument("--commit-message", default=None)
    ap.add_argument("--overwrite", action="store_true", help="re-upload a file whose content changed at the same path")
    ap.add_argument("--include-reports", action="store_true", help="also upload summary.json and compare reports (aggregates only)")
    ap.add_argument("--no-index", action="store_true", help="do not update index.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="check and list what would be uploaded; upload nothing")
    return ap


def main(argv=None, api=None):
    a = build_parser().parse_args(argv)
    token = a.hf_token or os.environ.get("HF_TOKEN") or None
    prefix = a.path_in_repo.strip("/")
    join = lambda rel: f"{prefix}/{rel}" if prefix else rel
    index_path = join("index.jsonl")

    # 1. local files: check, hash
    cand = collect(a.input, a.include_reports)
    if not cand:
        print("nothing to upload")
        return 0
    meta, bad = {}, []
    for p, rel in cand:
        try:
            meta[p] = check_bundle(p) if p.endswith(".tar.gz") else check_report(p)
        except (ValueError, tarfile.TarError, json.JSONDecodeError, KeyError, OSError) as e:
            bad.append(f"{rel}: {e}")
    if bad:
        print("refused (nothing was uploaded):\n  " + "\n  ".join(bad))
        return 2
    workers = max(1, a.num_workers)
    with cf.ThreadPoolExecutor(max_workers=min(workers, len(cand))) as ex:   # hashlib releases the GIL on large reads
        hashes = dict(ex.map(_hash_job, [p for p, _ in cand]))

    # 2. the repo: create (private by default), list what is there
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    if not a.dry_run:
        api.create_repo(a.repo_id, repo_type=a.repo_type, private=a.private, exist_ok=True)
    remote = {}
    try:
        for f in api.list_repo_tree(a.repo_id, recursive=True, repo_type=a.repo_type, revision=a.revision):
            if hasattr(f, "blob_id"):
                remote[f.path] = f
    except Exception as e:  # noqa: BLE001 - a new repo (dry run) or an empty branch
        if not a.dry_run:
            print(f"  (could not list the repo: {type(e).__name__}; treating it as empty)")

    # 3. decide per file
    from huggingface_hub import CommitOperationAdd

    ops, plan, now = [], [], dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for p, rel in cand:
        dest = join(rel.replace(os.sep, "/"))
        s256, s1, size = hashes[p]
        r = remote.get(dest)
        same = r is not None and ((getattr(r, "lfs", None) is not None and r.lfs.sha256 == s256) or r.blob_id == s1)
        if same:
            action = "unchanged (skip)"
        elif r is not None and not a.overwrite:
            action = "DIFFERENT on the Hub (skip; --overwrite to replace)"
        else:
            action = "replace" if r is not None else "add"
            if not a.dry_run:
                ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=p))
        plan.append((dest, size, action, dict(meta[p], path=dest, sha256=s256, size=size, uploaded_at=now)))
    print(f"{a.repo_id} ({a.repo_type}{', dry run' if a.dry_run else ''}):")
    for dest, size, action, _ in plan:
        print(f"  {action:<45} {dest}  ({size / 1e6:.1f} MB)")
    new = [m for _, _, act, m in plan if act in ("add", "replace")]
    if not new:
        print("everything is already on the Hub")
        return 0
    if a.dry_run:
        print(f"dry run: {len(new)} file(s) would be uploaded")
        return 0

    # 4. index.jsonl (merged with the one on the Hub) and a README for a repo that has none
    if not a.no_index and any(m["kind"] == "bundle" for m in new):
        rows = {}
        if index_path in remote:
            old = api.hf_hub_download(a.repo_id, index_path, repo_type=a.repo_type, revision=a.revision)
            for line in open(old, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    rows[r["path"]] = r
        for m in new:
            if m["kind"] == "bundle":
                rows[m["path"]] = m
        data = "".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in sorted(rows)).encode()
        ops.append(CommitOperationAdd(path_in_repo=index_path, path_or_fileobj=data))
    if "README.md" not in remote:
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CARD.encode()))

    bundles = [m for m in new if m["kind"] == "bundle"]
    msg = a.commit_message or (f"Add {len(bundles)} bundle(s)" + (f" and {len(new) - len(bundles)} report(s)" if len(new) > len(bundles) else ""))
    desc = "\n".join(f"- {m['path']}: run {m.get('run')}, experiment {m.get('experiment')}, condition {m.get('condition')}, "
                     f"seed {m.get('seed')}, passes {m.get('passes_used')}/{m.get('passes_budget')}, "
                     f"edit {m.get('edit_tensors')} tensor(s), digest {m.get('edit_digest')}" for m in bundles)
    info = api.create_commit(a.repo_id, ops, commit_message=msg, commit_description=desc or None,
                             repo_type=a.repo_type, revision=a.revision, create_pr=a.create_pr, num_threads=workers)
    print(f"committed {len(new)} file(s): {getattr(info, 'commit_url', info)}")
    if getattr(info, "pr_url", None):
        print(f"pull request: {info.pr_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
