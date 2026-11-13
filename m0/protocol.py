"""M0 configuration and task manifests; no environment construction here."""

import ast
import hashlib
import json
import random
import subprocess
from pathlib import Path

TAU_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, encoding="utf-8")


def verify_base(repo: Path, patch: Path | None = None) -> dict:
    commit = git(repo, "rev-parse", "HEAD").strip()
    if commit != TAU_COMMIT:
        raise ValueError(f"tau commit: expected {TAU_COMMIT}, got {commit}")
    status = git(repo, "status", "--porcelain", "--untracked-files=normal")
    if status:
        expected = patch.read_text(encoding="utf-8") if patch else ""
        if git(repo, "diff", "--binary").replace("\r\n", "\n") != expected:
            raise ValueError("tau worktree contains changes other than the approved patch")
        if any(line.startswith("??") for line in status.splitlines()):
            raise ValueError("tau worktree contains untracked files")
    return {"commit": commit, "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()
            if patch else None, "worktree": "approved_patch" if status else "clean"}


def build_manifest(repo: Path, seed: int = 20261029, train_count: int = 20,
                   test_count: int = 50, pilot_count: int = 5) -> dict:
    counts = {}
    for domain, split in [("retail", "dev"), ("retail", "train"),
                          ("retail", "test"), ("airline", "test")]:
        source = repo / f"tau_bench/envs/{domain}/tasks_{split}.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        counts[f"{domain}/{split}"] = sum(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Task"
            for n in ast.walk(tree))
    rng = random.Random(seed)
    def refs(domain, split, indexes):
        return [{"domain": domain, "split": split, "index": i} for i in sorted(indexes)]
    dev = refs("retail", "dev", range(counts["retail/dev"]))
    manifest = {"tau_commit": TAU_COMMIT, "seed": seed, "counts": counts,
        "development": dev + refs("retail", "train", rng.sample(range(counts["retail/train"]), train_count)),
        "test": refs("retail", "test", rng.sample(range(counts["retail/test"]), test_count)),
        "pilot": refs("retail", "dev", rng.sample(range(counts["retail/dev"]), pilot_count)),
        "airline_holdout": refs("airline", "test", range(counts["airline/test"])),
        "test_execution": "Only after M1 mechanism and configuration freeze",
        "airline_execution": "All tasks; do not run before M2 transfer evaluation"}
    validate_manifest(manifest)
    return manifest


def task_key(task: dict) -> str:
    return f"{task['domain']}/{task['split']}/{task['index']}"


def validate_manifest(manifest: dict) -> None:
    pools = {"development": {"retail/dev", "retail/train"}, "test": {"retail/test"},
             "pilot": {"retail/dev", "retail/train"}, "airline_holdout": {"airline/test"}}
    for pool, allowed in pools.items():
        seen = set()
        for task in manifest[pool]:
            source = f"{task['domain']}/{task['split']}"
            index = task["index"]
            if source not in allowed or type(index) is not int or not 0 <= index < manifest["counts"][source]:
                raise ValueError(f"Invalid {pool} task: {task}")
            key = task_key(task)
            if key in seen:
                raise ValueError(f"Duplicate {pool} task: {key}")
            seen.add(key)
    development = {task_key(t) for t in manifest["development"]}
    if not {task_key(t) for t in manifest["pilot"]} <= development:
        raise ValueError("pilot tasks must belong to development")


def load_config(path: Path, user_lock: Path | None = None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    lock = user_lock or path.with_name("user_simulator.lock.json")
    if config["user"] != json.loads(lock.read_text(encoding="utf-8")):
        raise ValueError("User simulator is frozen from M0; configuration differs from lock")
    if config["user_strategy"] != "llm" or config["concurrency"] != 1 or config["trials"] != 1:
        raise ValueError("M0 requires llm user, concurrency=1, trials=1")
    if config["prices"]["currency"] != "CNY" or config["prices"]["unit_tokens"] != 1000000:
        raise ValueError("M0 prices must be CNY per 1M tokens")
    return config


def write_manifest(manifest: dict, path: Path) -> None:
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Existing manifest differs; do not silently replace frozen tasks")
    path.write_text(text, encoding="utf-8")
