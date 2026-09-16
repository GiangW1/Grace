#!/usr/bin/env python3
"""Download official model/data and print export lines for training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ASSETS = {
    "model": ("Qwen/Qwen3-4B-Base", "model", "Qwen3-4B-Base"),
    "train": ("BytedTsinghua-SIA/DAPO-Math-17k", "dataset", "dapo"),
    "eval": ("HuggingFaceH4/MATH-500", "dataset", "math500"),
}
DEFAULT_MIRROR = "https://hf-mirror.com"


def _apply_endpoint(official: bool) -> str:
    """Prefer hf-mirror unless the user already set HF_ENDPOINT or asked for official."""
    if official:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    elif not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = DEFAULT_MIRROR
    return os.environ["HF_ENDPOINT"]


def _snapshot(repo_id: str, repo_type: str, dest: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(dest))
    meta = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "endpoint": os.environ.get("HF_ENDPOINT"),
        "local_dir": str(dest),
    }
    refs = dest / "refs" / "main"
    if refs.is_file():
        meta["hf_ref_main"] = refs.read_text(encoding="utf-8").strip()
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, repo_type=repo_type)
        meta["sha"] = getattr(info, "sha", None)
        last = getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
        meta["last_modified"] = None if last is None else str(last)
    except Exception as exc:
        meta["sha_error"] = f"{type(exc).__name__}: {exc}"
    import json

    (dest / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def _first_file(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    hits = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    return hits[0] if hits else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Qwen3-4B-Base, DAPO, MATH-500")
    parser.add_argument("--root", default="data", help="local download root")
    parser.add_argument("--only", choices=list(ASSETS), nargs="*", default=list(ASSETS))
    parser.add_argument(
        "--official",
        action="store_true",
        help="download from huggingface.co instead of hf-mirror.com",
    )
    args = parser.parse_args(argv)
    endpoint = _apply_endpoint(args.official)
    print(f"using HF_ENDPOINT={endpoint}", file=sys.stderr)
    root = Path(args.root).resolve()
    paths = {}
    for key in args.only:
        repo_id, repo_type, name = ASSETS[key]
        dest = root / name
        print(f"downloading {repo_id} -> {dest}", file=sys.stderr)
        _snapshot(repo_id, repo_type, dest)
        if key == "model":
            paths["MODEL"] = dest
        elif key == "train":
            data = _first_file(dest, (".parquet", ".jsonl", ".json"))
            if data is None:
                raise FileNotFoundError(f"no parquet/jsonl under {dest}")
            paths["TRAIN_DATA"] = data
        else:
            data = _first_file(dest, (".parquet", ".jsonl", ".json"))
            if data is None:
                raise FileNotFoundError(f"no parquet/jsonl under {dest}")
            paths["EVAL_DATA"] = data
    print("# add these to the shell, then run training")
    for key in ("MODEL", "TRAIN_DATA", "EVAL_DATA"):
        if key in paths:
            print(f"export {key}={paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
