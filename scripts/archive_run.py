"""Archive a run with hashes for both included and excluded files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


# These are the adapter formats produced/accepted by backends/weight_sync.py.
_ADAPTER_WEIGHTS = (
    "adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt",
    "pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin",
)


def archive_run(root: Path, output: Path, max_file_mib=20, include_checkpoints=False,
                *, include_paths=()):
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected = []
    for requested in include_paths:
        path = (root / requested).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"explicit include must be inside the run root: {requested}")
        if not path.exists():
            raise FileNotFoundError(path)
        selected.append(path)
    # Sidecar references are basenames in the NPZ's own directory. Include all
    # neighboring basis files conservatively, without unpickling checkpoints.
    basis_directories = {p.parent for p in selected if p.is_file() and p.suffix == ".npz"}
    manifest = {"run_root": str(root), "files": [],
                "selection_policy": {
                    "max_file_mib": max_file_mib,
                    "always_include_suffixes": [".jsonl"],
                    "always_include_names": ["batch_audit_means.npz"],
                    "include_checkpoints": include_checkpoints,
                    "checkpoint_suffixes": [".npz"],
                    "checkpoint_dependencies": ["basis-*.npy"],
                    "explicit_npz_dependencies": "all basis-*.npy in the selected NPZ's directory",
                    "adapter_weight_names": list(_ADAPTER_WEIGHTS),
                    "include_paths": [p.relative_to(root).as_posix() for p in selected],
                },
                "note": "Excluded states remain on the server; a hash is not a recoverable checkpoint."}
    # Snapshot before creating the archive, and never archive the archive itself.
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() != output
                   and p != root / "archive_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for path in paths:
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                manifest["files"].append({"path": rel, "included": False, "reason": "symlink"})
                continue
            before = path.stat()
            size = before.st_size
            digest = hashlib.sha256()
            if any(path.is_relative_to(p) for p in selected):
                inclusion_reason = "explicit_path"
            elif path.parent in basis_directories and path.match("basis-*.npy"):
                inclusion_reason = "checkpoint_dependency"
            elif path.suffix == ".jsonl" or path.name == "batch_audit_means.npz":
                inclusion_reason = "scientific_evidence"
            elif include_checkpoints and (path.suffix == ".npz" or path.name in _ADAPTER_WEIGHTS
                                          or path.match("basis-*.npy")):
                inclusion_reason = "checkpoint"
            elif size <= max_file_mib * 1024 * 1024:
                inclusion_reason = "within_size_limit"
            else:
                inclusion_reason = None
            include = inclusion_reason is not None
            with path.open("rb") as stream:
                if include:
                    class HashingReader:
                        def read(self, amount=-1):
                            data = stream.read(amount)
                            digest.update(data)
                            return data
                    entry = tar.gettarinfo(str(path), arcname=f"{root.name}/{rel}")
                    entry.size = size
                    tar.addfile(entry, HashingReader())
                else:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            after = path.stat()
            manifest["files"].append({"path": rel, "bytes": size, "sha256": digest.hexdigest(),
                                      "included": include, "reason": None if include else "size_limit",
                                      "inclusion_reason": inclusion_reason,
                                      "source_changed_during_archive": (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns),
                                      "hash_scope": "archive_bytes" if include else "source_read"})
        data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        entry = tarfile.TarInfo(f"{root.name}/archive_manifest.json")
        entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Archive a run with streaming hashes. JSONL evidence and batch_audit_means.npz "
        "are always retained, regardless of the size limit."))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-file-mib", type=float, default=20)
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="also retain all NPZ states, basis sidecars and LoRA adapter weights, regardless of size; not base HF weights")
    parser.add_argument("--include", action="append", type=Path, default=[], metavar="PATH",
                        help="retain a file or directory regardless of size; NPZ includes neighboring basis sidecars (repeatable, relative to root)")
    args = parser.parse_args()
    manifest = archive_run(args.root, args.output, args.max_file_mib, args.include_checkpoints,
                           include_paths=args.include)
    print(json.dumps({"archive": str(args.output), "included": sum(r["included"] for r in manifest["files"]),
                      "excluded": sum(not r["included"] for r in manifest["files"])}))
