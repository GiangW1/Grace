"""Archive a run with hashes for both included and excluded files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def archive_run(root: Path, output: Path, max_file_mib=20, include_checkpoints=False):
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = {"run_root": str(root), "files": [],
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
            include = size <= max_file_mib * 1024 * 1024 or (include_checkpoints and path.suffix == ".npz")
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
                                      "source_changed_during_archive": (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns),
                                      "hash_scope": "archive_bytes" if include else "source_read"})
        data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        entry = tarfile.TarInfo(f"{root.name}/archive_manifest.json")
        entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-file-mib", type=float, default=20)
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    manifest = archive_run(args.root, args.output, args.max_file_mib, args.include_checkpoints)
    print(json.dumps({"archive": str(args.output), "included": sum(r["included"] for r in manifest["files"]),
                      "excluded": sum(not r["included"] for r in manifest["files"])}))
