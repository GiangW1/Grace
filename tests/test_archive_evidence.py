import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts.archive_run import archive_run


def _write(root, relative, data=b"evidence" * 100):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("relative", [
    "seed-17/grace/audit/audit_bundles.jsonl",
    "seed-17/grace/audit-training/audit_raw_problems.jsonl",
    "seed-17/grace/batch-audit/batch_audit_samples.jsonl",
    "seed-17/grace/batch-audit/batch_audit_replicates.jsonl",
    "seed-17/grace/batch-audit/batch_audit_means.npz",
    "seed-17/grace/eval-40/eval_per_problem.jsonl",
    "seed-17/grace/eval-0/eval_requests.jsonl",
    "seed-17/grace/train/trajectories.jsonl",
    "seed-17/grace/train/logprob_probe.jsonl",
    "seed-17/grace/train/fresh_supervision.jsonl",
    "seed-17/grace/train/compute_ledger.jsonl",
])
def test_raw_scientific_evidence_survives_size_limit(tmp_path, relative):
    root = tmp_path / "run"
    source = _write(root, relative)
    original = source.read_bytes()
    output = tmp_path / "run.tar.gz"
    manifest = archive_run(root, output, max_file_mib=.0001)
    row, = manifest["files"]
    assert row["included"]
    assert row["inclusion_reason"] == "scientific_evidence"
    assert row["sha256"] == hashlib.sha256(original).hexdigest()
    with tarfile.open(output) as archive:
        assert archive.extractfile(f"run/{relative}").read() == original
    assert source.read_bytes() == original


def test_checkpoint_opt_in_includes_adapters_but_not_base_model(tmp_path):
    root = tmp_path / "run"
    checkpoints = ["train/checkpoint.npz", "train/checkpoints/step_0.npz"]
    adapters = ["adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt",
                "pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"]
    files = checkpoints + [f"eval-0/eval_lora/{name}" for name in adapters]
    files += ["base/model.safetensors", "base/pytorch_model.bin", "large-cache.bin"]
    for relative in files:
        _write(root, relative)
    output = tmp_path / "run.tar.gz"
    default = archive_run(root, output, max_file_mib=.0001)
    assert all(not row["included"] for row in default["files"])
    opted_in = archive_run(root, output, max_file_mib=.0001, include_checkpoints=True)
    included = {row["path"] for row in opted_in["files"] if row["included"]}
    assert included == set(files[:-3])
    excluded = [row for row in opted_in["files"] if not row["included"]]
    assert all(row["reason"] == "size_limit" and row["sha256"] for row in excluded)


def test_explicit_paths_select_only_requested_file_or_directory(tmp_path):
    root = tmp_path / "run"
    chosen = ["train/checkpoints/step_0.npz", "train/lora/step-40/adapter_model.safetensors",
              "train/lora/step-40/adapter_config.json", "base/model.safetensors"]
    omitted = ["train/checkpoints/step_20.npz", "train/lora/step-400/adapter_model.safetensors"]
    for relative in chosen + omitted:
        _write(root, relative)
    output = root / "run.tar.gz"
    manifest = archive_run(root, output, max_file_mib=.0001,
                           include_paths=[chosen[0], "train/lora/step-40", "base"])
    assert {row["path"] for row in manifest["files"] if row["included"]} == set(chosen)
    assert all(row["inclusion_reason"] == "explicit_path" for row in manifest["files"] if row["included"])
    with tarfile.open(output) as archive:
        saved = json.load(archive.extractfile("run/archive_manifest.json"))
        assert saved == manifest
        assert "run/run.tar.gz" not in archive.getnames()
    assert manifest["selection_policy"]["include_paths"] == [chosen[0], "train/lora/step-40", "base"]


@pytest.mark.parametrize("requested, error", [("missing.npz", FileNotFoundError), ("../outside.npz", ValueError)])
def test_explicit_missing_or_outside_path_is_not_silently_ignored(tmp_path, requested, error):
    root = tmp_path / "run"
    root.mkdir()
    _write(tmp_path, "outside.npz")
    with pytest.raises(error):
        archive_run(root, tmp_path / "run.tar.gz", include_paths=[requested])


def test_archive_hashes_stream_both_included_and_excluded_bytes(tmp_path, monkeypatch):
    root = tmp_path / "run"
    data = b"x" * (2 * 1024 * 1024 + 3)
    sources = [_write(root, "train/trajectories.jsonl", data), _write(root, "cache.bin", data)]
    original_open = Path.open
    reads = {path: [] for path in sources}

    class BoundedReader:
        def __init__(self, path, stream):
            self.path, self.stream = path, stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, amount=-1):
            assert 0 < amount <= 1024 * 1024
            reads[self.path].append(amount)
            return self.stream.read(amount)

    def tracked_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return BoundedReader(path, stream) if path in reads and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", tracked_open)
    manifest = archive_run(root, tmp_path / "run.tar.gz", max_file_mib=.0001)
    assert all(row["sha256"] == hashlib.sha256(data).hexdigest() for row in manifest["files"])
    assert all(len(amounts) > 1 for amounts in reads.values())
    assert {row["included"] for row in manifest["files"]} == {True, False}


def test_archive_cli_accepts_repeated_relative_includes(tmp_path):
    root = tmp_path / "run"
    files = ["train/checkpoints/step_0.npz", "train/checkpoint.npz", "base/model.safetensors"]
    for relative in files:
        _write(root, relative)
    output = tmp_path / "run.tar.gz"
    script = Path(__file__).resolve().parents[1] / "scripts/archive_run.py"
    result = subprocess.run([sys.executable, str(script), str(root), str(output),
                             "--max-file-mib", ".0001", "--include", files[0],
                             "--include", files[1]], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["included"] == 2
    with tarfile.open(output) as archive:
        assert f"run/{files[0]}" in archive.getnames()
        assert f"run/{files[1]}" in archive.getnames()
        assert f"run/{files[2]}" not in archive.getnames()
