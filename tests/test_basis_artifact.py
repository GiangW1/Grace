import shutil
import tarfile

import numpy as np
import pytest

from grace_gc.trainer import checkpoint
from grace_gc.versions import sha256_array
from scripts.archive_run import archive_run


def _payload(step=1, *, fixed=True):
    return {"step": step, "basis": {
        "u": np.arange(1024, dtype=np.float64).reshape(256, 4),
        "basis_id": 3, "predictor_synced_basis_id": 3, "fixed": fixed,
    }}


def _stored(path):
    with np.load(path, allow_pickle=True) as archive:
        return archive["payload"].item()


def test_fixed_basis_is_external_reused_and_restored_without_mutating_payload(tmp_path, monkeypatch):
    payload = _payload()
    path = tmp_path / "step_1.npz"
    stats = checkpoint.save_checkpoint(path, payload)
    name = f"basis-{sha256_array(payload['basis']['u'])}.npy"
    artifact = tmp_path / name
    assert stats["basis_artifact"] == name
    assert artifact.is_file()
    raw = _stored(path)["basis"]
    assert "u" not in raw
    assert raw["u_artifact"] == {
        "path": name, "sha256": sha256_array(payload["basis"]["u"]),
        "dtype": "<f8", "shape": [256, 4],
    }
    assert "u_artifact" not in payload["basis"]
    np.testing.assert_array_equal(np.load(artifact, allow_pickle=False), payload["basis"]["u"])
    original_stat = artifact.stat()

    def unexpected_save(*args, **kwargs):
        raise AssertionError("fixed U must not be serialized again")

    monkeypatch.setattr(checkpoint.np, "save", unexpected_save)
    later = tmp_path / "step_2.npz"
    assert checkpoint.save_checkpoint(later, payload)["basis_artifact"] == name
    assert artifact.stat().st_mtime_ns == original_stat.st_mtime_ns
    loaded = checkpoint.load_checkpoint(later)
    np.testing.assert_array_equal(loaded["basis"]["u"], payload["basis"]["u"])
    assert loaded["basis"]["basis_id"] == 3
    assert loaded["basis"]["predictor_synced_basis_id"] == 3


@pytest.mark.parametrize("fixed", [False, True])
def test_legacy_embedded_basis_loads_without_sidecar(tmp_path, fixed):
    payload = _payload(fixed=fixed)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, payload=np.array(payload, dtype=object))
    np.testing.assert_array_equal(checkpoint.load_checkpoint(path)["basis"]["u"], payload["basis"]["u"])


def test_nonfixed_basis_remains_embedded(tmp_path):
    path = tmp_path / "legacy.npz"
    stats = checkpoint.save_checkpoint(path, _payload(fixed=False))
    assert "u" in _stored(path)["basis"]
    assert stats.get("basis_artifact") is None
    assert not list(tmp_path.glob("basis-*.npy"))


def test_sidecar_write_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    checkpoint.save_checkpoint(path, {"step": 1})
    original = path.read_bytes()

    def interrupted(stream, *args, **kwargs):
        stream.write(b"partial basis")
        raise OSError("basis write interrupted")

    monkeypatch.setattr(checkpoint.np, "save", interrupted)
    with pytest.raises(OSError, match="basis write interrupted"):
        checkpoint.save_checkpoint(path, _payload(step=2))
    assert path.read_bytes() == original
    assert checkpoint.load_checkpoint(path)["step"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("basis-*.npy"))


def test_npz_failure_keeps_previous_checkpoint_and_its_basis(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    original_payload = _payload()
    checkpoint.save_checkpoint(path, original_payload)
    original = path.read_bytes()
    newer = _payload(step=2)
    newer["basis"]["u"] = newer["basis"]["u"] + 1

    def interrupted(stream, payload):
        stream.write(b"partial checkpoint")
        raise OSError("checkpoint write interrupted")

    monkeypatch.setattr(checkpoint, "_write_archive", interrupted)
    with pytest.raises(OSError, match="checkpoint write interrupted"):
        checkpoint.save_checkpoint(path, newer)
    assert path.read_bytes() == original
    np.testing.assert_array_equal(checkpoint.load_checkpoint(path)["basis"]["u"], original_payload["basis"]["u"])
    assert not list(tmp_path.glob("*.tmp"))


def test_sidecar_work_is_inside_serialization_and_copy_timers(tmp_path, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(checkpoint.time, "perf_counter", lambda: elapsed[0])
    monkeypatch.setattr(checkpoint.time, "process_time", lambda: elapsed[0])
    original_save = checkpoint.np.save

    def metered_save(*args, **kwargs):
        original_save(*args, **kwargs)
        elapsed[0] += 3.0

    monkeypatch.setattr(checkpoint.np, "save", metered_save)
    source = tmp_path / "steps" / "step.npz"
    stats = checkpoint.save_checkpoint(source, _payload())
    assert stats["serialize_write_wall_seconds"] == 3.0
    assert stats["serialize_write_cpu_seconds"] == 3.0
    original_copy = checkpoint.shutil.copyfileobj

    def metered_copy(*args, **kwargs):
        original_copy(*args, **kwargs)
        elapsed[0] += 2.0

    monkeypatch.setattr(checkpoint.shutil, "copyfileobj", metered_copy)
    copied = checkpoint.copy_checkpoint(source, tmp_path / "latest.npz", basis_artifact=stats["basis_artifact"])
    assert copied["latest_copy_wall_seconds"] == 4.0
    assert copied["latest_copy_cpu_seconds"] == 4.0


@pytest.mark.parametrize("damage", ["missing", "bytes", "values", "shape", "dtype"])
def test_missing_or_corrupt_basis_fails_clearly(tmp_path, damage):
    path = tmp_path / "checkpoint.npz"
    payload = _payload()
    stats = checkpoint.save_checkpoint(path, payload)
    artifact = tmp_path / stats["basis_artifact"]
    if damage == "missing":
        artifact.unlink()
        error = FileNotFoundError
    elif damage == "bytes":
        artifact.write_bytes(b"broken npy")
        error = ValueError
    else:
        u = payload["basis"]["u"]
        damaged = {"values": u + 1, "shape": u.reshape(128, 8), "dtype": u.view(np.int64)}[damage]
        np.save(artifact, damaged, allow_pickle=False)
        error = ValueError
    with pytest.raises(error, match="basis artifact"):
        checkpoint.load_checkpoint(path)


def test_saving_does_not_overwrite_existing_wrong_basis(tmp_path):
    path = tmp_path / "checkpoint.npz"
    checkpoint.save_checkpoint(path, {"step": 1})
    before = path.read_bytes()
    payload = _payload(step=2)
    artifact = tmp_path / f"basis-{sha256_array(payload['basis']['u'])}.npy"
    artifact.write_bytes(b"conflicting artifact")
    with pytest.raises(ValueError, match="basis artifact"):
        checkpoint.save_checkpoint(path, payload)
    assert artifact.read_bytes() == b"conflicting artifact"
    assert path.read_bytes() == before


def test_copy_to_latest_in_another_directory_and_move_whole_run(tmp_path):
    root = tmp_path / "run"
    source = root / "checkpoints" / "step_1.npz"
    payload = _payload()
    stats = checkpoint.save_checkpoint(source, payload)
    target = root / "checkpoint.npz"
    checkpoint.copy_checkpoint(source, target, basis_artifact=stats["basis_artifact"])
    assert target.read_bytes() == source.read_bytes()
    assert (target.parent / stats["basis_artifact"]).read_bytes() == (source.parent / stats["basis_artifact"]).read_bytes()
    moved = tmp_path / "moved"
    shutil.move(str(root), str(moved))
    for relative in ["checkpoint.npz", "checkpoints/step_1.npz"]:
        np.testing.assert_array_equal(checkpoint.load_checkpoint(moved / relative)["basis"]["u"], payload["basis"]["u"])


def test_copy_with_no_sidecar_does_not_unpack_legacy_checkpoint(tmp_path, monkeypatch):
    source, target = tmp_path / "step.npz", tmp_path / "latest.npz"
    checkpoint.save_checkpoint(source, {"step": 1})

    def unexpected_load(*args, **kwargs):
        raise AssertionError("copy must reuse the compressed checkpoint bytes")

    monkeypatch.setattr(checkpoint.np, "load", unexpected_load)
    checkpoint.copy_checkpoint(source, target)
    assert target.read_bytes() == source.read_bytes()


def test_copy_in_same_directory_reuses_basis(tmp_path, monkeypatch):
    source, target = tmp_path / "step.npz", tmp_path / "latest.npz"
    stats = checkpoint.save_checkpoint(source, _payload())
    artifact = tmp_path / stats["basis_artifact"]
    before = artifact.stat().st_mtime_ns
    checkpoint.copy_checkpoint(source, target, basis_artifact=stats["basis_artifact"])
    assert artifact.stat().st_mtime_ns == before
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("damage", ["missing_source", "wrong_target", "copy_failure"])
def test_sidecar_copy_failure_keeps_previous_latest(tmp_path, monkeypatch, damage):
    source = tmp_path / "steps" / "step.npz"
    target = tmp_path / "latest" / "checkpoint.npz"
    stats = checkpoint.save_checkpoint(source, _payload())
    checkpoint.save_checkpoint(target, {"step": 0})
    before = target.read_bytes()
    source_basis, target_basis = source.parent / stats["basis_artifact"], target.parent / stats["basis_artifact"]
    if damage == "missing_source":
        source_basis.unlink()
        error = FileNotFoundError
    elif damage == "wrong_target":
        target_basis.write_bytes(b"another artifact")
        error = ValueError
    else:
        def interrupted(original, stream, **kwargs):
            stream.write(b"partial basis")
            raise OSError("basis copy interrupted")
        monkeypatch.setattr(checkpoint.shutil, "copyfileobj", interrupted)
        error = OSError
    with pytest.raises(error, match="basis"):
        checkpoint.copy_checkpoint(source, target, basis_artifact=stats["basis_artifact"])
    assert target.read_bytes() == before
    assert not list(target.parent.glob("*.tmp"))
    if damage == "wrong_target":
        assert target_basis.read_bytes() == b"another artifact"
    else:
        assert not target_basis.exists()


def test_checkpoint_archive_retains_basis_dependencies_above_size_limit(tmp_path):
    root = tmp_path / "run"
    source = root / "train" / "checkpoints" / "step_1.npz"
    stats = checkpoint.save_checkpoint(source, _payload())
    target = root / "train" / "checkpoint.npz"
    checkpoint.copy_checkpoint(source, target, basis_artifact=stats["basis_artifact"])
    output = tmp_path / "run.tar.gz"
    default = archive_run(root, output, max_file_mib=0)
    assert all(not row["included"] for row in default["files"])
    included = archive_run(root, output, max_file_mib=0, include_checkpoints=True)
    assert all(row["included"] for row in included["files"])
    with tarfile.open(output) as archive:
        for folder in ["train", "train/checkpoints"]:
            assert f"run/{folder}/{stats['basis_artifact']}" in archive.getnames()


@pytest.mark.parametrize("requested", ["train/checkpoints/step_1.npz", "train/checkpoints"])
def test_explicit_checkpoint_archive_keeps_same_directory_basis_without_unpickling(tmp_path, monkeypatch, requested):
    root = tmp_path / "run"
    source = root / "train" / "checkpoints" / "step_1.npz"
    payload = _payload()
    stats = checkpoint.save_checkpoint(source, payload)
    # Exact references live inside pickle, so include all basis files beside an
    # explicitly selected NPZ, including an older basis with no selected NPZ.
    older_u = payload["basis"]["u"] + 1
    older_name = f"basis-{sha256_array(older_u)}.npy"
    np.save(source.parent / older_name, older_u, allow_pickle=False)
    checkpoint.save_checkpoint(root / "unrelated" / "checkpoint.npz", payload)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("archiving must not unpickle checkpoints")

    monkeypatch.setattr(checkpoint.np, "load", unexpected_load)
    output = tmp_path / "selected.tar.gz"
    manifest = archive_run(root, output, max_file_mib=0, include_paths=[requested])
    included = {row["path"] for row in manifest["files"] if row["included"]}
    expected = {f"train/checkpoints/{name}" for name in ["step_1.npz", stats["basis_artifact"], older_name]}
    assert included == expected
    with tarfile.open(output) as archive:
        assert all(f"run/{relative}" in archive.getnames() for relative in expected)


def test_default_small_archive_records_excluded_basis(tmp_path):
    root = tmp_path / "run"
    path = root / "checkpoint.npz"
    stats = checkpoint.save_checkpoint(path, _payload())
    manifest = archive_run(root, tmp_path / "small.tar.gz", max_file_mib=.002)
    rows = {row["path"]: row for row in manifest["files"]}
    assert rows["checkpoint.npz"]["included"]
    assert not rows[stats["basis_artifact"]]["included"]
    assert rows[stats["basis_artifact"]]["reason"] == "size_limit"
    assert rows[stats["basis_artifact"]]["sha256"]
