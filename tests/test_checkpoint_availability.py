import json
from types import SimpleNamespace

import pytest

from grace_gc.logging_util import forensics
from grace_gc.logging_util.run_dir import RunDirectory


def test_available_cost_is_after_publish_not_checkpoint_embedded_clock(tmp_path, monkeypatch):
    clock = {"now": 10.0}
    controller = SimpleNamespace(snapshot=lambda: {"global_wall_seconds": clock["now"]})
    state = SimpleNamespace(step=3)
    cfg = {"_cost_controller": controller}
    monkeypatch.setattr(forensics, "record_effective_config", lambda *a: {})

    def dump(path, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint")
        clock["now"] += 4
        return {}

    def copy(source, target):
        target.write_bytes(source.read_bytes())
        clock["now"] += 2
        return {}

    monkeypatch.setattr(forensics, "dump_train_state", dump)
    monkeypatch.setattr(forensics, "copy_checkpoint", copy)
    monkeypatch.delenv("GRACE_COMMAND_START_MONOTONIC", raising=False)
    run = RunDirectory(tmp_path)
    forensics._publish_snapshot(run, cfg, state, [], None)
    row = json.loads((tmp_path / "checkpoints.json").read_text())["steps"][0]
    assert state.cost_control["global_wall_seconds"] == 10
    assert row["available_global_wall_seconds"] == 16
    assert row["available_command_wall_seconds"] is None
    assert row["available_at"]
    assert row["sha256"]


def test_external_clock_is_invocation_only_and_resume_stays_unknown(tmp_path, monkeypatch):
    from grace_gc.logging_util.forensics import checkpoint_availability

    monkeypatch.setenv("GRACE_COMMAND_START_MONOTONIC", "100")
    monkeypatch.setattr(forensics.time, "monotonic", lambda: 130.)
    cfg = {"_cost_controller": SimpleNamespace(snapshot=lambda: {"global_wall_seconds": 28.})}
    assert checkpoint_availability(cfg)["available_command_wall_seconds"] == 30
    cfg["resume"] = "older.npz"
    result = checkpoint_availability(cfg)
    assert result["available_command_wall_seconds"] is None
    assert result["available_global_wall_seconds"] == 28


def test_measure_command_exposes_its_start_only_to_child(tmp_path, monkeypatch):
    import sys
    from scripts.measure_command import measure_command

    monkeypatch.delenv("GRACE_COMMAND_START_MONOTONIC", raising=False)
    output = tmp_path / "clock.txt"
    code = "import os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(os.environ['GRACE_COMMAND_START_MONOTONIC'])"
    assert measure_command([sys.executable, "-c", code, str(output)], tmp_path / "clock.jsonl", "train") == 0
    row = json.loads((tmp_path / "clock.jsonl").read_text())
    assert float(output.read_text()) == pytest.approx(row["started_monotonic"])
    import os
    assert "GRACE_COMMAND_START_MONOTONIC" not in os.environ
