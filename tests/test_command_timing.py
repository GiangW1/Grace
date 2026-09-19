import json
import sys

import pytest

from scripts.measure_command import measure_command, timing_summary


def test_real_command_and_failure_are_recorded(tmp_path):
    log = tmp_path / "timing.jsonl"
    output = tmp_path / "child.txt"
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')"
    assert measure_command([sys.executable, "-c", code, str(output)], log, "train") == 0
    assert measure_command([sys.executable, "-c", "raise SystemExit(7)"], log, "audit") == 7
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert output.read_text() == "ok"
    assert [r["exit_code"] for r in rows] == [0, 7]
    assert all(r["wall_seconds"] > 0 and r["started"] <= r["finished"] for r in rows)
    assert rows[0]["stage"] == "train"


def test_timing_gaps_are_not_claimed_as_compute():
    rows = [{"started_monotonic": 10., "finished_monotonic": 15., "wall_seconds": 5.},
            {"started_monotonic": 20., "finished_monotonic": 27., "wall_seconds": 7.}]
    result = timing_summary(rows)
    assert result["observed_envelope_seconds"] == 17
    assert result["command_wall_seconds"] == 12
    assert result["unattributed_gap_seconds"] == 5
    # Even if used with overlapping commands, don't double-count the envelope.
    rows.append({"started_monotonic": 12., "finished_monotonic": 22., "wall_seconds": 10.})
    assert timing_summary(rows)["unattributed_gap_seconds"] == 0


def test_launch_error_is_recorded_and_reraised(tmp_path):
    log = tmp_path / "timing.jsonl"
    with pytest.raises(FileNotFoundError):
        measure_command([str(tmp_path / "no-such-program")], log, "train")
    row = json.loads(log.read_text())
    assert row["exit_code"] is None and row["error_type"] == "FileNotFoundError"
