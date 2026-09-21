"""Controlled clocks for complete eval/audit envelopes; no GPU execution."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.data.math_data import MathRecord
from grace_gc.logging_util.run_dir import RunDirectory


@pytest.mark.parametrize("stage", ["eval", "audit", "audit_empty", "batch_audit"])
@pytest.mark.parametrize("failure", [None, "checkpoint", "environment", "generation", "summary"])
def test_stage_envelope_includes_setup_and_result_or_failure_persistence(
    tmp_path, monkeypatch, stage, failure
):
    from grace_gc.audit import batch_audit, run
    from grace_gc.evaluation import generate
    from grace_gc.trainer import checkpoint
    from grace_gc import versions

    module = generate if stage == "eval" else batch_audit if stage == "batch_audit" else run
    kind = "audit" if stage == "audit_empty" else stage
    clock = SimpleNamespace(wall=0., cpu=0., ledger_boundary=None, failed=False)

    def spend(seconds, phase=None):
        clock.wall += seconds
        clock.cpu += seconds / 4
        if failure == phase and phase is not None and not clock.failed:
            clock.failed = True
            raise RuntimeError("injected " + phase)

    class ControlledTimer:
        def __init__(self):
            self.wall, self.cpu = clock.wall, clock.cpu

        def elapsed(self):
            return clock.wall - self.wall

        def cpu_elapsed(self):
            return clock.cpu - self.cpu

    resolve = module.resolve_run_dir

    def resolve_run_dir(path):
        spend(3)
        return resolve(path)

    def load_checkpoint(path):
        spend(5, "checkpoint")
        return {"spec": "full_pg", "step": 1}

    def collect_environment(cfg):
        spend(7, "environment")
        return {}

    def generation(*args, **kwargs):
        spend(11, "generation")
        if stage == "batch_audit":
            return {"fixed_n": 1}
        return [] if stage in {"eval", "audit_empty"} else [SimpleNamespace(t=1)]

    write_json = RunDirectory.write_json
    append_jsonl = RunDirectory.append_jsonl

    def measured_write(self, name, payload):
        if name == "compute_ledger.json":
            clock.ledger_boundary = (clock.wall, clock.cpu)
            # The ledger cannot include its own final publication without
            # recursively rewriting it. This explicitly excluded tail is paid.
            spend(19)
        elif name in {kind + "_summary.json", "summary.json"}:
            spend(13, "summary" if name != "summary.json" and payload.get("finished") else None)
        return write_json(self, name, payload)

    def measured_append(self, name, payload):
        if name == "compute_ledger.jsonl":
            spend(23)
        return append_jsonl(self, name, payload)

    monkeypatch.setattr(module, "Timer", ControlledTimer)
    monkeypatch.setattr(module, "resolve_run_dir", resolve_run_dir)
    monkeypatch.setattr(module, "collect_environment", collect_environment)
    monkeypatch.setattr(checkpoint, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(versions, "sha256_file", lambda path: "checkpoint-hash")
    monkeypatch.setattr(run, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(batch_audit, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(RunDirectory, "write_json", measured_write)
    monkeypatch.setattr(RunDirectory, "append_jsonl", measured_append)
    monkeypatch.setattr(generate, "_generate_eval_items", generation)
    monkeypatch.setattr(run, "generate_bundles_gpu", generation)
    monkeypatch.setattr(run, "bundle_to_dict", lambda bundle: {"t": bundle.t})
    monkeypatch.setattr(run, "_audit_u", lambda *args: np.zeros((1, 1)))
    monkeypatch.setattr(run, "audit_bundles", lambda *args: {"n_bundles": 1})
    monkeypatch.setattr(batch_audit, "generate_bundles_gpu",
        lambda *args, cache, **kwargs: cache.update(engines=None, layout=None, encode_fn=None))
    monkeypatch.setattr(batch_audit, "_frozen_predictor", lambda *args: (None, None))
    monkeypatch.setattr(batch_audit, "audit_fixed_batches", generation)
    monkeypatch.setattr(batch_audit, "sha256_file", lambda *args: "checkpoint-hash")

    cfg = {"backend": "gpu_verl", "checkpoint": "unused.npz", "seed": 17,
           "hardware": {"n_gpu": 1, "name": "A100-80GB"},
           "decision_tokens": 1, "max_new_tokens": 2,
           "audit": {"max_new_tokens": 2}, "batch_audit": {"n_prompts": 1}}
    target = tmp_path / stage
    entry = getattr(module, "run_" + kind)
    if failure:
        with pytest.raises(RuntimeError, match="injected " + failure):
            entry([MathRecord("p", "q", "1")], cfg, target)
        assert json.loads((target / "summary.json").read_text())["run_status"] == "failed"
    else:
        entry([MathRecord("p", "q", "1")], cfg, target)

    ledger = json.loads((target / "compute_ledger.json").read_text())
    assert len(ledger["rows"]) == 1  # A failed summary must not add a completed envelope first.
    row = ledger["rows"][0]
    assert row["status"] == ("failed" if failure else "completed")
    assert (row["wall_seconds"], row["cpu_seconds"]) == clock.ledger_boundary
    assert row["gpu_reserved_seconds"] == row["wall_seconds"]
    assert ledger["hardware"] == row["hardware"] == "A100-80GB"
    assert "ledger publication" in row["timing_scope"]
    assert clock.wall == row["wall_seconds"] + 19 + 23
    persisted = [json.loads(line) for line in (target / "compute_ledger.jsonl").read_text().splitlines()]
    assert persisted == [row]
