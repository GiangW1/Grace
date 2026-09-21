"""Cost scheduling and lossless checkpoint regressions; no GPU timings."""

import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from grace_gc.trainer import checkpoint
from grace_gc.trainer.cost_control import CostControl, observe_batch_cost, restore_cost_control
from grace_gc.trainer.methods import method_spec


def test_fixed_overhead_is_not_inflated_when_main_n_changes():
    controller = CostControl({"cost_control": {"enabled": True, "target_step_seconds": 40., "ema_alpha": 1.}})
    assert controller.observe(16, 40., 1, group=4, main_seconds=32.) == 16
    # Same two seconds/main, same eight seconds overhead: halving N does not
    # turn that fixed overhead into an apparent increase in per-main cost.
    assert controller.observe(8, 24., 2, group=4, main_seconds=16.) == 16
    assert controller.state["ema_seconds_per_start"] == 2.
    assert controller.state["mean_fixed_seconds"] == 8.
    assert controller.state["last_batch_wall_seconds"] == 24.


def test_periodic_persistence_is_amortized_but_still_reduces_available_budget():
    controller = CostControl({"cost_control": {"enabled": True, "target_step_seconds": 40., "ema_alpha": .3}})
    for step in range(1, 5):
        controller.observe(16, 36., step, group=4, main_seconds=32.)
    # A 40-second periodic save affects the fixed average, not the slope.
    assert controller.observe(16, 76., 5, group=4, main_seconds=32.) == 12
    assert controller.state["ema_seconds_per_start"] == 2.
    assert controller.state["fixed_seconds_sum"] == 60.
    assert controller.state["mean_fixed_seconds"] == 12.
    assert controller.state["last_fixed_seconds"] == 44.
    # Actual main work becoming more expensive still lowers NEXT N.
    assert controller.observe(12, 52., 6, group=4, main_seconds=48.) == 8


def test_inferred_target_keeps_total_seconds_and_initial_group():
    controller = CostControl({"cost_control": {"enabled": True}})
    assert controller.observe(16, 48.54320781608112, 1, group=4, main_seconds=27.4987) == 16
    assert controller.state["target_step_seconds"] == 48.54320781608112
    assert controller.state["target_source"] == "first_observed_complete_batch"


def test_phase_feedback_records_all_fixed_costs_and_budget_includes_save():
    class Clock:
        now = 0.
        def elapsed(self):
            return self.now

    clock = Clock()
    controller = CostControl({"cost_control": {"enabled": True}, "run_wall_seconds": 50.}, clock)
    state = SimpleNamespace(spec=method_spec("grace"), n_ref=16, step=1, cost_control={})
    cfg = {"n_prompts": 4}
    last = {"n": 16, "timings": {"prefix": 5., "allocate": 1., "continue": 8., "backward": 2.,
            "audit": 1., "behavior_probe": 3., "prescan": 10., "predictor": 2., "fresh_supervision": 4.,
            "sync": 1., "update": 1.}}
    clock.now = 78.  # includes 40 seconds of persistence, not just phases
    observe_batch_cost(controller, cfg, state, last, 78.)
    assert controller.state["last_main_seconds"] == 20.
    assert controller.state["last_fixed_seconds"] == 58.
    assert controller.snapshot()["global_wall_seconds"] == 78.
    assert controller.snapshot()["run_overshoot_seconds"] == 28.
    assert controller.stop_reason() == "run_wall_seconds"


def test_feedback_resume_restores_decomposition_and_explicit_target(tmp_path):
    cfg = {"cost_control": {"enabled": True, "target_step_seconds": 40., "ema_alpha": .3}}
    live = CostControl(cfg)
    live.observe(16, 36., 1, group=4, main_seconds=32.)
    saved = live.snapshot()
    saved.update(event="batch_complete", step=1, global_wall_seconds=36.)
    run = tmp_path / "run"
    run.mkdir()
    (run / "cost_control.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")
    restored = restore_cost_control({**cfg, "resume": str(run / "checkpoint.npz")},
                                    SimpleNamespace(elapsed=lambda: 0.), {"step": 1})
    assert restored.state["fixed_seconds_sum"] == 4.
    assert restored.observe(8, 20., 2, group=4, main_seconds=16.) == live.observe(8, 20., 2, group=4, main_seconds=16.)
    fixed = restore_cost_control({"resume": str(run / "checkpoint.npz"), "cost_control": {"mode": "fixed"}},
                                 SimpleNamespace(elapsed=lambda: 1.), {"step": 1})
    assert fixed.snapshot()["mode"] == "fixed" and not fixed.enabled
    assert fixed.snapshot()["global_wall_seconds"] == 37.
    override = CostControl({"cost_control": {"enabled": True, "target_step_seconds": 24.}}, restored=saved)
    assert override.observe(8, 20., 2, group=4, main_seconds=16.) == 8


def test_legacy_slope_is_not_reused_as_marginal_slope():
    legacy = {"model": "complete_batch_wall_seconds_per_start_ema", "ema_seconds_per_start": 22.,
              "observations": 40, "next_n": 4, "target_step_seconds": 40.}
    controller = CostControl({"cost_control": {"enabled": True}}, restored=legacy)
    assert controller.observe(4, 12., 41, group=4, main_seconds=8.) == 16
    assert controller.state["ema_seconds_per_start"] == 2.
    assert controller.state["main_observations"] == 1
    assert controller.state["observations"] == 41


def test_zero_main_time_is_recorded_without_inventing_infinite_throughput():
    controller = CostControl({"cost_control": {"enabled": True}})
    assert controller.observe(8, 10., 1, group=4, main_seconds=0.) == 8
    assert controller.state["mean_fixed_seconds"] == 10.


def test_modes_are_explicit_and_legacy_enabled_is_compatible():
    from grace_gc.trainer.cost_control import start_count_mode
    assert start_count_mode({}) == "token"
    assert start_count_mode({"cost_control": {"enabled": True}}) == "wall"
    assert start_count_mode({"cost_control": {"enabled": False}}) == "token"
    for mode in ("fixed", "wall", "token"):
        controller = CostControl({"cost_control": {"mode": mode, "enabled": mode != "wall"}},
                                 restored={"mode": "wall", "enabled": True})
        assert controller.snapshot()["mode"] == mode
        assert controller.enabled == (mode == "wall")
    with pytest.raises(ValueError, match="mode"):
        start_count_mode({"cost_control": {"mode": "unknown"}})


def test_fixed_mode_keeps_run_budget_active():
    controller = CostControl({"cost_control": {"mode": "fixed", "enabled": True}, "run_wall_seconds": 5.},
                             timer=SimpleNamespace(elapsed=lambda: 7.))
    assert controller.observe(8, 7., 1, main_seconds=1.) is None
    assert controller.stop_reason() == "run_wall_seconds"
    assert controller.snapshot()["run_overshoot_seconds"] == 2.


def test_checkpoint_compacts_only_exact_float32_gradients_and_restores_dtype(tmp_path):
    rng = np.random.default_rng(17)
    gradients = [rng.normal(size=4096).astype(np.float32).astype(np.float64),
                 np.array([0., -0.], dtype=np.float64),
                 np.array([1. + 2.**-40], dtype=np.float64),
                 np.array([np.inf, -np.inf, np.nan], dtype=np.float64)]
    payload = {"reservoir": {"items": [{"g": g, "problem_id": str(i)} for i, g in enumerate(gradients)]},
               "basis": np.array([1. + 2.**-40]), "step": 4}
    path = tmp_path / "compact.npz"
    stats = checkpoint.save_checkpoint(path, payload)
    loaded = checkpoint.load_checkpoint(path)
    for original, item in zip(gradients, loaded["reservoir"]["items"]):
        assert item["g"].dtype == original.dtype
        np.testing.assert_array_equal(item["g"].view(np.uint64), original.view(np.uint64))
    # Live payload and full-precision basis are not changed by serialization.
    assert all(item["g"] is original for original, item in zip(gradients, payload["reservoir"]["items"]))
    np.testing.assert_array_equal(loaded["basis"], payload["basis"])
    with np.load(path, allow_pickle=True) as archive:
        stored = archive["payload"].item()["reservoir"]["items"]
        assert [item["g"].dtype for item in stored] == [np.dtype("float32"), np.dtype("float32"),
                                                       np.dtype("float64"), np.dtype("float64")]
    assert stats["reservoir_g_compacted"] == 2
    assert stats["reservoir_g_storage_bytes"] < stats["reservoir_g_original_bytes"]
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["payload.npy"]


def test_old_numpy_compressed_checkpoint_remains_loadable(tmp_path):
    path = tmp_path / "legacy.npz"
    payload = {"reservoir": {"items": [{"g": np.array([1. + 2.**-40])}]}, "step": 17}
    np.savez_compressed(path, payload=np.array(payload, dtype=object))
    loaded = checkpoint.load_checkpoint(path)
    assert loaded["step"] == 17
    np.testing.assert_array_equal(loaded["reservoir"]["items"][0]["g"], payload["reservoir"]["items"][0]["g"])


def test_failed_fast_archive_write_preserves_published_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    checkpoint.save_checkpoint(path, {"step": 1})
    before = path.read_bytes()
    def fail(stream, payload):
        stream.write(b"partial")
        raise OSError("interrupted")
    monkeypatch.setattr(checkpoint, "_write_archive", fail)
    with pytest.raises(OSError, match="interrupted"):
        checkpoint.save_checkpoint(path, {"step": 2})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
