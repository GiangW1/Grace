"""Independent/fixed baseline controls; synthetic CPU results only."""

import numpy as np
import pytest

from grace_gc.trainer.baseline import HistoricalBaseline


def test_fixed_baseline_ignores_history_and_all_update_paths():
    from grace_gc.trainer.baseline import baseline_from_config

    baseline = baseline_from_config({"mode": "fixed", "fixed_value": .4})
    baseline.values["a"] = 3.
    baseline.update("a", 1.)
    baseline.update_mean("a", [0., 0.])
    baseline.update_ht_mean("a", [1., None], [.2, .8])
    baseline.initialize_from_prescan("a", [1., 1.])
    assert baseline.get("a") == .4 and baseline.get("new", default=.9) == .4
    assert baseline.values == {"a": 3.}
    assert not baseline.uses_prescan


def test_default_prescan_remains_unsmoothed_and_does_not_overwrite_history():
    from grace_gc.trainer.baseline import baseline_from_config

    baseline = baseline_from_config({})
    assert baseline.uses_prescan
    assert baseline.initialize_from_prescan("a", [0., 1., 0., 0.]) == .25
    assert baseline.initialize_from_prescan("a", [1., 1., 1.]) == .25
    assert baseline.initialize_from_prescan("empty", []) == .5
    assert "empty" not in baseline.values


def test_smoothed_prescan_uses_explicit_prior_only_for_independent_samples():
    from grace_gc.trainer.baseline import baseline_from_config

    baseline = baseline_from_config({"ema_alpha": 1., "prescan_prior_strength": 1., "prescan_prior_mean": .5})
    assert baseline.initialize_from_prescan("fail", [0.] * 4) == pytest.approx(.1)
    assert baseline.initialize_from_prescan("pass", [1.] * 4) == pytest.approx(.9)
    baseline.update_ht_mean("fail", [1., None], [.2, .8])
    assert baseline.get("fail") == 2.5  # No smoothing/clipping of the HT history.


def test_fixed_raw_pg_preserves_reward_minus_b_and_stopper_null():
    from grace_gc.trainer.advantages import history_advantages
    from grace_gc.trainer.baseline import baseline_from_config

    baseline = baseline_from_config({"mode": "fixed", "fixed_value": .5})
    rewards = [1., 0., None]
    advantages = history_advantages(rewards, ["a"] * 3, baseline)
    np.testing.assert_allclose(advantages, [.5, -.5, 0.])
    assert rewards[2] is None


def test_prescan_estimate_is_read_only_and_does_not_count_missing_as_failure():
    baseline = HistoricalBaseline(prescan_prior_strength=1., prescan_prior_mean=.5)
    baseline.values["old"] = 1.8
    assert baseline.prescan_estimate([1., None]) == .75
    assert baseline.prescan_estimate([]) == .5
    assert baseline.values == {"old": 1.8}


@pytest.mark.parametrize("mode", ["ema", "fixed"])
def test_baseline_control_checkpoint_roundtrip(mode):
    from grace_gc.trainer.baseline import baseline_from_config

    original = baseline_from_config({"mode": mode, "ema_alpha": .3, "fixed_value": .2,
                                     "prescan_prior_strength": 2., "prescan_prior_mean": .4})
    original.values = {"a": 1.7}
    restored = HistoricalBaseline()
    restored.load_state_dict(original.state_dict())
    assert restored.configuration() == original.configuration()
    assert restored.state_dict() == original.state_dict()
    assert restored.get("a") == original.get("a")


def test_old_checkpoint_restores_legacy_ema_even_into_fixed_instance():
    from grace_gc.trainer.baseline import baseline_from_config

    restored = baseline_from_config({"mode": "fixed", "fixed_value": .2})
    restored.load_state_dict({"alpha": .7, "values": {"a": 1.8}})
    assert restored.mode == "ema" and restored.uses_prescan
    assert restored.get("a") == 1.8
    assert restored.configuration()["prescan_prior_strength"] == 0.
    restored.update_ht_mean("a", [1., None], [.5, .5])
    assert restored.get("a") == pytest.approx(.7 * 1. + .3 * 1.8)


@pytest.mark.parametrize("bad", [{"mode": "adaptive"}, {"fixed_value": float("nan")},
                                  {"prescan_prior_strength": -1.}, {"prescan_prior_mean": 2.}])
def test_baseline_control_rejects_undefined_inputs(bad):
    from grace_gc.trainer.baseline import baseline_from_config

    with pytest.raises(ValueError):
        baseline_from_config(bad)
