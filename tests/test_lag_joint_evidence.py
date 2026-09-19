"""Same-report paired LAG evidence; synthetic CPU observations only."""
from dataclasses import replace

import numpy as np
import pytest

from grace_gc.audit.prefix_audit import PrefixBundle, independent_report_statistics


def _bundle(pid, report_g, report_rewards, *, t=512, path="path"):
    return PrefixBundle(pid, t, np.array([0, 1, 0, 1, *report_rewards], dtype=float),
                        np.array([0, 2, 0, 2, *report_g], dtype=float)[:, None], path_id=path)


def test_marginal_means_can_pass_without_any_joint_prefix():
    bundles = [_bundle("a", [0, 0, 0, 0], [1, 1, 1, 0]),
               _bundle("b", [0, 2, 0, 2], [0, 1, 0, 1])]
    report = independent_report_statistics(bundles, {})
    assert report["curve"][0]["rho_l"] == pytest.approx(.5)
    assert report["curve"][0]["rho_a"] == pytest.approx(.875)
    joint = report["joint_lag"]
    assert joint["curve"][0]["joint_fraction"] == 0.
    assert joint["curve"][0]["n_paired_valid"] == 2
    assert joint["curve"][0]["n_joint"] == 0
    assert [row["joint"] for row in joint["rows"]] == [False, False]
    assert all(row["report_row_indices"] == [4, 5, 6, 7] for row in joint["rows"])


def test_report_uses_same_finite_pairs_and_keeps_undefined_denominator_visible():
    good = _bundle("good", [0, 0, 0, 0], [0, 1, 0, 1])
    bad = _bundle("zero_g_variance", [1, 1, 1, 1], [0, 1, 0, 1])
    bad.grads[:4] = 1.
    report = independent_report_statistics([good, bad], {})["joint_lag"]
    curve = report["curve"][0]
    assert curve["n_selected"] == 2 and curve["n_paired_valid"] == 1
    assert curve["n_selected_without_pair"] == 1 and curve["joint_fraction"] == 1.
    assert curve["rho_l_paired_mean"] == 0. and curve["rho_a_paired_mean"] == 1.
    invalid = report["rows"][1]
    assert invalid["rho_l"] is None and invalid["joint"] is None
    assert "nonpositive_or_nonfinite_gradient_denominator" in invalid["missing_reasons"]


def test_selection_and_denominators_never_read_reporting_half():
    bundle = _bundle("one", [0, 0, 0, 0], [0, 1, 0, 1])
    first = independent_report_statistics([bundle], {})["joint_lag"]["rows"][0]
    changed = replace(bundle, grads=bundle.grads.copy(), rewards=bundle.rewards.copy())
    changed.grads[4:] *= 100.
    changed.rewards[4:] = 1.
    second = independent_report_statistics([changed], {})["joint_lag"]["rows"][0]
    assert first["selected"] == second["selected"] is True
    assert first["gradient_denominator"] == second["gradient_denominator"]
    assert first["reward_denominator"] == second["reward_denominator"]
    assert first["joint"] is True and second["joint"] is False


def test_unsplit_or_no_selected_pairs_are_null_not_failure_zero():
    bundle = _bundle("short", [0, 0, 0, 0], [0, 1, 0, 1])
    bundle = replace(bundle, grads=bundle.grads[:4], rewards=bundle.rewards[:4])
    out = independent_report_statistics([bundle], {})["joint_lag"]
    assert out["curve"][0]["joint_fraction"] is None
    assert out["curve"][0]["n_selected"] == 0
    assert "independent_report_split_unavailable" in out["rows"][0]["missing_reasons"]
    empty = independent_report_statistics([], {})["joint_lag"]
    assert empty["curve"] == [] and empty["rows"] == []


def test_thresholds_and_context_are_analysis_metadata_only():
    bundle = _bundle("one", [0, 0, 0, 0], [0, 1, 0, 1])
    analysis = {"max_new_tokens": 2048, "joint_lag": {"rho_l_max": .2, "rho_a_min": 1.1},
                "joint_lag_context": {"checkpoint_sha256": "synthetic", "checkpoint_step": 40}}
    report = independent_report_statistics([bundle], analysis)
    assert report["n_selected"] == 1 and report["curve"][0]["rho_a"] == 1.
    joint = report["joint_lag"]
    assert joint["thresholds"] == {"rho_l_max": .2, "rho_a_min": 1.1}
    assert joint["context"]["max_new_tokens"] == 2048
    assert joint["context"]["checkpoint_sha256"] == "synthetic"
    assert joint["curve"][0]["joint_fraction"] == 0.
    assert joint["cluster_unit"] == "problem_id"
