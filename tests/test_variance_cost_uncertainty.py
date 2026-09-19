"""Conditional problem-cluster uncertainty, synthetic CPU data only."""
import numpy as np
import pytest

from grace_gc.audit.variance_cost import variance_times_cost


def test_cluster_interval_matches_complete_problem_resampling():
    from grace_gc.audit.variance_cost import cluster_bootstrap_variance_cost

    g = np.array([[0., 1.], [2., -1.], [1., 2.], [4., 0.], [5., 1.]])
    m = np.full_like(g, .3)
    p, prefix, suffix = np.array([.5, .5, .7, .8, .8]), np.arange(1., 6.), np.arange(2., 7.)
    ids = ["a", "a", "b", "c", "c"]
    report = cluster_bootstrap_variance_cost(g, m, p, prefix, suffix, ids, samples=80, seed=23)
    clusters = [np.flatnonzero(np.array(ids) == pid) for pid in ("a", "b", "c")]
    values = []
    for draw in np.random.default_rng(23).integers(0, 3, size=(80, 3)):
        ix = np.concatenate([clusters[j] for j in draw])
        values.append(variance_times_cost(g[ix], m[ix], p[ix], prefix[ix], suffix[ix])["ratio"])
    assert report["point_ratio"] == pytest.approx(variance_times_cost(g, m, p, prefix, suffix)["ratio"])
    valid = [value for value in values if value is not None]
    np.testing.assert_allclose(report["ratio_ci95"], np.quantile(valid, [.025, .975]), rtol=1e-12)
    assert report["cluster_unit"] == "problem_id" and report["n_clusters"] == 3
    assert [row["n_rows"] for row in report["cluster_sufficient_statistics"]] == [2, 1, 2]
    assert report["valid_replicates"] == len(valid) and report["undefined_replicates"] == 80 - len(valid)


def test_one_problem_keeps_point_without_fake_independence_interval():
    from grace_gc.audit.variance_cost import cluster_bootstrap_variance_cost

    g = np.arange(8.)[:, None]
    report = cluster_bootstrap_variance_cost(g, np.zeros_like(g), np.ones(8), 1., np.ones(8), ["a"] * 8)
    assert report["point_ratio"] == 1. and report["ratio_ci95"] is None
    assert report["unavailable_reason"] == "fewer_than_two_problem_clusters"
    assert report["n_rows"] == 8  # Eight suffixes do not become eight independent problems.


def test_undefined_bootstrap_denominators_remain_counted():
    from grace_gc.audit.variance_cost import cluster_bootstrap_variance_cost

    g = np.array([[1.], [1.], [2.], [2.]])
    report = cluster_bootstrap_variance_cost(g, np.zeros_like(g), np.ones(4), 1., np.ones(4),
                                              ["a", "a", "b", "b"], samples=100, seed=7)
    assert report["undefined_replicates"] > 0
    assert report["valid_replicates"] + report["undefined_replicates"] == 100
    np.testing.assert_allclose(report["ratio_ci95"], [1., 1.])
    assert "conditional on a defined ratio" in report["scope"]


def test_prefix_audit_adds_uncertainty_without_changing_point_target():
    from grace_gc.audit.prefix_audit import PrefixBundle, _restricted_variance_cost

    bundles = [PrefixBundle(str(i), 512, np.tile([0., 1.], 4),
                             (np.arange(8.) + i)[:, None], path_id=f"{i}:0",
                             suffix_cost=np.full(8, 12.), m_pred=np.zeros(1), r_hat=1., c_hat=12.)
               for i in range(3)]
    analysis = {"method": "grace", "variance_cost_bootstrap": {"samples": 50, "seed": 9}}
    result = _restricted_variance_cost(bundles, np.array([0]), np.array([1, 2]), analysis)
    assert result["uncertainty"]["point_ratio"] == pytest.approx(result["actual"]["ratio"])
    assert result["uncertainty"]["n_clusters"] == 2
    assert result["uncertainty"]["samples_requested"] == 50
