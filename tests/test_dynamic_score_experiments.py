from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from grace_gc.audit.benefit_replay import replay_means
from grace_gc.audit.dynamic_score import (feature_matrix, fit_prefix_coefficients,
                                           load_dynamic_replay, residual_metrics)
from grace_gc.versions import sha256_array, sha256_file
from scripts.expected_gain_dynamic_oracle import main as oracle_main
from scripts.expected_gain_dynamic_predictor import main as predictor_main
from scripts.replay_expected_gain import _apply_problem_baselines, _load_problem_baselines


def _write_replay(root: Path):
    replay = root / "replay"
    replay.mkdir()
    rng = np.random.default_rng(4)
    n, dimension = 6, 3
    basis = np.zeros((n, dimension, 1), dtype=np.float64)
    basis[:, :, 0] = np.asarray([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
                                 [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
    coefficients = np.arange(1., n + 1.)[:, None]
    target_a = basis[:, :, 0] * coefficients
    target_b = target_a + rng.normal(scale=.01, size=target_a.shape)
    norm_b = np.sum(target_b * target_b, axis=1) + .1
    np.save(replay / "half_mean_grads_a.npy", target_a)
    np.save(replay / "half_mean_grads_b.npy", target_b)
    np.save(replay / "half_mean_norm_sq_b.npy", norm_b)
    score = root / "score_gradients.npy"
    np.save(score, basis)
    with (replay / "prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(n):
            handle.write(json.dumps({
                "problem_id": f"p{index}", "features": [float(index), 1.],
                "prompt_features": [float(index), 1.], "prefix_token_ids": [1, index + 2],
                "mean_cost": 1., "mean_reward": .5, "baseline": .5,
                "half_mean_reward_a": [0., .5, 1.][index % 3],
                "half_mean_reward_b": .5,
                "half_mean_advantage": [[-.5, -.5], [0., 0.], [.5, .5]][index % 3],
            }) + "\n")
    split = {"train": ["p0", "p1", "p2"], "validation": ["p3"],
             "diagnostic": ["p4", "p5"]}
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return replay, score, split_path


def test_replay_can_store_independent_half_means():
    rows = [{"problem_id": "p", "path_id": "p:0", "t": 1,
             "continuation_records": [
                 {"token_ids": [1, 2, i + 3], "prompt_len": 1,
                  "reward": 1., "baseline": 0., "generated_suffix_tokens": 1}
                 for i in range(4)]}]
    with tempfile.TemporaryDirectory() as tmp:
        replay_means(rows, lambda tokens, *_: np.asarray([float(tokens[-1]), 0.]),
                     2, tmp, max_continuations=4, seed=7, store_half_means=True,
                     metric_weights=np.asarray([2., .5]), metric_name="test_metric")
        assert np.load(Path(tmp) / "half_mean_grads_a.npy").shape == (1, 2)
        assert np.load(Path(tmp) / "half_metric_norm_sq_b.npy").shape == (1,)
        metadata = json.loads((Path(tmp) / "prefixes.jsonl").read_text().splitlines()[0])
        assert metadata["half_counts"] == [2, 2]
        assert len(metadata["half_mean_norm_sq"]) == 2
        assert len(metadata["half_mean_advantage"]) == 2


def test_dynamic_oracle_and_predictor_use_a_b_split_without_torch():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        replay, score, split = _write_replay(root)
        oracle_main(["--replay-dir", str(replay), "--score-gradients", str(score),
                     "--split-manifest", str(split), "--run-dir", str(root / "oracle")])
        oracle = json.loads((root / "oracle" / "dynamic_oracle_summary.json").read_text())
        assert oracle["stage"] == "A"
        assert oracle["roles"]["diagnostic"]["oracle"]["residual_ratio_to_zero"] < 0.1
        assert oracle["q_stratification"]["available"] is True
        assert "reward_only" in oracle["roles"]["diagnostic"]["mechanism"]
        predictor_main(["--replay-dir", str(replay), "--score-gradients", str(score),
                        "--split-manifest", str(split), "--run-dir", str(root / "predictor"),
                        "--features", "legacy", "--models", "zero,ridge"])
        report = json.loads((root / "predictor" / "dynamic_predictor_summary.json").read_text())
        ridge = next(row for row in report["models"] if row["model"] == "ridge_l2_1")
        zero = next(row for row in report["models"] if row["model"] == "zero")
        assert ridge["validation"]["residual_mean"] < zero["validation"]["residual_mean"]
        assert (root / "predictor" / "predicted_coefficients_ridge_l2_1.npy").is_file()


def test_cheap_features_are_prefix_visible_and_metric_residual_is_weighted():
    rows = [{"prefix_token_ids": [1, 2], "baseline": .2,
             "mean_cost": 10., "mean_reward": 1.},
            {"prefix_token_ids": [1, 2, 3], "baseline": .4,
             "mean_cost": 20., "mean_reward": 0.}]
    first = feature_matrix(rows, "cheap")
    rows[0]["mean_cost"], rows[0]["mean_reward"] = 999., -999.
    np.testing.assert_array_equal(first, feature_matrix(rows, "cheap"))
    basis = np.asarray([[[1.], [0.]], [[0.], [1.]]])
    coefficients = np.asarray([[1.], [2.]])
    target = np.zeros((2, 2))
    weighted = residual_metrics(basis, coefficients, target, [1., 4.], [2., .5])
    assert weighted["residual_mean"] == 4.5
    weighted_basis = np.asarray([[[1.], [1.]]])
    labels = fit_prefix_coefficients(weighted_basis, np.asarray([[0., 2.]]), [2., .5])
    np.testing.assert_allclose(labels, [[.4]])


def test_dynamic_loader_rejects_mismatched_actor_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        replay, score, _ = _write_replay(root)
        (replay / "replay_provenance.json").write_text(
            json.dumps({"actor_sha256": "actor-a"}), encoding="utf-8")
        score.parent.joinpath("score_gradient_provenance.json").write_text(
            json.dumps({"replay_prefixes_sha256": sha256_file(replay / "prefixes.jsonl"),
                        "shape": [6, 3, 1],
                        "actor_sha256": "actor-b"}), encoding="utf-8")
        with pytest.raises(ValueError, match="provenance"):
            load_dynamic_replay(replay, score)


def test_dynamic_loader_can_read_euclidean_sidecar_from_weighted_replay():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        replay, score, _ = _write_replay(root)
        metric = np.asarray([2., .5, 1.])
        np.save(replay / "metric_weights.npy", metric)
        np.save(replay / "half_metric_norm_sq_b.npy",
                np.load(replay / "half_mean_norm_sq_b.npy") * 2.)
        (replay / "replay_provenance.json").write_text(
            json.dumps({"metric_name": "adam_diagonal",
                        "metric_weights_sha256": sha256_array(metric)}), encoding="utf-8")
        identity = root / "identity.npy"
        np.save(identity, np.ones(3))
        _, _, _, norm_b, _, loaded_metric, metric_name = load_dynamic_replay(
            replay, score, identity)
        np.testing.assert_allclose(norm_b, np.load(replay / "half_mean_norm_sq_b.npy"))
        np.testing.assert_allclose(loaded_metric, np.ones(3))
        assert metric_name == "euclidean"
        del norm_b, loaded_metric, _


def test_calibrated_baseline_rewrites_each_continuation_and_checks_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        calibration = root / "baseline.json"
        calibration.write_text(json.dumps({"checkpoint_sha256": "checkpoint-a",
                                            "problems": {"p": {"baseline": .75}}}),
                               encoding="utf-8")
        values = _load_problem_baselines(calibration, "checkpoint-a")
        rows = [{"problem_id": "p", "baseline": .5,
                 "continuation_records": [{"reward": 1., "baseline": .5}]}]
        updated = _apply_problem_baselines(rows, values)
        assert updated[0]["baseline"] == .75
        assert updated[0]["continuation_records"][0]["baseline"] == .75
        with pytest.raises(ValueError, match="checkpoint"):
            _load_problem_baselines(calibration, "checkpoint-b")
