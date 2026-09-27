from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from grace_gc.audit.benefit_replay import replay_means
from scripts.expected_gain_dynamic_oracle import main as oracle_main
from scripts.expected_gain_dynamic_predictor import main as predictor_main


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
                     2, tmp, max_continuations=4, seed=7, store_half_means=True)
        assert np.load(Path(tmp) / "half_mean_grads_a.npy").shape == (1, 2)
        metadata = json.loads((Path(tmp) / "prefixes.jsonl").read_text().splitlines()[0])
        assert metadata["half_counts"] == [2, 2]
        assert len(metadata["half_mean_norm_sq"]) == 2


def test_dynamic_oracle_and_predictor_use_a_b_split_without_torch():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        replay, score, split = _write_replay(root)
        oracle_main(["--replay-dir", str(replay), "--score-gradients", str(score),
                     "--split-manifest", str(split), "--run-dir", str(root / "oracle")])
        oracle = json.loads((root / "oracle" / "dynamic_oracle_summary.json").read_text())
        assert oracle["stage"] == "A"
        assert oracle["roles"]["diagnostic"]["oracle"]["residual_ratio_to_zero"] < 0.1
        predictor_main(["--replay-dir", str(replay), "--score-gradients", str(score),
                        "--split-manifest", str(split), "--run-dir", str(root / "predictor"),
                        "--features", "legacy", "--models", "zero,ridge"])
        report = json.loads((root / "predictor" / "dynamic_predictor_summary.json").read_text())
        ridge = next(row for row in report["models"] if row["model"] == "ridge_l2_1")
        zero = next(row for row in report["models"] if row["model"] == "zero")
        assert ridge["validation"]["residual_mean"] < zero["validation"]["residual_mean"]
