"""CPU checks for the scalar directional-value audit."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grace_gc.audit.benefit_replay import replay_means
from grace_gc.versions import sha256_array
from scripts.merge_expected_gain_replays import main as merge_main
from scripts.expected_gain_directional_audit import _loo_targets, main


def _write_replay(root, rows, means):
    root.mkdir()
    (root / "prefixes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    np.save(root / "mean_grads.npy", np.asarray(means, dtype=np.float64))
    (root / "replay_provenance.json").write_text(json.dumps({
        "actor_sha256": "actor", "layout_names": ["lora"], "layout_dim": means.shape[1],
        "model_path": "model", "lora": {}}), encoding="utf-8")


class DirectionalAuditTest(unittest.TestCase):
    def test_leave_one_problem_out_labels_exclude_own_problem(self):
        rows = [{"problem_id": "a"}, {"problem_id": "a"},
                {"problem_id": "b"}, {"problem_id": "c"}]
        means = np.array([[1., 0.], [3., 0.], [0., 2.], [0., 4.]])
        labels, norms = _loo_targets(rows, means)
        np.testing.assert_allclose(labels, [0., 0., 4., 4.])
        self.assertEqual(set(norms), {"a", "b", "c"})

    def test_fast_file_direction_audit_reports_n_grid_and_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"problem_id": f"p{i // 2}", "prefix_token_ids": [1, 2, i],
                     "mean_cost": 10. + i, "mean_reward": float(i % 2),
                     "baseline": .5} for i in range(12)]
            rng = np.random.default_rng(11)
            means = rng.normal(size=(len(rows), 4))
            replay = root / "replay"
            _write_replay(replay, rows, means)
            values = np.column_stack((means @ np.array([1., -1., .5, .25]),
                                      means @ np.array([.5, .25, -1., 1.]),
                                      means @ np.array([1., 0., 0., 0.]),
                                      means @ np.array([0., 1., 0., 0.])))
            np.save(replay / "directional_values.npy", values)
            direction = root / "direction.npy"
            np.save(direction, np.array([1., -1., .5, .25]))
            self.assertEqual(main(["--replay-dir", str(replay), "--direction", "file",
                                   "--direction-file", str(direction), "--features", "cheap",
                                   "--models", "zero,constant,ridge", "--n-grid", "2,4",
                                   "--run-dir", str(root / "result")]), 0)
            summary = json.loads((root / "result" / "directional_value_summary.json").read_text())
            self.assertEqual(summary["direction"], "frozen_direction_file")
            self.assertEqual([row["n"] for row in summary["n_grid"]], [2, 4])
            self.assertTrue(summary["predictors_on_mean_label"])

    def test_reference_direction_keeps_problem_pools_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"problem_id": f"p{i // 2}", "t": 512,
                     "features": [float(i), 1.]} for i in range(12)]
            replay = root / "replay"
            _write_replay(replay, rows, np.arange(48, dtype=np.float64).reshape(12, 4))
            reference_rows = [{"problem_id": f"r{i}", "t": 0,
                               "features": [0., 1.]} for i in range(3)]
            reference = root / "reference"
            _write_replay(reference, reference_rows,
                          np.tile(np.array([[1., 0., 0., 0.]]), (3, 1)))
            self.assertEqual(main(["--replay-dir", str(replay), "--direction", "reference",
                                   "--reference-dir", str(reference), "--features", "legacy",
                                   "--models", "constant", "--run-dir", str(root / "result")]), 0)
            summary = json.loads((root / "result" / "directional_value_summary.json").read_text())
            self.assertEqual(summary["direction"], "equal_problem_reference_gradient")
            self.assertIsNone(summary["reference_check_cosine"])

    def test_replay_stores_scalar_values_without_full_suffix_gradients(self):
        rows = [{"problem_id": "p", "path_id": "path", "t": 0,
                 "continuation_records": [
                     {"token_ids": [1, 2, 3 + i], "prompt_len": 1,
                      "reward": 1., "baseline": 0., "generated_suffix_tokens": 1}
                     for i in range(4)]}]
        with tempfile.TemporaryDirectory() as tmp:
            direction = np.array([1., 2.])
            replay_means(rows, lambda tokens, _prompt, _reward, _baseline:
                         np.array([float(tokens[-1]), 1.]), 2, tmp,
                         max_continuations=4, seed=3, reference_direction=direction)
            values = np.load(Path(tmp) / "directional_values.npy")
            self.assertEqual(values.shape, (1, 4))
            self.assertTrue(np.all(np.isfinite(values)))
            summary = json.loads((Path(tmp) / "replay_summary.json").read_text())
            self.assertEqual(summary["directional_values"]["shape"], [1, 4])

    def test_merge_preserves_scalar_values_when_direction_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = {"actor_sha256": "actor", "layout_names": ["lora"],
                          "layout_dim": 2, "model_path": "model", "lora": {},
                          "direction_sha256": sha256_array(np.array([1., 2.]))}
            for index in range(2):
                folder = root / f"replay-{index}"; folder.mkdir()
                row = {"problem_id": f"p{index}", "t": 512,
                       "features": [float(index), 1.]}
                (folder / "prefixes.jsonl").write_text(json.dumps(row) + "\n")
                np.save(folder / "mean_grads.npy", np.array([[index + 1., 2.]]))
                np.save(folder / "directional_values.npy",
                        np.array([[float(index), float(index + 1)]]))
                (folder / "replay_provenance.json").write_text(json.dumps(provenance))
            self.assertEqual(merge_main(["--replay", str(root / "replay-0"),
                                         "--replay", str(root / "replay-1"),
                                         "--run-dir", str(root / "merged")]), 0)
            values = np.load(root / "merged" / "directional_values.npy")
            np.testing.assert_allclose(values, [[0., 1.], [1., 2.]])


if __name__ == "__main__":
    unittest.main()
