"""CPU contract tests for the exact-gradient replay and held-out suite."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

from grace_gc.audit.benefit_replay import replay_means
from grace_gc.audit.expected_gain import (fit_global_basis, full_space_residual,
                                          load_replay, problem_split, project_means)
try:
    import yaml  # noqa: F401
except ImportError:
    # This unit suite also runs with the bundled NumPy-only Python runtime.
    sys.modules["yaml"] = types.ModuleType("yaml")
from scripts.expected_gain_suite import main as suite_main
from scripts.merge_expected_gain_replays import main as merge_main
from scripts.prepare_expected_gain_data import main as prepare_main


class ExpectedGainExperimentTest(unittest.TestCase):
    def test_data_preparation_reserves_disjoint_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            source.write_text("".join(json.dumps({"problem_id": f"p{i}",
                                                   "prompt": f"Solve problem {i}",
                                                   "answer": str(i)}) + "\n"
                                      for i in range(30)), encoding="utf-8")
            evaluation.write_text(json.dumps({"problem_id": "external",
                                              "prompt": "Solve problem 0",
                                              "answer": "0"}) + "\n", encoding="utf-8")
            prepare_main(["--data-path", str(source), "--eval-data-path", str(evaluation),
                          "--output-dir", str(root / "prepared"),
                          "--predictor-problems", "4", "--reference-problems", "2",
                          "--reference-check-problems", "2"])
            manifest = json.loads((root / "prepared" / "expected_gain_data_manifest.json").read_text())
            role_ids = [set(ids) for ids in manifest["roles"].values()]
            self.assertEqual(sum(map(len, role_ids)), 8)
            self.assertEqual(len(set.union(*role_ids)), 8)
            self.assertNotIn("p0", set.union(*role_ids))

    def test_replay_uses_original_norm_index_after_subsampling(self):
        rows = [{"problem_id": "p0", "path_id": "p0:0", "t": 512,
                 "finished": False, "features": [1., 2., 3., 4., 5.],
                 "continuation_records": [
                     {"token_ids": [1, 2, i + 3], "prompt_len": 1,
                      "reward": 1., "baseline": 0., "generated_suffix_tokens": i + 1}
                     for i in range(8)],
                 "true_grad_norm_sq": [float((i + 1) ** 2) for i in range(8)]}]
        def gradient(tokens, _prompt_len, _reward, _baseline):
            return np.array([float(tokens[-1] - 2), 0., 0.])
        with tempfile.TemporaryDirectory() as tmp:
            replay_means(rows, gradient, 3, tmp, max_continuations=3, seed=2)
            info, means = load_replay(tmp)
            chosen = info[0]["continuation_indices"]
            self.assertAlmostEqual(means[0, 0], np.mean([i + 1 for i in chosen]))
            self.assertAlmostEqual(info[0]["mean_norm_sq"],
                                   np.mean([(i + 1) ** 2 for i in chosen]))
            del means

    def test_full_space_residual_and_basis_rank(self):
        means = np.array([[1., 0., 0.], [0., 2., 0.], [1., 2., 0.]])
        with tempfile.TemporaryDirectory() as tmp:
            info = fit_global_basis(means, np.array([0, 1]),
                                    np.array([[1., 0.], [0., 1.]]),
                                    ["a", "b"], "mean_svd", 64,
                                    Path(tmp) / "basis.npy")
            self.assertEqual(info["actual_rank"], 2)
            basis = np.load(Path(tmp) / "basis.npy")
            coord = project_means(means, basis)
            residual = full_space_residual(np.sum(means * means, axis=1),
                                           coord, coord, np.asarray(info["basis_gram"]))
            np.testing.assert_allclose(residual, 0., atol=1e-10)

    def test_split_keeps_problem_groups_intact(self):
        rows = [{"problem_id": str(i)} for i in range(12) for _ in range(3)]
        split = problem_split(rows, seed=17)
        self.assertEqual(set(split["train"]) | set(split["validation"]) |
                         set(split["diagnostic"]), {str(i) for i in range(12)})
        self.assertFalse(set(split["train"]) & set(split["diagnostic"]))

    def test_suite_cpu_end_to_end(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {"actor_sha256": "frozen", "layout_names": ["lora"],
                      "layout_dim": 12, "model_path": "model"}
            for role, problem_count in (("predictor", 12), ("reference", 4),
                                        ("reference_check", 4)):
                folder = root / role
                folder.mkdir()
                n = problem_count * (2 if role == "predictor" else 1)
                gradients = rng.normal(size=(n, 12))
                np.save(folder / "mean_grads.npy", gradients)
                (folder / "replay_provenance.json").write_text(json.dumps(common))
                with (folder / "prefixes.jsonl").open("w") as handle:
                    for i, vec in enumerate(gradients):
                        row = {"problem_id": f"{role}-{i // (2 if role == 'predictor' else 1)}",
                               "features": rng.normal(size=8).tolist(),
                               "prompt_features": rng.normal(size=8).tolist(),
                               "mean_norm_sq": float(vec @ vec + 0.5),
                               "mean_cost": 50., "half_mean_dot": float(vec @ vec),
                               "mean_reward": 0.5}
                        handle.write(json.dumps(row) + "\n")
            suite_main(["--replay-dir", str(root / "predictor"),
                        "--reference-dir", str(root / "reference"),
                        "--reference-check-dir", str(root / "reference_check"),
                        "--run-dir", str(root / "result"),
                        "--models", "zero,constant,ridge"])
            report = json.loads((root / "result" / "expected_gain_summary.json").read_text())
            self.assertEqual(report["gradient_dimension"], 12)
            self.assertTrue(report["basis_rows"])
            self.assertTrue(any(row.get("feature_set") == "prompt_only"
                                for row in report["scalar_rows"]))
            self.assertEqual(len(report["allocation"]), 4)
            merge_main(["--replay", str(root / "predictor"),
                        "--replay", str(root / "reference"),
                        "--run-dir", str(root / "merged")])
            merged_rows, merged_means = load_replay(root / "merged")
            self.assertEqual(len(merged_rows), 28)
            self.assertEqual(merged_means.shape, (28, 12))
            del merged_means


if __name__ == "__main__":
    unittest.main()
