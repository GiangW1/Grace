"""CPU contract tests for the exact-gradient replay and held-out suite."""

from __future__ import annotations

import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from grace_gc.audit.benefit_replay import replay_means, replay_config, continuation_digest
from grace_gc.audit.expected_gain import (fit_global_basis, full_space_residual,
                                          load_replay, problem_split, project_means,
                                          reference_gradient, start_experiment_run)
from scripts.expected_gain_suite import main as suite_main, _allocation, _fixed_probability
from scripts.merge_expected_gain_replays import main as merge_main
from scripts.prepare_expected_gain_data import main as prepare_main
from scripts.probe_expected_gain_update import _checked_gradient


class ExpectedGainExperimentTest(unittest.TestCase):
    def test_zero_token_reference_starts_without_zero_token_generation(self):
        from grace_gc.audit import run
        from grace_gc.data.math_data import MathRecord
        class Engine:
            eos_id, last_rollout = None, None
            decode = staticmethod(lambda ids: "Answer: 1")
            def generate_prefix(self, *_args):
                raise AssertionError("reference must not send max_tokens=0 to vLLM")
            def continue_selected(self, prefixes, selected, length, rng):
                return [prefix + [3] * length for prefix in prefixes]
        with patch.object(run, "_policy_grad_vec", return_value=np.array([1., 2.])):
            bundles = run._bundles_from_engines(
                [MathRecord("p", "question", "1")], Engine(), SimpleNamespace(dim=2),
                1, 2, 0, 2, 17, lambda rec, n: ([[1]] * n, [rec.problem_id] * n, [rec.answer] * n),
                baseline_fn=lambda _pid: .5)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].t, 0)
        self.assertEqual(len(bundles[0].continuation_records), 2)

    def test_reference_is_unconditional_and_weights_problems_equally(self):
        rows = [{"problem_id": pid, "t": 0} for pid in ("a", "a", "b")]
        means = np.array([[2., 0.], [4., 0.], [0., 3.]])
        np.testing.assert_allclose(reference_gradient(rows, means), [1.5, 1.5])
        rows[0]["t"] = 512
        with self.assertRaisesRegex(ValueError, "unconditional"):
            reference_gradient(rows, means)

    def test_run_directories_preserve_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            first = start_experiment_run(path, "test", {})
            (first / "expected_gain_summary.json").write_text("old result")
            second = start_experiment_run(path, "test", {})
            self.assertNotEqual(first, second)
            self.assertEqual((first / "expected_gain_summary.json").read_text(), "old result")

    def test_replay_recovers_checkpoint_numerics(self):
        payload = {"run_config": {"lora": {"compute_dtype": "bfloat16"},
                                    "optim": {"grad_clip": .03},
                                    "data_path": "missing/source.jsonl"}}
        cfg = replay_config(payload, "base-model")
        self.assertEqual(cfg["lora"]["compute_dtype"], "bfloat16")
        self.assertEqual(cfg["optim"]["grad_clip"], .03)

    def test_allocation_uses_observed_cost_and_pointwise_probability(self):
        before = _fixed_probability([0., 1.], [1., 1.], 1e8, .2)[0]
        after = _fixed_probability([0., 1e12], [1., 1.], 1e8, .2)[0]
        self.assertEqual(before, after)
        result = _allocation("test", [1., 2.], [1., 2.], [1., 1.], [1., 1.],
                             [2., 2.], .5, .2, [1., 9.])
        self.assertAlmostEqual(result["predicted_test_cost_fraction"], .5)
        self.assertAlmostEqual(result["test_cost_fraction"], (1 / 3 + 9 * 2 / 3) / 10)
        full = _allocation("full", [1.], [1.], [1.], [1.], [1.], 1., .2)
        json.dumps(full, allow_nan=False)

    def test_probe_rejects_changed_source_trajectory(self):
        record = {"token_ids": [1, 2], "prompt_len": 1, "reward": 1., "baseline": 0.}
        replay = {"continuation_sha256": {"0": continuation_digest(record)}}
        record["token_ids"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "trajectory"):
            _checked_gradient(lambda *_: np.ones(2), {"continuation_records": [record]}, 0, replay)

    def test_adamw_probe_restores_moments_groups_and_shared_clip_order(self):
        import torch
        from grace_gc.audit.update_probe import adamw_update_direction
        from grace_gc.core.layout import collect_lora_layout
        from grace_gc.trainer.state_io import optimizer_state

        params = [torch.nn.Parameter(torch.tensor([.2, -.3])),
                  torch.nn.Parameter(torch.tensor([.5]))]
        optimizer = torch.optim.AdamW([{"params": [params[0]], "lr": .01},
                                        {"params": [params[1]], "lr": .02}], weight_decay=.02)
        params[0].grad, params[1].grad = torch.tensor([.4, -.5]), torch.tensor([.2])
        optimizer.step()
        named = [("q_proj.lora_A", params[0]), ("v_proj.lora_B", params[1])]
        layout = collect_lora_layout(named)
        actor = {key: value.detach().numpy().copy() for key, value in named}
        snapshot = optimizer_state(optimizer)
        original = copy.deepcopy(snapshot)
        g, ref = np.array([3., 4., 0.]), np.array([1., -2., 3.])
        first = adamw_update_direction(actor, layout, snapshot, {}, g, ref, .5)
        again = adamw_update_direction(actor, layout, snapshot, {}, g, ref, .5)
        self.assertEqual(first, again)
        for key, state in original["state"].items():
            for name, value in state.items():
                np.testing.assert_array_equal(snapshot["state"][key][name], value)
        starts = [param.detach().clone() for param in params]
        params[0].grad, params[1].grad = torch.tensor([-.3, -.4]), torch.tensor([0.])
        optimizer.step()
        delta = np.concatenate([(param.detach() - start).numpy()
                                for param, start in zip(params, starts)])
        self.assertAlmostEqual(first["reference_dot_delta"], float(delta @ ref), places=7)
        self.assertTrue(first["clip_triggered"])

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
            split = json.loads((root / "prepared" / "split.json").read_text())
            self.assertTrue(set(manifest["background_train_ids"]) <= set(split["train"]))

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
                               "t": 0,
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
                        "--models", "zero,constant,ridge,mlp64", "--mlp-epochs", "2"])
            report = json.loads((root / "result" / "expected_gain_summary.json").read_text())
            self.assertEqual(report["gradient_dimension"], 12)
            self.assertTrue(report["basis_rows"])
            self.assertTrue(any(row.get("feature_set") == "prompt_only"
                                for row in report["scalar_rows"]))
            self.assertEqual(len(report["allocation"]), 6)
            merge_main(["--replay", str(root / "predictor"),
                        "--replay", str(root / "reference"),
                        "--run-dir", str(root / "merged")])
            merged_rows, merged_means = load_replay(root / "merged")
            self.assertEqual(len(merged_rows), 28)
            self.assertEqual(merged_means.shape, (28, 12))
            del merged_means
            # Zero signal and one validation problem are valid negative/pilot results.
            np.save(root / "predictor" / "mean_grads.npy", np.zeros((24, 12)))
            pilot_split = {"train": [f"predictor-{i}" for i in range(8)],
                           "validation": ["predictor-8"],
                           "diagnostic": ["predictor-9", "predictor-10", "predictor-11"]}
            (root / "pilot.json").write_text(json.dumps(pilot_split))
            suite_main(["--replay-dir", str(root / "predictor"),
                        "--reference-dir", str(root / "reference"),
                        "--split-manifest", str(root / "pilot.json"),
                        "--run-dir", str(root / "zero-result"), "--models", "zero,constant"])
            zero = json.loads((root / "zero-result" / "expected_gain_summary.json").read_text())
            self.assertEqual(zero["selected_by_validation_residual"]["actual_rank"], 0)
            self.assertIn("constant", zero["risk_head_kind"])


if __name__ == "__main__":
    unittest.main()
