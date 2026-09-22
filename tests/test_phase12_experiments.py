from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_continuous_rollout_refills_and_keeps_all_starts(monkeypatch):
    import grace_gc.backends.vllm_continuous as module

    class FakeEngine:
        def __init__(self):
            self.pending = {}
            self.submissions = []
            self.step_count = 0

        def add_request(self, request_id, prompt, params, **_kwargs):
            self.pending[request_id] = (prompt, params)
            self.submissions.append(request_id)

        def step(self):
            self.step_count += 1
            outputs = []
            for request_id, (prompt, params) in list(self.pending.items()):
                # A prompt longer than the original prompt is a suffix request.
                if len(prompt) > 1:
                    completion = SimpleNamespace(token_ids=[12], finish_reason="stop", stop_reason=None)
                elif int(params.max_tokens) == 2:
                    completion = SimpleNamespace(token_ids=[10, 11], finish_reason="length", stop_reason=None)
                else:
                    completion = SimpleNamespace(token_ids=[12], finish_reason="stop", stop_reason=None)
                outputs.append(SimpleNamespace(request_id=request_id, finished=True, outputs=[completion]))
                del self.pending[request_id]
            return outputs

    engine = FakeEngine()
    llm = SimpleNamespace(llm_engine=engine)
    monkeypatch.setattr(module, "build_sampling_params", lambda max_tokens, *_args, **_kwargs: SimpleNamespace(max_tokens=max_tokens))
    monkeypatch.setattr(module, "_vllm_prompts", lambda prompts: prompts)
    starts = [{"problem_id": str(i), "prompt_token_ids": [1]} for i in range(5)]
    runner = module.ContinuousRollout(llm, eos_id=99, capacity=2)

    summary, records = runner.run(starts, mode="fixed_ht", decision_tokens=2,
                                   max_new_tokens=4, p=0.0, seed=4)

    assert len(records) == 5
    assert all(row.stage == "stopped" for row in records)
    assert summary["stopped"] == 5
    assert summary["continued"] == 0
    assert summary["max_active"] == 2
    assert summary["prefix_tokens"] == 10
    assert len(engine.submissions) == 5

    plan_a = module.make_request_plan(5, 0.5, 19)
    plan_b = module.make_request_plan(5, 0.5, 19)
    assert plan_a == plan_b
    assert len(set(plan_a["prefix_seeds"])) == 5


def test_predictor_diagnostic_splits_by_problem_and_reports_zero_ratio():
    from grace_gc.audit.prefix_audit import PrefixBundle, bundle_from_dict, bundle_to_dict
    from scripts.diagnose_predictor import diagnose

    bundles = []
    gram = [[1.0, 0.0], [0.0, 1.0]]
    for problem_id, x0 in (("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 3.0)):
        coords = [[x0, 0.0], [x0 + 0.2, 0.0]]
        bundles.append(PrefixBundle(
            problem_id=problem_id,
            t=2,
            rewards=np.asarray([0.0, 1.0]),
            grads=np.asarray(coords),
            features=np.asarray([x0, 1.0]),
            true_grad_norm_sq=[x0 * x0, (x0 + 0.2) ** 2],
            true_grad_coords=coords,
            basis_gram=gram,
        ))

    class Args:
        ridge_l2 = 1.0
        mlp_hidden = 8
        mlp_epochs = 1

    result, rows = diagnose(bundles, seed=9, test_fraction=0.5,
                            models=["zero", "simple"], args=Args())

    assert result["split"]["unit"] == "problem_id"
    assert result["split"]["train_problem_ids"]
    assert result["split"]["test_problem_ids"]
    assert not set(result["split"]["train_problem_ids"]) & set(result["split"]["test_problem_ids"])
    assert [row["model"] for row in rows] == ["zero", "simple"]
    assert all(row["n_test_problems"] == len(result["split"]["test_problem_ids"]) for row in rows)
    assert result["future_randomness"]["n_prefixes"] == len(result["split"]["test_problem_ids"])
    assert len(result["by_problem"]) == 2 * len(result["split"]["test_problem_ids"])
    restored = bundle_from_dict(bundle_to_dict(bundles[0]))
    assert np.array_equal(restored.features, bundles[0].features)


def test_predictor_diagnostic_ridge_and_mlp_candidates():
    pytest.importorskip("torch")
    from grace_gc.audit.prefix_audit import PrefixBundle
    from scripts.diagnose_predictor import diagnose

    bundles = []
    for index in range(8):
        x = float(index)
        bundles.append(PrefixBundle(
            problem_id=f"p{index}", t=2,
            rewards=np.asarray([0.0, 1.0]),
            grads=np.asarray([[x], [x + 0.1]]),
            features=np.asarray([x, 1.0]),
            true_grad_norm_sq=[x * x, (x + 0.1) ** 2],
            true_grad_coords=[[x], [x + 0.1]], basis_gram=[[1.0]],
        ))

    class Args:
        seed = 3
        ridge_l2 = 0.1
        mlp_hidden = 8
        mlp_epochs = 2

    result, rows = diagnose(bundles, seed=3, test_fraction=0.25,
                            models=["ridge", "mlp"], args=Args())
    assert [row["model"] for row in rows] == ["ridge", "mlp"]
    assert all(np.isfinite(row["test_residual_mean"]) for row in rows)
    assert result["oracle_basis"]["test_coordinate_error_mean"] < 1e-10
