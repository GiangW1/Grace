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


def test_serve_scheduler_refill_and_no_refill_keep_logical_slots(monkeypatch):
    from scripts import scheduler_probe_serve as module

    class FakeClient:
        def __init__(self):
            self.calls = []

        def complete(self, prompt_token_ids, max_tokens, seed, stop_token_ids, request_id, cache_salt):
            import time

            index = int(request_id.split("-")[3])
            if index % 3 == 0:
                time.sleep(0.002)
            self.calls.append((request_id, list(prompt_token_ids), int(max_tokens), cache_salt))
            if len(prompt_token_ids) > 1:
                token_ids, finish = [12], "stop"
            elif int(max_tokens) == 2:
                token_ids, finish = [10, 11], "length"
            else:
                token_ids, finish = [10, 11], "length"
            started = time.perf_counter()
            return module.ServeResponse(
                request_id=request_id, prompt_token_ids=list(prompt_token_ids), token_ids=token_ids,
                finish_reason=finish, stop_reason=None, usage={}, wall_seconds=0.001,
                submitted_at=started, returned_at=started + 0.001,
            )

    starts = [{"problem_id": str(i), "prompt_token_ids": [1]} for i in range(5)]
    plan = module.make_request_plan(5, 0.5, 19)
    for scheduler in ("no_refill", "refill"):
        client = FakeClient()
        summary, records, requests = module.run_scheduler(
            client, starts, scheduler=scheduler, capacity=2, all_concurrency=None,
            mode="fixed_ht", decision_tokens=2, max_new_tokens=4, p=0.5,
            request_plan=plan, eos_id=[99], cache_salt=f"test-{scheduler}",
        )
        assert len(records) == 5
        assert summary["completed"] + summary["stopped"] == 5
        assert summary["n_http_requests"] == len(requests)
        assert summary["client_capacity"] == 2
        assert all(row.request_count in {1, 2} for row in records)
        assert all(row[3] == f"test-{scheduler}" for row in client.calls)


def test_serve_scheduler_selected_suffix_does_not_consume_new_slot():
    from scripts import scheduler_probe_serve as module

    class FakeClient:
        def complete(self, prompt_token_ids, max_tokens, seed, stop_token_ids, request_id, cache_salt):
            import time

            now = time.perf_counter()
            if len(prompt_token_ids) > 1:
                token_ids, finish = [12], "stop"
            else:
                token_ids, finish = [10, 11], "length"
            return module.ServeResponse(
                request_id=request_id, prompt_token_ids=list(prompt_token_ids), token_ids=token_ids,
                finish_reason=finish, stop_reason=None, usage={}, wall_seconds=0.001,
                submitted_at=now, returned_at=now + 0.001,
            )

    starts = [{"problem_id": str(i), "prompt_token_ids": [1]} for i in range(3)]
    plan = module.make_request_plan(3, 1.0, 23)
    summary, records, requests = module.run_scheduler(
        FakeClient(), starts, scheduler="refill", capacity=2, all_concurrency=None,
        mode="fixed_ht", decision_tokens=2, max_new_tokens=4, p=1.0,
        request_plan=plan, eos_id=[99], cache_salt="test",
    )

    assert summary["completed"] == 3
    assert summary["stopped"] == 0
    assert summary["n_http_requests"] == 6
    assert all(row.request_count == 2 for row in records)
    assert any(request["stage"] == "suffix" for request in requests)


def test_serve_client_uses_tokenized_completion_request(monkeypatch):
    from scripts import scheduler_probe_serve as module

    client = module.VLLMServeClient("http://127.0.0.1:8000/v1", "served-model")
    captured = {}

    def fake_request(method, endpoint, payload=None):
        captured.update({"method": method, "endpoint": endpoint, "payload": payload})
        return {
            "choices": [{
                "prompt_token_ids": [1, 2], "token_ids": [10, 11],
                "finish_reason": "length", "stop_reason": None,
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }

    monkeypatch.setattr(client, "_request", fake_request)
    response = client.complete([1, 2], 2, 7, [99], "request-1", "salt")

    assert captured["method"] == "POST"
    assert captured["endpoint"] == "/v1/completions"
    assert captured["payload"]["prompt"] == [1, 2]
    assert captured["payload"]["return_token_ids"] is True
    assert captured["payload"]["add_special_tokens"] is False
    assert captured["payload"]["cache_salt"] == "salt"
    assert captured["payload"]["temperature"] == pytest.approx(0.2)
    assert response.token_ids == [10, 11]


def test_predictor_suite_separates_prefix_mean_from_suffix_noise():
    from scripts import diagnose_predictor_suite as module

    compact = module._without_large_grads(
        '{"problem_id":"p","grads":[[1,2],[3,4]],"coords":null,"rewards":[1]}'
    )
    assert '"grads":null' in compact
    assert '[[1,2],[3,4]]' not in compact

    def bundle(problem_id, coords):
        values = np.asarray(coords, dtype=np.float64).reshape(-1, 1)
        return SimpleNamespace(
            problem_id=problem_id, t=512, path_id=f"{problem_id}:0",
            features=np.arange(8, dtype=np.float64), true_grad_coords=values,
            true_grad_norm_sq=(values[:, 0] ** 2).tolist(),
            basis_gram=[[1.0]], rewards=np.ones(len(values)),
            suffix_cost=np.ones(len(values)),
        )

    examples = module._prefix_examples([bundle("a", [1.0, 3.0]), bundle("b", [0.0])])
    metrics = module._coordinate_metrics(examples, np.zeros((2, 1)))

    assert examples[0].coord_variance == pytest.approx(2.0)
    assert metrics["test_prefix_mean_coordinate_error"] == pytest.approx(2.0)
    assert metrics["test_suffix_residual_mean"] == pytest.approx(2.5)
    assert metrics["test_prefix_mean_ratio_to_zero"] == pytest.approx(1.0)
    assert module._feature_view(examples[0].features, "last").shape == (1,)

    nonorthogonal = bundle("c", [2.0])
    nonorthogonal.basis_gram = [[4.0]]
    nonorthogonal.true_grad_norm_sq = [1.0]
    projected = module._prefix_examples([nonorthogonal])
    exact = module._coordinate_metrics(projected, np.asarray([[0.5]]))
    assert exact["test_subspace_omission_mean"] == pytest.approx(0.0)
    assert exact["test_suffix_residual_mean"] == pytest.approx(0.0)


def test_two_checkpoint_ht_estimate_is_unbiased_and_nested():
    from grace_gc.core.estimator import two_checkpoint_ht_estimate

    g = np.asarray([[7.0, -2.0]])
    m1 = np.asarray([[2.0, 1.0]])
    m2 = np.asarray([[4.0, 0.0]])
    p1, p2 = 0.6, 0.4
    outcomes = [
        ((0, 0), 1 - p1),
        ((1, 0), p1 * (1 - p2)),
        ((1, 1), p1 * p2),
    ]
    mean = sum(prob * two_checkpoint_ht_estimate(
        g, m1, m2, [p1], [p2], [z1], [z2],
    )[0] for (z1, z2), prob in outcomes)
    assert np.allclose(mean, g[0])
    assert np.allclose(two_checkpoint_ht_estimate(g, m1, m2, [1], [1], [1], [1]), g)
    with pytest.raises(ValueError, match="nested"):
        two_checkpoint_ht_estimate(g, m1, m2, [p1], [p2], [0], [1])


def test_sparse_truncation_cost_match_and_trial_plan_are_stable():
    from scripts.truncation_refill_serve import make_trial_plan, matched_second_probability

    plan_a = make_trial_plan(4, 19)
    plan_b = make_trial_plan(4, 19)
    assert plan_a == plan_b
    assert len({*plan_a["prefix_seeds"]}) == 4
    result = matched_second_probability([2, 4, 6, 8], 2, 4, 0.5, 0.6)
    middle, tail = result["middle_tokens"], result["tail_tokens"]
    assert 0 < result["p_second"] <= 1
    assert 0.5 * (middle + tail) == pytest.approx(
        0.6 * middle + 0.6 * result["p_second"] * tail
    )


def test_sparse_truncation_refills_and_barrier_delays_suffix():
    from scripts import scheduler_probe_serve as serve
    from scripts.truncation_refill_serve import Arm, make_trial_plan, run_trial
    import threading
    import time

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.lock = threading.Lock()

        def complete(self, prompt_token_ids, max_tokens, seed, stop_token_ids,
                     request_id, cache_salt):
            stage = int(request_id.rsplit("-", 1)[-1])
            index = int(request_id.rsplit("-", 2)[-2])
            if stage == 0 and index == 1:
                time.sleep(0.02)
            with self.lock:
                self.calls.append((stage, index, cache_salt))
            now = time.perf_counter()
            return serve.ServeResponse(
                request_id=request_id, prompt_token_ids=list(prompt_token_ids),
                token_ids=[10 + stage] * int(max_tokens), finish_reason="length",
                stop_reason=None, usage={}, wall_seconds=0.001,
                submitted_at=now, returned_at=now,
            )

    starts = [{"problem_id": str(i), "prompt_token_ids": [1]} for i in range(3)]
    plan = make_trial_plan(3, 7)
    for scheduler in ("stream", "barrier"):
        client = FakeClient()
        arm = Arm(scheduler, (2,), scheduler=scheduler)
        summary, records, requests, _samples = run_trial(
            client, starts, arm, plan, capacity=2, max_new=4, eos_ids=[99],
            cache_salt=scheduler,
        )
        assert summary["n_starts"] == 3
        assert summary["n_http_requests"] == len(requests) == 6
        assert all(row["inclusion_probability"] == 1 for row in records)
        assert all(row["generated_tokens"] == 4 for row in records)
        first_suffix = next(i for i, (stage, _index, _salt) in enumerate(client.calls) if stage == 1)
        slow_prefix = next(i for i, (stage, index, _salt) in enumerate(client.calls)
                           if stage == 0 and index == 1)
        if scheduler == "stream":
            assert first_suffix < slow_prefix
        else:
            assert first_suffix > slow_prefix


def test_sparse_truncation_records_two_decisions_and_joint_probability():
    from scripts import scheduler_probe_serve as serve
    from scripts.truncation_refill_serve import Arm, make_trial_plan, run_trial
    import time

    class FakeClient:
        def complete(self, prompt_token_ids, max_tokens, seed, stop_token_ids,
                     request_id, cache_salt):
            now = time.perf_counter()
            return serve.ServeResponse(
                request_id=request_id, prompt_token_ids=list(prompt_token_ids),
                token_ids=[11] * int(max_tokens), finish_reason="length",
                stop_reason=None, usage={}, wall_seconds=0.001,
                submitted_at=now, returned_at=now,
            )

    class FakePredictor:
        def decide(self, checkpoint, token_ids, prompt_len, problem_id):
            return {"p": 0.8 if checkpoint == 2 else 0.3,
                    "risk": 1.0, "cost": 2.0, "f": [0.1],
                    "feature_seconds": 0.02, "prediction_seconds": 0.01}

    plan = make_trial_plan(1, 7)
    plan["selection_uniforms"] = [[0.1, 0.9]]
    arm = Arm("learned_two", (2, 4), learned=True)
    summary, records, requests, _samples = run_trial(
        FakeClient(), [{"problem_id": "p", "prompt_token_ids": [1]}], arm, plan,
        capacity=1, max_new=6, eos_ids=[99], cache_salt="test",
        predictor=FakePredictor(),
    )
    assert summary["stopped"] == 1
    assert summary["n_decisions"] == 2
    assert len(requests) == 2
    assert records[0]["inclusion_probability"] == pytest.approx(0.8 * 0.3)
    assert [row["selected"] for row in records[0]["decisions"]] == [True, False]
    assert records[0]["full_token_ids"] is None
    assert summary["feature_seconds"] == pytest.approx(0.04)


def test_sparse_predictor_batches_arriving_prefixes(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Condition, Thread
    from scripts.truncation_refill_serve import FrozenPredictor
    import grace_gc.backends.hf_actor as hf_actor

    sizes = []

    def fake_features(actor, prefixes, prompt_lens, baselines, pad_id, **kwargs):
        sizes.append(len(prefixes))
        return {"features": np.ones((len(prefixes), 2)),
                "cost_feat": np.ones((len(prefixes), 3))}

    class FakeModel:
        constant_cost = True

        def forward_numpy(self, features, cost_feat):
            n = len(features)
            return SimpleNamespace(f=np.zeros((n, 1)), r_hat=np.ones(n), c_hat=np.ones(n))

    monkeypatch.setattr(hf_actor, "prefix_feature_bundle", fake_features)
    predictor = FrozenPredictor.__new__(FrozenPredictor)
    predictor.actor = object()
    predictor.pad_id = 0
    predictor.eos_ids = [99]
    predictor.baseline = SimpleNamespace(get=lambda _problem_id: 0.5)
    predictor.models = {2: FakeModel()}
    predictor.feature_modes = {2: "legacy"}
    predictor.lambdas = {2: 1.0}
    predictor.uniform_p = {2: 0.5}
    predictor.p_min = 0.2
    predictor.uniform_shrink = 1.0
    predictor.batch_size = 4
    predictor.batch_wait_seconds = 0.05
    predictor._condition = Condition()
    predictor._pending = []
    predictor._closed = False
    predictor._worker = Thread(target=predictor._work, daemon=True)
    predictor._worker.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predictor.decide, 2, [1, 11, 11], 1, str(i))
                       for i in range(4)]
            results = [future.result() for future in futures]
    finally:
        predictor.close()
    assert sizes == [4]
    assert all(row["feature_batch_size"] == 4 for row in results)
    assert all(row["p"] == pytest.approx(0.5) for row in results)
