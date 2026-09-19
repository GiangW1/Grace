import json

import numpy as np
import pytest

from grace_gc.core.allocation import allocate_continuation
from grace_gc.data.reward import _compat_number_unit, rule_reward
from grace_gc.predictor.basis import refresh_basis
from grace_gc.predictor.risk import full_space_residual
from grace_gc.trainer.grace_step import decide_continuation, neyman_ready
from grace_gc.trainer.methods import method_spec


def test_neyman_ready_needs_synced_basis():
    assert neyman_ready(0, -1) is False
    assert neyman_ready(1, -1) is False
    assert neyman_ready(1, 0) is False
    assert neyman_ready(1, 1) is True
    assert neyman_ready(2, 1) is False


def test_grace_allocates_ones_until_basis_ready():
    spec = method_spec("grace")
    risk = np.array([9.0, 0.1, 1.0])
    cost = np.ones(3)
    finished = np.zeros(3, dtype=bool)
    p, _ = decide_continuation(spec, risk, cost, finished, 0.5, 0.2, warmup=False, basis_ready=False)
    np.testing.assert_allclose(p, np.ones(3))
    p2, _ = decide_continuation(spec, risk, cost, finished, 0.5, 0.2, warmup=False, basis_ready=True)
    assert p2.min() < 1.0 - 1e-12


def test_uniform_cv_does_not_wait_for_basis():
    spec = method_spec("uniform_cv")
    p, _ = decide_continuation(spec, np.ones(4), np.ones(4), np.zeros(4, dtype=bool), 0.5, 0.2, warmup=False)
    np.testing.assert_allclose(p, np.full(4, 0.5))


def test_allocate_p_invariant_to_risk_scale():
    risk = np.array([4.0, 1.0, 0.25])
    cost = np.ones(3)
    a = allocate_continuation(risk, cost, beta=0.5, p_min=0.2)
    b = allocate_continuation(100.0 * risk, cost, beta=0.5, p_min=0.2)
    np.testing.assert_allclose(a.p, b.p, atol=1e-10)


def test_full_space_residual_is_squared_norm():
    u = np.array([[1.0, 0.2], [0.0, 1.0], [0.0, 0.0]])
    g = np.array([1.0, 2.0, 3.0])
    f = np.array([0.5, -0.1])
    e = full_space_residual(g, f, u)
    assert e == pytest.approx(float(np.sum((g - u @ f) ** 2)), abs=1e-12)
    with pytest.raises(ValueError, match="not finite"):
        full_space_residual(np.array([np.nan, 0.0, 0.0]), f, u)


def test_refresh_basis_pads_rank_and_skips_zero():
    rank1 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    basis = refresh_basis(rank1, k=3, basis_id=1, seed=0)
    assert basis is not None
    assert basis.u.shape == (3, 3)
    assert basis.rank == 1
    assert np.linalg.norm(basis.u[:, 1]) == pytest.approx(0.0)
    assert np.linalg.norm(basis.u[:, 2]) == pytest.approx(0.0)
    assert refresh_basis(np.ones((5, 4)), k=2, basis_id=1, seed=0) is None
    collapsed = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    assert refresh_basis(collapsed, k=2, basis_id=1, seed=0, problem_ids=["a", "a", "b", "b"]) is None


def test_reservoir_stamp_basis_id():
    from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem

    res = GradientReservoir(capacity=4)
    res.add(ReservoirItem("a", np.ones(2), 1.0, 0.125, np.zeros(2), 1.0, 0))
    res.add(ReservoirItem("b", np.zeros(2), 1.0, 0.125, np.zeros(2), 0.0, 0))
    res.stamp_basis_id(3)
    assert [it.basis_id for it in res.items] == [3, 3]


def test_compat_number_unit_requires_numeric_core():
    assert _compat_number_unit("5", "5cm") is True
    assert _compat_number_unit("5/2", "5/2cm") is True
    assert _compat_number_unit("x", "xy") is False
    assert _compat_number_unit("5c", "5cm") is False
    assert rule_reward(r"Answer: 5", r"5\text{ cm}") == 1.0
    assert rule_reward("Answer: x", "xy") == 0.0


def test_prescan_does_not_consume_actor_token_rng():
    pytest.importorskip("torch")

    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.trainer.algorithm import TrainState, prescan_unseen_baselines
    from grace_gc.trainer.baseline import HistoricalBaseline
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    engines.reward_fn = lambda *a, **k: 1.0
    seen_rng = []

    def gen(prompt_ids, max_new, rng, stream):
        rng.integers(stream, 0, 10, size=len(prompt_ids))
        seen_rng.append(id(rng))
        return [list(p) + [1] for p in prompt_ids], np.zeros(len(prompt_ids), dtype=bool)

    engines.generate_prefix = gen
    state = TrainState(
        spec=method_spec("grace"),
        baseline=HistoricalBaseline(),
        rng=IsolatedRNG.create(0),
        layout=collect_lora_layout(actor.named_lora_params()),
        u=np.eye(2, 2, dtype=np.float64),
        predictor=None,
        reservoir=GradientReservoir(capacity=4),
    )
    before = dict(state.rng.counters)
    n = prescan_unseen_baselines(engines, state, [[1, 2]], ["a"], ["1"], 8, 2)
    assert n == 1
    assert state.rng.counters == before
    assert state.prescan_rng is not None
    assert id(state.prescan_rng) == seen_rng[0]
    assert state.prescan_rng.counters["token"] >= 1


def test_audit_actual_uses_zero_mean_without_predictor():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[4.0, 0.0], [4.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    bundle = PrefixBundle("p", 4, np.ones(4), g, suffix_cost=np.ones(4), r_hat=1.0, c_hat=1.0)
    out = audit_bundles(
        [bundle],
        np.eye(2, 1),
        {"beta": 0.5, "p_min": 0.2, "method": "uniform_ht"},
        np.random.default_rng(0),
    )
    assert "actual_m=0" in str(out["variance_cost_note"])
    assert out["variance_cost"]["extra_var"] != out["variance_cost_oracle"]["extra_var"]


def test_generate_phase_records_sampled_logprobs(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[7, 8],
                            finish_reason="length",
                            logprobs=[{7: SimpleNamespace(logprob=-0.5)}, {8: SimpleNamespace(logprob=-1.5)}],
                        )
                    ],
                )
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    phase = vllm_mod.generate_phase(_LLM(), [[1, 2]], 4, 1.0, 99, IsolatedRNG.create(0), "token")
    assert phase.logprob_sums == [pytest.approx(-2.0)]


def test_build_sampling_params_asks_for_logprobs(monkeypatch):
    import grace_gc.backends.vllm_two_phase as vllm_mod

    seen = {}

    class _SP:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (None, _SP))
    vllm_mod.build_sampling_params(4, 1.0, 1, eos_id=7)
    assert seen["logprobs"] == 1


def test_build_sampling_params_drops_unsupported_logprobs(monkeypatch):
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _SP:
        def __init__(self, **kwargs):
            if "logprobs" in kwargs:
                raise TypeError("logprobs")
            self.kwargs = kwargs

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (None, _SP))
    params = vllm_mod.build_sampling_params(4, 1.0, 1, eos_id=7)
    assert "logprobs" not in params.kwargs
    assert params.kwargs["stop_token_ids"] == [7]


def test_grace_first_step_stays_complete_then_syncs_basis(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.config import default_config, merge_configs
    from grace_gc.logging_util.ledger import ComputeLedger
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.predictor.basis import Basis
    from grace_gc.trainer.loop import run_tiny_training

    def fake_refresh(grads, k, basis_id, **kwargs):
        d = int(np.asarray(grads).shape[1])
        kk = int(k)
        return Basis(u=np.eye(d, kk, dtype=np.float64), basis_id=int(basis_id), k=kk, rank=kk)

    monkeypatch.setattr("grace_gc.trainer.algorithm.refresh_basis", fake_refresh)
    cfg = merge_configs(
        default_config(),
        {
            "method": "grace",
            "num_steps": 2,
            "n_start": 8,
            "n_prompts": 4,
            "decision_tokens": 3,
            "max_new_tokens": 6,
            "predictor": {
                "k": 2,
                "warmup_steps": 0,
                "refresh_every": 32,
                "audit_s": 1.0,
                "reservoir_size": 16,
                "epochs": 1,
            },
        },
    )
    run_tiny_training(cfg, RunDirectory(tmp_path), ComputeLedger(0, "cpu"))
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(steps) == 2
    assert steps[0]["all_p_one"] is True
    assert steps[0]["basis_id_at_allocate"] == 0
    assert steps[0]["allocating_with_init_basis"] is True
    assert steps[1]["basis_id_at_allocate"] == 1
    assert steps[1]["allocation_ready"] is True
    health = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert health["basis_id"] == 1
    assert health["predictor_synced_basis_id"] == 1
    assert health["token_cost_proxy_ratio"] is not None
    assert "mean_actual_response_tokens" in health


def test_environment_records_grpo_identity():
    from grace_gc.versions import collect_environment

    env = collect_environment({"method": "grpo", "predictor": {}})
    assert env["run_knobs"]["grpo_advantage"] == "group_mean_no_std"
    assert env["run_knobs"]["coord_kind"] == "mlp"
    ridge = collect_environment({"method": "grace", "predictor": {"coord_kind": "ridge"}})
    assert ridge["run_knobs"]["coord_kind"] == "ridge"


def test_affine_inverts_and_leaves_phi_untouched():
    from grace_gc.predictor.scale import FeatureScaler

    phi = np.array([[0.0, 10.0], [10.0, 10.0], [20.0, 10.0]], dtype=np.float64)
    stored = phi.copy()
    sc = FeatureScaler(enabled=True)
    sc.fit_features(phi)
    z = sc.transform(phi)
    np.testing.assert_allclose(z[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(z * sc.std + sc.mean, phi)
    np.testing.assert_allclose(phi, stored)
    f = np.array([[2.0, 0.0], [0.0, 4.0]])
    sc.fit_coord_scale(f)
    np.testing.assert_allclose(sc.unscale_coords(sc.scale_coords(f)), f)
    e = np.array([10.0, 30.0])
    sc.fit_risk_scale(e)
    assert sc.risk_scale == pytest.approx(20.0)
    np.testing.assert_allclose(sc.unscale_risk(sc.scale_risk(e)), e)
    off = FeatureScaler(enabled=False)
    off.fit_features(phi)
    np.testing.assert_allclose(off.transform(phi), phi)


def test_align_basis_sign_and_column_swap():
    from grace_gc.predictor.basis import align_basis

    rng = np.random.default_rng(0)
    old, _ = np.linalg.qr(rng.normal(size=(6, 3)))
    new = np.stack([-old[:, 1], old[:, 0], -old[:, 2]], axis=1)
    aligned = align_basis(new, old)
    np.testing.assert_allclose(aligned, old, atol=1e-12)
    padded_old = np.concatenate([old[:, :2], np.zeros((6, 1))], axis=1)
    padded_new = np.concatenate([-old[:, :1], old[:, 1:2], np.zeros((6, 1))], axis=1)
    aligned_pad = align_basis(padded_new, padded_old)
    np.testing.assert_allclose(aligned_pad[:, 0], old[:, 0], atol=1e-12)
    np.testing.assert_allclose(aligned_pad[:, 1], old[:, 1], atol=1e-12)
    np.testing.assert_allclose(aligned_pad[:, 2], 0.0, atol=1e-12)


def test_both_streams_use_the_same_gamma():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads

    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 3))
    heads = PredictorHeads(3, 2, coord_kind="ridge", use_affine=False, shrink_m=True)
    heads.train_coord(x, rng.normal(size=(8, 2)), np.ones(8), epochs=1)
    heads.m_shrink = 0.4
    raw = heads.predict_f(x, shrink=False)
    shrunk = heads.predict_f(x, shrink=True)
    np.testing.assert_allclose(shrunk, 0.4 * raw)
    np.testing.assert_allclose(heads.forward_numpy(x).f, shrunk)
    heads.shrink_m = False
    np.testing.assert_allclose(heads.forward_numpy(x).f, raw)


def test_ridge_coord_recovers_linear_map():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads, predictor_from_spec

    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3))
    weight = rng.normal(size=(3, 2))
    bias = rng.normal(size=(2,))
    y = x @ weight + bias
    heads = predictor_from_spec(
        3,
        2,
        {"coord_kind": "ridge", "ridge_l2": 1e-8, "use_affine": False, "shrink_m": False},
    )
    assert heads.coord_kind == "ridge"
    heads.train_coord(x, y, np.ones(40), epochs=1)
    pred = heads.predict_f(x, shrink=False)
    np.testing.assert_allclose(pred, y, atol=1e-4)
    linear = PredictorHeads(3, 2, coord_kind="linear", use_affine=False, shrink_m=False)
    assert linear.coord_kind == "ridge"


def test_design_shrink_gamma_recovers_and_clips():
    from grace_gc.predictor.scale import design_shrink_gamma

    u = np.eye(3, 2)
    f = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    m = f @ u.T
    p = np.full(3, 0.5)
    assert design_shrink_gamma(0.6 * m, f, u, p) == pytest.approx(0.6)
    assert design_shrink_gamma(5.0 * m, f, u, p) == pytest.approx(2.0)
    assert design_shrink_gamma(m, np.zeros_like(f), u, p) is None
    with pytest.raises(ValueError, match="p must"):
        design_shrink_gamma(m, f, u, np.zeros(3))


def test_duplicate_float_grads_are_zero_rank():
    g = np.array([[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]], dtype=np.float64)
    assert refresh_basis(g, k=2, basis_id=1, seed=0, problem_ids=["a", "a", "a"]) is None
    assert refresh_basis(g, k=2, basis_id=1, seed=0) is None


def test_sync_mark_requires_both_heads_after_basis_change():
    from types import SimpleNamespace

    from grace_gc.trainer.grace_step import maybe_mark_predictor_synced

    spec = method_spec("grace")
    state = SimpleNamespace(spec=spec, basis_id=2, predictor_synced_basis_id=1)
    maybe_mark_predictor_synced(state, {"coord_loss": None, "risk_loss": 0.2}, basis_changed=True)
    assert state.predictor_synced_basis_id == -1
    maybe_mark_predictor_synced(state, {"coord_loss": 0.1, "risk_loss": None}, basis_changed=True)
    assert state.predictor_synced_basis_id == -1
    maybe_mark_predictor_synced(state, {"coord_loss": 0.1, "risk_loss": 0.2}, basis_changed=True)
    assert state.predictor_synced_basis_id == 2
    maybe_mark_predictor_synced(state, {"coord_loss": None, "risk_loss": 0.2}, basis_changed=False)
    assert state.predictor_synced_basis_id == 2


def test_unready_checkpoint_audit_stays_complete():
    from grace_gc.audit.prefix_audit import PrefixBundle, _p_at_decision
    from grace_gc.trainer.grace_step import allocation_ready_from_checkpoint

    spec = method_spec("grace")
    payload = {"step": 0, "basis": {"basis_id": 0, "predictor_synced_basis_id": -1}}
    cfg = {"predictor": {"warmup_steps": 20}}
    assert allocation_ready_from_checkpoint(spec, payload, cfg) is False
    ready = {"step": 21, "basis": {"basis_id": 1, "predictor_synced_basis_id": 1}}
    assert allocation_ready_from_checkpoint(spec, ready, cfg) is True
    assert allocation_ready_from_checkpoint(method_spec("uniform_cv"), payload, cfg) is True
    bundle = PrefixBundle("p", 8, np.ones(2), np.ones((2, 2)), r_hat=8.0, c_hat=1.0)
    blocked = _p_at_decision([bundle], spec, {"beta": 0.5, "p_min": 0.2, "allocation_ready": False}, {}, {})
    opened = _p_at_decision([bundle], spec, {"beta": 0.5, "p_min": 0.2, "allocation_ready": True}, {}, {})
    np.testing.assert_allclose(blocked, np.ones(1))
    assert float(opened[0]) < 1.0 - 1e-12


def test_audit_constant_cost_uses_remaining_tokens():
    from grace_gc.trainer.grace_step import incremental_token_costs

    remain = [1536, 1024, 0]
    finished = [False, False, True]
    c = incremental_token_costs(remain, finished)
    np.testing.assert_allclose(c, [1536.0, 1024.0, 1.0])


def test_gamma_ipw_restores_start_distribution():
    from grace_gc.predictor.ipw import ipw_weights
    from grace_gc.predictor.scale import design_shrink_gamma

    u = np.ones((1, 1))
    f = np.ones((3, 1))
    g = np.array([[0.0], [2.0], [2.0]])
    p = np.array([0.25, 0.5, 0.5])
    s = np.ones(3)
    assert design_shrink_gamma(g, f, u, p) == pytest.approx(0.8)
    assert design_shrink_gamma(g, f, u, p, weights=ipw_weights(p, s)) == pytest.approx(0.5)


def test_missing_sampled_logprob_is_missing():
    from types import SimpleNamespace

    from grace_gc.backends.vllm_two_phase import _sum_sampled_logprobs
    from grace_gc.trainer.algorithm import StepEngines, _rollout_logprob_sum

    wrong_token = SimpleNamespace(token_ids=[7, 8], logprobs=[{9: SimpleNamespace(logprob=-0.5)}, {8: SimpleNamespace(logprob=-1.5)}])
    assert _sum_sampled_logprobs(wrong_token, 2) is None
    partial = SimpleNamespace(token_ids=[7, 8], logprobs=[{7: SimpleNamespace(logprob=-0.5)}, None])
    assert _sum_sampled_logprobs(partial, 2) is None
    engines = StepEngines(
        generate_prefix=None,
        continue_selected=None,
        prefix_features=None,
        logprob_sums=None,
        logprob_one=None,
        named_lora=None,
        trainable_params=None,
        reward_fn=None,
        last_rollout={"prefix_logprob_sums": [None], "continue_logprob_sums": {0: -1.0}},
    )
    assert _rollout_logprob_sum(engines, 0, 1.0) is None
    engines.last_rollout = {"prefix_logprob_sums": [-0.5], "continue_logprob_sums": {0: None}}
    assert _rollout_logprob_sum(engines, 0, 1.0) is None
    engines.last_rollout = {"prefix_logprob_sums": [-0.5], "continue_logprob_sums": {0: -1.5}}
    assert _rollout_logprob_sum(engines, 0, 1.0) == pytest.approx(-2.0)


def test_ledger_total_skips_nested_rows():
    from grace_gc.logging_util.ledger import ComputeLedger

    led = ComputeLedger(n_gpu=1, hardware="cpu")
    led.add("train_step", 10.0)
    led.add("phase_allocate", 4.0)
    led.add("phase_update", 6.0)
    led.add("train", 10.0)
    assert led.summary()["gpu_reserved_seconds"] == pytest.approx(10.0)


def test_incremental_token_costs_use_leftover_tokens():
    from grace_gc.trainer.grace_step import incremental_token_costs

    np.testing.assert_allclose(
        incremental_token_costs([10, 0, 3], [False, True, False]),
        [10.0, 1.0, 3.0],
    )
    np.testing.assert_allclose(incremental_token_costs([0, 5], [False, False]), [1.0, 5.0])
    with pytest.raises(ValueError, match="dimensions"):
        incremental_token_costs([1, 2], [False])
