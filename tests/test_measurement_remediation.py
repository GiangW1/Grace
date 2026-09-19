"""Regression checks for measured failures; no GPU results or sample-size gates."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit.prefix_audit import (
    PrefixBundle, audit_bundles, bundle_from_dict, bundle_to_dict,
    independent_report_statistics, jl_project, orthogonal_energy,
)
from grace_gc.audit.stats import rho_ucb
from grace_gc.data.math_data import MathRecord, load_math_records, select_records, selection_manifest
from grace_gc.data.reward import rule_reward


def test_case_sensitive_dedup_does_not_drop_distinct_math_questions(tmp_path):
    rows = [{"prompt": "Find A if A=1 and a=2.", "answer": "1"},
            {"prompt": "Find a if A=1 and a=2.", "answer": "2"},
            {"prompt": " Find A  if A=1 and a=2. ", "answer": "1"}]
    path = tmp_path / "math.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loaded = load_math_records(path)
    assert len(loaded) == 2
    assert {r.answer for r in loaded} == {"1", "2"}
    assert load_math_records.last["n_conflict_groups"] == 0


def test_compatibility_does_not_reward_mentions_or_algebra(monkeypatch):
    monkeypatch.setattr("grace_gc.data.reward.math_verify_fns", lambda: None)
    assert rule_reward("Evelyn was considered, but Carla is the fastest.", r"\text{Evelyn}") == 0
    assert rule_reward("Answer: 5x", "5") == 0
    assert rule_reward("Answer: X", "x") == 0
    assert rule_reward("The girl is Evelyn.", r"\text{Evelyn}") == 1
    assert rule_reward("Answer: 5", r"5\text{ cm}") == 1
    assert rule_reward(None, "0") is None


def test_windows_symbolic_worker_is_bounded_and_failure_not_a_miss(monkeypatch):
    from grace_gc.data import reward
    reward._windows_verify.cache_clear()
    seen = {}
    def run(args, **kwargs):
        seen.update(kwargs)
        assert args[0] and "parsing_timeout=None" in args[-1]
        return SimpleNamespace(stdout="true\n")
    monkeypatch.setattr(reward.subprocess, "run", run)
    assert reward._windows_verify("0.5", r"\frac{1}{2}")
    assert seen["timeout"] > 0 and seen["check"] and seen["capture_output"]
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("symbolic worker", 15)
    reward._windows_verify.cache_clear()
    monkeypatch.setattr(reward.subprocess, "run", timeout)
    monkeypatch.setattr(reward, "math_verify_fns", lambda: (str, reward._windows_verify))
    with pytest.raises(ValueError, match="not a scored miss"):
        rule_reward("Answer: 0.5", r"\frac{1}{2}")


def test_countsketch_preserves_one_hot_energy_and_general_span_projection():
    g = np.zeros((2, 40000)); g[0, 3] = 1; g[1, 30000] = 2
    sketched = jl_project(g, 256, 17)
    np.testing.assert_allclose(np.sum(sketched**2, axis=1), [1, 4])
    u = np.array([[2., 4.], [0., 0.], [0., 0.]])  # scaled and rank deficient
    assert orthogonal_energy(np.array([[3., 0., 0.]]), u) == pytest.approx(0, abs=1e-12)
    assert orthogonal_energy(np.array([[3., 4., 0.]]), u) == pytest.approx(16)


def test_missing_basis_does_not_invent_capture_or_replace_saved_prediction():
    from grace_gc.audit.run import _audit_u
    b = PrefixBundle("p", 2, np.array([0., 1.]), np.array([[1., 2.], [3., 4.]]),
                     coords=np.ones(8), m_pred=np.array([2., 3.]))
    u = _audit_u([b], {})
    assert u.shape == (2, 0)
    result = audit_bundles([b], u, {}, np.random.default_rng(0))
    assert result["orthogonal_energy"] is None
    assert result["residual_mean"] == pytest.approx(2)


def test_cost_matched_replays_preserve_cost_and_label_hindsight():
    from grace_gc.audit.prefix_audit import _allocation_diagnostics
    from grace_gc.audit.variance_cost import variance_times_cost
    g = np.array([[8., 0.], [-8., 0.], [1., 0.], [-1., 0.]])
    m = np.zeros_like(g)
    p = np.array([.25, .25, 1., 1.])
    prefix = np.ones(4)
    suffix = np.array([2., 4., 1., 1.])
    bundles = [PrefixBundle("a", 4, np.ones(2), g[:2], r_hat=1., c_hat=4.),
               PrefixBundle("b", 4, np.ones(2), g[2:], r_hat=9., c_hat=4.)]
    base = variance_times_cost(g, m, p, prefix, suffix)
    diagnostics = _allocation_diagnostics(bundles, [0, 0, 1, 1], g, m, p, prefix, suffix, .2)
    assert diagnostics["m_zero_same_p"]["ratio"] == pytest.approx(base["ratio"])
    for name in ("actual_m_uniform_same_cost", "pred_r_observed_c", "observed_r_pred_c", "observed_r_observed_c"):
        assert diagnostics[name]["actual_p_cost"] == pytest.approx(base["actual_p_cost"], abs=1e-10)
    assert diagnostics["observed_r_observed_c"]["ratio"] <= diagnostics["actual_m_uniform_same_cost"]["ratio"]
    assert "retrospective" in diagnostics["hindsight_note"]


def test_report_half_cannot_change_supplement_selection_or_denominators():
    g = np.array([[1., 0.], [2., 0.], [1., 0.], [2., 0.]] * 2)
    b = PrefixBundle("p", 8, np.array([0., 1., 0., 1.] * 2), g, path_id="p:0")
    analysis = {"nonzero_signal_kappa": 0., "wilson_lo": 0., "wilson_hi": 1.}
    before = independent_report_statistics([b], analysis)
    changed = replace(b, grads=np.concatenate([g[:4], g[4:] * 10]),
                      rewards=np.concatenate([b.rewards[:4], np.zeros(4)]))
    after = independent_report_statistics([changed], analysis)
    assert before["selected"] == after["selected"] and before["n_selected"] == 1
    assert after["curve"][0]["rho_l"] == pytest.approx(before["curve"][0]["rho_l"] * 100)
    assert after["curve"][0]["rho_a"] == 0
    assert np.isnan(rho_ucb([.1]))
    assert rho_ucb([.1, .2]) > .15
    small = independent_report_statistics([replace(b, grads=g[:4], rewards=b.rewards[:4])], analysis)
    assert small["n_unsplit_bundles"] == 1 and small["n_selected"] == 0


def test_seeded_selection_and_manifest_are_method_and_order_independent():
    rows = [MathRecord(str(i), f"Question {i}", str(i)) for i in range(32)]
    first = select_records(rows, 8, "seeded", 17)
    second = select_records(list(reversed(rows)), 8, "seeded", 17)
    assert [r.problem_id for r in first] == [r.problem_id for r in second]
    assert [r.problem_id for r in first] != [r.problem_id for r in rows[:8]]
    assert selection_manifest(first, "seeded", 17) == selection_manifest(second, "seeded", 17)
    assert selection_manifest(first, "seeded", 17)["ordered_records_sha256"] != selection_manifest(
        [replace(first[0], answer="changed"), *first[1:]], "seeded", 17)["ordered_records_sha256"]


def test_eval_uncertainty_has_the_right_target():
    from grace_gc.evaluation.eval_full import EvalItem, evaluate_items
    result = evaluate_items([EvalItem("a", "1", ["Answer: 1", "Answer: 2"], [False]*2),
                             EvalItem("b", "1", ["Answer: 1"]*2, [False]*2)], k=2)
    assert result["avg"] == .75 and result["pass_at_k"] == 1
    assert result["wilson"]["target"] == "problem_any_success"
    assert result["wilson"]["n_problems"] == 2
    assert result["avg_interval"]["resampling_unit"] == "problem"
    assert result["parse_rate"] == 1


def test_audit_records_are_rescorable_and_full_energy_precedes_sketch(monkeypatch):
    from grace_gc.audit import run
    from grace_gc.core.rng import IsolatedRNG
    class Engine:
        eos_id = None
        last_rollout = {}
        def decode(self, ids): return "Answer: 1"
        def generate_prefix(self, prompts, max_new, rng, stream):
            seeds = [int(rng.integers(stream, 0, 10000)) for _ in prompts]
            self.last_rollout = {"sampling": {"request_seeds": seeds},
                                 "prefix_finish_reasons": ["length"]*len(prompts)}
            return [p + [3]*max_new for p in prompts], np.zeros(len(prompts), bool)
        def continue_selected(self, prefixes, selected, max_new, rng):
            seed = int(rng.integers("continuation", 0, 10000))
            self.last_rollout = {"sampling": {"request_seeds": [seed]},
                                 "continue_finish_reasons": {0: "length"}}
            return [prefixes[0] + [4]*max_new]
    monkeypatch.setattr(run, "_policy_grad_vec", lambda *a: np.array([3., 4., 0.]))
    encode = lambda rec, n: ([[1, 2] for _ in range(n)], [rec.problem_id]*n, [rec.answer]*n)
    bundles = run._bundles_from_engines([MathRecord("p", "q", "1")], Engine(),
        SimpleNamespace(dim=3), 1, 8, [1, 2], 4, 17, encode, n_baseline=2)
    assert len(bundles) == 2 and len(bundles[0].baseline_samples) == 2
    for bundle in bundles:
        restored = bundle_from_dict(json.loads(json.dumps(bundle_to_dict(bundle))))
        assert restored.gold == "1" and restored.baseline == 1
        assert restored.prefix_request_seed is not None
        assert restored.true_grad_norm_sq == [25.]*8
        for raw in restored.continuation_records:
            assert raw["seed"] is not None and raw["finish_reason"] == "length"
            assert raw["reward"] == rule_reward(raw["text"], raw["gold"], raw["truncated"])
            assert raw["token_ids"][:len(restored.prefix_token_ids)] == restored.prefix_token_ids
    assert len(run._bundles_from_engines.last[0]["paths"]) == 1


def test_failed_eval_records_elapsed_cpu_and_wall(tmp_path, monkeypatch):
    from grace_gc.evaluation import generate
    monkeypatch.setattr(generate, "collect_environment", lambda cfg: {})
    def fail(*args): raise RuntimeError("intentional generator failure")
    monkeypatch.setattr(generate, "_generate_eval_items", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        generate.run_eval([MathRecord("p", "q", "1")], {"backend": "cpu_tiny"}, tmp_path / "eval")
    ledger = json.loads((tmp_path / "eval/compute_ledger.json").read_text())
    row = ledger["rows"][-1]
    assert row["status"] == "failed" and row["wall_seconds"] > 0
    assert row["cpu_seconds"] >= 0


def test_eval_batches_requests_without_reusing_or_reordering_seeds(monkeypatch):
    from grace_gc.backends import vllm_two_phase as vllm
    from grace_gc.data import tokenize
    from grace_gc.evaluation.generate import generate_answers_vllm
    class Sampling:
        def __init__(self, **kwargs): self.kwargs = kwargs
    seen = []
    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            seen.append([p.kwargs["seed"] for p in sampling_params])
            return [SimpleNamespace(prompt_token_ids=prompt,
                    outputs=[SimpleNamespace(token_ids=[p.kwargs["seed"]]*8, finish_reason="length")])
                    for prompt, p in zip(prompts, sampling_params)]
    monkeypatch.setattr(vllm, "_require_vllm", lambda: (LLM, Sampling))
    monkeypatch.setattr(vllm, "_vllm_prompts", lambda ids: ids)
    monkeypatch.setattr(tokenize, "encode_records_hf", lambda recs, tok, mx, **kw:
                        ([[1, 2]]*len(recs), ["p"]*len(recs), ["1"]*len(recs)))
    monkeypatch.setattr(tokenize, "decode_hf", lambda tok, ids: str(ids[0]))
    monkeypatch.setattr(tokenize, "collect_stop_token_ids", lambda tok: [99])
    result = generate_answers_vllm([MathRecord("p", "q", "1")], None, LLM(), 4, 8, .6, .95, 17,
                                  sample_batch_size=4)
    assert seen == [[17, 18, 19, 20]]
    assert result[0].sample_seeds == [17, 18, 19, 20]
    assert result[0].answers == ["17", "18", "19", "20"]


def test_tiny_audit_persists_replay_artifacts_and_success_cost(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.audit import run
    monkeypatch.setattr(run, "collect_environment", lambda cfg: {})
    cfg = {"backend": "cpu_tiny", "seed": 17, "max_new_tokens": 4, "decision_tokens": 2,
           "audit": {"n_prefixes": 1, "n_continuations": 8, "n_baseline": 2,
                     "selection": "seeded", "decision_grid": [2]}}
    run.run_audit([MathRecord("p", "1+1", "2")], cfg, tmp_path / "audit")
    raw = json.loads((tmp_path / "audit/audit_raw_problems.jsonl").read_text().splitlines()[0])
    assert raw["gold"] == "2" and len(raw["baseline_samples"]) == 2
    assert raw["prefix_rng_state"] and raw["baseline_rng_state"]
    manifest = json.loads((tmp_path / "audit/audit_manifest.json").read_text())
    assert manifest["reward_protocol_version"] == 2 and manifest["selection"] == "seeded"
    ledger = json.loads((tmp_path / "audit/compute_ledger.json").read_text())
    assert ledger["rows"][-1]["status"] == "completed"
    assert ledger["cpu_seconds"] >= 0 and ledger["gpu_reserved_seconds"] == 0


@pytest.mark.parametrize("stage,split", [("evaluate", "eval"), ("audit", "audit")])
def test_generation_cli_accepts_method_override(stage, split, tmp_path, monkeypatch):
    import importlib
    from grace_gc.data import math_data
    from grace_gc.trainer import loop
    entry = importlib.import_module(f"scripts.{stage}")
    seen = {}
    def config(paths, overrides):
        seen.update(overrides)
        return overrides
    monkeypatch.setattr(loop, "build_run_config", config)
    if stage == "audit":
        monkeypatch.setattr(entry, "build_run_config", config)
    monkeypatch.setattr(math_data, "load_math_records", lambda p: [MathRecord("p", "q", "1")])
    monkeypatch.setattr(math_data, "records_for_split", lambda recs, *a, **kw: recs)
    module = importlib.import_module("grace_gc.evaluation.generate" if stage == "evaluate" else "grace_gc.audit.run")
    def generate(recs, cfg, run_dir):
        assert cfg["method"] == "full_pg"
        return {"n_problems": 1, "avg": 1., "pass_at_k": 1., "parse_rate": 1., "truncate_rate": 0.}
    monkeypatch.setattr(module, "run_eval" if stage == "evaluate" else "run_audit", generate)
    assert entry.main(["--generate", "--method", "full_pg", "--data-path", "fixture.jsonl",
                       "--split", split, "--run-dir", str(tmp_path)]) == 0
    assert seen["method"] == "full_pg"


def test_optional_september17_archive_retains_original_rho_and_variance_cost():
    """Read-only real-artifact regression; skipped when the local archive is absent."""
    import yaml
    root = Path(__file__).resolve().parents[1] / "_minimal_review_20260917_071528" / (
        "minimal-chain-20260917-071528-completed-fullpg-grace/grace/audit")
    if not (root / "audit_bundles.jsonl").is_file():
        pytest.skip("September 17 local archive is not shipped with source")
    bundles = [bundle_from_dict(json.loads(line)) for line in
               (root / "audit_bundles.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    original = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    analysis = dict(cfg["analysis"], beta=cfg["allocation"]["beta"], p_min=cfg["allocation"]["p_min"],
                    method="grace", max_new_tokens=4096, allocation_ready=True)
    result = audit_bundles(bundles, np.zeros((bundles[0].grads.shape[1], 0)), analysis,
                           np.random.default_rng(17))
    assert result["variance_cost"]["ratio"] == pytest.approx(2.321800041853085)
    for field in ("rho_l_curve", "rho_a_curve", "rho_l_curve_all", "rho_a_curve_all"):
        np.testing.assert_allclose(result[field], original[field], equal_nan=True)
    replay = result["variance_cost_diagnostics"]
    expected = {"m_zero_same_p": 2.2621540059184366, "actual_m_uniform_same_cost": 1.6442526519196208,
                "pred_r_observed_c": 2.7022159828884265, "observed_r_pred_c": 1.2128383394724918,
                "observed_r_observed_c": 1.1633596263649626}
    for name, ratio in expected.items():
        assert replay[name]["ratio"] == pytest.approx(ratio)
        assert replay[name]["actual_p_cost"] == pytest.approx(original["variance_cost"]["actual_p_cost"])
