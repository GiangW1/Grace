"""Synchrony estimators against exact enumeration on a synthetic policy (CPU)."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from grace_gc.audit.streaming_replay import replay_statistics
from grace_gc.audit.synchrony import analyze, best_allocation_ratio, gain_ratio, prefix_stats

D = 4
G_H = [np.array([1., 0., 0., 0.]), np.array([0., 2., 0., 0.])]   # prefix scores
Q = [0.3, 0.6]                                                    # decision success per prefix type
DIRECTION = np.array([0., 0., 1., 0.])                            # decision-token direction
NOISE = [np.array([0., 0., 0., 1.]), np.array([0., 0., 0., -1.]),
         np.array([0., 0., .5, 0.]), np.array([0., 0., -.5, 0.])]  # zero-mean suffix noise
B = 0.5


def _score(kind, a, c):
    return G_H[kind] + (a - Q[kind]) * DIRECTION + NOISE[c]


def _truth(kind):
    q, outcomes = Q[kind], []
    for a, c in itertools.product((0, 1), range(len(NOISE))):
        prob = (q if a else 1 - q) / len(NOISE)
        outcomes.append((prob, (a - B) * _score(kind, a, c)))
    mu = sum(p * g for p, g in outcomes)
    second = sum(p * g @ g for p, g in outcomes)
    reward_only = (q - B) * G_H[kind]
    return {"mu": mu, "s": second, "mu2": mu @ mu, "r_oracle": second - mu @ mu,
            "c_energy": float((mu - reward_only) @ (mu - reward_only)),
            "r_reward": sum(p * (g - reward_only) @ (g - reward_only) for p, g in outcomes)}


def _rows(n_problems, n_cont, rng, iid_kinds=False):
    """Two fixed prefix types per problem, or iid prefix types (an on-policy draw)."""
    rows = []
    for pid in range(n_problems):
        kinds = rng.integers(2, size=4) if iid_kinds else (0, 1)
        for path, kind in enumerate(kinds):
            kind = int(kind)
            records = []
            for _ in range(n_cont):
                a = int(rng.random() < Q[kind])
                c = int(rng.integers(len(NOISE)))
                records.append({"token_ids": [9, 10 + kind, a, 2 + c], "prompt_len": 1,
                                "reward": float(a), "baseline": B})
            rows.append({"problem_id": f"p{pid}", "path_id": f"p{pid}:{path}", "t": 512,
                         "baseline": B, "prompt_token_ids": [9], "prefix_token_ids": [9, 10 + kind],
                         "answer_emitted": False, "finished": False, "continuation_records": records})
    return rows


def _replay(tmp_path, n_problems=600, n_cont=16, seed=5, iid_kinds=False):
    rows = _rows(n_problems, n_cont, np.random.default_rng(seed), iid_kinds)
    score = lambda tokens, _plen: _score(tokens[1] - 10, tokens[2], tokens[3] - 2)
    replay_statistics(rows, score, lambda row: G_H[row["prefix_token_ids"][1] - 10], D,
                      tmp_path, max_continuations=n_cont, seed=1)
    return [json.loads(line) for line in (tmp_path / "trajectory_scalars.jsonl").read_text().splitlines()]


def test_prefix_estimators_are_unbiased(tmp_path: Path):
    rows = _replay(tmp_path)
    for kind in (0, 1):
        truth = _truth(kind)
        mine = [prefix_stats(row) for row in rows if row["prefix_token_ids"][1] == 10 + kind]
        for key in ("s", "mu2", "r_oracle", "c_energy", "r_reward"):
            est = float(np.mean([m[key] for m in mine]))
            assert est == pytest.approx(truth[key], abs=0.03 + 0.03 * abs(truth[key])), key
        assert np.mean([m["var_r"] for m in mine]) == pytest.approx(Q[kind] * (1 - Q[kind]), abs=0.01)


def test_problem_cross_recovers_population_curves(tmp_path: Path):
    # Cross-prefix estimators assume same-problem prefixes are iid policy draws.
    _replay(tmp_path, n_problems=800, iid_kinds=True)
    report = analyze([tmp_path], bootstrap=20)
    row = report["positions"]["512"]
    t0, t1 = _truth(0), _truth(1)
    mean_g = .5 * (t0["mu"] + t1["mu"])
    v_x = .5 * (t0["s"] + t1["s"]) - mean_g @ mean_g
    q_x = np.mean(Q)
    assert row["rho_L"] == pytest.approx(.5 * (t0["r_oracle"] + t1["r_oracle"]) / v_x, rel=0.05)
    assert row["rho_A"] == pytest.approx(np.mean([q * (1 - q) for q in Q]) / (q_x * (1 - q_x)), rel=0.05)
    assert row["problems_missing_cross"] == 0
    assert abs(row["mean_suffix_prefix_cosine"]) < 0.05  # suffix noise is orthogonal to g_h here
    assert row["ci95"]["rho_L"][0] <= row["rho_L"] <= row["ci95"]["rho_L"][1]


def test_gain_ratio_matches_closed_form_and_needs_alpha_above_gamma():
    assert gain_ratio(0.05, 0.2) == 1.0
    assert gain_ratio(0.5, 0.2) == pytest.approx((np.sqrt(.1) + np.sqrt(.4)) ** 2)


def test_allocation_never_beats_full_with_homogeneous_irreducible_risk():
    risk = np.full(50, 4.0)          # oracle m explains nothing: r = V
    c0, c = np.full(50, 1.0), np.full(50, 3.0)
    assert best_allocation_ratio(risk, risk, c0, c, v_full=4.0)["ratio"] == pytest.approx(1.0)
    cheap = np.r_[np.full(25, 0.1), np.full(25, 7.9)]  # heterogeneous risk with the same mean
    assert best_allocation_ratio(cheap, cheap, c0, c, v_full=4.0)["ratio"] < 1.0


def test_tokens_only_audit_keeps_tokens_and_skips_gradients(tmp_path: Path):
    pytest.importorskip("torch")
    from grace_gc.audit.run import run_audit
    from grace_gc.data.math_data import MathRecord

    cfg = {"backend": "cpu_tiny", "n_prefixes": 2, "n_cont": 4, "decision_tokens": 2,
           "max_new_tokens": 4, "seed": 2, "audit": {"tokens_only": True}}
    run_audit([MathRecord("p", "1+1", "2")], cfg, tmp_path / "audit")
    lines = (tmp_path / "audit" / "audit_bundles.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        bundle = json.loads(line)
        assert bundle["true_grad_norm_sq"] is None
        assert len(bundle["continuation_records"]) == 4
        assert all(rec["token_ids"] for rec in bundle["continuation_records"])


def test_settled_problems_are_skipped_and_logged(tmp_path: Path):
    pytest.importorskip("torch")
    from grace_gc.audit import run as audit_run
    from grace_gc.data.math_data import MathRecord

    cfg = {"backend": "cpu_tiny", "n_prefixes": 2, "n_cont": 2, "decision_tokens": 2, "max_new_tokens": 4,
           "seed": 2, "audit": {"tokens_only": True, "skip_settled_problems": True, "n_baseline": 3}}
    audit_run.run_audit([MathRecord("p", "1+1", "2")], cfg, tmp_path / "audit")
    raw = [json.loads(line) for line in
           (tmp_path / "audit" / "audit_raw_problems.jsonl").read_text(encoding="utf-8").splitlines()]
    rewards = {row["reward"] for row in raw[0]["baseline_samples"]}
    assert (raw[0].get("skipped") == "settled_baseline") == (len(rewards) == 1)
