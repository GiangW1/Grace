import json
from pathlib import Path

import numpy as np

from grace_gc.audit.learning_completion import (analyze, answer_variance_terms, estimate,
                                                position_statistics, stacked_gram)
from scripts.analyze_learning_completion import main


def _simulate(rng, prefixes, n_half=8, noise=0.1, flip=None):
    """Binary-reward continuations with E[G|h] = (q-b) g + c exactly.

    prefixes: list of (problem_id, baseline, q, g, c).  The suffix score is
    (R-q)/(q(1-q)) c + xi, so E[(R-b) g_s | h] = c and the half means are
    correlated with their own reward counts as in the real replay.
    """
    rows, g_rows, a_rows, b_rows = [], [], [], []
    for index, (problem, baseline, q, g, c) in enumerate(prefixes):
        halves, sums = [], []
        for _half in range(2):
            rewards = (rng.random(n_half) < q).astype(np.float64)
            labels = [(r - baseline) * (g + (r - q) / (q * (1 - q)) * c +
                                        noise * rng.standard_normal(g.shape)) for r in rewards]
            halves.append(np.mean(labels, axis=0))
            sums.append(float(rewards.sum()))
        rows.append({"problem_id": problem, "path_id": f"{problem}:{index}", "t": 64,
                     "half_counts": [n_half, n_half], "half_reward_sum": sums,
                     "mean_baseline": baseline, "half_mean_baseline": [baseline, baseline],
                     "answer_emitted": False, "finished": False})
        g_rows.append(g)
        a_rows.append(halves[0])
        b_rows.append(halves[1])
    return rows, np.array(g_rows), np.array(a_rows), np.array(b_rows)


def _stats(rows, g, a, b, view="pooled"):
    stacked = np.concatenate([g, a, b])
    gram = stacked @ stacked.T
    scalars = [{"n_a": r["half_counts"][0], "n_b": r["half_counts"][1],
                "s_a": r["half_reward_sum"][0], "s_b": r["half_reward_sum"][1],
                "baseline": r["mean_baseline"]} for r in rows]
    return position_statistics(gram, scalars, [r["problem_id"] for r in rows], view)


def test_blockwise_gram_matches_dense_metric_gram():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((7, 23))
    metric = rng.random(23)
    expected = (vectors * metric) @ vectors.T
    assert np.allclose(stacked_gram(list(vectors), 23, metric, block_columns=5), expected)
    assert np.allclose(stacked_gram(list(vectors), 23, None, block_columns=23), vectors @ vectors.T)


def test_population_inner_products_are_unbiased_given_prefixes():
    rng = np.random.default_rng(1)
    dim, shared = 5, np.eye(5)[0]
    prefixes = []
    for problem in range(6):
        baseline = 0.3 + 0.07 * problem
        for _ in range(3):
            q = float(np.clip(baseline + 0.3 * rng.standard_normal(), 0.1, 0.9))
            prefixes.append((f"p{problem}", baseline, q, rng.standard_normal(dim),
                             0.4 * shared + 0.2 * rng.standard_normal(dim)))
    problems = sorted({p[0] for p in prefixes})
    p_sum = {x: sum((q - b) * g for pid, b, q, g, c in prefixes if pid == x) for x in problems}
    mu_sum = {x: sum((q - b) * g + c for pid, b, q, g, c in prefixes if pid == x) for x in problems}
    truth_pm = np.mean([p_sum[x] @ mu_sum[y] for x in problems for y in problems if x != y])
    truth_mm = np.mean([mu_sum[x] @ mu_sum[y] for x in problems for y in problems if x != y])
    truth_dpm = np.mean([p_sum[x] @ mu_sum[x] for x in problems])
    truth_dmm = np.mean([mu_sum[x] @ mu_sum[x] for x in problems])
    draws = []
    for _ in range(1500):
        rows, g, a, b = _simulate(rng, prefixes, n_half=4)
        point = estimate(_stats(rows, g, a, b), np.ones(len(problems)))
        draws.append([point[k][0] for k in ("U_PM", "U_MM", "D_PM", "D_MM")])
    draws = np.array(draws)
    mean, error = draws.mean(0), draws.std(0) / np.sqrt(len(draws))
    for value, truth, se in zip(mean, (truth_pm, truth_mm, truth_dpm, truth_dmm), error):
        assert abs(value - truth) < 4.5 * se + 1e-9


def test_population_share_separates_cancelling_prefix_terms_from_aligned_ones():
    rng = np.random.default_rng(2)
    dim, shared = 40, np.eye(40)[0]
    cancelling, aligned = [], []
    for problem in range(150):
        for _ in range(2):
            g = rng.standard_normal(dim)
            q_free = float(np.clip(0.5 + 0.3 * rng.standard_normal(), 0.1, 0.9))
            cancelling.append((f"p{problem}", 0.5, q_free, g, 0.3 * shared))
            q_tied = float(0.5 + 0.35 * np.tanh(g @ shared))
            aligned.append((f"p{problem}", 0.5, q_tied, g, np.zeros(dim)))
    first = estimate(_stats(*_simulate(rng, cancelling)), np.ones(150))
    assert first["F_prefix"][0] > 0.9
    assert first["F_pop"][0] < 0.3 and first["residual_pop"][0] > 0.7
    second = estimate(_stats(*_simulate(rng, aligned)), np.ones(150))
    assert second["F_pop"][0] > 0.85 and second["residual_pop"][0] < 0.15


def test_answer_variance_terms_are_unbiased_for_binary_rewards():
    rng = np.random.default_rng(3)
    within_total, total_total = 0.0, 0.0
    for _ in range(40):
        scalars, ids = [], []
        for problem in range(100):
            for _ in range(3):
                q = rng.beta(2.0, 3.0)
                s_a, s_b = rng.binomial(4, q), rng.binomial(4, q)
                scalars.append({"n_a": 4, "n_b": 4, "s_a": float(s_a), "s_b": float(s_b),
                                "baseline": 0.4})
                ids.append(f"p{problem}")
        within, total = answer_variance_terms(scalars, ids, sorted(set(ids)), "pooled")
        within_total += within.sum()
        total_total += total.sum()
    # Beta(2,3): E q(1-q) = 0.2, Var within the problem = 0.24.
    assert abs(within_total / total_total - 0.2 / 0.24) < 0.02


def test_bootstrap_never_pairs_a_problem_with_itself(tmp_path: Path):
    rng = np.random.default_rng(4)
    rows, g_rows, a_rows, b_rows = [], [], [], []
    for problem in range(5):
        for path in range(3):
            g, a, b = (np.zeros(10) for _ in range(3))
            for vector in (g, a, b):
                vector[2 * problem:2 * problem + 2] = rng.standard_normal(2) + 3.0
            rows.append({"problem_id": f"p{problem}", "path_id": f"p{problem}:{path}", "t": 32,
                         "half_counts": [2, 2], "half_reward_sum": [1.0, 2.0],
                         "mean_baseline": 0.5, "answer_emitted": False})
            g_rows.append(g)
            a_rows.append(a)
            b_rows.append(b)
    stacked = np.concatenate([g_rows, a_rows, b_rows])
    report = analyze(rows, {32: stacked @ stacked.T}, bootstrap=200, seed=5)
    entry = report["groups"]["all"]["by_t"][0]
    assert entry["estimates"]["U_MM"] == 0.0 and entry["estimates"]["D_MM"] > 0.0
    assert entry["ci95"]["U_MM"] == [0.0, 0.0]


def test_half_b_view_ignores_half_a_vectors():
    rng = np.random.default_rng(6)
    prefixes = [(f"p{x}", 0.5, 0.5, rng.standard_normal(6), 0.2 * rng.standard_normal(6))
                for x in range(4) for _ in range(3)]
    rows, g, a, b = _simulate(rng, prefixes)
    first = estimate(_stats(rows, g, a, b, "half_b"), np.ones(4))
    second = estimate(_stats(rows, g, rng.standard_normal(a.shape), b, "half_b"), np.ones(4))
    for key in ("U_PM", "U_MM", "U_PP", "rho_A"):
        assert np.isclose(first[key][0], second[key][0])
    assert np.isnan(first["F_prefix"][0])


def _write_shard(root: Path, rows, g, a, b):
    root.mkdir(parents=True)
    (root / "prefixes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    np.save(root / "prefix_score_gradients.npy", g[:, :, None])
    np.save(root / "half_mean_full_grads_a.npy", a)
    np.save(root / "half_mean_full_grads_b.npy", b)


def test_cli_combines_shards_and_reuses_cached_grams(tmp_path: Path):
    rng = np.random.default_rng(7)
    shards = []
    for shard in range(2):
        rows, gs, as_, bs = [], [], [], []
        for t in (32, 64):
            prefixes = [(f"s{shard}p{x}", 0.5, float(rng.uniform(0.2, 0.8)), rng.standard_normal(9),
                         0.1 * rng.standard_normal(9)) for x in range(3) for _ in range(2)]
            part, g, a, b = _simulate(rng, prefixes, n_half=4)
            for row in part:
                row["t"] = t
                row["path_id"] = f"{row['path_id']}:{t}"
            rows += part
            gs.append(g)
            as_.append(a)
            bs.append(b)
        root = tmp_path / f"shard{shard}"
        _write_shard(root, rows, np.concatenate(gs), np.concatenate(as_), np.concatenate(bs))
        shards.append((root, rows))
    probe = tmp_path / "probe.jsonl"
    probe.write_text("".join(json.dumps({"problem_id": r["problem_id"], "path_id": r["path_id"],
                                         "t": r["t"], "functional_recoverable": i % 3 == 0}) + "\n"
                             for _root, rows in shards for i, r in enumerate(rows)), encoding="utf-8")
    common = ["--replay-dir", *(str(root) for root, _ in shards), "--probe-rescores", str(probe),
              "--gram-cache", str(tmp_path / "cache"), "--bootstrap", "50",
              "--epsilon", "0.1", "--delta", "0.5"]
    assert main([*common, "--run-dir", str(tmp_path / "first")]) == 0
    first = json.loads((tmp_path / "first" / "learning_completion_summary.json").read_text())
    for root, _rows in shards:
        for name in ("prefix_score_gradients.npy", "half_mean_full_grads_a.npy",
                     "half_mean_full_grads_b.npy"):
            (root / name).unlink()
    assert main([*common, "--run-dir", str(tmp_path / "second")]) == 0
    second = json.loads((tmp_path / "second" / "learning_completion_summary.json").read_text())
    assert first["groups"] == second["groups"]
    assert first["positions"] == [32, 64] and first["n_problems"] == 6
    assert first["groups"]["functional_pre_answer"]["available"]
    assert set(first["groups"]["all"]["crossings"]) >= {"t_L_projection", "t_L_residual", "t_A"}
    assert first["groups"]["all"]["by_t"][0]["n_problems"] == 6
