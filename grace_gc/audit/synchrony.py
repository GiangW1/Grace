"""Update/answer synchrony analysis from streaming replay scalars (CPU only).

Inputs are ``trajectory_scalars.jsonl`` and ``problem_cross.jsonl`` written by
``streaming_replay.replay_statistics``. Per prefix h with A/B continuation
halves, every quantity below is unbiased because it pairs independent halves:

* ``s``       E||G||^2 | h                      (half second moments)
* ``mu2``     ||mu(h)||^2, mu = E[G|h]           (<Gbar_A, Gbar_B>)
* ``r_oracle`` tr Cov(G|h), residual of m = mu   (s - mu2)
* ``c_energy`` ||mu - (q-b) g_h||^2              (suffix cross term)
* ``r_reward`` E||G - (q-b) g_h||^2              (r_oracle + c_energy)
* ``r_free``   residual of the best scalar multiple of g_h

Problem-level ||E[G|x]||^2 and q_x^2 come from different prefixes of the same
problem (independent starts). Curves are ratios of sums over problems with at
least two live prefixes; the problem is the bootstrap unit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _var_r(k: float, n: int) -> float:
    return 0.0 if n < 2 else k * (n - k) / (n * (n - 1))


def prefix_stats(row: dict, metric: str = "euclidean") -> dict:
    stats = row["trajectory_statistics"][metric]
    n_a, n_b = (int(x) for x in row["half_counts"])
    q_a, q_b = (float(x) for x in row["half_mean_reward"])
    b = float(row["baseline"])
    n = n_a + n_b
    hh, ga, gb, ab = (float(stats[key]) for key in ("hh", "ga", "gb", "ab"))
    s = (n_a * float(stats["s_a"]) + n_b * float(stats["s_b"])) / n
    cross = (q_a - b) * gb + (q_b - b) * ga
    c_energy = ab - cross + (q_a - b) * (q_b - b) * hh
    k = n_a * q_a + n_b * q_b
    samples = row.get("samples") or []
    suffix = float(np.mean([s_["suffix_tokens"] for s_ in samples])) if samples else 0.0
    prefix = len(row.get("prefix_token_ids") or []) - len(row.get("prompt_token_ids") or [])
    # Martingale check: <g_s, g_h> from <G, g_h> = (R-b)(||g_h||^2 + <g_s, g_h>).
    cosines = []
    for sample in samples:
        adv = float(sample["reward"]) - b
        m = sample["metrics"][metric]
        if adv != 0.0 and hh > 0 and m["suffix_score_norm_sq"] > 0:
            dot = m["g_dot_prefix"] / adv - hh
            cosines.append(dot / math.sqrt(hh * m["suffix_score_norm_sq"]))
    return {
        "problem_id": str(row["problem_id"]), "t": row.get("t"), "baseline": b, "n": n,
        "q": k / n, "var_r": _var_r(k, n), "s": s, "mu2": ab, "r_oracle": s - ab,
        "c_energy": c_energy, "r_reward": s - ab + c_energy,
        "r_free": s - (ga * gb / hh if hh > 0 else 0.0), "hh": hh,
        "prefix_tokens": max(prefix, 0), "suffix_tokens": suffix,
        "settled": k in (0, n), "suffix_prefix_cosine": float(np.mean(cosines)) if cosines else None,
        "answer_emitted": bool(row.get("answer_emitted", False)),
    }


def problem_table(prefixes: list[dict], cross: dict) -> dict:
    """Group prefixes by problem; attach ||E[G|x]||^2 and Var(R|x) estimates."""
    groups: dict[str, list[dict]] = {}
    for row in prefixes:
        groups.setdefault(row["problem_id"], []).append(row)
    out = {}
    for pid, rows in groups.items():
        k = len(rows)
        if k < 2:
            continue
        q = np.array([r["q"] for r in rows])
        q2 = (q.sum() ** 2 - np.dot(q, q)) / (k * (k - 1))
        # The cross sum covers the replay's prefix set; a filtered subset cannot
        # reuse it, so V_x falls back to E||G||^2 (an upper bound) there.
        matched = pid in cross and cross[pid][1] == k
        mean_g2 = cross[pid][0] / (k * (k - 1)) if matched else 0.0
        out[pid] = {"rows": rows, "k": k, "var_r_x": float(q.mean() - q2),
                    "v_x": float(np.mean([r["s"] for r in rows]) - mean_g2),
                    "mean_g2": float(mean_g2), "has_cross": matched}
    return out


def gain_ratio(alpha: float, gamma: float) -> float:
    """Best variance x cost ratio of HT completion with homogeneous r and c."""
    if not (alpha > gamma):
        return 1.0
    return (math.sqrt(alpha * gamma) + math.sqrt((1 - alpha) * (1 - gamma))) ** 2


def costs(rows: list[dict], w_train: float, w_prefill: float) -> tuple[np.ndarray, np.ndarray]:
    """c0: prefix decode + decision prefill. c: suffix decode + full train pass."""
    pre = np.array([r["prefix_tokens"] for r in rows], dtype=np.float64)
    suf = np.array([r["suffix_tokens"] for r in rows], dtype=np.float64)
    return pre * (1.0 + w_prefill), suf + w_train * (pre + suf)


def best_allocation_ratio(risk, evaluated, c0, c, v_full, p_min=0.05, grid=200) -> dict:
    """min over lambda of V_new C_new / (V C) with clipped Neyman p from ``risk``.

    ``evaluated`` is the residual actually paid for skipping (the control
    variate's residual); ``risk`` only shapes p. In-sample oracle risk is an
    optimistic ceiling.
    """
    risk = np.maximum(np.asarray(risk, dtype=np.float64), 0.0)
    evaluated = np.maximum(np.asarray(evaluated, dtype=np.float64), 0.0)
    c = np.maximum(np.asarray(c, dtype=np.float64), 1e-12)
    full = v_full * float(np.mean(c0 + c))
    best = {"ratio": 1.0, "mean_p": 1.0}
    scale = float(np.mean(risk / c)) or 1.0
    for lam in np.logspace(-4, 4, grid) * scale:
        p = np.clip(np.sqrt(risk / (lam * c)), p_min, 1.0)
        ratio = (v_full + np.mean((1 / p - 1) * evaluated)) * float(np.mean(c0 + p * c)) / full
        if ratio < best["ratio"]:
            best = {"ratio": float(ratio), "mean_p": float(p.mean())}
    return best


def _q_bin_risk(rows: list[dict], values: np.ndarray, bins: int = 5) -> np.ndarray:
    """Leave-one-problem-out mean residual within q bins: a q-only risk model."""
    q = np.array([r["q"] for r in rows])
    which = np.minimum((q * bins).astype(int), bins - 1)
    pids = np.array([r["problem_id"] for r in rows])
    out = np.empty(len(rows))
    for i in range(len(rows)):
        mask = (which == which[i]) & (pids != pids[i])
        out[i] = values[mask].mean() if mask.any() else values[pids != pids[i]].mean()
    return out


def curves(problems: dict) -> dict:
    """Ratio-of-sums statistics over the given problems."""
    rows = [r for p in problems.values() for r in p["rows"]]
    if not rows:
        return {"n_problems": 0}
    total_v = sum(p["k"] * p["v_x"] for p in problems.values())
    total_vr = sum(p["k"] * p["var_r_x"] for p in problems.values())
    sum_ = lambda key: float(sum(r[key] for r in rows))
    s, mu2 = sum_("s"), sum_("mu2")
    ratio = lambda num, den: None if den <= 0 else num / den
    rho_l, rho_a = ratio(sum_("r_oracle"), total_v), ratio(sum_("var_r"), total_vr)
    return {
        "n_problems": len(problems), "n_prefixes": len(rows),
        "rho_L": rho_l, "rho_A": rho_a,
        "gap": None if rho_l is None or rho_a is None else rho_a - rho_l,
        "residual_reward_only": ratio(sum_("r_reward"), total_v),
        "residual_free_coefficient": ratio(sum_("r_free"), total_v),
        "suffix_cross_share": ratio(sum_("c_energy"), mu2),
        "alpha_upper": ratio(mu2, s),
        "settled_fraction": float(np.mean([r["settled"] for r in rows])),
        "settled_residual_share": ratio(sum(r["r_oracle"] for r in rows if r["settled"]), sum_("r_oracle")),
    }


def _stratum(b: float) -> str:
    return "hard" if b < 0.4 else ("medium" if b < 0.7 else "easy")


def analyze(replay_dirs, metric="euclidean", bootstrap=1000, seed=17,
            w_trains=(0.5, 1.0, 2.0), w_prefill=0.05, p_min=0.05) -> dict:
    by_t: dict = {}
    for directory in replay_dirs:
        directory = Path(directory)
        cross = {}
        cross_path = directory / "problem_cross.jsonl"
        if cross_path.is_file():
            for line in cross_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                cross[(str(row["problem_id"]), row.get("t"))] = (
                    float(row["metrics"][metric]["cross_mean_sum"]), int(row["n_prefixes"]))
        for line in (directory / "trajectory_scalars.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            stats = prefix_stats(row, metric)
            by_t.setdefault(stats["t"], {"rows": [], "cross": {}})["rows"].append(stats)
        for (pid, t), value in cross.items():
            by_t.setdefault(t, {"rows": [], "cross": {}})["cross"][pid] = value
    rng = np.random.default_rng(seed)
    report = {"metric": metric, "definitions": __doc__, "positions": {}}
    for t in sorted(by_t, key=lambda x: (x is None, x)):
        problems = problem_table(by_t[t]["rows"], by_t[t]["cross"])
        point = curves(problems)
        strata = {}
        for name in ("hard", "medium", "easy"):
            sub = {pid: p for pid, p in problems.items() if _stratum(p["rows"][0]["baseline"]) == name}
            strata[name] = curves(sub)
        # Pre-answer subset: rebuild problems from rows whose prefix has not emitted an answer.
        strata["pre_answer"] = curves(problem_table(
            [r for r in by_t[t]["rows"] if not r["answer_emitted"]], by_t[t]["cross"]))
        pids = sorted(problems)
        draws = {key: [] for key in ("rho_L", "rho_A", "gap", "alpha_upper", "suffix_cross_share")}
        for _ in range(int(bootstrap) if pids else 0):
            pick = rng.choice(len(pids), size=len(pids), replace=True)
            sample = {f"{j}:{pids[i]}": problems[pids[i]] for j, i in enumerate(pick)}
            stats = curves(sample)
            for key in draws:
                if stats.get(key) is not None:
                    draws[key].append(stats[key])
        ci = {key: (None if len(v) < 2 else [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
              for key, v in draws.items()}
        rows = [r for p in problems.values() for r in p["rows"]]
        allocation = {}
        if rows:
            v_full = float(np.mean([r["s"] for r in rows]))  # >= true variance: generous to completion
            r_oracle = np.array([r["r_oracle"] for r in rows])
            r_reward = np.array([r["r_reward"] for r in rows])
            for w_train in w_trains:
                c0, c = costs(rows, w_train, w_prefill)
                gamma = float(c0.mean() / (c0 + c).mean())
                alpha = point["alpha_upper"] or 0.0
                allocation[str(w_train)] = {
                    "gamma": gamma, "homogeneous_bound": gain_ratio(alpha, gamma),
                    "oracle_in_sample": best_allocation_ratio(r_oracle, r_oracle, c0, c, v_full, p_min),
                    "q_only_reward_cv": best_allocation_ratio(_q_bin_risk(rows, r_reward), r_reward,
                                                              c0, c, v_full, p_min),
                }
        cos = [r["suffix_prefix_cosine"] for r in rows if r["suffix_prefix_cosine"] is not None]
        report["positions"][str(t)] = {
            **point, "ci95": ci, "strata": strata, "allocation": allocation,
            "mean_suffix_prefix_cosine": float(np.mean(cos)) if cos else None,
            "problems_missing_cross": sum(not p["has_cross"] for p in problems.values()),
            "excluded_single_prefix_problems": len({r["problem_id"] for r in by_t[t]["rows"]}) - len(problems),
        }
    report["assumptions"] = {
        "alpha_upper": "drops ||E G||^2 from both terms, so it can only overstate completion gains",
        "v_full": "mean E||G||^2, an upper bound on the per-start variance (generous to completion)",
        "oracle_in_sample": "risk = estimated tr Cov(G|h) on the same rows; optimistic ceiling",
        "q_only_reward_cv": "risk from leave-one-problem-out q-bin means; m = (q-b) g_h",
        "costs": "c0 = prefix*(1+w_prefill); c = suffix + w_train*(prefix+suffix), token units",
        "p_min": p_min,
    }
    return report
