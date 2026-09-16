"""Same-prefix independent continuations and residual diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from grace_gc.audit.stats import (
    elf,
    headline_mask,
    lag_index,
    plc,
    rho_ucb,
    summarize_with_filters,
    t_answer,
    t_learn,
)
from grace_gc.audit.variance_cost import fit_eval_split, variance_times_cost
from grace_gc.core.allocation import allocate_continuation
from grace_gc.predictor.risk import full_space_residual


@dataclass
class PrefixBundle:
    problem_id: str
    t: int
    rewards: np.ndarray
    grads: np.ndarray
    coords: np.ndarray | None = None
    path_id: str | None = None
    suffix_cost: np.ndarray | None = None
    m_pred: np.ndarray | None = None
    r_hat: float | None = None
    c_hat: float | None = None
    finished: bool = False
    prefix_tokens: int | None = None
    answer_emitted: bool = False
    prefix_text: str | None = None
    suffix_texts: list[str] | None = None
    prompt_truncated: bool | None = None


_SKETCH_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _count_sketch(g: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Count-sketch JL. Dense Gaussian P would be d×dim and does not fit LoRA-d."""
    d = int(g.shape[1])
    key = (d, int(dim), int(seed))
    cached = _SKETCH_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, int(dim), size=d)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=d)
        cached = (idx, signs)
        _SKETCH_CACHE[key] = cached
    idx, signs = cached
    out = np.zeros((g.shape[0], int(dim)), dtype=np.float64)
    weighted = g * signs
    for i in range(g.shape[0]):
        np.add.at(out[i], idx, weighted[i])
    return out / np.sqrt(float(dim))


def jl_project(g: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Fixed-seed projection to `dim`. Same (d, dim, seed) reuses the same map."""
    g = np.asarray(g, dtype=np.float64)
    squeeze = False
    if g.ndim == 1:
        g = g[None, :]
        squeeze = True
    dim = int(dim)
    d = int(g.shape[1])
    if d <= dim:
        out = g.copy()
    elif d * dim * 8 <= 64 * 1024 * 1024:
        rng = np.random.default_rng(int(seed))
        p = rng.normal(size=(d, dim)) / np.sqrt(dim)
        out = g @ p
    else:
        out = _count_sketch(g, dim, seed)
    return out[0] if squeeze else out


def orthogonal_energy(g: np.ndarray, u: np.ndarray) -> float:
    g = np.asarray(g, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    proj = (g @ u) @ u.T if g.ndim == 2 else u @ (u.T @ g)
    resid = g - proj
    return float(np.mean(np.sum(resid * resid, axis=-1)))


def _detect_report_rows(g: np.ndarray, rewards: np.ndarray, min_split: int = 8):
    """Paper anti-leak: decide t_L on the first half, report ρ_L on the rest."""
    g = np.asarray(g, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if g.shape[0] < int(min_split):
        return g, r, g, r
    mid = int(g.shape[0]) // 2
    return g[:mid], r[:mid], g[mid:], r[mid:]


def _unbiased_mean_sq(centered_sq: np.ndarray, n: int) -> float:
    """M/(M-1) sample-variance factor from the paper ρ_L numerator."""
    if n <= 1:
        return 0.0
    return float(np.mean(centered_sq) * n / (n - 1))


def _cond_var(g: np.ndarray, finished: bool = False) -> float:
    g = np.asarray(g, dtype=np.float64)
    n = 1 if g.ndim == 1 else int(g.shape[0])
    if n <= 1:
        return 0.0 if finished else float("nan")
    if g.ndim == 1:
        return 0.0 if finished else float("nan")
    return _unbiased_mean_sq(np.sum((g - g.mean(axis=0)) ** 2, axis=1), n)


def _ab_energy(g: np.ndarray, prompt_norm_sq: float) -> float:
    g = np.asarray(g, dtype=np.float64)
    if g.ndim == 1 or g.shape[0] < 2:
        return 0.0
    mid = g.shape[0] // 2
    mean_a = g[:mid].mean(axis=0)
    mean_b = g[mid:].mean(axis=0)
    return float(np.dot(mean_a, mean_b) / max(prompt_norm_sq, 1e-12))


def _by_problem(bundles: list[PrefixBundle]) -> dict[str, list[PrefixBundle]]:
    out: dict[str, list[PrefixBundle]] = defaultdict(list)
    for bundle in bundles:
        out[bundle.problem_id].append(bundle)
    return out


def _prompt_level_bundles(group: list[PrefixBundle]) -> list[PrefixBundle]:
    """Var(G|x) / Var(R|x) use trajectories from x, not nested later slices."""
    if any(bundle.path_id is not None for bundle in group):
        chosen: dict[str, PrefixBundle] = {}
        for bundle in group:
            key = str(bundle.path_id) if bundle.path_id is not None else f"{bundle.problem_id}:{id(bundle)}"
            prev = chosen.get(key)
            if prev is None or int(bundle.t) < int(prev.t):
                chosen[key] = bundle
        return list(chosen.values())
    t0 = min(int(bundle.t) for bundle in group)
    return [bundle for bundle in group if int(bundle.t) == t0]


def _equal_prefix_rows(level: list[PrefixBundle], kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Each prefix has weight 1, so a finished 1-row prefix is not drowned by M continuations."""
    rows = []
    weights = []
    for bundle in level:
        if kind == "g":
            arr = np.asarray(bundle.grads, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
        else:
            arr = np.asarray(bundle.rewards, dtype=np.float64).reshape(-1, 1)
        n = max(int(arr.shape[0]), 1)
        w = 1.0 / n
        for i in range(arr.shape[0]):
            rows.append(arr[i])
            weights.append(w)
    if not rows:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.stack(rows, axis=0), np.asarray(weights, dtype=np.float64)


def _weighted_var(rows: np.ndarray, weights: np.ndarray) -> float:
    if rows.shape[0] == 0 or float(np.sum(weights)) <= 0.0:
        return 0.0
    w = weights / np.sum(weights)
    mean = (w[:, None] * rows).sum(axis=0)
    return float(np.sum(w * np.sum((rows - mean) ** 2, axis=1)))


def _prompt_stats(group: list[PrefixBundle]) -> tuple[float, float]:
    level = _prompt_level_bundles(group)
    g_all, w = _equal_prefix_rows(level, "g")
    n_pref = len(level)
    raw = _weighted_var(g_all, w)
    if n_pref <= 1:
        prompt_var_g = _cond_var(g_all, finished=all(b.finished for b in level))
    else:
        prompt_var_g = raw * n_pref / (n_pref - 1)
    if g_all.shape[0] and float(np.sum(w)) > 0.0:
        ww = w / np.sum(w)
        prompt_norm_sq = float(np.sum(ww * np.sum(g_all ** 2, axis=1)))
    else:
        prompt_norm_sq = 0.0
    return prompt_var_g, prompt_norm_sq


def _prompt_var_r(group: list[PrefixBundle]) -> float:
    level = _prompt_level_bundles(group)
    r_all, w = _equal_prefix_rows(level, "r")
    if r_all.shape[0] <= 1:
        return 0.0
    n_pref = len(level)
    raw = _weighted_var(r_all, w)
    if n_pref <= 1:
        flat = r_all.reshape(-1)
        return float(np.var(flat, ddof=1)) if flat.size > 1 else 0.0
    return raw * n_pref / (n_pref - 1)


def audit_bundles(
    bundles: list[PrefixBundle],
    u: np.ndarray,
    analysis: dict,
    rng: np.random.Generator,
) -> dict:
    if not bundles:
        return {"n_bundles": 0, "note": "no prefixes"}
    grouped = _by_problem(bundles)
    var_r = []
    cond_var = []
    energy = []
    residuals = []
    rho_l_vals = []
    rho_a_vals = []
    flat: list[PrefixBundle] = []
    for group in grouped.values():
        prompt_var_g, prompt_norm_sq = _prompt_stats(group)
        prompt_var_r = _prompt_var_r(group)
        for bundle in group:
            flat.append(bundle)
            r = bundle.rewards.astype(np.float64)
            g = bundle.grads.astype(np.float64)
            _g_det, _r_det, g_rep, r_rep = _detect_report_rows(g, r)
            vr = float(np.var(r, ddof=1) if r.size > 1 else 0.0)
            cv = _cond_var(g_rep, finished=bool(bundle.finished))
            var_r.append(vr)
            cond_var.append(cv)
            energy.append(_ab_energy(g, prompt_norm_sq))
            rho_l_vals.append(cv / prompt_var_g if prompt_var_g > 0.0 else float("nan"))
            rho_a_vals.append(
                float(np.var(r_rep, ddof=1) if r_rep.size > 1 else 0.0) / prompt_var_r
                if prompt_var_r > 0.0
                else float("nan")
            )
            if bundle.coords is not None:
                f = np.asarray(bundle.coords, dtype=np.float64)
                if f.ndim == 1:
                    f = np.broadcast_to(f, (g.shape[0], f.shape[0]))
                residuals.append(full_space_residual(g, f, u))
            elif bundle.m_pred is not None:
                m = np.asarray(bundle.m_pred, dtype=np.float64)
                residuals.append(float(np.mean(np.sum((g - m) ** 2, axis=1))))
    var_r = np.asarray(var_r)
    cond_var = np.asarray(cond_var)
    energy = np.asarray(energy)
    undecided = float(analysis.get("answer_undecided", 0.15))
    kappa = float(analysis.get("nonzero_signal_kappa", 0.05))
    mask = headline_mask(
        var_r,
        energy,
        undecided,
        kappa,
        [b.answer_emitted for b in flat],
        [b.rewards for b in flat],
        wilson_lo=float(analysis.get("wilson_lo", 0.05)),
        wilson_hi=float(analysis.get("wilson_hi", 0.95)),
        require_pre_emit=bool(analysis.get("require_pre_emit", True)),
    )
    gated = [bundle for bundle, keep in zip(flat, mask) if keep]
    prompt_var_g_by = {pid: _prompt_stats(group)[0] for pid, group in grouped.items()}
    prompt_var_r_by = {pid: _prompt_var_r(group) for pid, group in grouped.items()}
    keys = [
        str(b.path_id) if getattr(b, "path_id", None) is not None else str(b.problem_id)
        for b in bundles
    ]
    fit, eval_idx = fit_eval_split(len(bundles), rng, keys=keys)
    vc = _restricted_variance_cost(bundles, fit, eval_idx, analysis)
    times = np.array(sorted({int(b.t) for b in bundles}), dtype=np.float64)
    rho_l_curve = []
    rho_a_curve = []
    var_r_curve = []
    rho_l_detect = []
    rho_l_detect_ucb = []
    rho_l_curve_all = []
    rho_a_curve_all = []
    var_r_curve_all = []
    for t in times:
        sub_all = [b for b in bundles if int(b.t) == int(t)]
        sub = [b for b in gated if int(b.t) == int(t)]
        part = audit_rho_at_t(sub, prompt_var_g_by, prompt_var_r_by)
        part_det = audit_rho_at_t(sub, prompt_var_g_by, prompt_var_r_by, half="detect")
        part_all = audit_rho_at_t(sub_all, prompt_var_g_by, prompt_var_r_by)
        rho_l_curve.append(part["rho_l"])
        rho_a_curve.append(part["rho_a"])
        var_r_curve.append(part["var_r"])
        rho_l_detect.append(part_det["rho_l"])
        rho_l_detect_ucb.append(rho_ucb(part_det.get("rho_l_vals")))
        rho_l_curve_all.append(part_all["rho_l"])
        rho_a_curve_all.append(part_all["rho_a"])
        var_r_curve_all.append(part_all["var_r"])
    rho_l_curve = np.asarray(rho_l_curve, dtype=np.float64)
    rho_a_curve = np.asarray(rho_a_curve, dtype=np.float64)
    var_r_curve = np.asarray(var_r_curve, dtype=np.float64)
    rho_l_detect = np.asarray(rho_l_detect, dtype=np.float64)
    rho_l_detect_ucb = np.asarray(rho_l_detect_ucb, dtype=np.float64)
    n_g = int(sum(int(b.grads.shape[0]) for b in bundles))
    ortho = 0.0
    if u.size and n_g:
        for bundle in bundles:
            ortho += orthogonal_energy(bundle.grads, u) * int(bundle.grads.shape[0])
        ortho /= float(n_g)
    path_ids = [getattr(b, "path_id", None) for b in bundles]
    have_paths = all(pid is not None for pid in path_ids) and len({(b.path_id, b.t) for b in bundles}) > len({b.path_id for b in bundles})
    result = {
        "n_bundles": len(bundles),
        "rho_a_all": float(np.nanmean(rho_a_vals)) if np.any(np.isfinite(rho_a_vals)) else float("nan"),
        "rho_l_all": float(np.nanmean(rho_l_vals)) if np.any(np.isfinite(rho_l_vals)) else float("nan"),
        "times": [int(t) for t in times],
        "rho_l_curve": rho_l_curve.tolist(),
        "rho_l_detect_curve": rho_l_detect.tolist(),
        "rho_l_detect_ucb_curve": rho_l_detect_ucb.tolist(),
        "rho_a_curve": rho_a_curve.tolist(),
        "var_r_curve": var_r_curve.tolist(),
        "rho_l_curve_all": [float(x) for x in rho_l_curve_all],
        "rho_a_curve_all": [float(x) for x in rho_a_curve_all],
        "var_r_curve_all": [float(x) for x in var_r_curve_all],
        "n_gated": int(mask.sum()),
        "n_gated_pre_emit": int(sum(1 for b in gated if not b.answer_emitted)),
        "gates": {
            **summarize_with_filters(cond_var, mask),
            "rho_l": summarize_with_filters(np.asarray(rho_l_vals, dtype=np.float64), mask),
            "rho_a": summarize_with_filters(np.asarray(rho_a_vals, dtype=np.float64), mask),
        },
        "fit_index": fit.tolist(),
        "eval_index": eval_idx.tolist(),
        "variance_cost": vc["actual"],
        "variance_cost_oracle": vc["oracle"],
        "variance_cost_note": vc.get("note"),
        "orthogonal_energy": None if not u.size or n_g == 0 else float(ortho),
        "t_L": t_learn(rho_l_detect_ucb, times, float(analysis.get("epsilon_learn", 0.2))),
        "t_L_point": t_learn(rho_l_detect, times, float(analysis.get("epsilon_learn", 0.2))),
        # t_A is Var(R)≤δ on the trajectory, including prefixes that already left the Wilson set.
        "t_A": t_answer(np.asarray(var_r_curve_all, dtype=np.float64), times, float(analysis.get("delta_answer", 0.05))),
        "elf": elf(
            *_path_elf_arrays(
                gated,
                prompt_var_g_by,
                prompt_var_r_by,
                float(analysis.get("epsilon_learn", 0.2)),
                float(analysis.get("delta_answer", 0.05)),
                answer_bundles=flat,
            )
        )
        if have_paths
        else None,
        "elf_pre_emit": elf(
            *_path_elf_arrays(
                [b for b in gated if not b.answer_emitted],
                prompt_var_g_by,
                prompt_var_r_by,
                float(analysis.get("epsilon_learn", 0.2)),
                float(analysis.get("delta_answer", 0.05)),
                answer_bundles=flat,
            )
        )
        if have_paths
        else None,
        "elf_note": None if have_paths else "no cross-t path_id; ELF not computed",
        "lag_index": lag_index(
            rho_a_curve,
            rho_l_curve,
            times,
            length=analysis.get("max_new_tokens"),
        ),
        **_token_plc_fields(
            bundles,
            prompt_var_g_by,
            float(analysis.get("epsilon_learn", 0.2)),
        ),
        "residual_mean": None
        if not residuals
        else float(np.mean(np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float64)).reshape(-1) for x in residuals]))),
    }
    jl_dim = analysis.get("jl_dim")
    if jl_dim:
        result["jl_shape"] = [n_g, min(int(jl_dim), int(bundles[0].grads.shape[1]))]
    return result


def _token_plc_fields(bundles: list[PrefixBundle], prompt_var_g_by: dict[str, float], eps: float) -> dict:
    """Paper PLC: rollout tokens after t_L over total tokens. Backward is not metered."""
    by_path: dict[str, list[PrefixBundle]] = defaultdict(list)
    for bundle in bundles:
        if bundle.path_id is None:
            continue
        by_path[str(bundle.path_id)].append(bundle)
    if not by_path:
        return {"plc": None, "plc_note": "no path_id; PLC not computed"}
    after = 0.0
    total = 0.0
    for group in by_path.values():
        times = np.array(sorted({b.t for b in group}), dtype=np.float64)
        if times.size == 0:
            continue
        rho = []
        for t in times:
            bundle = next(b for b in group if b.t == t)
            pv = float(prompt_var_g_by.get(bundle.problem_id, 0.0))
            g_det, _r_det, _g_rep, _r_rep = _detect_report_rows(bundle.grads, bundle.rewards)
            cv = _cond_var(g_det, finished=bool(bundle.finished))
            rho.append(cv / pv if pv > 0.0 else np.nan)
        tl = t_learn(np.asarray(rho, dtype=np.float64), times, eps)
        last = max(group, key=lambda b: int(b.t))
        pref = float(last.prefix_tokens) if last.prefix_tokens is not None else float(last.t)
        extra = 0.0
        if not last.finished and last.suffix_cost is not None:
            extra = float(np.mean(np.asarray(last.suffix_cost, dtype=np.float64)))
        length = pref + extra
        if length <= 0.0:
            continue
        total += length
        if tl is not None:
            after += max(0.0, length - float(tl))
    if total <= 0.0:
        return {"plc": None, "plc_note": "no measured path token lengths"}
    return {
        "plc": plc(after, 0.0, total),
        "plc_note": "token proxy after t_L; backward not metered",
    }


def audit_rho_at_t(
    bundles: list[PrefixBundle],
    prompt_var_g_by: dict[str, float] | None = None,
    prompt_var_r_by: dict[str, float] | None = None,
    half: str = "report",
) -> dict[str, float]:
    if not bundles:
        return {"rho_l": float("nan"), "rho_a": float("nan"), "var_r": float("nan")}
    grouped = _by_problem(bundles)
    problem_rho_l = []
    problem_rho_a = []
    problem_var_r = []
    for pid, group in grouped.items():
        prompt_var_g = prompt_var_g_by[pid] if prompt_var_g_by and pid in prompt_var_g_by else _prompt_stats(group)[0]
        prompt_var_r = prompt_var_r_by[pid] if prompt_var_r_by and pid in prompt_var_r_by else _prompt_var_r(group)
        rl = []
        ra = []
        vr = []
        for bundle in group:
            g_det, r_det, g_rep, r_rep = _detect_report_rows(bundle.grads, bundle.rewards)
            g_use, r_use = (g_det, r_det) if half == "detect" else (g_rep, r_rep)
            var_r = float(np.var(r_use, ddof=1) if r_use.size > 1 else 0.0)
            cv = _cond_var(g_use, finished=bool(bundle.finished))
            vr.append(var_r)
            if prompt_var_g > 0.0:
                rl.append(cv / prompt_var_g)
            if prompt_var_r > 0.0:
                ra.append(var_r / prompt_var_r)
        if rl:
            problem_rho_l.append(float(np.mean(rl)))
        if ra:
            problem_rho_a.append(float(np.mean(ra)))
        if vr:
            problem_var_r.append(float(np.mean(vr)))
    return {
        "rho_l": float(np.mean(problem_rho_l)) if problem_rho_l else float("nan"),
        "rho_a": float(np.mean(problem_rho_a)) if problem_rho_a else float("nan"),
        "var_r": float(np.mean(problem_var_r)) if problem_var_r else float("nan"),
        "rho_l_vals": problem_rho_l,
    }


def _path_times(bundles: list[PrefixBundle], prompt_var_g_by: dict[str, float], kind: str, thresh: float) -> dict[str, float]:
    by_path: dict[str, list[PrefixBundle]] = defaultdict(list)
    for bundle in bundles:
        if bundle.path_id is None:
            continue
        by_path[str(bundle.path_id)].append(bundle)
    out: dict[str, float] = {}
    for pid, group in by_path.items():
        times = np.array(sorted({b.t for b in group}), dtype=np.float64)
        if times.size == 0:
            continue
        vals = []
        for t in times:
            bundle = next(b for b in group if b.t == t)
            if kind == "learn":
                pv = float(prompt_var_g_by.get(bundle.problem_id, 0.0))
                g_det, _r_det, _g_rep, _r_rep = _detect_report_rows(bundle.grads, bundle.rewards)
                cv = _cond_var(g_det, finished=bool(bundle.finished))
                vals.append(cv / pv if pv > 0.0 else np.nan)
            else:
                r = bundle.rewards.astype(np.float64)
                vals.append(float(np.var(r, ddof=1) if r.size > 1 else 0.0))
        hit = t_learn(np.asarray(vals, dtype=np.float64), times, thresh) if kind == "learn" else t_answer(
            np.asarray(vals, dtype=np.float64), times, thresh
        )
        if hit is not None:
            out[pid] = float(hit)
    return out


def _path_elf_arrays(
    bundles: list[PrefixBundle],
    prompt_var_g_by: dict[str, float],
    prompt_var_r_by: dict[str, float],
    eps: float = 0.2,
    delta: float = 0.05,
    answer_bundles: list[PrefixBundle] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _ = prompt_var_r_by
    t_l_by = _path_times(bundles, prompt_var_g_by, "learn", eps)
    t_a_by = _path_times(answer_bundles if answer_bundles is not None else bundles, prompt_var_g_by, "answer", delta)
    paths = sorted(set(t_l_by) | set(t_a_by))
    # Paths that appear only as prefixes still belong in the ELF denominator.
    if bundles or answer_bundles:
        seen = set()
        for bundle in list(bundles) + list(answer_bundles or []):
            if bundle.path_id is None:
                continue
            seen.add(str(bundle.path_id))
        paths = sorted(seen)
    t_l = [t_l_by.get(pid, float("nan")) for pid in paths]
    t_a = [t_a_by.get(pid, float("nan")) for pid in paths]
    return np.asarray(t_l, dtype=np.float64), np.asarray(t_a, dtype=np.float64)


def _p_at_decision(eval_b: list[PrefixBundle], spec, analysis: dict, problem_risk: dict, problem_cost: dict) -> np.ndarray:
    """One λ per visible decision set. Mixed-t prefixes are not a training batch."""
    from grace_gc.trainer.methods import uniform_p

    if not eval_b:
        return np.zeros(0, dtype=np.float64)
    finished = np.array([bool(b.finished) for b in eval_b], dtype=bool)
    beta = float(analysis.get("beta", 0.5))
    p_min = float(analysis.get("p_min", 0.2))
    if spec.name in {"full_pg", "grpo", "grpo_short"} or spec.objective == "grpo":
        return np.ones(len(eval_b), dtype=np.float64)
    if spec.name in {"uniform_ht", "uniform_cv"} or not spec.use_allocation:
        p_alloc = np.full(len(eval_b), uniform_p(len(eval_b), beta, p_min), dtype=np.float64)
        p_alloc[finished] = 1.0
        return p_alloc
    if spec.use_predictor and all(b.r_hat is not None and b.c_hat is not None for b in eval_b):
        risk = np.array([float(b.r_hat) for b in eval_b], dtype=np.float64)
        cost = np.array([float(b.c_hat) for b in eval_b], dtype=np.float64)
    else:
        risk = np.array([problem_risk.get(b.problem_id, 1.0) for b in eval_b], dtype=np.float64)
        cost = np.array([problem_cost.get(b.problem_id, 1.0) for b in eval_b], dtype=np.float64)
    return allocate_continuation(risk, cost, beta=beta, p_min=p_min, finished=finished).p


def _restricted_variance_cost(
    bundles: list[PrefixBundle],
    fit: np.ndarray,
    eval_idx: np.ndarray,
    analysis: dict,
) -> dict:
    from grace_gc.trainer.methods import method_spec

    fit_b = [bundles[i] for i in fit]
    eval_b = [bundles[i] for i in eval_idx] or list(bundles)
    problem_risk = {}
    problem_cost = {}
    for group_id, group in _by_problem(fit_b).items():
        g = np.concatenate([b.grads for b in group], axis=0)
        cv = _cond_var(g, finished=all(b.finished for b in group))
        problem_risk[group_id] = 1e-8 if not np.isfinite(cv) else max(float(cv), 1e-8)
        paid = []
        for bundle in group:
            if bundle.suffix_cost is not None:
                paid.extend(np.asarray(bundle.suffix_cost, dtype=np.float64).reshape(-1).tolist())
        if paid:
            problem_cost[group_id] = float(np.mean(paid))
    spec = method_spec(str(analysis.get("method", "grace")))
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, bundle in enumerate(eval_b):
        by_t[int(bundle.t)].append(i)
    p_alloc = np.ones(len(eval_b), dtype=np.float64)
    for idxs in by_t.values():
        group = [eval_b[i] for i in idxs]
        p_t = _p_at_decision(group, spec, analysis, problem_risk, problem_cost)
        for loc, i in enumerate(idxs):
            p_alloc[i] = float(p_t[loc])
    use_pred = bool(spec.use_predictor)
    g_rows = []
    m_oracle = []
    m_actual = []
    p_rows = []
    suffix = []
    prefix_rows = []
    used_within = False
    for bundle, p_i in zip(eval_b, p_alloc):
        g = np.asarray(bundle.grads, dtype=np.float64)
        if g.ndim == 1:
            g = g[None, :]
        costs_b = None if bundle.suffix_cost is None else np.asarray(bundle.suffix_cost, dtype=np.float64).reshape(-1)
        if g.shape[0] >= 2:
            used_within = True
            mid = g.shape[0] // 2
            oracle = g[:mid].mean(axis=0)
            rows = g[mid:]
            row_cost = None if costs_b is None else costs_b[mid:]
        else:
            continue
        if use_pred and bundle.m_pred is not None:
            actual = np.asarray(bundle.m_pred, dtype=np.float64)
        else:
            actual = oracle
        for j, row in enumerate(rows):
            g_rows.append(row)
            m_oracle.append(oracle)
            m_actual.append(actual)
            p_rows.append(float(p_i))
            pc = analysis.get("prefix_cost")
            if pc is not None:
                prefix_rows.append(float(pc))
            elif getattr(bundle, "prefix_tokens", None) is not None:
                prefix_rows.append(float(max(int(bundle.prefix_tokens), 0)))
            else:
                prefix_rows.append(float(bundle.t))
            if row_cost is not None and row_cost.size:
                suffix.append(float(row_cost[min(j, row_cost.size - 1)]))
            else:
                suffix.append(1.0)
    if not g_rows:
        empty = {
            "full_var": float("nan"),
            "extra_var": float("nan"),
            "full_cost": float("nan"),
            "full_product": float("nan"),
            "actual_p_var": float("nan"),
            "actual_p_cost": float("nan"),
            "actual_p_product": float("nan"),
            "ratio": None,
        }
        return {"oracle": empty, "actual": empty, "note": "token_proxy_cost; no independent same-prefix rows"}
    if use_pred and any(b.m_pred is not None for b in eval_b):
        note = "token_proxy_cost; frozen_predictor; same-prefix oracle"
    elif used_within:
        note = "token_proxy_cost; same-prefix oracle"
    else:
        note = "token_proxy_cost"
    g_rows = np.stack(g_rows, axis=0)
    return {
        "oracle": variance_times_cost(
            g_rows, np.stack(m_oracle, axis=0), np.asarray(p_rows), prefix_cost=np.asarray(prefix_rows), suffix_cost=np.asarray(suffix)
        ),
        "actual": variance_times_cost(
            g_rows, np.stack(m_actual, axis=0), np.asarray(p_rows), prefix_cost=np.asarray(prefix_rows), suffix_cost=np.asarray(suffix)
        ),
        "note": note,
    }
