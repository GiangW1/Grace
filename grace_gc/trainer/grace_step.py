"""One GRACE batch: frozen actor/predictor, then a single update."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grace_gc.core.allocation import allocate_continuation
from grace_gc.core.estimator import dual_stream_mean, optimizer_grad_from_ghat
from grace_gc.core.losses import prediction_grad_correction, real_stream_loss_scale
from grace_gc.trainer.methods import MethodSpec, uniform_p


@dataclass
class StartRecord:
    problem_id: str
    finished: bool
    p: float
    z: float
    f: np.ndarray
    r_hat: float
    c_hat: float
    reward: float | None
    advantage: float | None
    g: np.ndarray | None
    audited: bool
    leak_flag: bool = False


@dataclass
class GraceBatchState:
    n: int
    records: list[StartRecord] = field(default_factory=list)
    allocation_deviation: float = 0.0
    leak_warned: bool = False


def decide_continuation(
    spec: MethodSpec,
    risk: np.ndarray,
    cost: np.ndarray,
    finished: np.ndarray,
    beta: float,
    p_min: float,
    warmup: bool,
    iters: int = 20,
) -> tuple[np.ndarray, float]:
    n = int(risk.shape[0])
    if warmup or spec.name in {"full_pg", "grpo", "grpo_short"} or spec.objective == "grpo":
        return np.ones(n, dtype=np.float64), 0.0
    if spec.name == "uniform_ht" or spec.name == "uniform_cv" or not spec.use_allocation:
        p = np.full(n, uniform_p(n, beta, p_min), dtype=np.float64)
        p[finished] = 1.0
        return p, 0.0
    result = allocate_continuation(risk, cost, beta=beta, p_min=p_min, finished=finished, iters=int(iters))
    return result.p, result.budget_deviation


def assemble_ghat(
    spec: MethodSpec,
    records: list[StartRecord],
    u: np.ndarray,
    n: int,
) -> np.ndarray:
    z = np.array([r.z for r in records], dtype=np.float64)
    p = np.array([r.p for r in records], dtype=np.float64)
    f = np.stack([r.f for r in records], axis=0) if records else np.zeros((0, u.shape[1]))
    completed = [r for r in records if r.z >= 1.0]
    if any(r.g is None for r in completed):
        raise ValueError("assemble_ghat needs G for every completed start")
    if spec.name == "full_pg" or spec.objective == "grpo":
        if not completed:
            return np.zeros(u.shape[0], dtype=np.float64)
        g = np.stack([r.g for r in completed], axis=0)
        return g.sum(axis=0) / n
    if spec.name == "uniform_ht":
        u0 = np.zeros_like(u)
        f0 = np.zeros_like(f)
        g_c = np.stack([r.g for r in completed], axis=0) if completed else np.zeros((0, u.shape[0]))
        p_c = np.array([r.p for r in completed], dtype=np.float64)
        return dual_stream_mean(g_c, p_c, u0, f0, z, p, n)
    g_c = np.stack([r.g for r in completed], axis=0) if completed else np.zeros((0, u.shape[0]))
    p_c = np.array([r.p for r in completed], dtype=np.float64)
    return dual_stream_mean(g_c, p_c, u, f, z, p, n)


def batch_token_costs(
    prompt_lens,
    prefixes: list[list[int]],
    finished: np.ndarray,
    p: np.ndarray,
    max_new: int,
) -> tuple[float, float]:
    """Mean expected tokens paid vs mean full-budget tokens. Prefix is not free.

    Finished starts have no remaining budget. Unfinished starts pay the prefix
    and, in expectation, `p` of the leftover budget. Full-PG (`p=1`) therefore
    has paid == full and keeps N_ref.
    """
    if not prefixes:
        return 0.0, 0.0
    finished = np.asarray(finished, dtype=bool).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if finished.shape[0] != len(prefixes) or p.shape[0] != len(prefixes):
        raise ValueError("finished/p dimensions do not match prefixes")
    paid = np.empty(len(prefixes), dtype=np.float64)
    full = np.empty(len(prefixes), dtype=np.float64)
    budget = int(max_new)
    for i, prefix in enumerate(prefixes):
        pref = max(len(prefix) - int(prompt_lens[i]), 0)
        rem = 0 if finished[i] else max(budget - pref, 0)
        # `p` may be the planned probability or the realized Z. Full-PG has
        # p=Z=1, so paid==full and N stays N_ref. GRACE uses realized Z.
        paid[i] = float(pref) + float(p[i]) * float(rem)
        full[i] = float(pref) + float(rem)
    return float(np.mean(paid)), float(np.mean(full))


def next_start_count(
    history_costs: list[float],
    n_ref: int,
    c_full: float,
    min_n: int = 1,
    group: int = 1,
) -> int:
    if not history_costs:
        n = int(n_ref)
    else:
        hat = float(np.mean(history_costs[-10:]))
        if hat <= 0.0:
            n = int(n_ref)
        else:
            n = max(min_n, int(np.ceil(n_ref * c_full / hat)))
    group = max(1, int(group))
    return max(group, (int(n) // group) * group)


def grace_batch_update(
    spec: MethodSpec,
    records: list[StartRecord],
    u: np.ndarray,
    n: int,
) -> dict:
    ghat = assemble_ghat(spec, records, u, n)
    grad = optimizer_grad_from_ghat(ghat)
    z = np.array([r.z for r in records], dtype=np.float64)
    p = np.array([r.p for r in records], dtype=np.float64)
    f = np.stack([r.f for r in records], axis=0)
    pred_grad = prediction_grad_correction(u, f, z, p, n) if spec.use_ht_correction and spec.use_predictor else np.zeros_like(grad)
    scales = real_stream_loss_scale(
        np.array([0.0 if r.advantage is None else r.advantage for r in records], dtype=np.float64),
        p,
        z,
        n,
    )
    leak = any(r.leak_flag for r in records)
    return {
        "ghat": ghat,
        "optimizer_grad": grad,
        "prediction_grad": pred_grad,
        "loss_scales": scales,
        "n_completed": int(z.sum()),
        "leak_warned": leak,
    }
