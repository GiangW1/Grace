"""One GRACE batch: frozen actor/predictor, then a single update."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from grace_gc.core.allocation import allocate_continuation
from grace_gc.core.estimator import dual_stream_mean, optimizer_grad_from_ghat
from grace_gc.core.losses import prediction_grad_correction, real_stream_loss_scale
from grace_gc.data.reward import first_parseable_index
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
    prompt_len: int = 0
    prefix_tokens: int = 0
    response_tokens: int = 0
    suffix_tokens: int = 0
    finish_reason: str = "unknown"
    truncated: bool = False
    text: str | None = None
    prefix_text: str | None = None
    gold: str | None = None
    extracted: str | None = None
    answer_first_token: int | None = None
    baseline_b: float | None = None
    q_hat: float | None = None
    g_norm_sq: float | None = None
    rollout_logprob_sum: float | None = None
    rollout_token_logprobs: list[float | None] | None = None
    request_seeds: dict | None = None
    prompt_token_ids: list[int] | None = None
    prefix_token_ids: list[int] | None = None
    full_token_ids: list[int] | None = None
    prompt_truncated: bool = False
    untruncated_prompt_len: int | None = None
    used_chat_template: bool | None = None
    thinking_closed: bool | None = None
    vllm_finish_reason: str | None = None
    vllm_stop_reason: Any = None


def finish_reason(z: float, natural_finish: bool, truncated: bool) -> str:
    if float(z) < 1.0:
        return "stopped"
    if natural_finish:
        return "eos"
    if truncated:
        return "length"
    return "completed"


def answer_first_token(decode, token_ids) -> int | None:
    if decode is None or not token_ids:
        return None
    return first_parseable_index([decode([int(tok)]) for tok in token_ids])


@dataclass
class GraceBatchState:
    n: int
    records: list[StartRecord] = field(default_factory=list)
    allocation_deviation: float = 0.0
    leak_warned: bool = False


def neyman_ready(basis_id: int, predictor_synced_basis_id: int) -> bool:
    """Residual-risk Neyman needs a real U and heads trained on that U."""
    return int(basis_id) > 0 and int(predictor_synced_basis_id) == int(basis_id)


def heads_trained_on_current_u(spec: MethodSpec, pred_metrics: dict | None) -> bool:
    """True only if this method's coord head and risk head ran on the live U."""
    metrics = pred_metrics or {}
    if metrics.get("coord_loss") is None:
        return False
    mode = str(getattr(spec, "risk_mode", "full") or "full")
    if mode == "none":
        return True
    if mode == "reward":
        return metrics.get("reward_risk_loss") is not None
    return metrics.get("risk_loss") is not None


def maybe_mark_predictor_synced(state, pred_metrics: dict | None, basis_changed: bool) -> None:
    """Invalidate sync on a new U; restore it only after both required heads train."""
    if basis_changed:
        state.predictor_synced_basis_id = -1
    if int(getattr(state, "basis_id", 0) or 0) > 0 and heads_trained_on_current_u(state.spec, pred_metrics):
        state.predictor_synced_basis_id = int(state.basis_id)


def allocation_ready_from_checkpoint(spec: MethodSpec, payload: dict | None, cfg: dict | None) -> bool:
    """Same warmup / U-sync gate the next training batch would use."""
    if not spec.use_allocation:
        return True
    raw = payload or {}
    warmup = int(((cfg or {}).get("predictor") or {}).get("warmup_steps", 0) or 0)
    if int(raw.get("step", 0) or 0) < warmup:
        return False
    basis = raw.get("basis") or {}
    return neyman_ready(int(basis.get("basis_id", 0) or 0), int(basis.get("predictor_synced_basis_id", -1)))


def decide_continuation(
    spec: MethodSpec,
    risk: np.ndarray,
    cost: np.ndarray,
    finished: np.ndarray,
    beta: float,
    p_min: float,
    warmup: bool,
    iters: int = 20,
    basis_ready: bool = True,
    uniform_shrink: float = 0.0,
) -> tuple[np.ndarray, float]:
    n = int(risk.shape[0])
    if beta == 1.0 or warmup or not basis_ready or spec.name in {"full_pg", "grpo", "grpo_short"} or spec.objective == "grpo":
        return np.ones(n, dtype=np.float64), 0.0
    if spec.name == "uniform_ht" or spec.name == "uniform_cv" or not spec.use_allocation:
        p = np.full(n, uniform_p(n, beta, p_min), dtype=np.float64)
        p[finished] = 1.0
        return p, 0.0
    result = allocate_continuation(risk, cost, beta=beta, p_min=p_min, finished=finished,
                                   iters=int(iters), uniform_shrink=uniform_shrink)
    return result.p, result.budget_deviation


def control_variate_coordinates(f: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Ablate m in the estimator without changing predictor training or risk p."""
    values = np.asarray(f, dtype=np.float64)
    return values if enabled else np.zeros_like(values)


def assemble_ghat(
    spec: MethodSpec,
    records: list[StartRecord],
    u: np.ndarray,
    n: int,
    control_variate_enabled: bool = True,
) -> np.ndarray:
    z = np.array([r.z for r in records], dtype=np.float64)
    p = np.array([r.p for r in records], dtype=np.float64)
    f = np.stack([r.f for r in records], axis=0) if records else np.zeros((0, u.shape[1]))
    f = control_variate_coordinates(f, control_variate_enabled)
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


def incremental_token_costs(remainings, finished) -> np.ndarray:
    """Neyman ĉ is leftover tokens, not a learned head, when cost is constant."""
    c = np.maximum(np.asarray(remainings, dtype=np.float64).reshape(-1), 1.0)
    fin = np.asarray(finished, dtype=bool).reshape(-1)
    if fin.shape[0] != c.shape[0]:
        raise ValueError("remainings/finished dimensions do not match")
    c[fin] = 1.0
    return c


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
    control_variate_enabled: bool = True,
) -> dict:
    ghat = assemble_ghat(spec, records, u, n, control_variate_enabled=control_variate_enabled)
    grad = optimizer_grad_from_ghat(ghat)
    z = np.array([r.z for r in records], dtype=np.float64)
    p = np.array([r.p for r in records], dtype=np.float64)
    f = np.stack([r.f for r in records], axis=0)
    f = control_variate_coordinates(f, control_variate_enabled)
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
