"""Reversible feature/target scales and a weighted ridge coordinate fit.

Stored φ is unchanged. Affine and unit scales are fit on historical items,
frozen for the next actor batch, and inverted before HT / residual risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_std(x: np.ndarray, axis: int = 0) -> np.ndarray:
    std = np.std(np.asarray(x, dtype=np.float64), axis=axis)
    return np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)


@dataclass
class FeatureScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    coord_scale: np.ndarray | None = None
    risk_scale: float = 1.0
    enabled: bool = True

    def fit_features(self, features: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        self.mean = x.mean(axis=0)
        self.std = _finite_std(x, axis=0)

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if not self.enabled or self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std

    def fit_coord_scale(self, coords: np.ndarray) -> None:
        y = np.asarray(coords, dtype=np.float64)
        if y.ndim == 1:
            y = y[None, :]
        rms = np.sqrt(np.mean(y * y, axis=0))
        self.coord_scale = np.where(np.isfinite(rms) & (rms > 1e-8), rms, 1.0)

    def scale_coords(self, coords: np.ndarray) -> np.ndarray:
        y = np.asarray(coords, dtype=np.float64)
        if not self.enabled or self.coord_scale is None:
            return y
        return y / self.coord_scale

    def unscale_coords(self, coords: np.ndarray) -> np.ndarray:
        y = np.asarray(coords, dtype=np.float64)
        if not self.enabled or self.coord_scale is None:
            return y
        return y * self.coord_scale

    def fit_risk_scale(self, residuals: np.ndarray) -> bool:
        """Return whether labels supplied a usable scale, rather than the fallback.

        Zero residuals still train the risk head. Their temporary unit scale
        must not permanently lock a fixed scaler before nonzero labels arrive.
        """
        e = np.asarray(residuals, dtype=np.float64).reshape(-1)
        mean = float(np.mean(e)) if e.size else 0.0
        fitted = bool(np.isfinite(mean) and mean > 1e-8)
        self.risk_scale = mean if fitted else 1.0
        return fitted

    def scale_risk(self, residuals: np.ndarray) -> np.ndarray:
        e = np.asarray(residuals, dtype=np.float64)
        if not self.enabled:
            return e
        return e / float(self.risk_scale)

    def unscale_risk(self, risk: np.ndarray) -> np.ndarray:
        r = np.asarray(risk, dtype=np.float64)
        if not self.enabled:
            return r
        return r * float(self.risk_scale)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "coord_scale": self.coord_scale,
            "risk_scale": float(self.risk_scale),
            "enabled": bool(self.enabled),
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.mean = None if state.get("mean") is None else np.asarray(state["mean"], dtype=np.float64)
        self.std = None if state.get("std") is None else np.asarray(state["std"], dtype=np.float64)
        self.coord_scale = (
            None if state.get("coord_scale") is None else np.asarray(state["coord_scale"], dtype=np.float64)
        )
        self.risk_scale = float(state.get("risk_scale", 1.0) or 1.0)
        if self.risk_scale <= 0.0 or not np.isfinite(self.risk_scale):
            self.risk_scale = 1.0
        self.enabled = bool(state.get("enabled", True))


def fit_weighted_ridge(features: np.ndarray, targets: np.ndarray, weights: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Return (d+1, k) weights, last row intercept, intercept unpenalized."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[:, None]
    n = x.shape[0]
    if n != y.shape[0] or n != w.shape[0]:
        raise ValueError("ridge features/targets/weights do not match")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("ridge l2 must be finite and nonnegative")
    # Preserve the sum-weighted objective and unpenalized intercept.
    w = np.maximum(w, 0.0)
    if not np.any(w > 0):
        return np.zeros((x.shape[1] + 1, y.shape[1]), dtype=np.float64)
    if float(l2) > 0.0 and n < x.shape[1] and np.sum(w) > 0:
        xbar = np.average(x, axis=0, weights=w)
        ybar = np.average(y, axis=0, weights=w)
        sw = np.sqrt(w)[:, None]
        xc, yc = (x - xbar) * sw, (y - ybar) * sw
        gram = xc @ xc.T + float(l2) * np.eye(n)
        coef = xc.T @ np.linalg.solve(gram, yc)
        return np.vstack([coef, ybar - xbar @ coef])
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    sw = np.sqrt(np.maximum(w, 0.0))
    xw = xb * sw[:, None]
    yw = y * sw[:, None]
    if float(l2) == 0:
        return np.linalg.lstsq(xw, yw, rcond=None)[0]
    dim = xb.shape[1]
    gram = xw.T @ xw + float(l2) * np.eye(dim, dtype=np.float64)
    gram[-1, -1] -= float(l2)
    rhs = xw.T @ yw
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def ridge_predict(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    if xb.shape[1] != coef.shape[0]:
        raise ValueError("ridge feature dim does not match coefficients")
    return xb @ coef


def design_shrink_gamma(
    grads: np.ndarray,
    f: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray | None = None,
) -> float | None:
    """γ* = E[w a <G,m>] / E[w a ||m||²], a = 1/p − 1. None if the denominator is 0.

    `weights` is the reservoir IPW 1/(p s) so the average is over starts, not audits.
    """
    g = np.asarray(grads, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if g.ndim == 1:
        g = g[None, :]
        f = f[None, :]
    if g.shape[0] != f.shape[0] or g.shape[0] != p.shape[0]:
        raise ValueError("shrink gamma dimensions do not match")
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in (0, 1]")
    if weights is None:
        w = np.ones(p.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != p.shape[0]:
            raise ValueError("shrink gamma weights do not match")
    return design_shrink_from_projections(g @ u, f, u.T @ u, p, w)


def design_shrink_from_projections(coords, f, gram, p, weights=None) -> float | None:
    """Same full-space γ using GᵀU and UᵀU, without an n×D prediction array."""
    gamma = design_shrink_diagnostics(coords, f, gram, p, weights)["raw_gamma"]
    return None if gamma is None else float(np.clip(gamma, 0.0, 2.0))


def design_shrink_diagnostics(coords, f, gram, p, weights=None) -> dict:
    """Report the existing calibration objective, including unclipped γ.

    These statistics explain clipping or a missing estimate; they add no
    sample-count or confidence requirement to calibration.
    """
    coords = np.asarray(coords, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    w = np.ones(p.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.any(p <= 0) or np.any(p > 1) or not np.all(np.isfinite(p)):
        raise ValueError("p must be finite and in (0, 1]")
    a = w * ((1.0 / p) - 1.0)
    num = float(np.sum(a * np.sum(coords * f, axis=1)))
    den = float(np.sum(a * np.sum((f @ gram) * f, axis=1)))
    gamma = num / den if np.isfinite(den) and den > 1e-12 else None
    if gamma is not None and not np.isfinite(gamma):
        gamma = None
    return {"n": len(p), "numerator": num if np.isfinite(num) else None,
            "denominator": den if np.isfinite(den) else None,
            "raw_gamma": gamma, "design_positive_n": int(np.count_nonzero(a > 0)),
            "design_effective_n": float(a.sum() ** 2 / np.dot(a, a)) if np.any(a > 0) else 0.}
