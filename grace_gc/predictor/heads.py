"""Coordinate, risk, and cost heads. Torch is imported only here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.predictor.scale import FeatureScaler, fit_weighted_ridge, ridge_predict

REWARD_RISK_DIM = 5


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _numpy_opt_state(optimizer) -> dict:
    if optimizer is None:
        return {}
    raw = optimizer.state_dict()
    out = {"param_groups": raw["param_groups"], "state": {}}
    for key, item in raw["state"].items():
        out["state"][str(key)] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in item.items()
        }
    return out


def _load_opt_state(optimizer, payload: dict) -> None:
    torch, _nn, _F = _torch()
    state = {"param_groups": payload["param_groups"], "state": {}}
    for key, item in payload.get("state", {}).items():
        conv = {}
        for k, v in item.items():
            conv[k] = torch.as_tensor(v) if isinstance(v, np.ndarray) else v
        state["state"][int(key) if str(key).isdigit() else key] = conv
    optimizer.load_state_dict(state)


@dataclass
class PredictorOutput:
    f: np.ndarray
    r_hat: np.ndarray
    c_hat: np.ndarray


class PredictorHeads:
    def __init__(
        self,
        in_dim: int,
        k: int,
        hidden_coord: int = 256,
        hidden_risk: int = 64,
        constant_cost: bool = True,
        lr: float = 1e-3,
        coord_kind: str = "mlp",
        ridge_l2: float = 1.0,
        use_affine: bool = True,
        shrink_m: bool = True,
        feature_scaler: str = "fixed",
        shrink_calibration: str = "fit",
        calibration_fraction: float = 0.2,
        ridge_weight_normalization: str = "sum",
        train_auxiliary: bool = True,
    ):
        torch, nn, _F = _torch()
        if in_dim <= 0 or k <= 0:
            raise ValueError("in_dim and k must be positive")
        kind = str(coord_kind or "mlp").lower()
        if kind not in {"mlp", "ridge", "linear"}:
            raise ValueError(f"coord_kind must be mlp or ridge, got {coord_kind}")
        if kind == "linear":
            kind = "ridge"
        self.in_dim = int(in_dim)
        self.k = int(k)
        self.hidden_coord = int(hidden_coord)
        self.hidden_risk = int(hidden_risk)
        self.constant_cost = bool(constant_cost)
        self.coord_kind = kind
        self.ridge_l2 = float(ridge_l2)
        self.ridge_w: np.ndarray | None = None
        self.scaler = FeatureScaler(enabled=bool(use_affine))
        self.shrink_m = bool(shrink_m)
        self.m_shrink = 1.0
        if feature_scaler not in {"fixed", "refit"}:
            raise ValueError("feature_scaler must be fixed or refit")
        if shrink_calibration not in {"fit", "holdout"}:
            raise ValueError("shrink_calibration must be fit or holdout")
        if not 0.0 < float(calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        if ridge_weight_normalization not in {"sum", "mean"}:
            raise ValueError("ridge_weight_normalization must be sum or mean")
        self.feature_scaler = feature_scaler
        self.shrink_calibration = shrink_calibration
        self.calibration_fraction = float(calibration_fraction)
        self.ridge_weight_normalization = ridge_weight_normalization
        self.train_auxiliary = bool(train_auxiliary)
        self.seen_problem_ids: set[str] = set()
        self.diagnostic_history_complete = True
        self.risk_scale_fitted = False
        self.coord = nn.Sequential(
            nn.Linear(in_dim, hidden_coord),
            nn.ReLU(),
            nn.Linear(hidden_coord, k),
        )
        self.risk = nn.Sequential(
            nn.Linear(in_dim, hidden_risk),
            nn.ReLU(),
            nn.Linear(hidden_risk, 1),
        )
        self.cost = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.reward_risk = nn.Sequential(
            nn.Linear(REWARD_RISK_DIM, hidden_risk),
            nn.ReLU(),
            nn.Linear(hidden_risk, 1),
        )
        self.success = nn.Sequential(
            nn.Linear(in_dim, hidden_risk),
            nn.ReLU(),
            nn.Linear(hidden_risk, 1),
        )
        # Instantiate in legacy order before releasing unused heads, preserving
        # the RNG state and initial values of all retained heads.
        if kind == "ridge":
            self.coord = None
        if self.constant_cost:
            self.cost = None
        if not self.train_auxiliary:
            self.reward_risk = None
            self.success = None
        # Separate Adam states: a shared optimizer applies stale momentum to
        # heads that were not in that loss, which silently moves f and r_hat.
        self.opt_coord = None if self.coord is None else torch.optim.AdamW(self.coord.parameters(), lr=lr, weight_decay=0.0)
        self.opt_risk = torch.optim.AdamW(self.risk.parameters(), lr=lr, weight_decay=0.0)
        self.opt_cost = None if self.cost is None else torch.optim.AdamW(self.cost.parameters(), lr=lr, weight_decay=0.0)
        self.opt_reward_risk = None if self.reward_risk is None else torch.optim.AdamW(self.reward_risk.parameters(), lr=lr, weight_decay=0.0)
        self.opt_success = None if self.success is None else torch.optim.AdamW(self.success.parameters(), lr=lr, weight_decay=0.0)

    def fit_feature_scaler(self, features: np.ndarray) -> None:
        """Fixed scaling keeps persistent neural heads in the same coordinates."""
        if self.feature_scaler == "refit" or self.scaler.mean is None:
            self.scaler.fit_features(features)

    def _to_tensor(self, x: np.ndarray):
        torch, _nn, _F = _torch()
        return torch.as_tensor(np.asarray(x, dtype=np.float32))

    def _as_2d(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"feature dim {x.shape[-1]} != {self.in_dim}")
        return x

    def _coord_forward_np(self, scaled_feat: np.ndarray) -> np.ndarray:
        if self.coord_kind == "ridge":
            if self.ridge_w is None:
                return np.zeros((scaled_feat.shape[0], self.k), dtype=np.float64)
            return ridge_predict(scaled_feat, self.ridge_w)
        torch, _nn, _F = _torch()
        with torch.no_grad():
            f = self.coord(self._to_tensor(scaled_feat))
        return f.cpu().numpy().astype(np.float64)

    def predict_f(self, features: np.ndarray, shrink: bool | None = None) -> np.ndarray:
        x = self.scaler.transform(self._as_2d(features))
        f = self.scaler.unscale_coords(self._coord_forward_np(x))
        if bool(self.shrink_m if shrink is None else shrink):
            f = float(self.m_shrink) * f
        return f

    def forward_numpy(self, features: np.ndarray, cost_feat: np.ndarray | None = None) -> PredictorOutput:
        torch, _nn, F = _torch()
        raw = self._as_2d(features)
        feat = self._to_tensor(self.scaler.transform(raw))
        with torch.no_grad():
            r = F.softplus(self.risk(feat)) + 1e-8
            if self.constant_cost:
                c = torch.ones(feat.shape[0], 1)
            else:
                if cost_feat is None:
                    cost_feat = np.zeros((feat.shape[0], 3), dtype=np.float32)
                c = F.softplus(self.cost(self._to_tensor(cost_feat).reshape(feat.shape[0], 3))) + 1e-8
        f = self.predict_f(raw)
        return PredictorOutput(
            f=np.asarray(f, dtype=np.float64),
            r_hat=self.scaler.unscale_risk(r.cpu().numpy().reshape(-1).astype(np.float64)),
            c_hat=c.cpu().numpy().reshape(-1).astype(np.float64),
        )

    def train_coord(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        x = self.scaler.transform(self._as_2d(features))
        y = self.scaler.scale_coords(np.asarray(targets, dtype=np.float64))
        if y.ndim == 1:
            y = y[None, :]
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if self.coord_kind == "ridge":
            fit_weights = w / w.sum() if self.ridge_weight_normalization == "mean" and w.sum() > 0 else w
            self.ridge_w = fit_weighted_ridge(x, y, fit_weights, l2=self.ridge_l2)
            pred = ridge_predict(x, self.ridge_w)
            err = np.sum((pred - y) ** 2, axis=1)
            return float(np.mean(w * err))
        torch, _nn, _F = _torch()
        xt = self._to_tensor(x)
        yt = self._to_tensor(y)
        wt = self._to_tensor(w)
        last = 0.0
        for _ in range(epochs):
            self.opt_coord.zero_grad()
            pred = self.coord(xt)
            err = ((pred - yt) ** 2).sum(dim=-1)
            loss = (wt * err).mean()
            loss.backward()
            self.opt_coord.step()
            last = float(loss.detach().cpu())
        return last

    def risk_diagnostics(self, features: np.ndarray) -> dict:
        """Observe numerical saturation without changing outputs or optimizer."""
        torch, _nn, F = _torch()
        x = self._to_tensor(self.scaler.transform(self._as_2d(features)))
        with torch.no_grad():
            logits = self.risk(x).reshape(-1)
            positive = F.softplus(logits)
        values = logits.cpu().numpy().astype(np.float64)
        positive = positive.cpu().numpy().astype(np.float64)
        finite = np.isfinite(values)
        return {"n": int(values.size), "logit_finite_n": int(finite.sum()),
                "logit_min": float(values[finite].min()) if np.any(finite) else None,
                "logit_max": float(values[finite].max()) if np.any(finite) else None,
                "softplus_finite_n": int(np.isfinite(positive).sum()),
                "softplus_near_floor_fraction": float(np.mean(positive <= 1e-8)) if values.size else None,
                "softplus_zero_n": int(np.count_nonzero(positive == 0))}

    def train_risk(self, features: np.ndarray, residuals: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        torch, _nn, F = _torch()
        x = self._to_tensor(self.scaler.transform(self._as_2d(features)))
        e = self._to_tensor(self.scaler.scale_risk(residuals)).reshape(-1)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_risk.zero_grad()
            r = (F.softplus(self.risk(x)) + 1e-8).reshape(-1)
            nll = e / r + torch.log(r)
            loss = (w * nll).mean()
            loss.backward()
            self.opt_risk.step()
            last = float(loss.detach().cpu())
        return last

    def train_cost(self, cost_feat: np.ndarray, targets: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        if self.constant_cost:
            return 0.0
        torch, _nn, F = _torch()
        x = self._to_tensor(cost_feat).reshape(-1, 3)
        y = self._to_tensor(targets).reshape(-1)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_cost.zero_grad()
            pred = (F.softplus(self.cost(x)) + 1e-8).reshape(-1)
            loss = (w * (pred - y) ** 2).mean()
            loss.backward()
            self.opt_cost.step()
            last = float(loss.detach().cpu())
        return last

    def forward_success(self, features: np.ndarray) -> np.ndarray:
        """Prefix-dependent pass-rate q(h) for Reward-CV risk features."""
        torch, _nn, F = _torch()
        if self.success is None:
            raise ValueError("success head is disabled for this method")
        feat = self._to_tensor(self.scaler.transform(self._as_2d(features)))
        with torch.no_grad():
            q = torch.sigmoid(self.success(feat)).clamp(1e-6, 1.0 - 1e-6)
        return q.cpu().numpy().reshape(-1).astype(np.float64)

    def train_success(self, features: np.ndarray, rewards: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        if self.success is None:
            return 0.0
        torch, _nn, _F = _torch()
        x = self._to_tensor(self.scaler.transform(self._as_2d(features)))
        y = self._to_tensor(rewards).reshape(-1).clamp(0.0, 1.0)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_success.zero_grad()
            q = torch.sigmoid(self.success(x)).clamp(1e-6, 1.0 - 1e-6).reshape(-1)
            nll = -(y * torch.log(q) + (1.0 - y) * torch.log(1.0 - q))
            loss = (w * nll).mean()
            loss.backward()
            self.opt_success.step()
            last = float(loss.detach().cpu())
        return last

    def forward_reward_risk(self, reward_feat: np.ndarray) -> np.ndarray:
        if self.reward_risk is None:
            raise ValueError("reward risk head is disabled for this method")
        torch, _nn, F = _torch()
        x = self._to_tensor(reward_feat)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != REWARD_RISK_DIM:
            raise ValueError(f"reward feature dim {x.shape[-1]} != {REWARD_RISK_DIM}")
        with torch.no_grad():
            r = F.softplus(self.reward_risk(x)) + 1e-8
        return r.cpu().numpy().reshape(-1).astype(np.float64)

    def train_reward_risk(self, reward_feat: np.ndarray, residuals: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        if self.reward_risk is None:
            return 0.0
        torch, _nn, F = _torch()
        x = self._to_tensor(reward_feat)
        e = self._to_tensor(residuals).reshape(-1)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_reward_risk.zero_grad()
            r = (F.softplus(self.reward_risk(x)) + 1e-8).reshape(-1)
            nll = e / r + torch.log(r)
            loss = (w * nll).mean()
            loss.backward()
            self.opt_reward_risk.step()
            last = float(loss.detach().cpu())
        return last

    def state_dict(self) -> dict:
        def module_state(module):
            return {} if module is None else {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}

        return {
            "in_dim": self.in_dim,
            "k": self.k,
            "hidden_coord": self.hidden_coord,
            "hidden_risk": self.hidden_risk,
            "constant_cost": self.constant_cost,
            "coord_kind": self.coord_kind,
            "ridge_l2": self.ridge_l2,
            "ridge_w": self.ridge_w,
            "use_affine": bool(self.scaler.enabled),
            "scaler": self.scaler.state_dict(),
            "shrink_m": self.shrink_m,
            "m_shrink": float(self.m_shrink),
            "feature_scaler": self.feature_scaler,
            "shrink_calibration": self.shrink_calibration,
            "calibration_fraction": self.calibration_fraction,
            "ridge_weight_normalization": self.ridge_weight_normalization,
            "train_auxiliary": self.train_auxiliary,
            "seen_problem_ids": sorted(self.seen_problem_ids),
            "diagnostic_history_complete": self.diagnostic_history_complete,
            "risk_scale_fitted": self.risk_scale_fitted,
            "coord": module_state(self.coord),
            "risk": module_state(self.risk),
            "cost": module_state(self.cost),
            "opt_coord": _numpy_opt_state(self.opt_coord),
            "opt_risk": _numpy_opt_state(self.opt_risk),
            "opt_cost": _numpy_opt_state(self.opt_cost),
            "reward_risk": module_state(self.reward_risk),
            "opt_reward_risk": _numpy_opt_state(self.opt_reward_risk),
            "success": module_state(self.success),
            "opt_success": _numpy_opt_state(self.opt_success),
        }

    def load_state_dict(self, state: dict) -> None:
        torch, _nn, _F = _torch()
        def _load(module, payload):
            if module is None or not payload:
                return
            tensors = {k: torch.as_tensor(v) for k, v in payload.items()}
            module.load_state_dict(tensors)

        if state.get("coord"):
            _load(self.coord, state["coord"])
        _load(self.risk, state["risk"])
        _load(self.cost, state.get("cost"))
        if state.get("opt_coord") and self.opt_coord is not None:
            _load_opt_state(self.opt_coord, state["opt_coord"])
        if state.get("opt_risk"):
            _load_opt_state(self.opt_risk, state["opt_risk"])
        if state.get("opt_cost") and self.opt_cost is not None:
            _load_opt_state(self.opt_cost, state["opt_cost"])
        if state.get("reward_risk"):
            _load(self.reward_risk, state["reward_risk"])
        if state.get("opt_reward_risk") and self.opt_reward_risk is not None:
            _load_opt_state(self.opt_reward_risk, state["opt_reward_risk"])
        if state.get("success"):
            _load(self.success, state["success"])
        if state.get("opt_success") and self.opt_success is not None:
            _load_opt_state(self.opt_success, state["opt_success"])
        if state.get("coord_kind"):
            kind = str(state["coord_kind"]).lower()
            self.coord_kind = "ridge" if kind == "linear" else kind
        if state.get("ridge_l2") is not None:
            self.ridge_l2 = float(state["ridge_l2"])
        if "ridge_w" in state:
            self.ridge_w = None if state["ridge_w"] is None else np.asarray(state["ridge_w"], dtype=np.float64)
        self.scaler.load_state_dict(state.get("scaler"))
        self.feature_scaler = str(state.get("feature_scaler", "refit"))
        self.shrink_calibration = str(state.get("shrink_calibration", "fit"))
        self.calibration_fraction = float(state.get("calibration_fraction", .2))
        self.ridge_weight_normalization = str(state.get("ridge_weight_normalization", "sum"))
        self.seen_problem_ids = set(state.get("seen_problem_ids", []))
        self.diagnostic_history_complete = bool(state.get("diagnostic_history_complete", False))
        self.risk_scale_fitted = bool(state.get("risk_scale_fitted", False))
        if "shrink_m" in state:
            self.shrink_m = bool(state["shrink_m"])
        if state.get("m_shrink") is not None:
            self.m_shrink = float(state["m_shrink"])
            if not np.isfinite(self.m_shrink):
                self.m_shrink = 1.0


def predictor_from_spec(in_dim: int, k: int, spec: dict | None = None, **overrides) -> PredictorHeads:
    raw = dict(spec or {})
    raw.update(overrides)
    return PredictorHeads(
        in_dim=int(in_dim),
        k=int(k),
        hidden_coord=int(raw.get("hidden_coord", 256)),
        hidden_risk=int(raw.get("hidden_risk", 64)),
        constant_cost=bool(raw.get("constant_cost", True)),
        lr=float(raw.get("lr", 1e-3)),
        coord_kind=str(raw.get("coord_kind", "mlp")),
        ridge_l2=float(raw.get("ridge_l2", 1.0)),
        use_affine=bool(raw.get("use_affine", True)),
        shrink_m=bool(raw.get("shrink_m", True)),
        feature_scaler=str(raw.get("feature_scaler", "refit")),
        shrink_calibration=str(raw.get("shrink_calibration", "fit")),
        calibration_fraction=float(raw.get("calibration_fraction", .2)),
        ridge_weight_normalization=str(raw.get("ridge_weight_normalization", "sum")),
        train_auxiliary=bool(raw.get("train_auxiliary", True)),
    )
