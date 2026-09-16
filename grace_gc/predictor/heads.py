"""Coordinate, risk, and cost heads. Torch is imported only here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REWARD_RISK_DIM = 5


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _numpy_opt_state(optimizer) -> dict:
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
    ):
        torch, nn, _F = _torch()
        if in_dim <= 0 or k <= 0:
            raise ValueError("in_dim and k must be positive")
        self.in_dim = int(in_dim)
        self.k = int(k)
        self.hidden_coord = int(hidden_coord)
        self.hidden_risk = int(hidden_risk)
        self.constant_cost = bool(constant_cost)
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
        # Separate Adam states: a shared optimizer applies stale momentum to
        # heads that were not in that loss, which silently moves f and r_hat.
        self.opt_coord = torch.optim.AdamW(self.coord.parameters(), lr=lr, weight_decay=0.0)
        self.opt_risk = torch.optim.AdamW(self.risk.parameters(), lr=lr, weight_decay=0.0)
        self.opt_cost = torch.optim.AdamW(self.cost.parameters(), lr=lr, weight_decay=0.0)
        self.opt_reward_risk = torch.optim.AdamW(self.reward_risk.parameters(), lr=lr, weight_decay=0.0)
        self.opt_success = torch.optim.AdamW(self.success.parameters(), lr=lr, weight_decay=0.0)

    def _to_tensor(self, x: np.ndarray):
        torch, _nn, _F = _torch()
        return torch.as_tensor(np.asarray(x, dtype=np.float32))

    def forward_numpy(self, features: np.ndarray, cost_feat: np.ndarray | None = None) -> PredictorOutput:
        torch, _nn, F = _torch()
        feat = self._to_tensor(features)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        if feat.shape[-1] != self.in_dim:
            raise ValueError(f"feature dim {feat.shape[-1]} != {self.in_dim}")
        with torch.no_grad():
            f = self.coord(feat)
            r = F.softplus(self.risk(feat)).clamp_min(1e-8)
            if self.constant_cost:
                c = torch.ones(feat.shape[0], 1)
            else:
                if cost_feat is None:
                    cost_feat = np.zeros((feat.shape[0], 3), dtype=np.float32)
                c = F.softplus(self.cost(self._to_tensor(cost_feat).reshape(feat.shape[0], 3))).clamp_min(1e-8)
        return PredictorOutput(
            f=f.cpu().numpy().astype(np.float64),
            r_hat=r.cpu().numpy().reshape(-1).astype(np.float64),
            c_hat=c.cpu().numpy().reshape(-1).astype(np.float64),
        )

    def train_coord(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        torch, _nn, _F = _torch()
        x = self._to_tensor(features)
        y = self._to_tensor(targets)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_coord.zero_grad()
            pred = self.coord(x)
            err = ((pred - y) ** 2).sum(dim=-1)
            loss = (w * err).mean()
            loss.backward()
            self.opt_coord.step()
            last = float(loss.detach().cpu())
        return last

    def train_risk(self, features: np.ndarray, residuals: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        torch, _nn, F = _torch()
        x = self._to_tensor(features)
        e = self._to_tensor(residuals).reshape(-1)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_risk.zero_grad()
            r = F.softplus(self.risk(x)).clamp_min(1e-8).reshape(-1)
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
            pred = F.softplus(self.cost(x)).clamp_min(1e-8).reshape(-1)
            loss = (w * (pred - y) ** 2).mean()
            loss.backward()
            self.opt_cost.step()
            last = float(loss.detach().cpu())
        return last

    def forward_success(self, features: np.ndarray) -> np.ndarray:
        """Prefix-dependent pass-rate q(h) for Reward-CV risk features."""
        torch, _nn, F = _torch()
        feat = self._to_tensor(features)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        if feat.shape[-1] != self.in_dim:
            raise ValueError(f"feature dim {feat.shape[-1]} != {self.in_dim}")
        with torch.no_grad():
            q = torch.sigmoid(self.success(feat)).clamp(1e-6, 1.0 - 1e-6)
        return q.cpu().numpy().reshape(-1).astype(np.float64)

    def train_success(self, features: np.ndarray, rewards: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        torch, _nn, _F = _torch()
        x = self._to_tensor(features)
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
        torch, _nn, F = _torch()
        x = self._to_tensor(reward_feat)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != REWARD_RISK_DIM:
            raise ValueError(f"reward feature dim {x.shape[-1]} != {REWARD_RISK_DIM}")
        with torch.no_grad():
            r = F.softplus(self.reward_risk(x)).clamp_min(1e-8)
        return r.cpu().numpy().reshape(-1).astype(np.float64)

    def train_reward_risk(self, reward_feat: np.ndarray, residuals: np.ndarray, weights: np.ndarray, epochs: int = 2) -> float:
        torch, _nn, F = _torch()
        x = self._to_tensor(reward_feat)
        e = self._to_tensor(residuals).reshape(-1)
        w = self._to_tensor(weights).reshape(-1)
        last = 0.0
        for _ in range(epochs):
            self.opt_reward_risk.zero_grad()
            r = F.softplus(self.reward_risk(x)).clamp_min(1e-8).reshape(-1)
            nll = e / r + torch.log(r)
            loss = (w * nll).mean()
            loss.backward()
            self.opt_reward_risk.step()
            last = float(loss.detach().cpu())
        return last

    def state_dict(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "k": self.k,
            "hidden_coord": self.hidden_coord,
            "hidden_risk": self.hidden_risk,
            "constant_cost": self.constant_cost,
            "coord": {k: v.detach().cpu().numpy() for k, v in self.coord.state_dict().items()},
            "risk": {k: v.detach().cpu().numpy() for k, v in self.risk.state_dict().items()},
            "cost": {k: v.detach().cpu().numpy() for k, v in self.cost.state_dict().items()},
            "opt_coord": _numpy_opt_state(self.opt_coord),
            "opt_risk": _numpy_opt_state(self.opt_risk),
            "opt_cost": _numpy_opt_state(self.opt_cost),
            "reward_risk": {k: v.detach().cpu().numpy() for k, v in self.reward_risk.state_dict().items()},
            "opt_reward_risk": _numpy_opt_state(self.opt_reward_risk),
            "success": {k: v.detach().cpu().numpy() for k, v in self.success.state_dict().items()},
            "opt_success": _numpy_opt_state(self.opt_success),
        }

    def load_state_dict(self, state: dict) -> None:
        torch, _nn, _F = _torch()
        def _load(module, payload):
            tensors = {k: torch.as_tensor(v) for k, v in payload.items()}
            module.load_state_dict(tensors)

        _load(self.coord, state["coord"])
        _load(self.risk, state["risk"])
        _load(self.cost, state["cost"])
        if state.get("opt_coord"):
            _load_opt_state(self.opt_coord, state["opt_coord"])
        if state.get("opt_risk"):
            _load_opt_state(self.opt_risk, state["opt_risk"])
        if state.get("opt_cost"):
            _load_opt_state(self.opt_cost, state["opt_cost"])
        if state.get("reward_risk"):
            _load(self.reward_risk, state["reward_risk"])
        if state.get("opt_reward_risk"):
            _load_opt_state(self.opt_reward_risk, state["opt_reward_risk"])
        if state.get("success"):
            _load(self.success, state["success"])
        if state.get("opt_success"):
            _load_opt_state(self.opt_success, state["opt_success"])
