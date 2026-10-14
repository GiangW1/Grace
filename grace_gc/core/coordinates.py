"""O(1)-storage identity map for direct full-trainable-space prediction."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IdentityBasis:
    dim: int
    __array_priority__ = 10000
    ndim = 2

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("identity dimension must be positive")

    @property
    def shape(self):
        return (self.dim, self.dim)

    @property
    def size(self):
        return self.dim * self.dim

    @property
    def T(self):
        return self

    def __matmul__(self, other):
        if isinstance(other, IdentityBasis):
            if other.dim != self.dim:
                raise ValueError("identity dimensions disagree")
            return self
        array = np.asarray(other, dtype=np.float64)
        if array.shape[0] != self.dim:
            raise ValueError("identity input dimension mismatch")
        return array.copy()

    def __rmatmul__(self, other):
        array = np.asarray(other, dtype=np.float64)
        if array.shape[-1] != self.dim:
            raise ValueError("identity input dimension mismatch")
        return array.copy()

    def tolist(self):
        return {"identity_dim": self.dim}


def as_basis(value):
    if isinstance(value, dict) and "identity_dim" in value:
        return IdentityBasis(int(value["identity_dim"]))
    return value if isinstance(value, IdentityBasis) else np.asarray(value, dtype=np.float64)


def basis_pinv(value):
    value = as_basis(value)
    return value if isinstance(value, IdentityBasis) else np.linalg.pinv(value)


def prediction_basis(dim, cfg):
    mode = cfg.get("target_space", "projected")
    if mode == "direct":
        return IdentityBasis(dim)
    if mode != "projected":
        raise ValueError("predictor.target_space must be projected or direct")
    return np.eye(dim, min(int(cfg.get("k", 8)), dim), dtype=np.float64)
