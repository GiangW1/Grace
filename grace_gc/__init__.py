"""GRACE-GC reference implementation.

Core math is CPU-importable. GPU backends are imported only from
``grace_gc.backends`` entry points.
"""

from grace_gc.core.allocation import allocate_continuation
from grace_gc.core.estimator import (
    batch_ht_mean,
    dual_stream_mean,
    ht_estimate,
    optimizer_grad_from_ghat,
    prediction_correction,
)

__version__ = "0.1.0"

__all__ = [
    "allocate_continuation",
    "batch_ht_mean",
    "dual_stream_mean",
    "ht_estimate",
    "optimizer_grad_from_ghat",
    "prediction_correction",
]
