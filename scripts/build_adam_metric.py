#!/usr/bin/env python3
"""Export the fixed diagonal AdamW quadratic metric from a GRACE checkpoint.

The exported weight for coordinate ``i`` is
``1 / (sqrt(v_i) + eps)**2``.  It is computed before replay and saved as a
standalone ``.npy`` sidecar so the metric is fixed independently of replay
outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.versions import sha256_array, sha256_file


def adam_diagonal_from_checkpoint(payload: dict, eps: float = 1e-8) -> np.ndarray:
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    names = list(payload.get("layout_names") or [])
    actor = payload.get("actor") or {}
    optimizer = payload.get("optimizer") or {}
    groups = optimizer.get("param_groups") or []
    state = optimizer.get("state") or {}
    param_ids = [param_id for group in groups for param_id in group.get("params", [])]
    actor_names = list(actor)
    if not names or not actor_names or len(param_ids) != len(actor_names):
        raise ValueError("checkpoint lacks an ordered actor/optimizer layout")
    by_name = {}
    for name, param_id in zip(actor_names, param_ids):
        item = state.get(param_id, state.get(str(param_id)))
        if item is None:
            raise ValueError(f"optimizer state is missing {name}")
        moment = item.get("exp_avg_sq")
        if moment is None:
            raise ValueError(f"optimizer state for {name} lacks exp_avg_sq")
        array = np.asarray(moment, dtype=np.float64)
        expected = np.asarray(actor[name]).shape
        if array.shape != expected:
            raise ValueError(f"optimizer moment shape for {name} does not match actor")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"optimizer moment for {name} is invalid")
        by_name[name] = array
    pieces = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"optimizer state lacks layout parameter {name}")
        pieces.append(by_name[name].reshape(-1))
    moment = np.concatenate(pieces).astype(np.float64, copy=False)
    weights = 1.0 / np.square(np.sqrt(moment) + float(eps))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Adam metric contains nonfinite or nonpositive weights")
    return weights


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--metric-name", default="adam_diagonal")
    args = parser.parse_args(argv)
    payload = load_checkpoint(args.checkpoint)
    values = adam_diagonal_from_checkpoint(payload, args.eps)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, values)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "metric_name": args.metric_name,
        "metric_kind": "adam_diagonal",
        "formula": "1/(sqrt(exp_avg_sq)+eps)^2",
        "eps": args.eps,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "layout_names": list(payload.get("layout_names") or []),
        "layout_dim": int(values.size),
        "metric_weights_sha256": sha256_array(values),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "sidecar": str(sidecar.resolve()),
                      "layout_dim": int(values.size),
                      "metric_weights_sha256": sha256_array(values)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
