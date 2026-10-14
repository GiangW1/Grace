#!/usr/bin/env python3
"""Score a frozen direct-gradient predictor on fresh audit continuations, without refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from grace_gc.logging_util.run_dir import atomic_write_text


def prediction_row(raw):
    row = {"problem_id": raw["problem_id"], "t": raw["t"], "path_id": raw.get("path_id"),
           "finished": bool(raw.get("finished", False))}
    f = raw.get("effective_coords")
    if f is None or raw.get("true_grad_coords") is None:
        return {**row, "status": "unavailable", "reason": "missing frozen prediction or full-gradient labels"}
    gram = raw.get("basis_gram")
    if not isinstance(gram, dict) or "identity_dim" not in gram:
        raise ValueError("direct-gradient report requires an implicit identity basis, not PCA/sketched coordinates")
    g, f = np.asarray(raw["true_grad_coords"], float), np.asarray(f, float)
    if g.ndim != 2 or g.shape[1] != int(gram["identity_dim"]) or f.shape != (g.shape[1],):
        raise ValueError("direct-gradient label/prediction dimensions disagree")
    if not np.isfinite(g).all() or not np.isfinite(f).all():
        raise ValueError("non-finite gradient labels or predictions")
    if len(g) == 0:
        return {**row, "status": "unavailable", "reason": "no continuations"}
    mean = g.mean(axis=0)
    energy = float(np.mean(np.sum(g * g, axis=1)))
    residual = float(np.mean(np.sum((g - f) ** 2, axis=1)))
    noise = float(np.sum(np.var(g, axis=0, ddof=1))) if len(g) > 1 else None
    mean_energy = float(mean @ mean)
    mean_error = float((mean - f) @ (mean - f))
    denominator = np.linalg.norm(mean) * np.linalg.norm(f)
    return {**row, "status": "available", "n_continuations": len(g), "dimension": g.shape[1],
            "zero_prediction_mse": energy, "frozen_prediction_mse": residual,
            "empirical_mean_error_sq": mean_error,
            "conditional_noise_trace": noise,
            # Negative unbiased estimates are legitimate small-sample outcomes. Do not clip.
            "noise_corrected_mean_energy": None if noise is None else mean_energy - noise / len(g),
            "noise_corrected_mean_error_sq": None if noise is None else mean_error - noise / len(g),
            "prediction_dot_empirical_mean": float(f @ mean),
            "prediction_norm_sq": float(f @ f),
            "cosine_to_empirical_mean": float(np.clip((f @ mean) / denominator, -1, 1)) if denominator > 0 else None}


def aggregate(rows, p):
    available = [r for r in rows if r["status"] == "available"]
    out = {"n_prefixes": len(rows), "n_available": len(available),
           "n_problems": len({r["problem_id"] for r in available})}
    for key in ("zero_prediction_mse", "frozen_prediction_mse", "conditional_noise_trace",
                "noise_corrected_mean_energy", "noise_corrected_mean_error_sq", "cosine_to_empirical_mean"):
        values = [r[key] for r in available if r[key] is not None]
        out[key] = float(np.mean(values)) if values else None
    zero, predicted = out["zero_prediction_mse"], out["frozen_prediction_mse"]
    out["realized_gradient_mse_ratio_vs_zero"] = predicted / zero if zero is not None and zero > 0 else None
    out["fixed_p"] = p
    # Finished trajectories are always included (p_i=1), even when their G is nonzero.
    # Average contributions over the original available prefixes, retaining their zero variance.
    for name, key in (("uniform_ht_added_variance_trace", "zero_prediction_mse"),
                      ("uniform_cv_added_variance_trace", "frozen_prediction_mse")):
        contributions = [0. if r["finished"] else (1 - p) / p * r[key] for r in available]
        out[name] = float(np.mean(contributions)) if contributions else None
    return out


def analyze(audit_dir, p=0.5):
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    root = Path(audit_dir)
    rows = []
    with (root / "audit_bundles.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                # Sufficient direct-coordinate labels are in JSON; don't load huge duplicate sidecars.
                rows.append(prediction_row(json.loads(line)))
    eligible = [r for r in rows if not r["finished"]]
    source = root / "audit_summary.json"
    context = json.loads(source.read_text()) if source.is_file() else {}
    return {"checkpoint": context.get("checkpoint"), "audit_manifest": context.get("audit_manifest"),
            "all_prefixes": aggregate(rows, p), "eligible_prefixes": aggregate(eligible, p),
            "by_t": [{"t": t, **aggregate([r for r in eligible if r["t"] == t], p)}
                     for t in sorted({r["t"] for r in rows})], "prefixes": rows,
            "scope": "Frozen effective prediction (including checkpoint shrinkage), no fitting on audit labels. Equal weight per prefix; repeated continuations are not independent problems.",
            "variance_note": "Equal-prefix mean of (1-p_i)/p_i E||G-f||^2: fixed_p applies only to unfinished prefixes; finished prefixes have p_i=1 and zero added stopping variance. This isolates independent Bernoulli completion on the same actor/prefix distribution. It excludes full-PG variance and predictor cost; it is NOT measured online GRACE variance or speedup.",
            "mean_note": "Subtract sample covariance trace/n to remove continuation noise from squared mean estimates. Negative estimates are retained. Cosine uses the noisy empirical mean. No training-seed confidence is implied."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args(argv)
    result = analyze(args.audit_dir, args.p)
    dest = Path(args.audit_dir) / "direct_prediction_summary.json"
    atomic_write_text(dest, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
