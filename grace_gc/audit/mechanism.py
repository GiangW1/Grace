"""Small full-space sufficient statistics, separate from rho headline metrics."""
from __future__ import annotations

import numpy as np


def full_space_decomposition(norm_sq, true_coords, gram, predicted_coords):
    """||G-Uf||² = ||G-P_span(U)G||² + ||P_span(U)G-Uf||².

    true_coords is U.T @ G, not projected coefficients when U is nonorthogonal.
    This describes realized gradient labels; it is not a conditional-mean oracle.
    """
    missing = [name for name, value in (("true_grad_norm_sq", norm_sq), ("true_grad_coords", true_coords),
                                       ("basis_gram", gram)) if value is None]
    if missing:
        return {"status": "unavailable", "missing": missing, "n_labels": 0}
    energy, coords, gram = np.asarray(norm_sq, float).reshape(-1), np.asarray(true_coords, float), np.asarray(gram, float)
    if energy.size == 0 or gram.size == 0:
        return {"status": "unavailable", "missing": ["full_space_labels_or_basis"], "n_labels": 0}
    if coords.ndim != 2 or coords.shape[0] != len(energy) or gram.shape != (coords.shape[1], coords.shape[1]):
        raise ValueError("full-space norm/coordinate/Gram dimensions disagree")
    projected_coefficients = coords @ np.linalg.pinv(gram)
    captured = np.maximum(np.sum(projected_coefficients * coords, axis=1), 0.)
    omitted = np.maximum(energy-captured, 0.)
    coordinate_error = residual = None
    if predicted_coords is not None:
        f = np.broadcast_to(np.asarray(predicted_coords, float), coords.shape)
        coordinate_error = np.maximum(captured-2*np.sum(coords*f, axis=1)+np.sum((f@gram)*f, axis=1), 0.)
        residual = omitted+coordinate_error
    return {"status": "available" if predicted_coords is not None else "partial", "n_labels": len(energy),
            "gradient_energy": energy.tolist(), "captured_energy": captured.tolist(),
            "subspace_omission_energy": omitted.tolist(),
            "coordinate_prediction_error_energy": None if coordinate_error is None else coordinate_error.tolist(),
            "prediction_error_energy": None if residual is None else residual.tolist(),
            "missing": [] if predicted_coords is not None else ["effective_prediction_coordinates"]}


def summarize_decompositions(rows):
    """Label-weighted descriptions only; no changes to problem-level rho averages."""
    result = {"n_bundles": len(rows), "n_unavailable_bundles": sum(r["status"] == "unavailable" for r in rows)}
    for key in ("gradient_energy", "captured_energy", "subspace_omission_energy",
                "coordinate_prediction_error_energy", "prediction_error_energy"):
        values = [value for row in rows for value in (row.get(key) or [])]
        result[key] = {"n_labels": len(values), "mean": float(np.mean(values)) if values else None}
    paired = [row for row in rows if row.get("captured_energy") is not None]
    total = sum(sum(row["gradient_energy"]) for row in paired)
    result["captured_fraction_of_realized_gradient_energy"] = (
        sum(sum(row["captured_energy"]) for row in paired)/total if total > 0 else None)
    return result


def paired_geometry(reference, actual):
    reference, actual = np.asarray(reference, float), np.asarray(actual, float)
    ref_sq, actual_sq = float(reference@reference), float(actual@actual)
    error = actual-reference
    error_sq = float(error@error)
    return {"reference_norm": float(np.sqrt(ref_sq)), "actual_norm": float(np.sqrt(actual_sq)),
            "difference_norm": float(np.sqrt(error_sq)), "squared_error": error_sq,
            "relative_squared_error": error_sq/ref_sq if ref_sq > 0 else None,
            "cosine": float(np.clip(reference@actual/np.sqrt(ref_sq*actual_sq), -1., 1.))
                      if ref_sq > 0 and actual_sq > 0 else None}


def summarize_geometries(rows):
    return {key: {"n_defined": sum(row[key] is not None for row in rows),
                  "mean": float(np.mean([row[key] for row in rows if row[key] is not None]))
                          if any(row[key] is not None for row in rows) else None}
            for key in ("cosine", "squared_error", "relative_squared_error")}
