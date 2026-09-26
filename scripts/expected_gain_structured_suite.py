#!/usr/bin/env python3
"""Compare a global 64-D basis with layer/q-v blocks and fixed window pooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.expected_gain import (fit_predict, full_space_residual, load_replay,
                                           problem_split, reference_gradient, start_experiment_run)
from grace_gc.audit.structured_gain import (fit_projected_basis, layer_qv_blocks,
                                             reference_block_diagnostics)
from grace_gc.versions import sha256_array, sha256_file
from scripts.expected_gain_suite import _check_replay_identity, _directional_labels, _indices, _metrics, _provenance


def _split_roles(rows, manifest, seed):
    split = (json.loads(Path(manifest).read_text(encoding="utf-8")) if manifest else
             problem_split(rows, seed=seed))
    roles = {role: _indices(rows, split[role]) for role in ("train", "validation", "diagnostic")}
    if (any(not len(index) for index in roles.values()) or
            sum(map(len, roles.values())) != len(rows) or
            any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"),
                                                            ("train", "diagnostic"),
                                                            ("validation", "diagnostic")))):
        raise ValueError("split must partition every predictor problem with nonempty roles")
    return split, roles


def _predict_rows(x, target, roles, settings):
    tr = roles["train"]
    for name, model, l2 in settings:
        predicted = fit_predict(model, x[tr], target[tr], x,
                                ridge_l2=l2)
        yield name, predicted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--reference-check-dir", required=True)
    parser.add_argument("--feature-dir", required=True,
                        help="output of extract_expected_gain_windows.py for this replay")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None,
                        help="reuse the original expected-gain problem split")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    started = perf_counter()
    rows, means = load_replay(args.replay_dir)
    ref_rows, ref_means = load_replay(args.reference_dir)
    check_rows, check_means = load_replay(args.reference_check_dir)
    for source in (args.reference_dir, args.reference_check_dir):
        _check_replay_identity(args.replay_dir, source)
    if means.shape[1] != ref_means.shape[1] or means.shape[1] != check_means.shape[1]:
        raise ValueError("replay gradient dimensions differ")
    pools = [{str(row["problem_id"]) for row in group} for group in (rows, ref_rows, check_rows)]
    if any(pools[a] & pools[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("predictor and reference problem pools must be disjoint")
    reference = reference_gradient(ref_rows, ref_means)
    check = reference_gradient(check_rows, check_means)
    replay_meta = _provenance(args.replay_dir)
    stored_direction = replay_meta.get("reference_direction_sha256")
    if stored_direction and stored_direction != sha256_array(reference):
        raise ValueError("replay gain labels use another reference direction")

    feature_dir = Path(args.feature_dir)
    feature_meta = json.loads((feature_dir / "feature_provenance.json").read_text(encoding="utf-8"))
    for key in ("checkpoint_sha256", "actor_sha256", "layout_names", "layout_dim", "model_path", "lora"):
        if feature_meta.get(key) != replay_meta.get(key):
            raise ValueError(f"window features and replay differ in {key}")
    if feature_meta.get("replay_prefix_sha256") != sha256_file(Path(args.replay_dir) / "prefixes.jsonl"):
        raise ValueError("window feature rows do not match this replay")
    windows = np.load(feature_dir / "multiwindow_features.npy", mmap_mode="r")
    entries = json.loads((feature_dir / "layout_entries.json").read_text(encoding="utf-8"))
    if ([entry["name"] for entry in entries] != replay_meta["layout_names"] or
            windows.ndim != 2 or len(windows) != len(rows) or
            not np.all(np.isfinite(windows))):
        raise ValueError("window features or packed LoRA layout are inconsistent")
    per_layer, groups = layer_qv_blocks(entries, means.shape[1])
    split, roles = _split_roles(rows, args.split_manifest, args.seed)
    if not all(row.get("features") is not None for row in rows):
        raise ValueError("replay lacks legacy prefix features")
    legacy = np.stack([np.asarray(row["features"], dtype=np.float64) for row in rows])
    if not np.all(np.isfinite(legacy)):
        raise ValueError("legacy features contain nonfinite values")
    views = {"legacy": legacy, "legacy_plus_multiwindow": np.concatenate((legacy, windows), axis=1)}
    norms = np.asarray([row["mean_norm_sq"] for row in rows], dtype=np.float64)
    gains = _directional_labels(means, reference)
    tr, va, te = (roles[role] for role in ("train", "validation", "diagnostic"))
    settings = [("zero", "zero", 1.), ("constant", "constant", 1.)] + [
        (f"ridge_l2_{l2:g}", "ridge", l2) for l2 in (1., 10., 100.)]
    destination = start_experiment_run(args.run_dir, "expected_gain_structured_suite", vars(args))
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    scalar_rows = []
    for view_name, features in views.items():
        for model_name, pred in _predict_rows(features, gains, roles, settings):
            scalar_rows.append({"feature_set": view_name, "model": model_name,
                                "validation_mse": float(np.mean((pred[va, 0] - gains[va]) ** 2)),
                                "diagnostic": _metrics(pred[te, 0], gains[te], np.mean(gains[tr]))})

    basis_rows, coefficients, basis_blocks = [], {}, {}
    selected = None
    for basis_name, partitions in (("global64", {"global": [(0, means.shape[1])]}),
                                   ("layer_qv64", groups)):
        coordinates, ref_coords = [], []
        block_info = {}
        for name, segments in partitions.items():
            coord, ref_coord, coeff = fit_projected_basis(means, tr, segments,
                                                           64 if basis_name == "global64" else 8,
                                                           reference)
            coordinates.append(coord)
            ref_coords.append(ref_coord)
            coefficients[f"{basis_name}_{name}"] = coeff
            block_info[name] = {"dimension": sum(end - start for start, end in segments),
                                "actual_rank": coeff.shape[1], "segments": segments}
        basis_blocks[basis_name] = block_info
        coord = np.concatenate(coordinates, axis=1)
        ref_coord = np.concatenate(ref_coords)
        if coord.shape[1] == 0:
            basis_rows.append({"basis": basis_name, "status": "unavailable", "blocks": block_info})
            continue
        oracle_energy = np.sum(coord * coord, axis=1)
        for view_name, features in views.items():
            for model_name, pred in _predict_rows(features, coord, roles, settings):
                val_residual = full_space_residual(norms[va], coord[va], pred[va])
                test_residual = full_space_residual(norms[te], coord[te], pred[te])
                row = {"basis": basis_name, "requested_rank": 64,
                       "actual_rank": coord.shape[1], "feature_set": view_name,
                       "model": model_name,
                       "validation_full_residual_mean": float(np.mean(val_residual)),
                       "diagnostic_full_residual_mean": float(np.mean(test_residual)),
                       "diagnostic_zero_residual_mean": float(np.mean(norms[te])),
                       "diagnostic_oracle_mean_energy_fraction":
                           float(np.sum(oracle_energy[te]) / max(float(np.sum(norms[te])), 1e-30)),
                       "diagnostic_gain_from_gradient": _metrics(pred[te] @ ref_coord,
                                                                gains[te], np.mean(gains[tr]))}
                basis_rows.append(row)
                if selected is None or row["validation_full_residual_mean"] < selected[0]:
                    selected = (row["validation_full_residual_mean"], row)
        print(f"basis={basis_name} rank={coord.shape[1]}", flush=True)
    np.savez(destination / "basis_coefficients.npz", **coefficients)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(check))
    result = {"split": split, "reference_check_cosine":
              None if denominator == 0 else float(reference @ check / denominator),
              "per_layer_reference": reference_block_diagnostics(reference, check, per_layer),
              "block_reference": reference_block_diagnostics(reference, check, groups),
              "basis_blocks": basis_blocks,
              "basis_coefficients": "U_block = G_train_block.T @ basis_coefficients[block]",
              "train_indices": tr.tolist(), "scalar_rows": scalar_rows,
              "basis_rows": basis_rows,
              "selected_by_validation_residual": None if selected is None else selected[1],
              "wall_seconds": perf_counter() - started}
    (destination / "structured_gain_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination),
                      "selected": result["selected_by_validation_residual"],
                      "wall_seconds": result["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
