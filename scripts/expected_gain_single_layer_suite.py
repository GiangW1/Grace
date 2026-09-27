#!/usr/bin/env python3
"""Fast single-layer gradient predictability audit on frozen expected-gain replays."""

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

from grace_gc.audit.expected_gain import (fit_predict, load_replay, problem_split,
                                           reference_gradient, start_experiment_run)
from grace_gc.audit.structured_gain import fit_projected_basis, layer_qv_blocks
from grace_gc.versions import sha256_array, sha256_file
from scripts.expected_gain_suite import (_check_replay_identity, _indices, _metrics,
                                         _provenance)


def _split_roles(rows, manifest, seed):
    split = (json.loads(Path(manifest).read_text(encoding="utf-8")) if manifest else
             problem_split(rows, seed=seed))
    roles = {role: _indices(rows, split[role])
             for role in ("train", "validation", "diagnostic")}
    if (any(not len(index) for index in roles.values()) or
            sum(map(len, roles.values())) != len(rows) or
            any(set(split[a]) & set(split[b])
                for a, b in (("train", "validation"), ("train", "diagnostic"),
                             ("validation", "diagnostic")))):
        raise ValueError("split must partition every predictor problem with nonempty roles")
    return split, roles


def _csv_ints(value, name):
    try:
        result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not result:
        raise ValueError(f"{name} must contain at least one integer")
    return result


def _model_settings(names):
    for name in names:
        if name in {"zero", "constant"}:
            yield name, name, 1.0
        elif name == "ridge":
            for l2 in (1.0, 10.0, 100.0):
                yield f"ridge_l2_{l2:g}", "ridge", l2
        elif name in {"mlp64", "mlp256"}:
            yield name, "mlp", int(name[3:])
        else:
            raise ValueError(f"unknown predictor model: {name}")


def _predict(name, model, scale, x_train, y_train, x_eval, args, offset):
    return fit_predict(model, x_train, y_train, x_eval,
                       ridge_l2=scale if model == "ridge" else 1.0,
                       mlp_hidden=scale if model == "mlp" else 64,
                       mlp_epochs=args.mlp_epochs, seed=args.seed + offset,
                       device=args.device)


def _layer_slice(values, segments):
    return np.concatenate([np.asarray(values[:, left:right], dtype=np.float64)
                           for left, right in segments], axis=1)


def _reference_diagnostic(reference, check, segments):
    first = np.concatenate([reference[left:right] for left, right in segments])
    second = np.concatenate([check[left:right] for left, right in segments])
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return first, {"dimension": int(len(first)),
                   "reference_norm": float(np.linalg.norm(first)),
                   "reference_energy": float(first @ first),
                   "reference_check_cosine": (None if denominator == 0 else
                                               float(first @ second / denominator))}


def _check_feature_identity(feature_dir, replay_dir, replay_meta):
    feature_dir = Path(feature_dir)
    metadata_path = feature_dir / "feature_provenance.json"
    entries_path = feature_dir / "layout_entries.json"
    if not metadata_path.exists() or not entries_path.exists():
        raise ValueError("feature-dir needs feature_provenance.json and layout_entries.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("checkpoint_sha256", "actor_sha256", "layout_names", "layout_dim",
                "model_path", "lora"):
        if metadata.get(key) != replay_meta.get(key):
            raise ValueError(f"feature and replay differ in {key}")
    if metadata.get("replay_prefix_sha256") != sha256_file(Path(replay_dir) / "prefixes.jsonl"):
        raise ValueError("feature rows do not match replay prefixes")
    entries = json.loads(entries_path.read_text(encoding="utf-8"))
    if [entry["name"] for entry in entries] != replay_meta["layout_names"]:
        raise ValueError("feature layout names do not match replay layout")
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--reference-check-dir", required=True)
    parser.add_argument("--feature-dir", required=True,
                        help="feature directory produced by extract_expected_gain_windows.py")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--layers", default="0,12,24,35",
                        help="layer ids to screen; use one id for a single-layer run")
    parser.add_argument("--ranks", default="8,64")
    parser.add_argument("--models", default="zero,constant,ridge",
                        help="fast default; mlp64/mlp256 are optional")
    parser.add_argument("--include-multiwindow", action="store_true",
                        help="also evaluate legacy plus extracted fixed-window features")
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    layers = _csv_ints(args.layers, "layers")
    ranks = _csv_ints(args.ranks, "ranks")
    if any(layer < 0 for layer in layers) or any(rank <= 0 for rank in ranks):
        raise ValueError("layers must be nonnegative and ranks must be positive")
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("choose at least one predictor model")
    started = perf_counter()

    rows, means = load_replay(args.replay_dir)
    ref_rows, ref_means = load_replay(args.reference_dir)
    check_rows, check_means = load_replay(args.reference_check_dir)
    for source in (args.reference_dir, args.reference_check_dir):
        _check_replay_identity(args.replay_dir, source)
    if means.shape[1] != ref_means.shape[1] or means.shape[1] != check_means.shape[1]:
        raise ValueError("replay gradient dimensions differ")
    pools = [{str(row["problem_id"]) for row in group}
             for group in (rows, ref_rows, check_rows)]
    if any(pools[a] & pools[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("predictor and reference problem pools must be disjoint")

    reference = reference_gradient(ref_rows, ref_means)
    check = reference_gradient(check_rows, check_means)
    replay_meta = _provenance(args.replay_dir)
    stored_direction = replay_meta.get("reference_direction_sha256")
    if stored_direction and stored_direction != sha256_array(reference):
        raise ValueError("replay gain labels use another reference direction")
    entries = _check_feature_identity(args.feature_dir, args.replay_dir, replay_meta)

    if not all(row.get("features") is not None for row in rows):
        raise ValueError("replay lacks legacy prefix features; run the audit with store_features")
    legacy = np.stack([np.asarray(row["features"], dtype=np.float64) for row in rows])
    if legacy.ndim != 2 or not np.all(np.isfinite(legacy)):
        raise ValueError("legacy features are inconsistent or nonfinite")
    views = {"legacy": legacy}
    if args.include_multiwindow:
        windows = np.load(Path(args.feature_dir) / "multiwindow_features.npy", mmap_mode="r")
        if windows.ndim != 2 or len(windows) != len(rows) or not np.all(np.isfinite(windows)):
            raise ValueError("multiwindow features are inconsistent or nonfinite")
        views["legacy_plus_multiwindow"] = np.concatenate((legacy, windows), axis=1)

    per_layer, _ = layer_qv_blocks(entries, means.shape[1])
    available_layers = sorted({int(name.split("_")[1]) for name in per_layer})
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"requested layers are absent from layout: {missing}")
    split, roles = _split_roles(rows, args.split_manifest, args.seed)
    tr, va, te = (roles[name] for name in ("train", "validation", "diagnostic"))
    settings = list(_model_settings(model_names))
    destination = start_experiment_run(args.run_dir, "expected_gain_single_layer_suite", vars(args))
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    scalar_rows, basis_rows, layer_diagnostics = [], [], {}
    coefficients = {}
    selected = {}
    for layer in layers:
        segments = per_layer[f"layer_{layer}_q"] + per_layer[f"layer_{layer}_v"]
        layer_reference, diagnostics = _reference_diagnostic(reference, check, segments)
        layer_means = _layer_slice(means, segments)
        layer_norm_sq = np.sum(layer_means * layer_means, axis=1)
        total_mean_energy = float(np.sum(layer_norm_sq[te]))
        diagnostics["diagnostic_mean_gradient_energy"] = total_mean_energy
        diagnostics["segments"] = segments
        diagnostics["q_dimension"] = int(sum(right - left
                                              for left, right in per_layer[f"layer_{layer}_q"]))
        diagnostics["v_dimension"] = int(sum(right - left
                                              for left, right in per_layer[f"layer_{layer}_v"]))
        layer_diagnostics[str(layer)] = diagnostics
        gains = layer_means @ layer_reference
        for view_name, features in views.items():
            for offset, (model_name, model, scale) in enumerate(settings):
                predicted = _predict(model_name, model, scale, features[tr], gains[tr, None],
                                     features, args, offset=layer * 10000 + offset)
                scalar_rows.append({"layer": layer, "feature_set": view_name,
                                    "model": model_name, "target": "direct_layer_gain",
                                    "validation_mse": float(np.mean((predicted[va, 0] - gains[va]) ** 2)),
                                    "diagnostic": _metrics(predicted[te, 0], gains[te],
                                                            np.mean(gains[tr]))})
        for rank in ranks:
            coordinates, ref_coords, coeff = fit_projected_basis(
                means, tr, segments, rank, reference)
            key = f"layer_{layer}_rank_{rank}"
            coefficients[key] = coeff
            oracle_energy = np.sum(coordinates * coordinates, axis=1)
            energy_fraction = (None if total_mean_energy == 0 else
                               float(np.sum(oracle_energy[te]) / total_mean_energy))
            best = None
            for view_name, features in views.items():
                for offset, (model_name, model, scale) in enumerate(settings):
                    predicted = _predict(model_name, model, scale, features[tr], coordinates[tr],
                                         features, args,
                                         offset=layer * 100000 + rank * 1000 + offset)
                    target = coordinates[te]
                    pred = predicted[te]
                    residual = (layer_norm_sq[te] - 2. * np.sum(pred * target, axis=1) +
                                np.sum(pred * pred, axis=1))
                    projected_gain = pred @ ref_coords
                    row = {"layer": layer, "rank": rank, "feature_set": view_name,
                           "model": model_name, "actual_rank": int(coordinates.shape[1]),
                           "validation_coordinate_mse": float(np.mean((predicted[va] - coordinates[va]) ** 2)),
                           "diagnostic_coordinate_mse": float(np.mean((pred - target) ** 2)),
                           "validation_layer_residual_mean": float(
                               np.mean(layer_norm_sq[va] - 2. * np.sum(predicted[va] * coordinates[va], axis=1) +
                                       np.sum(predicted[va] * predicted[va], axis=1))),
                           "diagnostic_layer_residual_mean": float(np.mean(residual)),
                           "diagnostic_zero_mean_gradient_residual_mean": float(np.mean(layer_norm_sq[te])),
                           "diagnostic_projected_energy_fraction": energy_fraction,
                           "diagnostic_projected_gain": _metrics(projected_gain, gains[te],
                                                                  np.mean(gains[tr]))}
                    basis_rows.append(row)
                    if best is None or row["validation_layer_residual_mean"] < best[0]:
                        best = (row["validation_layer_residual_mean"], row)
            selected[f"layer_{layer}_rank_{rank}"] = None if best is None else best[1]
        print(f"layer={layer} ranks={ranks}", flush=True)

    np.savez(destination / "basis_coefficients.npz", **coefficients)
    result = {"split": split, "layers": layers, "ranks": ranks,
              "models": model_names, "feature_sets": list(views),
              "gradient_dimension": int(means.shape[1]),
              "layer_diagnostics": layer_diagnostics, "scalar_rows": scalar_rows,
              "basis_rows": basis_rows, "selected_by_validation": selected,
              "wall_seconds": perf_counter() - started}
    (destination / "single_layer_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination), "selected": selected,
                      "wall_seconds": result["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
