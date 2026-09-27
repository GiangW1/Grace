"""CPU checks for exact block coverage, window semantics and replay alignment."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grace_gc.audit.structured_gain import fit_projected_basis, layer_qv_blocks
from grace_gc.predictor.features import multiwindow_hidden_features
from grace_gc.versions import sha256_file
from scripts.expected_gain_structured_suite import main as structured_main


def _layout(n_layers=4):
    entries = []
    offset = 0
    for layer in range(n_layers):
        for target in ("q", "v"):
            for matrix in ("A", "B"):
                entries.append({"name": f"model.layers.{layer}.self_attn.{target}_proj.lora_{matrix}.default.weight",
                                "offset": offset, "numel": 2})
                offset += 2
    return entries


class StructuredGainTest(unittest.TestCase):
    def test_blocks_cover_all_qv_coordinates_once(self):
        entries = _layout()
        per_layer, groups = layer_qv_blocks(entries, 32)
        self.assertEqual(len(per_layer), 8)
        self.assertEqual(len(groups), 8)
        used = [coordinate for segments in groups.values() for a, b in segments
                for coordinate in range(a, b)]
        self.assertEqual(sorted(used), list(range(32)))
        with self.assertRaisesRegex(ValueError, "overlap"):
            layer_qv_blocks(entries + [entries[0]], 32)

    def test_gram_projection_matches_explicit_svd(self):
        rng = np.random.default_rng(3)
        means = rng.normal(size=(12, 30))
        reference = rng.normal(size=30)
        train = np.arange(8)
        segments = [(0, 5), (12, 21)]
        coord, ref_coord, coeff = fit_projected_basis(means, train, segments, 4, reference, block=3)
        selected = np.concatenate([means[:, a:b] for a, b in segments], axis=1)
        basis = selected[train].T @ coeff
        np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(coord, selected @ basis, atol=1e-10)
        np.testing.assert_allclose(ref_coord, np.concatenate([reference[a:b] for a, b in segments]) @ basis)

    def test_multiwindow_uses_prompt_and_response_segments(self):
        hidden = np.arange(10, dtype=np.float64)[:, None]
        got = multiwindow_hidden_features(hidden, 2, width=2)
        np.testing.assert_allclose(got, [.5, 2.5, 5.5, 8.5])
        short = multiwindow_hidden_features(hidden[:4], 2, width=2)
        self.assertEqual(short[2], 0.)

    def test_suite_uses_aligned_features_and_problem_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = _layout()
            metadata = {"actor_sha256": "actor", "checkpoint_sha256": "checkpoint",
                        "layout_names": [entry["name"] for entry in entries],
                        "layout_dim": 32, "model_path": "model", "lora": {"rank": 2}}
            rng = np.random.default_rng(11)
            for role, ids, t in (("replay", range(6), 512),
                                 ("reference", range(6, 9), 0),
                                 ("check", range(9, 12), 0)):
                directory = root / role
                directory.mkdir()
                rows = [{"problem_id": f"p{i}", "path_id": f"path{i}", "t": t,
                         "features": [float(i), 1.], "mean_norm_sq": 40.,
                         "baseline": 0.} for i in ids]
                (directory / "prefixes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                np.save(directory / "mean_grads.npy", rng.normal(size=(len(rows), 32)))
                (directory / "replay_provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
            feature_dir = root / "features"
            feature_dir.mkdir()
            np.save(feature_dir / "multiwindow_features.npy", rng.normal(size=(6, 7)))
            (feature_dir / "layout_entries.json").write_text(json.dumps(entries), encoding="utf-8")
            feature_meta = {**metadata, "replay_prefix_sha256": sha256_file(root / "replay" / "prefixes.jsonl")}
            (feature_dir / "feature_provenance.json").write_text(json.dumps(feature_meta), encoding="utf-8")
            command = ["--replay-dir", str(root / "replay"), "--reference-dir", str(root / "reference"),
                       "--reference-check-dir", str(root / "check"), "--feature-dir", str(feature_dir),
                       "--run-dir", str(root / "result")]
            self.assertEqual(structured_main(command), 0)
            result = json.loads((root / "result" / "structured_gain_summary.json").read_text())
            self.assertEqual({row["basis"] for row in result["basis_rows"]}, {"global64", "layer_qv64"})
            self.assertEqual(len(result["scalar_rows"]), 10)
            self.assertEqual(len(result["per_layer_reference"]), 8)
            self.assertTrue(result["selected_by_validation_residual"])
            feature_meta["replay_prefix_sha256"] = "wrong"
            (feature_dir / "feature_provenance.json").write_text(json.dumps(feature_meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rows do not match"):
                structured_main(command)
            feature_meta["replay_prefix_sha256"] = sha256_file(root / "replay" / "prefixes.jsonl")
            feature_meta["lora"] = {"rank": 4}
            (feature_dir / "feature_provenance.json").write_text(json.dumps(feature_meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ in lora"):
                structured_main(command)


if __name__ == "__main__":
    unittest.main()
