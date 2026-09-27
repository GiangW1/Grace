"""CPU checks for the fast single-layer expected-gain suite."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grace_gc.versions import sha256_file
from scripts.expected_gain_single_layer_suite import main as single_layer_main


def _layout(n_layers=3):
    entries = []
    offset = 0
    for layer in range(n_layers):
        for target in ("q", "v"):
            for matrix in ("A", "B"):
                entries.append({"name": f"model.layers.{layer}.self_attn.{target}_proj.lora_{matrix}.default.weight",
                                "offset": offset, "numel": 2})
                offset += 2
    return entries


class SingleLayerSuiteTest(unittest.TestCase):
    def test_single_layer_suite_reports_direct_and_projected_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = _layout()
            metadata = {"actor_sha256": "actor", "checkpoint_sha256": "checkpoint",
                        "layout_names": [entry["name"] for entry in entries],
                        "layout_dim": 24, "model_path": "model", "lora": {"rank": 2}}
            rng = np.random.default_rng(13)
            for role, ids in (("replay", range(9)), ("reference", range(9, 12)),
                              ("check", range(12, 15))):
                directory = root / role
                directory.mkdir()
                rows = [{"problem_id": f"p{i}", "path_id": f"path{i}", "t": 0,
                         "features": [float(i), 1.], "mean_norm_sq": 40.,
                         "baseline": 0.} for i in ids]
                (directory / "prefixes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                np.save(directory / "mean_grads.npy", rng.normal(size=(len(rows), 24)))
                (directory / "replay_provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
            feature_dir = root / "features"
            feature_dir.mkdir()
            (feature_dir / "layout_entries.json").write_text(json.dumps(entries), encoding="utf-8")
            feature_meta = {**metadata, "replay_prefix_sha256": sha256_file(root / "replay" / "prefixes.jsonl")}
            (feature_dir / "feature_provenance.json").write_text(json.dumps(feature_meta), encoding="utf-8")
            command = ["--replay-dir", str(root / "replay"),
                       "--reference-dir", str(root / "reference"),
                       "--reference-check-dir", str(root / "check"),
                       "--feature-dir", str(feature_dir), "--layers", "1",
                       "--ranks", "2", "--models", "zero,constant",
                       "--run-dir", str(root / "result")]
            self.assertEqual(single_layer_main(command), 0)
            result = json.loads((root / "result" / "single_layer_summary.json").read_text())
            self.assertEqual(result["layers"], [1])
            self.assertEqual(result["ranks"], [2])
            self.assertTrue(result["scalar_rows"])
            self.assertTrue(result["basis_rows"])
            self.assertIn("layer_1_rank_2", result["selected_by_validation"])


if __name__ == "__main__":
    unittest.main()
