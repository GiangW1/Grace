"""Synthetic training-seed summaries, separate from problem bootstrap."""
import json
import sys

import numpy as np
import pytest

from scripts.summarize_seeds import summarize


def _write_seed(root, seed, delta, *, pass4=.8, starts=None):
    folder = root / f"seed-{seed}"
    folder.mkdir()
    protocol = {"ordered_records_sha256": "questions", "reward_protocol_version": "test",
                "samples_per_problem": 4, "temperature": .6, "top_p": .95,
                "max_new_tokens": 4096, "sample_batch_size": 4, "sample_seed_start": seed}
    metadata = {"actual_seed": seed, "shared_initial_actor_hash": f"actor-{seed}",
                "evaluation_manifest": protocol, "final_training_step": 2}
    rows = [{"method": "grace", "final_avg4": .5 + delta, "final_pass4": pass4, **metadata},
            {"method": "full_pg", "final_avg4": .5, "final_pass4": .7, **metadata}]
    if starts is not None:
        train = folder / "grace" / "train"
        train.mkdir(parents=True)
        rows[0]["stage_paths"] = {"train": "grace/train"}
        (train / "steps.jsonl").write_text("\n".join(
            json.dumps({"step": i + 1, "n": n}) for i, n in enumerate(starts)), encoding="utf-8")
    (folder / "comparison.json").write_text(json.dumps({"methods": rows}), encoding="utf-8")
    return folder, rows


def test_paired_seed_t_interval_uses_seed_differences_and_df(tmp_path):
    pytest.importorskip("scipy.stats")
    for seed, delta in [(17, .01), (23, .03), (41, .05)]:
        _write_seed(tmp_path, seed, delta)
    (tmp_path / "experiment.json").write_text(json.dumps({"requested_seeds": ["17", "23", "41", "59"]}))
    out = summarize(tmp_path)
    paired = out["paired"][0]
    half = 4.302652729696142 * .02 / np.sqrt(3.)  # t(.975, df=2)
    assert paired["n_seeds"] == 3 and paired["seed_ci95_df"] == 2
    assert paired["mean_delta"] == pytest.approx(.03)
    assert paired["seed_sd"] == pytest.approx(.02)
    np.testing.assert_allclose(paired["seed_ci95"], [.03 - half, .03 + half], atol=1e-12)
    assert paired["seed_ci95_unavailable_reason"] is None
    assert "Student-t" in paired["seed_ci95_method"]
    assert "not a problem bootstrap" in paired["note"]
    assert out["missing_seed_reports"] == ["59"]


def test_single_seed_keeps_observation_without_seed_interval(tmp_path):
    _write_seed(tmp_path, 17, .08)
    paired = summarize(tmp_path)["paired"][0]
    assert paired["mean_delta"] == pytest.approx(.08)
    assert paired["seed_sd"] is None and paired["seed_ci95"] is None
    assert paired["n_seeds"] == 1 and paired["seed_ci95_df"] == 0
    assert paired["seed_ci95_unavailable_reason"] == "fewer_than_two_paired_training_seeds"


def test_optional_scipy_missing_preserves_mean_sd_and_output(tmp_path, monkeypatch):
    _write_seed(tmp_path, 17, .02)
    _write_seed(tmp_path, 23, .06)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    paired = summarize(tmp_path)["paired"][0]
    assert paired["mean_delta"] == pytest.approx(.04)
    assert paired["seed_sd"] == pytest.approx(np.std([.02, .06], ddof=1))
    assert paired["seed_ci95"] is None
    assert paired["seed_ci95_unavailable_reason"] == "scipy_unavailable_install_analysis_extra"
    assert (tmp_path / "seeds_comparison.json").is_file()


def test_pass4_and_complete_all_starts_are_summarized_per_seed(tmp_path):
    _write_seed(tmp_path, 17, .02, pass4=.75, starts=[4, 8])
    _write_seed(tmp_path, 23, .04, pass4=.85, starts=[8, 16])
    method = next(row for row in summarize(tmp_path)["methods"] if row["method"] == "grace")
    assert method["n_seeds_with_final_pass4"] == 2
    assert method["final_pass4_mean"] == pytest.approx(.8)
    assert method["final_pass4_seed_sd"] == pytest.approx(np.std([.75, .85], ddof=1))
    assert method["n_seeds_with_all_starts"] == 2
    assert method["all_starts_mean"] == 18.
    assert method["all_starts_seed_sd"] == pytest.approx(np.std([12, 24], ddof=1))
    assert [row["all_starts"] for row in method["per_seed"]] == [12, 24]
    assert all(row["all_starts_source"] == "complete_training_steps" for row in method["per_seed"])


def test_partial_steps_are_unknown_and_explicit_row_count_is_preserved(tmp_path):
    _write_seed(tmp_path, 17, .02, starts=[8])  # Endpoint says 2; the log has only step 1.
    folder, rows = _write_seed(tmp_path, 23, .04)
    rows[0]["all_starts"] = 32
    (folder / "comparison.json").write_text(json.dumps({"methods": rows}))
    method = next(row for row in summarize(tmp_path)["methods"] if row["method"] == "grace")
    assert method["n_seeds_with_all_starts"] == 1 and method["all_starts_mean"] == 32
    assert method["per_seed"][0]["all_starts"] is None
    assert method["per_seed"][1]["all_starts_source"] == "comparison_row"


def test_mismatched_and_missing_final_seeds_remain_visible_outside_ci(tmp_path):
    _write_seed(tmp_path, 17, .02)
    folder, rows = _write_seed(tmp_path, 23, .9)
    rows[0]["shared_initial_actor_hash"] = "other"
    (folder / "comparison.json").write_text(json.dumps({"methods": rows}))
    folder, rows = _write_seed(tmp_path, 41, .8)
    rows[1]["final_avg4"] = None
    (folder / "comparison.json").write_text(json.dumps({"methods": rows}))
    paired = summarize(tmp_path)["paired"][0]
    assert paired["n_seeds"] == 1 and paired["mean_delta"] == pytest.approx(.02)
    assert paired["seed_ci95"] is None
    assert {row["seed"] for row in paired["incomparable"]} == {"seed-23", "seed-41"}
