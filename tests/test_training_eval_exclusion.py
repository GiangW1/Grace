import json
import pytest

from grace_gc.data.format_prompt import DAPO_SOLVE_PREFIX, DAPO_SOLVE_SUFFIX
from grace_gc.trainer.loop import build_run_config, run_training


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_external_eval_excluded_before_sft_and_sampling(tmp_path, monkeypatch):
    from grace_gc.trainer import format_warmup, loop

    train = write_rows(tmp_path / "train.jsonl", [
        {"problem_id": "leak", "prompt": DAPO_SOLVE_PREFIX + "A + 1 ?" + DAPO_SOLVE_SUFFIX, "answer": "wrong", "split": "train"},
        {"problem_id": "keep", "prompt": "a + 1 ?", "answer": "2", "split": "train"},
        {"problem_id": "keep2", "prompt": "2 + 1 ?", "answer": "3", "split": "train"},
    ])
    evaluation = write_rows(tmp_path / "eval.jsonl", [
        {"problem_id": "different-id", "prompt": "A  +\n1 ?", "answer": "2"},
    ])
    originals = [(path, path.read_bytes()) for path in (tmp_path / "train.jsonl", tmp_path / "eval.jsonl")]
    seen = {}

    def warmup(actor, records, vocab, cfg, ledger):
        seen["sft"] = [r.problem_id for r in records]
        return {"steps": 0}

    old_sample = loop.sample_starts

    def sample(records, *args, **kwargs):
        seen["sampling_pool"] = [r.problem_id for r in records]
        return old_sample(records, *args, **kwargs)

    monkeypatch.setattr(format_warmup, "run_format_warmup_tiny", warmup)
    monkeypatch.setattr(loop, "sample_starts", sample)
    cfg = build_run_config([], {"backend": "cpu_tiny", "method": "full_pg", "num_steps": 1,
        "n_start": 2, "n_prompts": 1, "max_new_tokens": 4, "decision_tokens": 2,
        "data_path": train, "eval_data_path": evaluation, "format_warmup": {"steps": 1},
        "baseline": {"mode": "fixed", "value": 0.5, "prescan_per_prompt": 0}})
    result = run_training(cfg, tmp_path / "run")
    assert result["run_status"] == "complete"
    assert seen["sft"] == seen["sampling_pool"] == ["keep", "keep2"]
    report = json.loads((tmp_path / "run/data_exclusions.json").read_text())
    assert report["removed"] == [{"split": "train", "problem_id": "leak",
        "prompt_sha256": report["removed"][0]["prompt_sha256"], "eval_problem_ids": ["different-id"]}]
    assert report["eval_source"]["sha256"]
    inventory = json.loads((tmp_path / "run/data_splits.json").read_text())
    assert inventory["counts"]["train"] == 2
    assert inventory["load"]["n_raw"] == 3  # never the last-loaded eval file's report
    assert all(path.read_bytes() == before for path, before in originals)


def test_filter_preserves_split_assignments_and_heldout_pool(tmp_path):
    from grace_gc.data.math_data import load_training_data

    rows = [{"problem_id": str(i), "prompt": f"q{i}", "answer": "1", "split": split}
            for i, split in enumerate(("train", "calib", "audit", "eval", "train"))]
    train = write_rows(tmp_path / "train.jsonl", rows)
    evaluation = write_rows(tmp_path / "eval.jsonl", [
        {"problem_id": "e" + str(i), "prompt": f"q{i}", "answer": "2"} for i in range(4)])
    buckets, report = load_training_data(train, evaluation, seed=17)
    assert {key: [r.problem_id for r in values] for key, values in buckets.items()} == {
        "train": ["4"], "calib": [], "audit": [], "eval": ["3"]}
    assert len(report["eval_exclusion"]["removed"]) == 3


def test_no_external_eval_is_reported_without_blocking_training(tmp_path):
    from grace_gc.data.math_data import load_training_data

    train = write_rows(tmp_path / "train.jsonl", [{"problem_id": "p", "prompt": "q", "answer": "1"}])
    buckets, report = load_training_data(train)
    assert len(buckets["train"]) == 1
    assert report["eval_exclusion"]["status"] == "not_provided"


def test_exclusion_evidence_survives_an_empty_training_pool(tmp_path, monkeypatch):
    from grace_gc.trainer import loop

    monkeypatch.setattr(loop, "collect_environment", lambda *a, **k: {})
    path = write_rows(tmp_path / "data.jsonl", [{"problem_id": "p", "prompt": "q", "answer": "1"}])
    cfg = build_run_config([], {"backend": "cpu_tiny", "method": "full_pg", "num_steps": 0,
                               "data_path": path, "eval_data_path": path})
    with pytest.raises(ValueError, match="training split is empty"):
        run_training(cfg, tmp_path / "run")
    report = json.loads((tmp_path / "run/data_exclusions.json").read_text())
    assert report["removed"][0]["problem_id"] == "p"
    assert json.loads((tmp_path / "run/data_splits.json").read_text())["counts"]["train"] == 0
