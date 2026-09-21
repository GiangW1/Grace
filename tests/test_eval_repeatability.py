import json
from pathlib import Path
from types import SimpleNamespace

import yaml


def eval_fixture(root, tokens=(3, 4), batch=1, actor="actor", seed=17):
    root.mkdir(parents=True)
    manifest = {"ordered_records_sha256": "problems", "n_problems": 1,
                "reward_protocol_version": 2, "samples_per_problem": 1,
                "temperature": .6, "top_p": .95, "max_new_tokens": 8,
                "sample_batch_size": batch, "sample_seed_start": seed}
    summary = {"seed": seed, "checkpoint_step": 40, "evaluation_manifest": manifest,
               "checkpoint_identity": {"actor_state_sha256": actor, "scope": "checkpoint_lora_only"}}
    (root/"eval_summary.json").write_text(json.dumps(summary))
    (root/"config.yaml").write_text(yaml.safe_dump({"backend": "gpu_verl", "model_path": "model",
        "prompt_max_tokens": 1024, "lora": {"rank": 16}, "vllm": {"seed": 17}}))
    (root/"environment.json").write_text(json.dumps({"packages": {"vllm": "test"}, "env": {
        key: None for key in ("CUDA_VISIBLE_DEVICES", "VLLM_ATTENTION_BACKEND", "VLLM_ENABLE_V1_MULTIPROCESSING",
                             "VLLM_BATCH_INVARIANT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "model": {"files": {"config.json": {"sha256": "model-config"}}}}))
    (root/"vllm_engine.json").write_text(json.dumps({"accepted": {"seed": 17, "dtype": "bfloat16"}}))
    (root/"tokenizer.json").write_text(json.dumps({"eos_token_id": 9}))
    (root/"eval_per_problem.jsonl").write_text(json.dumps({"problem_id": "p", "answers": ["same text"],
        "token_ids": [list(tokens)], "sample_seeds": [seed], "truncated": [False], "finish_reasons": ["eos"]})+"\n")
    (root/"eval_requests.jsonl").write_text(json.dumps({"problem_id": "p", "sample_index": 0,
        "request_seed": seed, "prompt_token_sha256": "prompt", "chunk_index": 0,
        "requested_batch_size": batch, "execution_batch_size": batch, "serial_fallback": False})+"\n")
    return root


def test_same_identity_reports_exact_tokens_and_first_divergence(tmp_path):
    from scripts.check_eval_repeatability import compare_runs
    left, right = eval_fixture(tmp_path/"a"), eval_fixture(tmp_path/"b", tokens=(3, 5))
    result = compare_runs(left, right)
    assert result["classification"] == "same_protocol_repeat"
    assert result["exact_tokens_equal"] is False
    sample = result["samples"][0]
    assert sample["first_difference"] == {"response_index": 1, "left_token": 4, "right_token": 5}
    assert sample["left_token_sha256"] != sample["right_token_sha256"]
    assert sample["text_equal"] is True  # decoded equality is not token equality
    assert "not full base-model" in result["weight_identity_scope"]


def test_batch_only_change_is_a_single_factor_control_not_repeat(tmp_path):
    from scripts.check_eval_repeatability import compare_runs
    result = compare_runs(eval_fixture(tmp_path/"a"), eval_fixture(tmp_path/"b", batch=4))
    assert result["classification"] == "batch_size_single_factor"
    assert result["exact_tokens_equal"] is True
    assert "requested_batch_size" in result["samples"][0]["request_differences"]


def test_actor_seed_and_missing_token_evidence_are_not_repeatability_success(tmp_path):
    from scripts.check_eval_repeatability import compare_runs
    left, right = eval_fixture(tmp_path/"a"), eval_fixture(tmp_path/"b", actor="different", seed=23)
    result = compare_runs(left, right)
    assert result["classification"] == "confounded"
    assert {row["field"] for row in result["identity_issues"]} >= {"actor_state", "seed"}
    (right/"eval_per_problem.jsonl").write_text(json.dumps({"problem_id": "p", "answers": ["same text"]})+"\n")
    assert compare_runs(left, right)["exact_tokens_equal"] is None


def test_length_divergence_and_missing_identity_remain_observable(tmp_path):
    from scripts.check_eval_repeatability import compare_runs
    left, right = eval_fixture(tmp_path/"a"), eval_fixture(tmp_path/"b", tokens=(3,))
    result = compare_runs(left, right)
    assert result["samples"][0]["first_difference"] == {"response_index": 1, "left_token": 4, "right_token": None}
    (right/"vllm_engine.json").unlink()
    result = compare_runs(left, right)
    assert result["classification"] == "unverified"
    assert {row["field"] for row in result["identity_issues"]} >= {"engine"}


def test_old_and_new_reports_compare_the_same_loaded_actor_hash_scheme(tmp_path):
    from scripts.check_eval_repeatability import compare_runs
    left, right = eval_fixture(tmp_path/"old"), eval_fixture(tmp_path/"new")
    for root in (left, right):
        (root/"actor_source.json").write_text(json.dumps({"actor_sha256": "loaded-same", "layout": "all_qv_lora_A_B"}))
    path = left/"eval_summary.json"
    summary = json.loads(path.read_text())
    summary.pop("checkpoint_identity")
    path.write_text(json.dumps(summary))
    assert compare_runs(left, right)["classification"] == "same_protocol_repeat"


def test_fallback_differences_are_recorded_without_claiming_different_requested_protocol(tmp_path):
    from scripts.check_eval_repeatability import compare_runs
    left, right = eval_fixture(tmp_path/"a", batch=4), eval_fixture(tmp_path/"b", batch=4)
    path = right/"eval_requests.jsonl"
    request = json.loads(path.read_text())
    request.update(serial_fallback=True, execution_batch_size=1, serial_fallback_reason="old API")
    path.write_text(json.dumps(request)+"\n")
    result = compare_runs(left, right)
    assert result["classification"] == "same_protocol_repeat"
    assert set(result["samples"][0]["request_differences"]) == {
        "execution_batch_size", "serial_fallback", "serial_fallback_reason"}


def test_repeat_launches_fresh_evaluate_processes_and_preserves_outputs(tmp_path, monkeypatch):
    from scripts import check_eval_repeatability as check
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        child = command[command.index("--")+1:]
        assert Path(child[1]).name == "evaluate.py" and "--generate" in child
        config = yaml.safe_load(Path(child[child.index("--config")+1]).read_text())
        output = Path(child[child.index("--run-dir")+1])
        eval_fixture(output, batch=config["eval"]["sample_batch_size"])
        kwargs["stdout"].write("run_dir " + str(output) + "\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(check.subprocess, "run", execute)
    result = check.repeat_evaluations({"checkpoint": "source.npz", "data_path": "eval.jsonl",
                                     "eval": {"sample_batch_size": 1}}, tmp_path/"diagnostic", repeats=2, batch_sizes=[1, 4])
    assert len(calls) == 4 and result["status"] == "completed"
    assert [row["classification"] for row in result["comparisons"]] == [
        "same_protocol_repeat", "same_protocol_repeat", "batch_size_single_factor"]
    assert all(Path(row["stdout"]).is_file() and Path(row["run_dir"]).is_dir() for row in result["runs"])


def test_repeat_preserves_failed_command_and_stops_without_spending_next_run(tmp_path, monkeypatch):
    from scripts import check_eval_repeatability as check
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("synthetic child failed\n")
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(check.subprocess, "run", execute)
    result = check.repeat_evaluations({"checkpoint": "source.npz", "data_path": "eval.jsonl"}, tmp_path/"diagnostic")
    assert len(calls) == 1 and result["status"] == "failed" and result["exit_code"] == 7
    assert Path(result["runs"][0]["stdout"]).read_text() == "synthetic child failed\n"
