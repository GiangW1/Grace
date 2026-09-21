import json
import pytest

from scripts.summarize_mechanism import summarize


def _paired_fixture(root):
    protocol = {"ordered_records_sha256": "same-prompts-and-golds", "reward_protocol_version": 2,
                "samples_per_problem": 4, "temperature": .6, "top_p": .95,
                "max_new_tokens": 4096, "sample_batch_size": 4, "sample_seed_start": 17}
    (root/"mechanism.json").write_text(json.dumps({"arms": {"grace": "grace", "m0": "m0"}}))
    for name, avg in (("grace", .75), ("m0", .5)):
        folder = root/name/"seed-17"
        train, final = folder/"grace/train", folder/"grace/eval-40"
        train.mkdir(parents=True); final.mkdir()
        (folder/"stages.json").write_text(json.dumps({"grace": {"train": "grace/train", "eval-final": "grace/eval-40"}}))
        (train/"config.yaml").write_text("seed: 17\n")
        (train/"initial_actor.json").write_text(json.dumps({"actor_sha256": "shared"}))
        (train/"summary.json").write_text(json.dumps({"step": 40, "run_status": "complete"}))
        (train/"steps.jsonl").write_text("".join(json.dumps({"step": step, "n": 16, "warmup": step <= 20})+"\n" for step in range(1, 41)))
        (train/"trajectories.jsonl").write_text("".join((json.dumps({"step": step, "problem_id": "p", "prompt_token_ids": [1, 2]})+"\n")*16 for step in range(1, 41)))
        (final/"eval_summary.json").write_text(json.dumps({"checkpoint_step": 40, "seed": 17,
            "checkpoint": str(train/"checkpoints/step_40.npz"), "evaluation_manifest": protocol,
            "avg": avg, "per_problem": [{"problem_id": "p", "avg": avg}]}))
    return root/"m0/seed-17/grace"


@pytest.mark.parametrize("mismatch", ["actor", "seed", "protocol", "endpoint", "endpoint_actor", "source", "missing_seed", "input", "n", "whole_step_missing"])
def test_final_comparison_requires_matched_evidence_but_keeps_scores(tmp_path, mismatch):
    method = _paired_fixture(tmp_path)
    assert summarize(tmp_path)["seeds"]["seed-17"]["final_grace_minus_arm"]["m0"]["available"] is True
    final_path = method/"eval-40/eval_summary.json"
    final = json.loads(final_path.read_text())
    if mismatch == "actor":
        (method/"train/initial_actor.json").write_text(json.dumps({"actor_sha256": "different"}))
    elif mismatch == "seed":
        (method/"train/config.yaml").write_text("seed: 23\n")
    elif mismatch == "missing_seed":
        (method/"train/config.yaml").unlink()
    elif mismatch == "protocol":
        final["evaluation_manifest"]["temperature"] = .7
    elif mismatch == "endpoint":
        final["checkpoint_step"] = 20
    elif mismatch == "endpoint_actor":
        (method/"eval-40/actor_source.json").write_text(json.dumps({"actor_sha256": "wrong-final-actor"}))
        (method/"train/steps.jsonl").write_text(json.dumps({"step": 40, "n": 16, "post_update_snapshot_sha": "recorded-final-actor"})+"\n")
    elif mismatch == "source":
        final["checkpoint"] = str(tmp_path/"other/seed-17/grace/train/checkpoints/step_40.npz")
    elif mismatch == "input":
        (method/"train/trajectories.jsonl").write_text((json.dumps({"step": 40, "problem_id": "different", "prompt_token_ids": [1, 3]})+"\n")*16)
    elif mismatch == "n":
        (method/"train/steps.jsonl").write_text(json.dumps({"step": 40, "n": 8})+"\n")
        (method/"train/trajectories.jsonl").write_text((json.dumps({"step": 40, "problem_id": "p", "prompt_token_ids": [1, 2]})+"\n")*8)
    elif mismatch == "whole_step_missing":
        for name in ("steps.jsonl", "trajectories.jsonl"):
            path = method/"train"/name
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows if row["step"] != 3))
    final_path.write_text(json.dumps(final))
    result = summarize(tmp_path)["seeds"]["seed-17"]
    pair = result["final_grace_minus_arm"]["m0"]
    assert pair["available"] is False and pair["issues"]
    assert "ci95" not in pair and "effect" not in pair
    assert result["arms"]["m0"]["final_avg"] == .5


def test_training_token_breakdown_includes_auxiliary_and_preserves_missing(tmp_path):
    from scripts.summarize_mechanism import training_costs
    steps = [{"step": 1, "n": 1, "warmup": True}, {"step": 2, "n": 1, "warmup": False}]
    main = [{"step": 1, "generated_response_tokens": 10}, {"step": 2, "prefix_tokens": 3, "suffix_tokens": 4}]
    def write(name, rows):
        (tmp_path/name).write_text("".join(json.dumps(row)+"\n" for row in rows))
    write("trajectories.jsonl", main)
    write("prescan.jsonl", [{"step": 1, "prompt_token_ids": [1], "full_token_ids": [1, 2, 3]},
                           {"step": 2, "prompt_token_ids": [1], "full_token_ids": [1, 2, 3, 4]}])
    write("fresh_supervision.jsonl", [{"step": 2, "response_tokens": 5}])
    write("cost_control.jsonl", [{"event": "batch_complete", "step": 1, "last_batch_wall_seconds": 4},
                                {"event": "batch_complete", "step": 2, "last_batch_wall_seconds": 6}])
    result = training_costs(tmp_path, steps, {"step": 2, "cumulative_wall_seconds": 15})
    assert result["all_training"] == {"main_tokens": 17, "prescan_tokens": 5, "fresh_tokens": 5,
                                      "total_generation_tokens": 27, "wall_seconds": 15}
    assert result["post_warmup"] == {"main_tokens": 7, "prescan_tokens": 3, "fresh_tokens": 5,
                                     "total_generation_tokens": 15, "wall_seconds": 6}
    (tmp_path/"fresh_supervision.jsonl").unlink()
    missing = training_costs(tmp_path, steps, {})
    assert missing["all_training"]["fresh_tokens"] is None
    assert missing["all_training"]["total_generation_tokens"] is None
    assert missing["all_training"]["wall_seconds"] is None
    assert missing["post_warmup"]["total_generation_tokens"] is None
    (tmp_path/"cost_control.jsonl").unlink()
    assert training_costs(tmp_path, steps, {})["post_warmup"]["wall_seconds"] is None


def test_absent_auxiliary_logs_zero_only_from_complete_runtime_step_facts(tmp_path):
    from scripts.summarize_mechanism import training_costs
    steps = [{"step": i, "n": 1, "warmup": i == 1, "n_prescan": 0,
              "predictor": {"fresh_supervision": {"n": 0, "generated_tokens": 0}}} for i in (1, 2)]
    (tmp_path/"trajectories.jsonl").write_text("".join(json.dumps({"step": i, "generated_response_tokens": 3})+"\n" for i in (1, 2)))
    result = training_costs(tmp_path, steps, {"step": 2})
    assert result["all_training"]["prescan_tokens"] == result["all_training"]["fresh_tokens"] == 0
    assert result["post_warmup"]["prescan_tokens"] == result["post_warmup"]["fresh_tokens"] == 0
    assert result["all_training"]["total_generation_tokens"] == 6
    steps[0]["predictor"]["fresh_supervision"] = {"n": 1, "generated_tokens": 20}
    result = training_costs(tmp_path, steps, {"step": 2})
    assert result["all_training"]["fresh_tokens"] is None
    assert result["post_warmup"]["fresh_tokens"] == 0
    assert result["post_warmup"]["total_generation_tokens"] == 3
    del steps[1]["n_prescan"]
    result = training_costs(tmp_path, steps, {"step": 2})
    assert result["all_training"]["prescan_tokens"] is None
    assert result["post_warmup"]["prescan_tokens"] is None
    partial = training_costs(tmp_path, steps[:1], {"step": 2})
    assert partial["post_warmup"]["fresh_tokens"] is None
    assert partial["all_training"]["total_generation_tokens"] is None
    assert "training_step_range_missing_or_incomplete" in partial["issues"]


def test_missing_evidence_is_not_successful_pairing(tmp_path):
    (tmp_path/"mechanism.json").write_text(json.dumps({"arms": {"grace": "grace"}}))
    (tmp_path/"grace/seed-17").mkdir(parents=True)
    result = summarize(tmp_path)
    assert result["seeds"]["seed-17"]["pairing"] == {
        "shared_initial_actor": None, "same_problem_and_prompt_token_sequence": None, "fixed_n16_all_steps": None}


def test_four_arm_pairing_uses_prompt_tokens_and_fixed_n(tmp_path):
    names = ("grace", "p1", "m0", "uniform")
    (tmp_path/"mechanism.json").write_text(json.dumps({"arms": {name: name for name in names}}))
    for name in names:
        folder = tmp_path/name/"seed-17"
        train = folder/"grace/train"; train.mkdir(parents=True)
        (folder/"stages.json").write_text(json.dumps({"grace": {"train": "grace/train"}}))
        (train/"initial_actor.json").write_text(json.dumps({"actor_sha256": "common"}))
        (train/"summary.json").write_text(json.dumps({"step": 1}))
        (train/"steps.jsonl").write_text(json.dumps({"step": 1, "n": 16})+"\n")
        (train/"trajectories.jsonl").write_text((json.dumps({"step": 1, "problem_id": "p", "prompt_token_ids": [1, 2]})+"\n")*16)
    pairing = summarize(tmp_path)["seeds"]["seed-17"]["pairing"]
    assert all(pairing.values())
    path = tmp_path/"m0/seed-17/grace/train/trajectories.jsonl"
    path.write_text((json.dumps({"step": 1, "problem_id": "p", "prompt_token_ids": [1, 3]})+"\n")*16)
    assert summarize(tmp_path)["seeds"]["seed-17"]["pairing"]["same_problem_and_prompt_token_sequence"] is False
    path.write_text(json.dumps({"step": 1, "problem_id": "p", "prompt_token_ids": [1, 3]})+"\n")
    assert summarize(tmp_path)["seeds"]["seed-17"]["pairing"]["same_problem_and_prompt_token_sequence"] is None
