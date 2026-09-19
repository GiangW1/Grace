import json

import pytest

from scripts.summarize_training_dynamics import summarize_training


def _run(tmp_path, method, steps, rows, prescan=None, fresh=None, endpoint=None):
    root = tmp_path / method / "train"
    root.mkdir(parents=True)
    for name, values in (("steps.jsonl", steps), ("trajectories.jsonl", rows),
                         ("prescan.jsonl", prescan), ("fresh_supervision.jsonl", fresh)):
        if values is not None:
            (root / name).write_text("".join(json.dumps(r) + "\n" for r in values), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"step": endpoint or max(r["step"] for r in steps)}))
    (root / "config.yaml").write_text(json.dumps({"method": method, "baseline": {"prescan": 2}}))
    return root


def _step(step, n, warmup=True, **extra):
    return {"step": step, "n": n, "warmup": warmup, "n_prescan": 0,
            "predictor": {"fresh_supervision": {"n": 0, "generated_tokens": 0}}, **extra}


def _row(step, pid, reward, advantage, baseline=.0, **extra):
    return {"step": step, "problem_id": pid, "reward": reward, "advantage": advantage,
            "baseline_b": baseline, "z_continue": 0 if reward is None else 1,
            "generated_response_tokens": 8, "truncated": False, **extra}


def test_pg_uses_r_minus_b_and_never_scores_stoppers(tmp_path):
    root = _run(tmp_path, "grace", [_step(1, 3)], [
        _row(1, "a", 0., 0.), _row(1, "a", 1., 1.), _row(1, "a", None, None)])
    report = summarize_training(root)
    main = report["phases"]["warmup"]["main"]
    assert report["objective"] == "R_minus_b"
    assert main["starts"] == 3 and main["completed"] == 2 and main["stopped"] == 1
    assert main["reward"]["observed"] == 2 and main["reward"]["mean"] == .5
    assert main["advantage"]["zero"] == 1
    assert main["generated_tokens"]["sum"] == 24
    assert report["groups"][0]["reward_sample_variance"] == .5
    assert report["integrity"]["advantage_mismatches"] == []


def test_grpo_uses_group_mean_not_saved_history_baseline(tmp_path):
    root = _run(tmp_path, "grpo", [_step(1, 4)], [
        _row(1, "same", 1., 0., .2), _row(1, "same", 1., 0., .2),
        _row(1, "mixed", 0., -.5, .2), _row(1, "mixed", 1., .5, .2)])
    report = summarize_training(root)
    assert report["objective"] == "group_mean_advantage"
    assert report["integrity"]["advantage_mismatches"] == []
    phase = report["phases"]["warmup"]
    assert phase["group_counts"]["all_completed_adv_zero"] == 1
    assert phase["main"]["baseline"]["used_by_objective"] is False


def test_missing_gradient_is_not_zero_and_zero_gradient_can_update(tmp_path):
    root = _run(tmp_path, "full_pg", [_step(1, 2, grad_norm_preclip=0., parameter_update_norm=.1)], [
        _row(1, "a", 0., 0., true_grad_norm_sq=0.), _row(1, "b", 1., 1.)])
    phase = summarize_training(root)["phases"]["warmup"]
    observed = phase["main"]["recorded_gradient_norm_sq"]
    assert observed["zero"] == 1 and observed["missing"] == 1
    assert observed["sum"] is None
    assert phase["updates"]["zero_gradient_nonzero_parameter_update_steps"] == [1]
    assert phase["group_counts"]["singleton_completed_groups"] == 2


def test_auxiliary_missing_is_unknown_unless_step_facts_prove_zero(tmp_path):
    steps = [_step(1, 1), _step(2, 1, False, n_prescan=1,
              predictor={"fresh_supervision": {"n": 1, "generated_tokens": 9}})]
    root = _run(tmp_path, "grace", steps, [_row(1, "a", 0., 0.), _row(2, "b", 0., 0.)])
    phases = summarize_training(root)["phases"]
    assert phases["warmup"]["auxiliary"]["prescan"]["generated_tokens"] == 0
    assert phases["warmup"]["auxiliary"]["fresh"]["generated_tokens"] == 0
    assert phases["post_warmup"]["auxiliary"]["prescan"]["generated_tokens"] is None
    assert phases["post_warmup"]["auxiliary"]["fresh"]["generated_tokens"] is None
    assert phases["all"]["total_generation_tokens"] is None


def test_prescan_counts_questions_not_samples_and_phase_token_sums(tmp_path):
    root = _run(tmp_path, "grace", [_step(1, 1, n_prescan=1), _step(2, 2, False)],
                [_row(1, "a", 0., -.5, .5, generated_response_tokens=2),
                 _row(2, "b", 0., 0., generated_response_tokens=10),
                 _row(2, "b", 0., 0., generated_response_tokens=10)],
                prescan=[{"step": 1, "problem_id": "a", "reward": reward,
                          "baseline_after_prescan": .5, "prompt_token_ids": [4, 5],
                          "full_token_ids": [4, 5, 6, 7, 8], "truncated": False}
                         for reward in (0., 1.)])
    report = summarize_training(root)
    phase = report["phases"]["all"]
    assert phase["main"]["generated_tokens"]["mean"] == pytest.approx(22 / 3)
    assert phase["auxiliary"]["prescan"]["generated_tokens"] == 6
    assert phase["auxiliary"]["prescan"]["group_counts"]["observed"] == 1
    assert phase["total_generation_tokens"] == 28
    assert phase["auxiliary"]["prescan"]["complete"] is True


def test_advantage_mismatch_and_invalid_stopper_reward_are_reported(tmp_path):
    root = _run(tmp_path, "grace", [_step(1, 2)], [
        _row(1, "a", 1., 0.), _row(1, "b", 0., 0., z_continue=0)])
    integrity = summarize_training(root)["integrity"]
    assert len(integrity["advantage_mismatches"]) == 1
    assert len(integrity["stopped_with_reward"]) == 1


def test_missing_entire_step_prevents_complete_cost_claim(tmp_path):
    root = _run(tmp_path, "full_pg", [_step(1, 1), _step(3, 1)],
                [_row(1, "a", 0., 0.), _row(3, "c", 0., 0.)], endpoint=3)
    report = summarize_training(root)
    assert not report["integrity"]["complete_step_range"]
    assert report["phases"]["all"]["total_generation_tokens"] is None
    assert report["phases"]["all"]["main"]["generated_tokens"]["sum"] == 16
    assert report["scope"].startswith("Observed log rows")


def test_partial_prescan_group_cannot_be_claimed_complete(tmp_path):
    root = _run(tmp_path, "full_pg", [_step(1, 1, n_prescan=1)], [_row(1, "a", 0., 0.)],
                prescan=[{"step": 1, "problem_id": "a", "reward": 0., "response_tokens": 6}])
    auxiliary = summarize_training(root)["phases"]["all"]["auxiliary"]["prescan"]
    assert auxiliary["rows_observed"] == 1
    assert auxiliary["complete"] is False
    assert auxiliary["generated_tokens"] is None


def test_grpo_group_is_local_to_step_and_unknown_phase_stays_unknown(tmp_path):
    one, two = _step(1, 2), _step(2, 2)
    two.pop("warmup")
    root = _run(tmp_path, "grpo", [one, two], [
        _row(1, "a", 0., 0.), _row(1, "a", 0., 0.),
        _row(2, "a", 1., 0.), _row(2, "a", 1., 0.)])
    report = summarize_training(root)
    assert report["integrity"]["advantage_mismatches"] == []
    assert len(report["groups"]) == 2
    assert report["phases"]["unknown_phase"]["recorded_steps"] == [2]
    assert report["phases"]["post_warmup"]["recorded_steps"] == []
    assert report["phases"]["all"]["updates"]["clip"]["observed"] == 0
    assert report["phases"]["all"]["updates"]["clip"]["rate_among_observed"] is None


@pytest.mark.parametrize("partial", ["prescan_sample", "prescan_problem", "prescan_balance", "fresh_count", "fresh_tokens"])
def test_mechanism_costs_reject_auxiliary_rows_inconsistent_with_step_facts(tmp_path, partial):
    from scripts.summarize_mechanism import training_costs
    step = _step(1, 1)
    prescan, fresh = None, None
    if partial.startswith("prescan"):
        step["n_prescan"] = 1 if partial == "prescan_sample" else 2
        prescan = [{"step": 1, "problem_id": "a", "reward": 0., "response_tokens": 6}]
        if partial == "prescan_problem":
            prescan *= 4  # right total sample count, but the second problem is missing
        elif partial == "prescan_balance":
            prescan += [{"step": 1, "problem_id": "b", "reward": 0., "response_tokens": 6}] * 3
        key = "prescan_tokens"
    else:
        step["predictor"]["fresh_supervision"] = {"n": 2, "generated_tokens": 12} if partial == "fresh_count" else {"n": 1, "generated_tokens": 8}
        fresh = [{"step": 1, "problem_id": "a", "response_tokens": 6}]
        key = "fresh_tokens"
    root = _run(tmp_path, "grace", [step], [_row(1, "a", 0., 0.)], prescan=prescan, fresh=fresh)
    result = training_costs(root, [step], {"step": 1})
    assert result["all_training"][key] is None
    assert result["all_training"]["total_generation_tokens"] is None
    assert any("inconsistent_with_step_facts" in issue for issue in result["issues"])
    kind = "prescan" if partial.startswith("prescan") else "fresh"
    assert summarize_training(root)["phases"]["all"]["auxiliary"][kind]["generated_tokens"] is None


def test_bad_warmup_auxiliary_does_not_erase_complete_postwarmup_cost(tmp_path):
    from scripts.summarize_mechanism import training_costs
    steps = [_step(1, 1, n_prescan=1), _step(2, 1, False, n_prescan=1)]
    prescan = [{"step": 1, "problem_id": "a", "reward": 0., "response_tokens": 6}]
    prescan += [{"step": 2, "problem_id": "b", "reward": 0., "response_tokens": 6}] * 2
    root = _run(tmp_path, "grace", steps, [_row(1, "a", 0., 0.), _row(2, "b", 0., 0.)], prescan=prescan)
    result = training_costs(root, steps, {"step": 2})
    assert result["all_training"]["prescan_tokens"] is None
    assert result["post_warmup"]["prescan_tokens"] == 12
    assert result["post_warmup"]["total_generation_tokens"] == 20
