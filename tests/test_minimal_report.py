import json

from scripts.summarize_minimal import paired_delta, summarize


def test_paired_comparison_aligns_problem_ids():
    left = {"per_problem": [{"problem_id": "a", "avg": 1.0}, {"problem_id": "b", "avg": 0.5}]}
    right = {"per_problem": [{"problem_id": "b", "avg": 0.25}, {"problem_id": "a", "avg": 0.75}]}
    result = paired_delta(left, right)
    assert result["effect"] == 0.25
    assert result["ci95"] == [0.25, 0.25]


def test_partial_report_preserves_missing_and_undefined_results(tmp_path):
    audit = tmp_path / "grace/audit"
    audit.mkdir(parents=True)
    (audit / "audit_summary.json").write_text(json.dumps({
        "n_bundles": 1, "n_gated": 0, "times": [512],
        "rho_l_curve": [float("nan")], "rho_a_curve": [float("nan")],
        "variance_cost": {"ratio": float("nan")},
    }))
    summarize(tmp_path)
    raw = (tmp_path / "comparison.json").read_text()
    assert "NaN" not in raw
    payload = json.loads(raw)
    assert payload["methods"][0]["final_avg4"] is None
    assert payload["methods"][0]["variance_cost_ratio"] is None
    assert all(row["target_met_observed"] is None for row in payload["paper_targets"])


def test_report_uses_completed_audit_retry(tmp_path):
    failed = tmp_path / "full_pg/audit"
    failed.mkdir(parents=True)
    (failed / "summary.json").write_text(json.dumps({"run_status": "failed"}))
    retry = tmp_path / "full_pg/audit-retry-memory"
    retry.mkdir()
    (retry / "audit_summary.json").write_text(json.dumps({
        "n_bundles": 4, "finished": "2026-09-16T14:00:00+00:00",
    }))
    summarize(tmp_path)
    row = json.loads((tmp_path / "comparison.json").read_text())["methods"][0]
    assert row["audit_bundles"] == 4
    assert row["audit_run_dir"] == str(retry)
