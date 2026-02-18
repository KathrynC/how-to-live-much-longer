"""Tests for ten_types_audit deterministic scoring and outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import ten_types_audit as tta


def test_run_audit_returns_all_canonical_types():
    report = tta.run_audit()
    type_ids = [row["type_id"] for row in report["type_assessments"]]

    assert len(type_ids) == 10
    assert type_ids == [
        "profit_model",
        "network",
        "structure",
        "process",
        "product_performance",
        "product_system",
        "service",
        "channel",
        "brand",
        "customer_engagement",
    ]


def test_strict_summary_counts_only_differentiating():
    synthetic = [
        {"maturity_level": "differentiating", "maturity_score": 2},
        {"maturity_level": "me_too", "maturity_score": 1},
        {"maturity_level": "no_activity", "maturity_score": 0},
    ]

    summary = tta.build_summary(synthetic)

    assert summary["innovative_count"] == 1
    assert summary["me_too_count"] == 1
    assert summary["no_activity_count"] == 1
    assert summary["innovation_coverage_pct"] == 33.33


def test_profit_model_is_displayed_as_value_model():
    report = tta.run_audit()
    row = next(item for item in report["type_assessments"] if item["type_id"] == "profit_model")

    assert row["label"] == "Value Model"
    assert row["canonical_label"] == "Profit Model"


def test_output_generation_writes_valid_json_and_markdown(tmp_path: Path):
    report = tta.run_audit()
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    written = tta._write_outputs(report, output_format="both", out_json=out_json, out_md=out_md)

    assert out_json in written
    assert out_md in written
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(parsed.keys()) >= {
        "timestamp",
        "scope",
        "scoring_model",
        "type_assessments",
        "summary",
        "category_balance",
        "priority_gaps",
        "evidence_index",
    }

    md_text = out_md.read_text(encoding="utf-8")
    assert "# Ten Types of Innovation Audit" in md_text
    assert "Value Model" in md_text


def test_cli_smoke_with_temp_outputs(tmp_path: Path):
    out_json = tmp_path / "cli_audit.json"
    out_md = tmp_path / "cli_audit.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(tta.PROJECT / "ten_types_audit.py"),
            "--format",
            "both",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--as-of",
            "2026-02-18",
        ],
        cwd=str(tta.PROJECT),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Ten Types audit complete" in completed.stdout
    assert out_json.exists()
    assert out_md.exists()


def test_missing_evidence_repo_degrades_gracefully(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tta, "PROJECT", tmp_path)

    report = tta.run_audit()

    assert len(report["type_assessments"]) == 10
    assert all(row["maturity_level"] == "no_activity" for row in report["type_assessments"])
    assert report["summary"]["innovative_count"] == 0


def test_docs_wiring_mentions_module_and_artifacts():
    readme = (tta.PROJECT / "README.md").read_text(encoding="utf-8")
    guide = (tta.PROJECT / "docs/guide.md").read_text(encoding="utf-8")

    assert "ten_types_audit.py" in readme
    assert "artifacts/ten_types_innovation_audit.json" in readme
    assert "artifacts/ten_types_innovation_audit.md" in readme
    assert "ten_types_audit.py" in guide
    assert "ten_types_audit.md" in guide
