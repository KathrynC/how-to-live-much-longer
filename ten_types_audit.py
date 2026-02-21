#!/usr/bin/env python3
"""Ten Types of Innovation audit for the whole repository platform.

This module implements a deterministic, repeatable audit over repository
artifacts. It uses a strict 3-level maturity model:
  - no_activity (0)
  - me_too (1)
  - differentiating (2)

Only ``differentiating`` counts as true innovation in summary coverage.

The canonical Ten Types are preserved, with one explicit adaptation:
``profit_model`` is displayed as ``Value Model`` for this non-commercial
research repository context.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT / "artifacts"
DEFAULT_JSON_PATH = ARTIFACTS_DIR / "ten_types_innovation_audit.json"
DEFAULT_MD_PATH = ARTIFACTS_DIR / "ten_types_innovation_audit.md"

MaturityLevel = str

MATURITY_LEVELS: dict[MaturityLevel, int] = {
    "no_activity": 0,
    "me_too": 1,
    "differentiating": 2,
}

TYPE_TAXONOMY = [
    {
        "type_id": "profit_model",
        "label": "Value Model",
        "canonical_label": "Profit Model",
        "category": "Configuration",
    },
    {
        "type_id": "network",
        "label": "Network",
        "canonical_label": "Network",
        "category": "Configuration",
    },
    {
        "type_id": "structure",
        "label": "Structure",
        "canonical_label": "Structure",
        "category": "Configuration",
    },
    {
        "type_id": "process",
        "label": "Process",
        "canonical_label": "Process",
        "category": "Configuration",
    },
    {
        "type_id": "product_performance",
        "label": "Product Performance",
        "canonical_label": "Product Performance",
        "category": "Offering",
    },
    {
        "type_id": "product_system",
        "label": "Product System",
        "canonical_label": "Product System",
        "category": "Offering",
    },
    {
        "type_id": "service",
        "label": "Service",
        "canonical_label": "Service",
        "category": "Experience",
    },
    {
        "type_id": "channel",
        "label": "Channel",
        "canonical_label": "Channel",
        "category": "Experience",
    },
    {
        "type_id": "brand",
        "label": "Brand",
        "canonical_label": "Brand",
        "category": "Experience",
    },
    {
        "type_id": "customer_engagement",
        "label": "Customer Engagement",
        "canonical_label": "Customer Engagement",
        "category": "Experience",
    },
]


TYPE_RULES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "profit_model": {
        "basic": [
            {
                "id": "readme_present",
                "kind": "path_exists",
                "path": "README.md",
                "description": "Repository-level value proposition is documented.",
            },
            {
                "id": "license_present",
                "kind": "path_exists",
                "path": "LICENSE",
                "description": "Licensing clarifies value-sharing boundaries.",
            },
            {
                "id": "deviation_doc_present",
                "kind": "path_exists",
                "path": "docs/model_deviations.md",
                "description": "Value assumptions and known deviations are documented.",
            },
        ],
        "differentiating": [
            {
                "id": "conformance_matrix_present",
                "kind": "path_exists",
                "path": "docs/book_conformance_appendix2.md",
                "description": "Value-accountability matrix ties claims to tests.",
            },
            {
                "id": "conformance_tests_present",
                "kind": "path_exists",
                "path": "tests/test_book_conformance.py",
                "description": "Executable conformance checks protect value integrity.",
            },
            {
                "id": "research_scope_statement",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "hypothesis-generating research artifacts",
                "description": "Non-commercial value lens is explicitly stated.",
            },
        ],
    },
    "network": {
        "basic": [
            {
                "id": "zimmerman_bridge_present",
                "kind": "path_exists",
                "path": "zimmerman_bridge.py",
                "description": "External toolkit bridge is implemented.",
            },
            {
                "id": "kcramer_bridge_present",
                "kind": "path_exists",
                "path": "kcramer_bridge.py",
                "description": "Second toolkit bridge is implemented.",
            },
        ],
        "differentiating": [
            {
                "id": "cross_tool_runner_present",
                "kind": "path_exists",
                "path": "zimmerman_analysis.py",
                "description": "Integrated multi-tool runner orchestrates networked capabilities.",
            },
            {
                "id": "readme_mentions_kcramer_bridge",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "K-Cramer Toolkit bridge",
                "description": "Networked integration is exposed in public docs.",
            },
            {
                "id": "readme_mentions_zimmerman_bridge",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "Zimmerman Simulator protocol adapter",
                "description": "Cross-project protocol adapter is documented.",
            },
        ],
    },
    "structure": {
        "basic": [
            {
                "id": "constants_present",
                "kind": "path_exists",
                "path": "constants.py",
                "description": "Core configuration is centralized.",
            },
            {
                "id": "schemas_present",
                "kind": "path_exists",
                "path": "schemas.py",
                "description": "Data contracts are explicitly modeled.",
            },
            {
                "id": "docs_guide_present",
                "kind": "path_exists",
                "path": "docs/guide.md",
                "description": "Information architecture is maintained in guide docs.",
            },
        ],
        "differentiating": [
            {
                "id": "plan_tracking_present",
                "kind": "path_exists",
                "path": "docs/plans/progress.json",
                "description": "Program-level planning structure is tracked.",
            },
            {
                "id": "test_harness_present",
                "kind": "path_exists",
                "path": "tests/conftest.py",
                "description": "Shared test harness enforces structure.",
            },
            {
                "id": "core_pipeline_doc",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "### Core Pipeline",
                "description": "System structure is externally legible.",
            },
        ],
    },
    "process": {
        "basic": [
            {
                "id": "simulator_present",
                "kind": "path_exists",
                "path": "simulator.py",
                "description": "Core simulation process is codified.",
            },
            {
                "id": "analytics_present",
                "kind": "path_exists",
                "path": "analytics.py",
                "description": "Outcome analytics process is codified.",
            },
            {
                "id": "tests_dir_present",
                "kind": "path_exists",
                "path": "tests",
                "description": "Verification process exists.",
            },
        ],
        "differentiating": [
            {
                "id": "full_test_command",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "pytest tests/ -v",
                "description": "Repeatable full-suite test process is documented.",
            },
            {
                "id": "strict_conformance_gate",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "STRICT_BOOK_CONFORMANCE=1",
                "description": "Optional strict process gate is available.",
            },
            {
                "id": "conformance_matrix_present",
                "kind": "path_exists",
                "path": "docs/book_conformance_appendix2.md",
                "description": "Process traceability is mapped claim-to-test.",
            },
        ],
    },
    "product_performance": {
        "basic": [
            {
                "id": "simulator_present",
                "kind": "path_exists",
                "path": "simulator.py",
                "description": "Core product engine exists.",
            },
            {
                "id": "core_constants_present",
                "kind": "path_exists",
                "path": "constants.py",
                "description": "Core model constants are maintained.",
            },
        ],
        "differentiating": [
            {
                "id": "simulator_tests_present",
                "kind": "path_exists",
                "path": "tests/test_simulator.py",
                "description": "Core performance behavior is regression-tested.",
            },
            {
                "id": "falsifier_artifact_present",
                "kind": "path_exists",
                "path": "artifacts/falsifier_report_2026-02-15.md",
                "description": "Critical quality fixes are archived as evidence.",
            },
            {
                "id": "readme_ode_corrections",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "ODE Corrections Applied",
                "description": "Performance correction history is documented.",
            },
        ],
    },
    "product_system": {
        "basic": [
            {
                "id": "schemas_present",
                "kind": "path_exists",
                "path": "schemas.py",
                "description": "Typed interfaces support system integration.",
            },
            {
                "id": "llm_common_present",
                "kind": "path_exists",
                "path": "llm_common.py",
                "description": "Shared LLM utilities provide subsystem cohesion.",
            },
            {
                "id": "prompt_templates_present",
                "kind": "path_exists",
                "path": "prompt_templates.py",
                "description": "Prompt layer integrates with simulator pipeline.",
            },
        ],
        "differentiating": [
            {
                "id": "zimmerman_bridge_present",
                "kind": "path_exists",
                "path": "zimmerman_bridge.py",
                "description": "Simulator protocol adapter enables ecosystem composition.",
            },
            {
                "id": "grief_adapter_present",
                "kind": "path_exists",
                "path": "grief_mito_simulator.py",
                "description": "Cross-domain system extension is implemented.",
            },
            {
                "id": "kcramer_tools_runner_present",
                "kind": "path_exists",
                "path": "kcramer_tools_runner.py",
                "description": "System-level workflows are orchestrated through a unified runner.",
            },
        ],
    },
    "service": {
        "basic": [
            {
                "id": "guide_present",
                "kind": "path_exists",
                "path": "docs/guide.md",
                "description": "Service-level guidance exists for users.",
            },
            {
                "id": "simulator_docs_present",
                "kind": "path_exists",
                "path": "docs/simulator.md",
                "description": "Core usage and behavior documentation is available.",
            },
            {
                "id": "analytics_docs_present",
                "kind": "path_exists",
                "path": "docs/analytics.md",
                "description": "Outcome interpretation service is documented.",
            },
        ],
        "differentiating": [
            {
                "id": "resilience_viz_present",
                "kind": "path_exists",
                "path": "resilience_viz.py",
                "description": "Service layer includes ready-to-run visual diagnostics.",
            },
            {
                "id": "visualize_present",
                "kind": "path_exists",
                "path": "visualize.py",
                "description": "Core visualization services are part of the platform.",
            },
            {
                "id": "resilience_metrics_docs_present",
                "kind": "path_exists",
                "path": "docs/resilience_metrics.md",
                "description": "Operational resilience service semantics are documented.",
            },
        ],
    },
    "channel": {
        "basic": [
            {
                "id": "setup_section_present",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "## Setup",
                "description": "Primary channel onboarding is documented.",
            },
            {
                "id": "simulator_cli_command_present",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "python simulator.py",
                "description": "Direct execution channel is documented.",
            },
            {
                "id": "analytics_cli_command_present",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "python analytics.py",
                "description": "Second execution channel is documented.",
            },
        ],
        "differentiating": [
            {
                "id": "kcramer_runner_present",
                "kind": "path_exists",
                "path": "kcramer_tools_runner.py",
                "description": "Dedicated cross-tool access channel exists.",
            },
            {
                "id": "zimmerman_analysis_present",
                "kind": "path_exists",
                "path": "zimmerman_analysis.py",
                "description": "High-level orchestration channel exists.",
            },
            {
                "id": "claude_discovery_tools_section",
                "kind": "file_contains",
                "path": "CLAUDE.md",
                "pattern": "Tier 5 — Discovery Tools",
                "description": "Agent-facing channel map is maintained.",
            },
        ],
    },
    "brand": {
        "basic": [
            {
                "id": "readme_title_present",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "# How to Live Much Longer",
                "description": "Project identity is clearly named in primary docs.",
            },
            {
                "id": "license_present",
                "kind": "path_exists",
                "path": "LICENSE",
                "description": "Legal identity and usage boundaries are explicit.",
            },
        ],
        "differentiating": [
            {
                "id": "project_state_artifact_present",
                "kind": "path_exists",
                "path": "artifacts/project_state_2026-02-15.md",
                "description": "Narrative continuity is preserved through state artifacts.",
            },
            {
                "id": "compliance_impact_artifact_present",
                "kind": "path_exists",
                "path": "artifacts/compliance_impact_replication_advantage_2026-02-17.md",
                "description": "Identity claims are reinforced by compliance artifacts.",
            },
            {
                "id": "readme_citation_positioning",
                "kind": "file_contains",
                "path": "README.md",
                "pattern": "Cramer, J.G. (forthcoming 2026)",
                "description": "Consistent external positioning is maintained.",
            },
        ],
    },
    "customer_engagement": {
        "basic": [
            {
                "id": "prompt_templates_present",
                "kind": "path_exists",
                "path": "prompt_templates.py",
                "description": "Interaction prompts are productized.",
            },
            {
                "id": "posiwid_audit_present",
                "kind": "path_exists",
                "path": "posiwid_audit.py",
                "description": "Intention-vs-outcome feedback loop exists.",
            },
            {
                "id": "clinical_consensus_present",
                "kind": "path_exists",
                "path": "clinical_consensus.py",
                "description": "Multi-model feedback and comparison channel exists.",
            },
        ],
        "differentiating": [
            {
                "id": "character_seed_present",
                "kind": "path_exists",
                "path": "character_seed_experiment.py",
                "description": "Large-scale user-facing prompt archetype exploration exists.",
            },
            {
                "id": "archetype_matchmaker_present",
                "kind": "path_exists",
                "path": "archetype_matchmaker.py",
                "description": "Personalization layer connects archetypes to outcomes.",
            },
            {
                "id": "guide_workflow_present",
                "kind": "file_contains",
                "path": "docs/guide.md",
                "pattern": "## Typical Workflow",
                "description": "Engagement loop is documented as an iterative workflow.",
            },
        ],
    },
}


GAP_RECOMMENDATIONS = {
    "profit_model": "Strengthen explicit value-sustainment metrics across release cycles.",
    "network": "Add additional cross-project integration pathways and benchmark evidence.",
    "structure": "Formalize more role/process ownership artifacts for subsystem governance.",
    "process": "Expand process gates so more scripts have strict reproducibility checks.",
    "product_performance": "Add more targeted performance regression checks for edge scenarios.",
    "product_system": "Increase interoperability tests across bridges and orchestration runners.",
    "service": "Broaden service-level docs with explicit SLA-like expectations for outputs.",
    "channel": "Add additional user channels (packaged CLI presets or workflow wrappers).",
    "brand": "Codify brand-positioning checks to keep narrative consistency over time.",
    "customer_engagement": "Expand closed-loop feedback artifacts and scenario-level follow-up analyses.",
}


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _evaluate_probe(root: Path, type_id: str, tier: str, probe: dict[str, Any]) -> dict[str, Any]:
    kind = str(probe["kind"])
    rel_path = str(probe["path"])
    path = root / rel_path

    matched = False
    evidence_refs: list[str] = []
    detail = ""

    if kind == "path_exists":
        matched = path.exists()
        if matched:
            evidence_refs = [rel_path]
        detail = f"exists={matched}"

    elif kind == "file_contains":
        text = _safe_read_text(path)
        pattern = str(probe.get("pattern", ""))
        matched = bool(text) and pattern.lower() in text.lower()
        if matched:
            evidence_refs = [rel_path]
        detail = f"pattern_present={matched}"

    else:
        raise ValueError(f"Unsupported probe kind: {kind}")

    probe_key = f"{type_id}.{probe['id']}"
    return {
        "probe_key": probe_key,
        "type_id": type_id,
        "tier": tier,
        "description": str(probe["description"]),
        "criterion": {
            "kind": kind,
            "path": rel_path,
            "pattern": probe.get("pattern"),
        },
        "matched": matched,
        "evidence_refs": evidence_refs,
        "detail": detail,
    }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _assess_type(root: Path, type_meta: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    type_id = type_meta["type_id"]
    rules = TYPE_RULES[type_id]

    probe_results: list[dict[str, Any]] = []
    for probe in rules["basic"]:
        probe_results.append(_evaluate_probe(root, type_id, "basic", probe))
    for probe in rules["differentiating"]:
        probe_results.append(_evaluate_probe(root, type_id, "differentiating", probe))

    basic_hits = sum(1 for p in probe_results if p["tier"] == "basic" and p["matched"])
    differentiating_hits = sum(
        1 for p in probe_results if p["tier"] == "differentiating" and p["matched"]
    )
    total_hits = basic_hits + differentiating_hits

    if differentiating_hits >= 1 and total_hits >= 3:
        maturity = "differentiating"
        rationale = "Differentiating signal detected with broad supporting evidence."
    elif total_hits >= 1:
        maturity = "me_too"
        rationale = "Activity present, but differentiation threshold not fully met."
    else:
        maturity = "no_activity"
        rationale = "No meaningful repository evidence detected for this type."

    evidence_refs = _dedupe_preserve_order(
        [ref for probe in probe_results for ref in probe["evidence_refs"]]
    )

    assessment = {
        "type_id": type_id,
        "label": type_meta["label"],
        "canonical_label": type_meta["canonical_label"],
        "category": type_meta["category"],
        "maturity_level": maturity,
        "maturity_score": MATURITY_LEVELS[maturity],
        "counts_as_innovation": maturity == "differentiating",
        "hit_counts": {
            "basic": basic_hits,
            "differentiating": differentiating_hits,
            "total": total_hits,
        },
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }
    return assessment, probe_results


def build_summary(type_assessments: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(type_assessments)
    innovative_count = sum(1 for t in type_assessments if t["maturity_level"] == "differentiating")
    me_too_count = sum(1 for t in type_assessments if t["maturity_level"] == "me_too")
    no_activity_count = sum(1 for t in type_assessments if t["maturity_level"] == "no_activity")
    weighted_score = sum(int(t["maturity_score"]) for t in type_assessments)
    max_weighted = total * MATURITY_LEVELS["differentiating"] if total else 0

    return {
        "n_types": total,
        "strict_count_rule": "Only differentiating counts as true innovation.",
        "innovative_count": innovative_count,
        "me_too_count": me_too_count,
        "no_activity_count": no_activity_count,
        "innovation_coverage_pct": round((innovative_count / total) * 100.0, 2) if total else 0.0,
        "weighted_maturity_pct": round((weighted_score / max_weighted) * 100.0, 2)
        if max_weighted
        else 0.0,
    }


def _build_category_balance(type_assessments: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({row["category"] for row in type_assessments})
    out: dict[str, Any] = {}
    for category in categories:
        rows = [r for r in type_assessments if r["category"] == category]
        n = len(rows)
        out[category] = {
            "n_types": n,
            "differentiating": sum(1 for r in rows if r["maturity_level"] == "differentiating"),
            "me_too": sum(1 for r in rows if r["maturity_level"] == "me_too"),
            "no_activity": sum(1 for r in rows if r["maturity_level"] == "no_activity"),
            "mean_maturity_score": round(
                sum(int(r["maturity_score"]) for r in rows) / n, 3
            )
            if n
            else 0.0,
        }
    return out


def _build_priority_gaps(type_assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = [row for row in type_assessments if row["maturity_level"] != "differentiating"]
    gaps.sort(key=lambda row: (row["maturity_score"], row["type_id"]))

    out = []
    for row in gaps:
        out.append(
            {
                "type_id": row["type_id"],
                "label": row["label"],
                "category": row["category"],
                "current_level": row["maturity_level"],
                "recommended_next_step": GAP_RECOMMENDATIONS[row["type_id"]],
            }
        )
    return out


def run_audit(
    scope: str = "whole_platform",
    scoring_model: str = "strict_3_level",
    as_of: str | None = None,
) -> dict[str, Any]:
    """Run the Ten Types audit.

    Args:
        scope: Audit scope. v1 supports only ``whole_platform``.
        scoring_model: Scoring mode. v1 supports only ``strict_3_level``.
        as_of: Optional date string (YYYY-MM-DD) to stamp report context.

    Returns:
        Dictionary report containing type assessments, summary, category balance,
        priority gaps, and full evidence index.
    """
    if scope != "whole_platform":
        raise ValueError("Only scope='whole_platform' is supported in v1")
    if scoring_model != "strict_3_level":
        raise ValueError("Only scoring_model='strict_3_level' is supported in v1")

    if as_of is None:
        as_of = time.strftime("%Y-%m-%d")
    else:
        time.strptime(as_of, "%Y-%m-%d")

    type_assessments: list[dict[str, Any]] = []
    all_probe_results: list[dict[str, Any]] = []

    for type_meta in TYPE_TAXONOMY:
        assessment, probe_results = _assess_type(PROJECT, type_meta)
        type_assessments.append(assessment)
        all_probe_results.extend(probe_results)

    summary = build_summary(type_assessments)
    category_balance = _build_category_balance(type_assessments)
    priority_gaps = _build_priority_gaps(type_assessments)
    evidence_index = {probe["probe_key"]: probe for probe in all_probe_results}

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of,
        "scope": scope,
        "scoring_model": scoring_model,
        "type_assessments": type_assessments,
        "summary": summary,
        "category_balance": category_balance,
        "priority_gaps": priority_gaps,
        "evidence_index": evidence_index,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines: list[str] = [
        "# Ten Types of Innovation Audit",
        "",
        f"Generated: {report['timestamp']}",
        f"As of: {report['as_of']}",
        f"Scope: {report['scope']}",
        f"Scoring: {report['scoring_model']} (strict)",
        "",
        "## Summary",
        f"- Types assessed: {summary['n_types']}",
        f"- Innovative (differentiating only): {summary['innovative_count']}",
        f"- Me-too: {summary['me_too_count']}",
        f"- No activity: {summary['no_activity_count']}",
        f"- Innovation coverage: {summary['innovation_coverage_pct']}%",
        f"- Weighted maturity: {summary['weighted_maturity_pct']}%",
        "",
        "## Type Assessments",
        "",
        "| Type | Canonical | Category | Maturity | Score | Evidence refs |",
        "|---|---|---|---|---:|---|",
    ]

    for row in report["type_assessments"]:
        refs = ", ".join(row["evidence_refs"]) if row["evidence_refs"] else "-"
        lines.append(
            f"| {row['label']} | {row['canonical_label']} | {row['category']} | "
            f"{row['maturity_level']} | {row['maturity_score']} | {refs} |"
        )

    lines.extend(["", "## Category Balance", ""])
    for category, details in report["category_balance"].items():
        lines.append(
            f"- **{category}**: diff={details['differentiating']}, me-too={details['me_too']}, "
            f"no-activity={details['no_activity']}, mean-score={details['mean_maturity_score']}"
        )

    lines.extend(["", "## Priority Gaps", ""])
    if report["priority_gaps"]:
        for gap in report["priority_gaps"]:
            lines.append(
                f"- **{gap['label']}** ({gap['current_level']}): {gap['recommended_next_step']}"
            )
    else:
        lines.append("- No priority gaps; all types currently score as differentiating.")

    lines.extend([
        "",
        "## Strict Rule",
        "",
        f"- {summary['strict_count_rule']}",
    ])

    return "\n".join(lines) + "\n"


def _write_outputs(
    report: dict[str, Any],
    output_format: str,
    out_json: Path,
    out_md: Path,
) -> list[Path]:
    written: list[Path] = []
    if output_format in {"json", "both"}:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        written.append(out_json)
    if output_format in {"markdown", "both"}:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(report), encoding="utf-8")
        written.append(out_md)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ten Types of Innovation audit for this repository."
    )
    parser.add_argument(
        "--scope",
        default="whole_platform",
        choices=["whole_platform"],
        help="Audit scope (v1 supports only whole_platform).",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="both",
        choices=["json", "markdown", "both"],
        help="Output format.",
    )
    parser.add_argument(
        "--out-json",
        default=str(DEFAULT_JSON_PATH),
        help="JSON output path.",
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_MD_PATH),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Optional report context date (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    report = run_audit(scope=args.scope, scoring_model="strict_3_level", as_of=args.as_of)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    written = _write_outputs(
        report=report,
        output_format=args.output_format,
        out_json=out_json,
        out_md=out_md,
    )

    print("Ten Types audit complete")
    print(f"Scope: {report['scope']}")
    print(
        "Summary: "
        f"diff={report['summary']['innovative_count']} "
        f"me-too={report['summary']['me_too_count']} "
        f"no-activity={report['summary']['no_activity_count']}"
    )
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
