"""run_protocol_pipeline.py — End-to-end protocol curation pipeline.

Ported from rosetta-motion's run_pipeline.py. Orchestrates:
  1. Ingest protocols from various sources
  2. Run simulation + compute analytics (if not already present)
  3. Enrich records (complexity, clinical signature, prototype)
  4. Classify outcomes (multi-method pipeline)
  5. Apply rewrite rules (normalization, clinical flags)
  6. Route low-confidence to review queue
  7. Save dictionary + report

Usage:
    python run_protocol_pipeline.py --source dark_matter
    python run_protocol_pipeline.py --intervention '{"rapamycin_dose":0.5}'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol_record import ProtocolRecord
from protocol_dictionary import ProtocolDictionary
from protocol_enrichment import enrich_record
from protocol_classifier import multi_classify
from protocol_rewrite_rules import apply_rules
from protocol_review import needs_review, append_to_review_queue

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "protocol_pipeline"
DEFAULT_RULES = ROOT / "patterns" / "protocol_rewrite_rules.v1.json"


def ingest_from_simulation(
    intervention: dict[str, Any],
    patient: dict[str, Any],
    source: str = "manual",
    method: str = "",
) -> ProtocolRecord:
    """Create a ProtocolRecord by running a simulation and computing analytics."""
    from simulator import simulate
    from analytics import compute_all

    result = simulate(intervention=intervention, patient=patient)
    baseline = simulate(patient=patient)
    analytics = compute_all(result, baseline)

    final_states = result["states"][-1]
    het = result["heteroplasmy"][-1]

    return ProtocolRecord(
        intervention=intervention,
        patient=patient,
        source=source,
        method=method,
        analytics=analytics,
        simulation={
            "final_atp": float(final_states[2]),
            "final_het": float(het),
            "final_nad": float(final_states[4]),
            "final_sen": float(final_states[5]),
        },
    )


def ingest_dark_matter(path: Path) -> list[ProtocolRecord]:
    """Import protocols from dark_matter.py JSON artifact."""
    data = json.loads(path.read_text())
    records = []
    for patient_key, patient_data in data.items():
        categories = patient_data.get("by_category", {})
        for category, cat_data in categories.items():
            for example in cat_data.get("examples", []):
                records.append(ProtocolRecord(
                    intervention=example.get("intervention", {}),
                    patient={},  # dark_matter doesn't store full patient dict
                    source="dark_matter",
                    method="random_sample",
                    outcome_class=category,
                    simulation={
                        "final_atp": example.get("final_atp"),
                        "final_het": example.get("final_het"),
                    },
                    meta={"patient_key": patient_key},
                ))
    return records


def run_pipeline(
    records: list[ProtocolRecord],
    output_dir: Path | None = None,
    rules_path: Path | None = None,
    classify_pipeline: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full protocol curation pipeline.

    Steps:
      1. Simulate (if analytics missing)
      2. Enrich (complexity, clinical signature, prototype)
      3. Classify (multi-method)
      4. Apply rewrite rules
      5. Route to review if needed
      6. Save dictionary + report
    """
    out_dir = Path(output_dir or DEFAULT_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    rules_file = Path(rules_path or DEFAULT_RULES)

    dict_path = out_dir / "protocol_dictionary.json"
    review_path = out_dir / "review_queue.jsonl"
    report_path = out_dir / "pipeline_report.json"

    pd = ProtocolDictionary(dict_path)
    classify_pipe = classify_pipeline or ["rule", "analytics_fit"]

    # Load rewrite rules if available
    rules_data = None
    if rules_file.exists():
        rules_data = json.loads(rules_file.read_text())

    processed = 0
    reviewed = 0

    for rec in records:
        # Step 1: Simulate if needed
        if not rec.analytics and rec.intervention:
            try:
                rec = ingest_from_simulation(
                    rec.intervention, rec.patient,
                    source=rec.source, method=rec.method,
                )
            except Exception:
                pass  # Keep record without analytics

        # Step 2: Enrich
        rec = enrich_record(rec)

        # Step 3: Classify
        sim = rec.simulation
        final_atp = sim.get("final_atp", 0.0) or 0.0
        final_het = sim.get("final_het", 0.0) or 0.0
        baseline_atp = 0.6  # approximate no-treatment baseline
        baseline_het = rec.patient.get("baseline_heteroplasmy", 0.30)

        cls_result = multi_classify(
            final_atp=final_atp, final_het=final_het,
            baseline_atp=baseline_atp, baseline_het=baseline_het,
            analytics=rec.analytics,
            pipeline=classify_pipe,
        )
        rec.outcome_class = cls_result.get("outcome_class")
        rec.confidence = cls_result.get("confidence")
        rec.classifications = cls_result.get("methods", {})

        # Step 4: Apply rewrite rules
        if rules_data is not None:
            flat = rec.to_dict()
            # Merge simulation fields into flat dict for rule matching
            flat.update(rec.simulation)
            flat.update(rec.intervention)
            updated, _trace = apply_rules(flat, rules_data)
            # Extract flags back
            if "flags" in updated:
                rec.meta["flags"] = updated["flags"]

        # Step 5: Route to review if needed
        if needs_review(rec.confidence, rec.outcome_class):
            reason = "paradoxical" if rec.outcome_class == "paradoxical" else "low_confidence"
            append_to_review_queue(review_path, rec, reason=reason)
            reviewed += 1

        pd.add(rec)
        processed += 1

    pd.save()

    # Step 6: Generate report
    report = {
        "total_processed": processed,
        "total_reviewed": reviewed,
        "summary": pd.summary(),
    }
    report_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Protocol curation pipeline")
    ap.add_argument("--source", choices=["dark_matter", "manual"],
                    default="manual", help="Data source to ingest")
    ap.add_argument("--artifact", type=str, default=None,
                    help="Path to source artifact (e.g. dark_matter.json)")
    ap.add_argument("--intervention", type=str, default=None,
                    help="JSON intervention dict for manual mode")
    ap.add_argument("--patient", type=str, default=None,
                    help="JSON patient dict for manual mode")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                    help="Output directory")
    args = ap.parse_args()

    records: list[ProtocolRecord] = []

    if args.source == "dark_matter" and args.artifact:
        records = ingest_dark_matter(Path(args.artifact))
    elif args.intervention:
        iv = json.loads(args.intervention)
        pt = json.loads(args.patient) if args.patient else {
            "baseline_age": 70.0, "baseline_heteroplasmy": 0.30,
            "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
            "metabolic_demand": 1.0, "inflammation_level": 0.25,
        }
        rec = ingest_from_simulation(iv, pt, source="manual", method="cli")
        records = [rec]

    if not records:
        print("No records to process. Use --source or --intervention.")
        return

    result = run_pipeline(records, output_dir=Path(args.output))
    print(f"Pipeline complete: {result['total_processed']} processed, "
          f"{result['total_reviewed']} sent to review")
    print(f"Output: {args.output}/protocol_dictionary.json")


if __name__ == "__main__":
    main()
