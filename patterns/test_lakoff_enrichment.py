#!/usr/bin/env python3
"""
Test Lakoff archetype enrichment integration.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from protocol_record import ProtocolRecord
from protocol_enrichment import enrich_record
from simulator import simulate
from analytics import compute_all

def test_single_protocol():
    """Create a protocol record, enrich it, and check Lakoff archetype."""
    intervention = {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.25,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }
    patient = {
        "baseline_age": 70.0,
        "baseline_heteroplasmy": 0.3,
        "baseline_nad_level": 0.6,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.25,
    }
    
    # Simulate
    result = simulate(intervention=intervention, patient=patient)
    baseline = simulate(patient=patient)
    analytics = compute_all(result, baseline)
    
    # Create record
    record = ProtocolRecord(
        intervention=intervention,
        patient=patient,
        source="test",
        method="test",
        analytics=analytics,
        simulation={
            "final_atp": float(result['states'][-1, 2]),
            "final_het": float(result['heteroplasmy'][-1]),
        }
    )
    
    # Enrich
    enriched = enrich_record(record)
    
    print("Enrichment fields:")
    for key, value in enriched.enrichment.items():
        if key == "lakoff_archetype":
            print(f"  {key}:")
            if isinstance(value, dict):
                for subk, subv in value.items():
                    if isinstance(subv, dict):
                        print(f"    {subk}: ...")
                    else:
                        print(f"    {subk}: {subv}")
        else:
            print(f"  {key}: {value}")
    
    # Check Lakoff archetype
    lakoff = enriched.enrichment.get("lakoff_archetype", {})
    if isinstance(lakoff, dict):
        arch = lakoff.get("best_archetype")
        score = lakoff.get("best_score")
        print(f"\nLakoff archetype: {arch} (score: {score})")
    assert "lakoff_archetype" in enriched.enrichment
    assert isinstance(lakoff, dict)

if __name__ == "__main__":
    print("Testing Lakoff archetype enrichment...")
    test_single_protocol()
