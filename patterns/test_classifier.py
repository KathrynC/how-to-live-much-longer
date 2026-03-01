#!/usr/bin/env python3
"""
Test the Lakoff archetype classifier with adjusted archetypes.
"""

import json
from pathlib import Path
import sys
import pytest

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.lakoff_classifier import load_adjusted_archetypes, classify_analytics
from simulator import simulate
from analytics import compute_all

def test_with_dummy_simulation():
    """Run a dummy simulation and classify its analytics."""
    print("Testing classifier with dummy simulation...")
    
    # Simple intervention cocktail
    intervention = {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.25,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }
    
    # Default patient
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
    
    print(f"Simulation complete. Final ATP: {result['states'][-1,2]:.3f}, Final het: {result['heteroplasmy'][-1]:.3f}")
    
    # Load adjusted archetypes
    library = load_adjusted_archetypes()
    
    # Classify
    classification = classify_analytics(analytics, library)
    
    print("\nClassification Results:")
    print(f"Best archetype: {classification['best_archetype']}")
    print(f"Similarity score: {classification['best_score']:.3f}")
    print(f"Grounding stats: {classification['grounding_stats']}")
    
    # Show similarity vector
    print("\nSimilarity vector:")
    for arch, score in classification['similarity_vector'].items():
        print(f"  {arch}: {score:.3f}")
    
    assert classification["best_archetype"] is not None
    assert 0.0 <= classification["best_score"] <= 1.0

def test_with_protocol_dictionary_sample():
    """Test classifier on a sample from protocol dictionary."""
    dict_path = ROOT / "artifacts" / "protocol_pipeline" / "protocol_dictionary.json"
    if not dict_path.exists():
        pytest.skip(f"Protocol dictionary not found at {dict_path}")
    
    print(f"\nLoading protocol dictionary sample from {dict_path}")
    with open(dict_path, 'r') as f:
        data = json.load(f)
    
    records = data.get("records", [])
    if not records:
        pytest.skip("No records found.")
    
    # Take first 5 records
    sample = records[:5]
    
    from patterns.lakoff_classifier import batch_classify_records
    library = load_adjusted_archetypes()
    results = batch_classify_records(sample, library)
    
    print(f"\nClassified {len(results)} records:")
    for i, res in enumerate(results):
        if "error" in res:
            print(f"Record {i}: ERROR {res['error']}")
        else:
            print(f"Record {i}: {res['existing_archetype']} -> {res['lakoff_archetype']} (score: {res['lakoff_score']:.3f})")
    
    assert len(results) == len(sample)

if __name__ == "__main__":
    print("=" * 70)
    print("Lakoff Archetype Classifier Test")
    print("=" * 70)
    
    # Test 1: Dummy simulation
    test_with_dummy_simulation()
    
    # Test 2: Protocol dictionary sample
    test_with_protocol_dictionary_sample()
    
    print("\n" + "=" * 70)
    print("Test completed.")
