#!/usr/bin/env python3
"""
Test updated transplant_focused archetype with young patient.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.lakoff_classifier import load_adjusted_archetypes, classify_analytics
from simulator import simulate
from analytics import compute_all

print("Testing updated transplant_focused archetype...")

# Young patient (20 years old, het=0.3)
patient = {
    "baseline_age": 20.0,
    "baseline_heteroplasmy": 0.3,
    "baseline_nad_level": 0.6,
    "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0,
    "inflammation_level": 0.25,
}

# Fixed intervention (no transplant)
intervention = {
    "rapamycin_dose": 0.5,
    "nad_supplement": 0.75,
    "senolytic_dose": 0.25,
    "yamanaka_intensity": 0.0,
    "transplant_rate": 0.0,  # NO TRANSPLANT
    "exercise_level": 0.5,
}

print(f"Patient: age={patient['baseline_age']}, het={patient['baseline_heteroplasmy']}")
print(f"Intervention: transplant_rate={intervention['transplant_rate']}")

# Simulate
result = simulate(intervention=intervention, patient=patient)
baseline = simulate(patient=patient)
analytics = compute_all(result, baseline)

print(f"Results:")
print(f"  Final ATP: {result['states'][-1,2]:.3f}")
print(f"  Final het: {result['heteroplasmy'][-1]:.3f}")
print(f"  Het benefit: {patient['baseline_heteroplasmy'] - result['heteroplasmy'][-1]:.3f}")

# Classify
library = load_adjusted_archetypes()
classification = classify_analytics(analytics, library)

print(f"\nClassification:")
print(f"  Best archetype: {classification['best_archetype']}")
print(f"  Similarity score: {classification['best_score']:.3f}")
print(f"  Similarity vector: {classification['similarity_vector']}")

# Check transplant_focused specifically
if 'transplant_focused' in classification['similarity_vector']:
    tf_score = classification['similarity_vector']['transplant_focused']
    print(f"\nTransplant_focused similarity: {tf_score:.3f}")
    if tf_score > 0.5:
        print("  WARNING: Young patient still matches transplant_focused!")
    else:
        print("  Good: Young patient does not strongly match transplant_focused")

# Also test an older patient with higher deletion het
print("\n" + "="*50)
print("Testing older patient (70 years, het=0.5)...")

older_patient = patient.copy()
older_patient["baseline_age"] = 70.0
older_patient["baseline_heteroplasmy"] = 0.5

result2 = simulate(intervention=intervention, patient=older_patient)
baseline2 = simulate(patient=older_patient)
analytics2 = compute_all(result2, baseline2)

print(f"  Final ATP: {result2['states'][-1,2]:.3f}")
print(f"  Final het: {result2['heteroplasmy'][-1]:.3f}")
print(f"  Het benefit: {older_patient['baseline_heteroplasmy'] - result2['heteroplasmy'][-1]:.3f}")

classification2 = classify_analytics(analytics2, library)
print(f"  Best archetype: {classification2['best_archetype']}")
print(f"  Similarity vector: {classification2['similarity_vector']}")