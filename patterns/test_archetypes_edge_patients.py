#!/usr/bin/env python3
"""
Test archetypes on edge-case, near-cliff patients.

Loads edge patient population, filters for damaged patients,
runs aggressive intervention (scenario D), and classifies outcomes
with refined archetypes to see if archetype differentiation emerges.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lakoff_integration import (
    ArchetypeLibrary, extract_features_from_analytics,
    create_refined_archetypes, get_feature_layer
)
from scenario_definitions import get_example_scenarios, Scenario, InterventionProfile
from scenario_runner import run_scenario
from analytics import compute_all

def load_edge_patients():
    """Load edge patient population."""
    edge_path = ROOT / "artifacts" / "sample_patients_edge.json"
    with open(edge_path, 'r') as f:
        data = json.load(f)
    return data.get("patients", [])

def filter_damaged_patients(patients, min_het=0.5, max_age=90):
    """Filter patients with high heteroplasmy (damaged)."""
    damaged = []
    for p in patients:
        het = p.get('baseline_heteroplasmy', 0)
        age = p.get('baseline_age', 0)
        if het >= min_het and age <= max_age:
            damaged.append(p)
    return damaged

def filter_cliff_boundary(patients):
    """Filter patients in cliff_boundary category."""
    return [p for p in patients if p.get('_category') == 'cliff_boundary']

def create_scenario_for_patient(patient_dict, intervention_profile, scenario_name="Edge test"):
    """Create a Scenario for a given patient with intervention profile."""
    # Extract core patient params (6D)
    core_params = {
        'baseline_age': patient_dict['baseline_age'],
        'baseline_heteroplasmy': patient_dict['baseline_heteroplasmy'],
        'baseline_nad_level': patient_dict['baseline_nad_level'],
        'genetic_vulnerability': patient_dict['genetic_vulnerability'],
        'metabolic_demand': patient_dict['metabolic_demand'],
        'inflammation_level': patient_dict['inflammation_level'],
    }
    # Add expanded params if available
    expanded_params = dict(core_params)
    # Add other fields that might be needed for ParameterResolver
    # For simplicity, use core params only
    
    return Scenario(
        name=f"{scenario_name} - {patient_dict.get('_label', patient_dict.get('_id', 'unknown'))}",
        description=f"Edge patient {patient_dict.get('_category', 'unknown')} with het={patient_dict['baseline_heteroplasmy']:.2f}, age={patient_dict['baseline_age']}",
        patient_params=expanded_params,
        interventions=intervention_profile,
        duration_years=30.0,
    )

def classify_outcome(analytics_dict, library):
    """Classify analytics using refined archetype library."""
    features = extract_features_from_analytics(analytics_dict)
    best_arch, best_score = library.best_match(features)
    similarity_vector = library.similarity_vector(features)
    
    # Grounding statistics
    grounded = sum(1 for f in features if get_feature_layer(f) == "grounded")
    linking = len(features) - grounded
    grounding_ratio = grounded / len(features) if features else 0.0
    
    return {
        'best_archetype': best_arch.name if best_arch else None,
        'best_score': best_score,
        'similarity_vector': similarity_vector,
        'grounding_ratio': grounding_ratio,
        'grounded_features': grounded,
        'linking_features': linking,
    }

def main():
    print("=" * 70)
    print("Test Archetypes on Edge-Case, Damaged Patients")
    print("=" * 70)
    
    # Load refined archetypes
    library = create_refined_archetypes()
    print(f"Refined archetypes: {[a.name for a in library.archetypes]}")
    
    # Load edge patients
    all_patients = load_edge_patients()
    print(f"Total edge patients: {len(all_patients)}")
    
    # Filter for damaged patients (high heteroplasmy)
    damaged = filter_damaged_patients(all_patients, min_het=0.5)
    print(f"Damaged patients (het ≥ 0.5): {len(damaged)}")
    
    # Filter for cliff boundary patients
    cliff = filter_cliff_boundary(all_patients)
    print(f"Cliff boundary patients: {len(cliff)}")
    
    # Use scenario D (most aggressive) intervention profile
    example_scenarios = get_example_scenarios()
    scenario_d = example_scenarios[3]  # D: C + Experimental
    intervention_profile = scenario_d.interventions
    
    print(f"\nUsing intervention profile: {scenario_d.name}")
    print(f"  Transplant rate: {intervention_profile.transplant_rate}")
    print(f"  Yamanaka intensity: {intervention_profile.yamanaka_intensity}")
    print(f"  Rapamycin dose: {intervention_profile.rapamycin_dose}")
    
    # Test on a sample of damaged patients (limit to 5 for speed)
    sample_size = min(5, len(damaged))
    test_patients = damaged[:sample_size]
    
    print(f"\nTesting on {sample_size} damaged patients:")
    
    results = []
    for i, patient in enumerate(test_patients):
        print(f"\n[{i+1}/{sample_size}] {patient.get('_label', patient.get('_id', 'unknown'))}")
        print(f"  Category: {patient.get('_category', 'unknown')}")
        print(f"  Age: {patient['baseline_age']}, Het: {patient['baseline_heteroplasmy']:.3f}")
        print(f"  NAD: {patient['baseline_nad_level']:.3f}, Infl: {patient['inflammation_level']:.3f}")
        
        # Create scenario
        scenario = create_scenario_for_patient(
            patient, intervention_profile,
            scenario_name="D+Edge"
        )
        
        # Run simulation
        try:
            result = run_scenario(scenario, years=30, include_annotations=False)
            core = result['core']
            
            # Compute analytics
            analytics = compute_all(core, None)
            
            # Classify outcome
            classification = classify_outcome(analytics, library)
            
            # Extract key metrics
            energy = analytics.get('energy', {})
            damage = analytics.get('damage', {})
            dynamics = analytics.get('dynamics', {})
            
            results.append({
                'patient_id': patient.get('_id', 'unknown'),
                'patient_label': patient.get('_label', 'unknown'),
                'patient_category': patient.get('_category', 'unknown'),
                'baseline_age': patient['baseline_age'],
                'baseline_het': patient['baseline_heteroplasmy'],
                'atp_final': energy.get('atp_final', 0),
                'het_final': damage.get('het_final', 0),
                'het_delta': damage.get('delta_het', 0),
                'deletion_het_final': damage.get('deletion_het_final', 0),
                'ros_amplitude': dynamics.get('ros_amplitude', 0),
                'classification': classification,
            })
            
            print(f"  Outcome: ATP={energy.get('atp_final', 0):.3f}, Het={damage.get('het_final', 0):.3f}")
            print(f"  Archetype: {classification['best_archetype']} (score: {classification['best_score']:.3f})")
            print(f"  Grounding ratio: {classification['grounding_ratio']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Summary analysis
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        archetype_counts = {}
        for r in results:
            arch = r['classification']['best_archetype']
            archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        
        print(f"Archetype distribution:")
        for arch, count in archetype_counts.items():
            print(f"  {arch}: {count} patients")
        
        # Check if we see differentiation beyond conservative
        conservative_only = all(r['classification']['best_archetype'] == 'conservative' for r in results)
        if conservative_only:
            print("\nAll patients classified as 'conservative'.")
            print("Possible reasons:")
            print("  1. Archetype criteria too strict for damaged patients")
            print("  2. Intervention not aggressive enough for severe damage")
            print("  3. Patient baseline too close to collapse")
        else:
            print("\nArchetype differentiation observed!")
            print("Damaged patients produce varied archetype outcomes.")
        
        # Save results
        output_path = ROOT / "patterns" / "edge_patient_archetype_test.json"
        output_data = {
            'test_config': {
                'intervention_profile': scenario_d.name,
                'sample_size': sample_size,
                'patient_filter': 'damaged (het ≥ 0.5)',
            },
            'results': results,
            'archetype_distribution': archetype_counts,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to {output_path}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()