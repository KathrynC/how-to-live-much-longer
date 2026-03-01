#!/usr/bin/env python3
"""
Validate heuristic archetype mappings against actual simulation outcomes.

Compares:
1. Heuristic mapping (scenario A→conservative, B→metabolic_optimizer, etc.)
2. Actual archetype classification based on simulation analytics using refined archetypes
3. Grounding compliance (proportion of grounded features)

Runs all 4 scenarios (A-D), extracts analytics, classifies with refined archetypes,
and reports agreement/disagreement with heuristic mapping.
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
from scenario_definitions import get_example_scenarios
from scenario_runner import run_scenario
from analytics import compute_all

def run_scenario_with_analytics(scenario):
    """Run scenario and compute analytics."""
    result = run_scenario(scenario, years=30, include_annotations=False)
    core = result['core']
    baseline = None  # We could compute baseline, but for archetype classification we just need the treatment analytics
    # For archetype classification, we need the analytics dict
    analytics_dict = compute_all(core, baseline)
    return {
        'result': result,
        'analytics': analytics_dict,
        'scenario_name': scenario.name
    }

def classify_with_refined_archetypes(analytics_dict, library):
    """Classify analytics using refined archetype library."""
    features = extract_features_from_analytics(analytics_dict)
    best_arch, best_score = library.best_match(features)
    similarity_vector = library.similarity_vector(features)
    
    # Get grounding statistics
    grounded_count = 0
    linking_count = 0
    for feature_name in features.keys():
        layer = get_feature_layer(feature_name)
        if layer == "grounded":
            grounded_count += 1
        else:
            linking_count += 1
    
    return {
        'best_archetype': best_arch.name if best_arch else None,
        'best_score': best_score,
        'similarity_vector': similarity_vector,
        'grounded_features': grounded_count,
        'linking_features': linking_count,
        'grounding_ratio': grounded_count / (grounded_count + linking_count) if (grounded_count + linking_count) > 0 else 0.0
    }

def get_heuristic_mapping():
    """Return heuristic mapping from scenario names to archetypes."""
    return {
        "A: Sleep + Alcohol Cessation": "conservative",
        "B: A + OTC Supplements + Keto": "metabolic_optimizer",
        "C: B + Prescription": "aggressive",
        "D: C + Experimental": "transplant_focused",
    }

def main():
    print("=" * 70)
    print("Validate Archetype Mappings: Heuristic vs Actual Classification")
    print("=" * 70)
    
    # Load refined archetypes
    library = create_refined_archetypes()
    print(f"Loaded refined archetypes: {[a.name for a in library.archetypes]}")
    
    # Get heuristic mapping
    heuristic_mapping = get_heuristic_mapping()
    
    # Run all scenarios
    scenarios = get_example_scenarios()
    results = []
    
    for scenario in scenarios:
        print(f"\nProcessing {scenario.name}...")
        
        # Run simulation and compute analytics
        data = run_scenario_with_analytics(scenario)
        
        # Classify with refined archetypes
        classification = classify_with_refined_archetypes(data['analytics'], library)
        
        # Get heuristic archetype
        heuristic_archetype = heuristic_mapping.get(scenario.name, "")
        
        # Determine agreement
        agreement = classification['best_archetype'] == heuristic_archetype
        agreement_str = "✓ MATCH" if agreement else "✗ MISMATCH"
        
        results.append({
            'scenario_name': scenario.name,
            'heuristic_archetype': heuristic_archetype,
            'actual_archetype': classification['best_archetype'],
            'similarity_score': classification['best_score'],
            'agreement': agreement,
            'grounding_ratio': classification['grounding_ratio'],
            'grounded_features': classification['grounded_features'],
            'linking_features': classification['linking_features'],
            'similarity_vector': classification['similarity_vector']
        })
        
        print(f"  Heuristic: {heuristic_archetype}")
        print(f"  Actual: {classification['best_archetype']} (score: {classification['best_score']:.3f})")
        print(f"  Agreement: {agreement_str}")
        print(f"  Grounding ratio: {classification['grounding_ratio']:.2f} ({classification['grounded_features']} grounded, {classification['linking_features']} linking)")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = len(results)
    matches = sum(1 for r in results if r['agreement'])
    match_percentage = 100 * matches / total if total > 0 else 0
    
    print(f"Total scenarios: {total}")
    print(f"Matches: {matches}/{total} ({match_percentage:.1f}%)")
    
    # Detailed breakdown
    print("\nDetailed breakdown:")
    for r in results:
        status = "✓" if r['agreement'] else "✗"
        print(f"  {status} {r['scenario_name']}")
        print(f"    Heuristic: {r['heuristic_archetype']}")
        print(f"    Actual: {r['actual_archetype']} (score: {r['similarity_score']:.3f})")
    
    # Grounding analysis
    print("\nGrounding analysis:")
    avg_grounding = sum(r['grounding_ratio'] for r in results) / total if total > 0 else 0
    print(f"Average grounding ratio: {avg_grounding:.3f}")
    for r in results:
        print(f"  {r['scenario_name']}: {r['grounding_ratio']:.3f}")
    
    # Save results
    output_path = ROOT / "archetype_validation_results.json"
    output_data = {
        'validation_date': "2026-02-22",
        'total_scenarios': total,
        'matches': matches,
        'match_percentage': match_percentage,
        'average_grounding_ratio': avg_grounding,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if match_percentage == 100:
        print("Perfect agreement! Heuristic mapping matches actual archetype classification.")
        print("This validates the heuristic mapping based on intervention profiles.")
    elif match_percentage >= 75:
        print("Good agreement. Heuristic mapping is mostly accurate.")
        print("Consider refining mappings for mismatched scenarios.")
    elif match_percentage >= 50:
        print("Moderate agreement. Heuristic mapping has some predictive value.")
        print("Review mismatches to understand why intervention profiles don't match outcomes.")
    else:
        print("Poor agreement. Heuristic mapping does not predict actual archetypes well.")
        print("Consider revising heuristic mapping or archetype definitions.")
    
    # Analyze mismatches
    mismatches = [r for r in results if not r['agreement']]
    if mismatches:
        print(f"\nMismatches to investigate ({len(mismatches)}):")
        for r in mismatches:
            print(f"  {r['scenario_name']}: heuristic={r['heuristic_archetype']}, actual={r['actual_archetype']}")
            # Show similarity vector to understand classification
            sim_vec = r['similarity_vector']
            if isinstance(sim_vec, dict):
                print(f"    Similarities: {', '.join(f'{k}:{v:.3f}' for k, v in sim_vec.items())}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()