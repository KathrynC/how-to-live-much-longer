#!/usr/bin/env python3
"""
Test refined archetypes on protocol dictionary data.

Loads refined archetypes (Lakoff Maxim 7 compliant) and classifies
protocol records from the protocol dictionary. Compares with default
archetype classifications to assess impact of grounding criteria refinement.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Any
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lakoff_integration import (
    ArchetypeLibrary, MetaphorAuditor, extract_features_from_analytics,
    create_default_archetypes, create_refined_archetypes, get_feature_layer
)

def load_protocol_dictionary_sample(path: Path, sample_size: int = 50) -> List[Dict[str, Any]]:
    """Load a sample of protocol records from dictionary JSON."""
    print(f"Loading protocol dictionary from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    records = data.get("records", [])
    print(f"Total records: {len(records)}")
    
    # Take a random sample (but deterministic for reproducibility)
    np.random.seed(42)
    if len(records) > sample_size:
        indices = np.random.choice(len(records), sample_size, replace=False)
        sample = [records[i] for i in indices]
    else:
        sample = records
    
    print(f"Sample size: {len(sample)}")
    return sample

def classify_with_library(record: Dict[str, Any], library: ArchetypeLibrary) -> Dict[str, Any]:
    """Classify a single protocol record using given archetype library."""
    analytics = record.get("analytics", {})
    if not analytics:
        return {"error": "No analytics data"}
    
    features = extract_features_from_analytics(analytics)
    
    # Find best matching archetype
    best_arch, best_score = library.best_match(features)
    
    # Get similarity vector
    similarity_vector = library.similarity_vector(features)
    
    # Get existing classification for comparison
    existing_class = record.get("enrichment", {}).get("prototype", {}).get("archetype", "unknown")
    
    return {
        "protocol_id": record.get("_id", "unknown"),
        "existing_archetype": existing_class,
        "best_archetype": best_arch.name if best_arch else None,
        "best_score": best_score,
        "similarity_vector": similarity_vector,
        "features_sample": {k: v for i, (k, v) in enumerate(features.items()) if i < 3},
    }

def compare_classifications(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare classifications using default vs refined archetypes."""
    default_lib = create_default_archetypes()
    refined_lib = create_refined_archetypes()
    
    print(f"\nDefault archetypes: {[a.name for a in default_lib.archetypes]}")
    print(f"Refined archetypes: {[a.name for a in refined_lib.archetypes]}")
    
    results = []
    agreement_counts = {"match": 0, "partial": 0, "mismatch": 0}
    
    for i, record in enumerate(sample[:10]):  # First 10 for quick comparison
        default_result = classify_with_library(record, default_lib)
        refined_result = classify_with_library(record, refined_lib)
        
        default_arch = default_result["best_archetype"]
        refined_arch = refined_result["best_archetype"]
        default_score = default_result["best_score"]
        refined_score = refined_result["best_score"]
        
        # Determine agreement
        if default_arch == refined_arch:
            agreement = "match"
        elif default_score > 0.5 and refined_score > 0.5:
            agreement = "partial"
        else:
            agreement = "mismatch"
        
        agreement_counts[agreement] += 1
        
        results.append({
            "protocol_id": record.get("_id", "unknown"),
            "existing": default_result["existing_archetype"],
            "default_archetype": default_arch,
            "default_score": default_score,
            "refined_archetype": refined_arch,
            "refined_score": refined_score,
            "agreement": agreement,
        })
        
        if i < 5:  # Print first 5 for inspection
            print(f"  Protocol {i}:")
            print(f"    Existing: {default_result['existing_archetype']}")
            print(f"    Default: {default_arch} ({default_score:.2f})")
            print(f"    Refined: {refined_arch} ({refined_score:.2f})")
            print(f"    Agreement: {agreement}")
    
    # Summary statistics
    total = sum(agreement_counts.values())
    summary = {
        "total_classified": total,
        "agreement_counts": agreement_counts,
        "match_percentage": 100 * agreement_counts["match"] / total if total > 0 else 0,
        "partial_percentage": 100 * agreement_counts["partial"] / total if total > 0 else 0,
        "mismatch_percentage": 100 * agreement_counts["mismatch"] / total if total > 0 else 0,
    }
    
    # Archetype distribution
    default_dist = {}
    refined_dist = {}
    for result in results:
        if result["default_archetype"]:
            default_dist[result["default_archetype"]] = default_dist.get(result["default_archetype"], 0) + 1
        if result["refined_archetype"]:
            refined_dist[result["refined_archetype"]] = refined_dist.get(result["refined_archetype"], 0) + 1
    
    summary["default_distribution"] = default_dist
    summary["refined_distribution"] = refined_dist
    
    return {"results": results, "summary": summary}

def analyze_grounding_criteria():
    """Analyze grounding criteria differences between default and refined archetypes."""
    default_lib = create_default_archetypes()
    refined_lib = create_refined_archetypes()
    
    analysis = {}
    
    for arch_name in ["conservative", "aggressive", "transplant_focused", "metabolic_optimizer"]:
        default_arch = default_lib.get(arch_name)
        refined_arch = refined_lib.get(arch_name)
        
        if not default_arch or not refined_arch:
            continue
        
        default_criteria = default_arch.grounding_criteria
        refined_criteria = refined_arch.grounding_criteria
        
        # Count grounded vs linking features
        default_grounded = sum(1 for gc in default_criteria if get_feature_layer(gc.feature) == "grounded")
        default_linking = len(default_criteria) - default_grounded
        
        refined_grounded = sum(1 for gc in refined_criteria if get_feature_layer(gc.feature) == "grounded")
        refined_linking = len(refined_criteria) - refined_grounded
        
        analysis[arch_name] = {
            "default": {"total": len(default_criteria), "grounded": default_grounded, "linking": default_linking},
            "refined": {"total": len(refined_criteria), "grounded": refined_grounded, "linking": refined_linking},
            "grounding_improvement": refined_grounded - default_grounded,
        }
    
    return analysis

def main():
    """Main test function."""
    print("=" * 70)
    print("Refined Archetype Test on Protocol Dictionary")
    print("=" * 70)
    
    # Path to protocol dictionary
    dict_path = ROOT.parent / "artifacts" / "protocol_pipeline" / "protocol_dictionary.json"
    if not dict_path.exists():
        print(f"Protocol dictionary not found at {dict_path}")
        return
    
    # Load protocol sample
    sample = load_protocol_dictionary_sample(dict_path, sample_size=50)
    
    # Compare classifications
    print("\n1. Comparing default vs refined archetype classifications...")
    comparison = compare_classifications(sample)
    
    print("\n2. Classification agreement summary:")
    summary = comparison["summary"]
    print(f"   Total classified: {summary['total_classified']}")
    print(f"   Match: {summary['agreement_counts']['match']} ({summary['match_percentage']:.1f}%)")
    print(f"   Partial: {summary['agreement_counts']['partial']} ({summary['partial_percentage']:.1f}%)")
    print(f"   Mismatch: {summary['agreement_counts']['mismatch']} ({summary['mismatch_percentage']:.1f}%)")
    
    print("\n3. Archetype distribution:")
    print("   Default:", summary["default_distribution"])
    print("   Refined:", summary["refined_distribution"])
    
    # Analyze grounding criteria
    print("\n4. Grounding criteria analysis:")
    grounding_analysis = analyze_grounding_criteria()
    for arch_name, data in grounding_analysis.items():
        print(f"   {arch_name}:")
        print(f"     Default: {data['default']['grounded']}/{data['default']['total']} grounded")
        print(f"     Refined: {data['refined']['grounded']}/{data['refined']['total']} grounded")
        print(f"     Improvement: +{data['grounding_improvement']} grounded criteria")
    
    # Save results
    output_path = ROOT / "refined_archetype_test.json"
    output_data = {
        "sample_size": len(sample),
        "comparison": comparison,
        "grounding_analysis": grounding_analysis,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()