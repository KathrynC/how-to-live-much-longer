#!/usr/bin/env python3
"""
Test Lakoff archetype classification on existing protocol dictionary data.

Loads protocol dictionary, extracts analytics features, applies Lakoff archetype
classification, and compares with existing prototype labels.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path to import lakoff_integration
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lakoff_integration import (
    ArchetypeLibrary, MetaphorAuditor, extract_features_from_analytics,
    create_default_archetypes, get_feature_layer
)

def load_protocol_dictionary_sample(path: Path, sample_size: int = 100) -> List[Dict[str, Any]]:
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

def classify_protocol_record(record: Dict[str, Any], auditor: MetaphorAuditor) -> Dict[str, Any]:
    """Classify a single protocol record using Lakoff archetypes."""
    # Extract analytics features
    analytics = record.get("analytics", {})
    if not analytics:
        return {"error": "No analytics data"}
    
    features = extract_features_from_analytics(analytics)
    
    # Run metaphor audit
    audit_results = auditor.audit(features)
    
    # Find best matching archetype
    best_arch = None
    best_score = -1.0
    for arch_name, result in audit_results.items():
        if result["similarity"] > best_score:
            best_score = result["similarity"]
            best_arch = arch_name
    
    # Get existing classification for comparison
    existing_class = record.get("enrichment", {}).get("prototype", {}).get("archetype", "unknown")
    
    return {
        "protocol_id": record.get("_id", "unknown"),
        "existing_archetype": existing_class,
        "lakoff_best_archetype": best_arch,
        "lakoff_best_score": best_score,
        "audit_results": audit_results,
        "features_sample": {k: v for i, (k, v) in enumerate(features.items()) if i < 5},
    }

def analyze_feature_layers(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze feature layer distribution in protocol records."""
    layer_counts = {"grounded": 0, "linking": 0}
    feature_occurrence = {}
    
    for record in records[:10]:  # Analyze first 10 records
        analytics = record.get("analytics", {})
        if not analytics:
            continue
        
        features = extract_features_from_analytics(analytics)
        for feature in features.keys():
            layer = get_feature_layer(feature)
            layer_counts[layer] += 1
            feature_occurrence[feature] = feature_occurrence.get(feature, 0) + 1
    
    total = sum(layer_counts.values())
    if total > 0:
        grounded_pct = 100 * layer_counts["grounded"] / total
        linking_pct = 100 * layer_counts["linking"] / total
    else:
        grounded_pct = linking_pct = 0.0
    
    # Most common features
    sorted_features = sorted(feature_occurrence.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "layer_counts": layer_counts,
        "grounded_percentage": grounded_pct,
        "linking_percentage": linking_pct,
        "most_common_features": sorted_features[:10],
        "unique_features_count": len(feature_occurrence),
    }

def main():
    """Main test function."""
    print("=" * 70)
    print("Lakoff Archetype Classification Test on Protocol Dictionary")
    print("=" * 70)
    
    # Path to protocol dictionary
    dict_path = ROOT.parent / "artifacts" / "protocol_pipeline" / "protocol_dictionary.json"
    if not dict_path.exists():
        print(f"Protocol dictionary not found at {dict_path}")
        return
    
    # Load Lakoff archetypes
    print("\n1. Loading Lakoff archetypes...")
    library = create_default_archetypes()
    auditor = MetaphorAuditor(library)
    print(f"   Loaded {len(library.archetypes)} archetypes")
    
    # Load protocol sample
    print("\n2. Loading protocol dictionary sample...")
    sample = load_protocol_dictionary_sample(dict_path, sample_size=50)
    
    # Analyze feature layers
    print("\n3. Analyzing feature layers...")
    layer_analysis = analyze_feature_layers(sample)
    print(f"   Grounded features: {layer_analysis['grounded_percentage']:.1f}%")
    print(f"   Linking features: {layer_analysis['linking_percentage']:.1f}%")
    print(f"   Unique features: {layer_analysis['unique_features_count']}")
    print(f"   Most common features:")
    for feat, count in layer_analysis['most_common_features']:
        layer = get_feature_layer(feat)
        print(f"     {feat:40s} ({layer:8s}) : {count}")
    
    # Classify sample protocols
    print("\n4. Classifying sample protocols...")
    classifications = []
    agreement_counts = {"match": 0, "partial": 0, "mismatch": 0}
    
    for i, record in enumerate(sample[:20]):  # First 20 for quick test
        result = classify_protocol_record(record, auditor)
        classifications.append(result)
        
        # Compare with existing classification
        existing = result["existing_archetype"]
        lakoff = result["lakoff_best_archetype"]
        score = result["lakoff_best_score"]
        
        # Simple agreement heuristic
        if existing.lower() in lakoff.lower() or lakoff.lower() in existing.lower():
            agreement = "match"
        elif score > 0.5:
            agreement = "partial"
        else:
            agreement = "mismatch"
        
        agreement_counts[agreement] += 1
        
        if i < 5:  # Print first 5 for inspection
            print(f"   Protocol {i}:")
            print(f"     Existing: {existing}")
            print(f"     Lakoff: {lakoff} (score: {score:.2f})")
            print(f"     Agreement: {agreement}")
    
    # Summary statistics
    print("\n5. Classification summary:")
    total_classified = sum(agreement_counts.values())
    print(f"   Total classified: {total_classified}")
    print(f"   Matches: {agreement_counts['match']} ({100*agreement_counts['match']/total_classified:.1f}%)")
    print(f"   Partial: {agreement_counts['partial']} ({100*agreement_counts['partial']/total_classified:.1f}%)")
    print(f"   Mismatches: {agreement_counts['mismatch']} ({100*agreement_counts['mismatch']/total_classified:.1f}%)")
    
    # Archetype distribution
    print("\n6. Lakoff archetype distribution:")
    arch_counts = {}
    for result in classifications:
        arch = result["lakoff_best_archetype"]
        if arch:
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
    
    for arch, count in sorted(arch_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {arch:20s}: {count:3d} protocols")
    
    # Save results
    output_path = ROOT / "lakoff_protocol_classification_test.json"
    output_data = {
        "sample_size": len(sample),
        "classified_count": len(classifications),
        "layer_analysis": layer_analysis,
        "agreement_counts": agreement_counts,
        "archetype_distribution": arch_counts,
        "classifications": classifications[:10],  # Save first 10 for inspection
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()