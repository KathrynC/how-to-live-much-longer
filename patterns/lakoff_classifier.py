#!/usr/bin/env python3
"""
Lakoff archetype classifier for mitochondrial intervention protocols.

Loads adjusted archetypes from JSON and provides classification functions
for protocol records, analytics dicts, or simulation results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lakoff_integration import (
    ArchetypeLibrary, MetaphorAuditor, extract_features_from_analytics,
    get_feature_layer
)

ADJUSTED_ARCHETYPES_PATH = Path(__file__).resolve().parent / "lakoff_archetypes_adjusted.json"

def load_adjusted_archetypes() -> ArchetypeLibrary:
    """Load adjusted archetypes from JSON file."""
    if not ADJUSTED_ARCHETYPES_PATH.exists():
        raise FileNotFoundError(f"Adjusted archetypes file not found: {ADJUSTED_ARCHETYPES_PATH}")
    library = ArchetypeLibrary.load(ADJUSTED_ARCHETYPES_PATH)
    print(f"Loaded {len(library.archetypes)} adjusted archetypes: {[a.name for a in library.archetypes]}")
    return library

def classify_analytics(analytics_dict: Dict[str, Dict[str, float]], 
                       library: Optional[ArchetypeLibrary] = None) -> Dict[str, Any]:
    """
    Classify a 4-pillar analytics dict into Lakoff archetypes.
    
    Args:
        analytics_dict: Output from analytics.compute_all()
        library: Optional ArchetypeLibrary (loads adjusted archetypes if None)
        
    Returns:
        Dict with classification results:
        - best_archetype: name of best matching archetype
        - best_score: similarity score (0-1)
        - similarity_vector: dict mapping archetype names to scores
        - grounding_stats: counts of grounded vs linking features used
        - audit_results: full metaphor audit dict
    """
    if library is None:
        library = load_adjusted_archetypes()
    
    features = extract_features_from_analytics(analytics_dict)
    auditor = MetaphorAuditor(library)
    audit_results = auditor.audit(features)
    
    # Find best match
    best_arch = None
    best_score = -1.0
    for arch_name, result in audit_results.items():
        if result["similarity"] > best_score:
            best_score = result["similarity"]
            best_arch = arch_name
    
    # Compute grounding statistics
    grounded = linking = 0
    for feature in features.keys():
        layer = get_feature_layer(feature)
        if layer == "grounded":
            grounded += 1
        else:
            linking += 1
    total = grounded + linking
    grounding_ratio = grounded / total if total > 0 else 0.0
    
    return {
        "best_archetype": best_arch,
        "best_score": best_score,
        "similarity_vector": {arch_name: result["similarity"] for arch_name, result in audit_results.items()},
        "grounding_stats": {
            "grounded": grounded,
            "linking": linking,
            "grounding_ratio": grounding_ratio
        },
        "audit_results": audit_results
    }

def classify_protocol_record(record: Dict[str, Any], 
                             library: Optional[ArchetypeLibrary] = None) -> Dict[str, Any]:
    """
    Classify a protocol dictionary record (from protocol dictionary).
    
    Expects record to have "analytics" field with 4-pillar analytics.
    Returns classification results merged with record metadata.
    """
    analytics = record.get("analytics")
    if not analytics:
        return {"error": "No analytics in record"}
    
    classification = classify_analytics(analytics, library)
    
    # Extract existing classification for comparison
    existing = record.get("enrichment", {}).get("prototype", {}).get("archetype", "unknown")
    
    return {
        "record_id": record.get("_id", "unknown"),
        "existing_archetype": existing,
        "lakoff_archetype": classification["best_archetype"],
        "lakoff_score": classification["best_score"],
        "similarity_vector": classification["similarity_vector"],
        "grounding_stats": classification["grounding_stats"],
        "audit_summary": {k: v["similarity"] for k, v in classification["audit_results"].items()}
    }

def batch_classify_records(records: List[Dict[str, Any]], 
                           library: Optional[ArchetypeLibrary] = None) -> List[Dict[str, Any]]:
    """Classify multiple protocol records."""
    if library is None:
        library = load_adjusted_archetypes()
    
    results = []
    for i, record in enumerate(records):
        try:
            result = classify_protocol_record(record, library)
            results.append(result)
        except Exception as e:
            print(f"Error classifying record {i}: {e}")
            results.append({"error": str(e), "record_index": i})
    
    return results

if __name__ == "__main__":
    # Quick test
    print("Testing Lakoff archetype classifier...")
    library = load_adjusted_archetypes()
    print("Successfully loaded adjusted archetypes.")