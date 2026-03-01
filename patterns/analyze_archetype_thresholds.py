#!/usr/bin/env python3
"""
Analyze archetype criteria thresholds based on protocol dictionary distributions.

Loads protocol dictionary (2292 records), extracts features, computes
distributions for each feature used in archetype grounding criteria,
and suggests threshold adjustments based on percentiles.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lakoff_integration import (
    extract_features_from_analytics, create_refined_archetypes,
    get_feature_layer
)

def load_protocol_dictionary(path: Path, max_records: int = 1000) -> List[Dict[str, Any]]:
    """Load protocol records from dictionary JSON."""
    print(f"Loading protocol dictionary from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    records = data.get("records", [])
    print(f"Total records: {len(records)}")
    
    # Limit for performance
    if len(records) > max_records:
        np.random.seed(42)
        indices = np.random.choice(len(records), max_records, replace=False)
        records = [records[i] for i in indices]
        print(f"Sampled records: {len(records)}")
    
    return records

def extract_features_from_records(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Extract features from all records, returning feature→values mapping."""
    feature_values = {}
    
    for i, record in enumerate(records):
        analytics = record.get("analytics", {})
        if not analytics:
            continue
        
        features = extract_features_from_analytics(analytics)
        
        for feature, value in features.items():
            if feature not in feature_values:
                feature_values[feature] = []
            feature_values[feature].append(float(value))
    
    print(f"Extracted {len(feature_values)} unique features from {len(records)} records")
    return feature_values

def load_archetype_criteria():
    """Load archetype criteria from refined archetypes JSON."""
    archetypes_path = ROOT / "patterns" / "lakoff_archetypes_refined.json"
    with open(archetypes_path, 'r') as f:
        archetypes = json.load(f)
    
    criteria_by_feature = {}
    archetype_info = {}
    
    for arch in archetypes:
        arch_name = arch["name"]
        archetype_info[arch_name] = {
            "description": arch["description"],
            "criteria": []
        }
        
        for criterion in arch.get("grounding_criteria", []):
            feature = criterion["feature"]
            predicate = criterion["predicate"]
            value = criterion["value"]
            tolerance = criterion.get("tolerance", 0.0)
            rationale = criterion.get("rationale", "")
            layer = criterion.get("layer", "grounded")
            
            key = (feature, predicate)
            if key not in criteria_by_feature:
                criteria_by_feature[key] = []
            
            criteria_by_feature[key].append({
                "archetype": arch_name,
                "value": value,
                "tolerance": tolerance,
                "rationale": rationale,
                "layer": layer,
            })
            
            archetype_info[arch_name]["criteria"].append({
                "feature": feature,
                "predicate": predicate,
                "value": value,
                "tolerance": tolerance,
            })
    
    return criteria_by_feature, archetype_info

def compute_distributions(feature_values: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute distribution statistics for each feature."""
    distributions = {}
    
    for feature, values in feature_values.items():
        if len(values) < 2:
            continue
        
        arr = np.array(values)
        # Remove infinities and NaNs
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue
        
        distributions[feature] = {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }
    
    print(f"Computed distributions for {len(distributions)} features")
    return distributions

def analyze_thresholds(criteria_by_feature, distributions, archetype_info):
    """Analyze current thresholds vs feature distributions."""
    analysis = {}
    
    for (feature, predicate), criteria_list in criteria_by_feature.items():
        if feature not in distributions:
            continue
        
        dist = distributions[feature]
        
        for criterion in criteria_list:
            arch_name = criterion["archetype"]
            current_value = criterion["value"]
            tolerance = criterion["tolerance"]
            
            # Determine appropriate percentile based on predicate and archetype intent
            if predicate == "lt":  # less than threshold
                # For "lt", we want threshold to be at high percentile (e.g., p75 or p90)
                # because we want to catch values below that threshold
                suggested_percentile = 75  # conservative: p75
                suggested_value = dist["p75"]
                comparison = "current < suggested" if current_value < suggested_value else "current ≥ suggested"
                adjustment = suggested_value - current_value
                
            elif predicate == "gt":  # greater than threshold
                # For "gt", we want threshold to be at low percentile (e.g., p25)
                suggested_percentile = 25
                suggested_value = dist["p25"]
                comparison = "current > suggested" if current_value > suggested_value else "current ≤ suggested"
                adjustment = suggested_value - current_value
                
            elif predicate == "between":
                # For "between", we have value as lower bound, tolerance as upper bound offset
                # Actually, in our JSON, "between" uses value as lower, tolerance as range
                lower = current_value
                upper = current_value + tolerance
                # Suggest based on interquartile range
                suggested_lower = dist["p25"]
                suggested_upper = dist["p75"]
                comparison = f"current [{lower:.3f}, {upper:.3f}] vs IQR [{suggested_lower:.3f}, {suggested_upper:.3f}]"
                adjustment_lower = suggested_lower - lower
                adjustment_upper = suggested_upper - upper
                suggested_value = suggested_lower
                tolerance = suggested_upper - suggested_lower
                adjustment = (adjustment_lower, adjustment_upper)
            else:
                continue
            
            # Calculate what percentage of data meets current criterion
            values = np.array([v for v in distributions[feature].get("_values", []) if np.isfinite(v)])
            if len(values) == 0:
                meet_pct = 0
            else:
                if predicate == "lt":
                    meet_pct = 100 * np.sum(values < current_value) / len(values)
                elif predicate == "gt":
                    meet_pct = 100 * np.sum(values > current_value) / len(values)
                elif predicate == "between":
                    meet_pct = 100 * np.sum((values >= current_value) & (values <= current_value + tolerance)) / len(values)
                else:
                    meet_pct = 0
            
            key = f"{arch_name}.{feature}.{predicate}"
            analysis[key] = {
                "archetype": arch_name,
                "feature": feature,
                "predicate": predicate,
                "current_value": current_value,
                "current_tolerance": tolerance,
                "distribution": {
                    "mean": dist["mean"],
                    "std": dist["std"],
                    "min": dist["min"],
                    "max": dist["max"],
                    "p25": dist["p25"],
                    "p50": dist["p50"],
                    "p75": dist["p75"],
                },
                "suggested_percentile": suggested_percentile,
                "suggested_value": suggested_value,
                "suggested_tolerance": tolerance if predicate == "between" else None,
                "comparison": comparison,
                "adjustment": adjustment,
                "meet_percentage": meet_pct,
                "rationale": criterion.get("rationale", ""),
            }
    
    return analysis

def main():
    print("=" * 70)
    print("Archetype Criteria Threshold Analysis")
    print("=" * 70)
    
    # Load archetype criteria
    criteria_by_feature, archetype_info = load_archetype_criteria()
    print(f"Loaded {len(archetype_info)} archetypes")
    for arch_name, info in archetype_info.items():
        print(f"  {arch_name}: {len(info['criteria'])} criteria")
    
    # Load protocol dictionary
    dict_path = ROOT / "artifacts" / "protocol_pipeline" / "protocol_dictionary.json"
    if not dict_path.exists():
        print(f"Protocol dictionary not found at {dict_path}")
        return
    
    records = load_protocol_dictionary(dict_path, max_records=500)
    if not records:
        print("No records loaded")
        return
    
    # Extract features
    feature_values = extract_features_from_records(records)
    
    # Compute distributions
    distributions = compute_distributions(feature_values)
    
    # Store raw values for percentage calculations
    for feature, values in feature_values.items():
        if feature in distributions:
            distributions[feature]["_values"] = values
    
    # Analyze thresholds
    analysis = analyze_thresholds(criteria_by_feature, distributions, archetype_info)
    
    print(f"\nAnalyzed {len(analysis)} criteria")
    
    # Group by archetype
    by_archetype = {}
    for key, data in analysis.items():
        arch = data["archetype"]
        if arch not in by_archetype:
            by_archetype[arch] = []
        by_archetype[arch].append(data)
    
    # Print summary by archetype
    print("\n" + "=" * 70)
    print("SUMMARY BY ARCHETYPE")
    print("=" * 70)
    
    for arch, criteria_list in by_archetype.items():
        print(f"\n{arch.upper()}:")
        print(f"  {'Feature':<30} {'Pred':<6} {'Current':<8} {'Suggested':<10} {'Meet%':<6} {'Note'}")
        print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*20}")
        
        for data in sorted(criteria_list, key=lambda x: x["feature"]):
            feature = data["feature"]
            predicate = data["predicate"]
            current = data["current_value"]
            suggested = data["suggested_value"]
            meet_pct = data["meet_percentage"]
            
            if predicate == "between":
                tol = data["current_tolerance"]
                current_str = f"{current:.3f}+{tol:.3f}"
                suggested_str = f"{suggested:.3f}+{data['suggested_tolerance']:.3f}"
            else:
                current_str = f"{current:.3f}"
                suggested_str = f"{suggested:.3f}"
            
            note = ""
            if meet_pct < 10:
                note = "VERY STRICT (<10%)"
            elif meet_pct < 25:
                note = "STRICT (<25%)"
            elif meet_pct > 90:
                note = "VERY LOOSE (>90%)"
            elif meet_pct > 75:
                note = "LOOSE (>75%)"
            else:
                note = "MODERATE"
            
            print(f"  {feature:<30} {predicate:<6} {current_str:<8} {suggested_str:<10} {meet_pct:<6.1f} {note}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    strict_criteria = [d for d in analysis.values() if d["meet_percentage"] < 25]
    loose_criteria = [d for d in analysis.values() if d["meet_percentage"] > 75]
    
    if strict_criteria:
        print(f"\nSTRICT CRITERIA (meet <25% of protocols):")
        for data in strict_criteria[:5]:  # limit output
            print(f"  {data['archetype']}.{data['feature']}.{data['predicate']}:")
            print(f"    Current: {data['current_value']:.3f}, Meet: {data['meet_percentage']:.1f}%")
            print(f"    Suggested: {data['suggested_value']:.3f} (p{data['suggested_percentile']})")
            print(f"    Rationale: {data['rationale'][:80]}...")
    
    if loose_criteria:
        print(f"\nLOOSE CRITERIA (meet >75% of protocols):")
        for data in loose_criteria[:5]:
            print(f"  {data['archetype']}.{data['feature']}.{data['predicate']}:")
            print(f"    Current: {data['current_value']:.3f}, Meet: {data['meet_percentage']:.1f}%")
            print(f"    Suggested: {data['suggested_value']:.3f} (p{data['suggested_percentile']})")
    
    # Save detailed analysis
    output_path = ROOT / "patterns" / "archetype_threshold_analysis.json"
    output_data = {
        "analysis": analysis,
        "summary": {
            "total_criteria": len(analysis),
            "strict_criteria": len(strict_criteria),
            "loose_criteria": len(loose_criteria),
            "moderate_criteria": len(analysis) - len(strict_criteria) - len(loose_criteria),
        },
        "by_archetype": {arch: len(crit) for arch, crit in by_archetype.items()},
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed analysis saved to {output_path}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()