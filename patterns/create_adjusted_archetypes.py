#!/usr/bin/env python3
"""
Create adjusted archetype definitions based on threshold analysis.

Uses analysis from archetype_threshold_analysis.json to adjust criteria
to be more reasonable (target 25-75% satisfaction rate).
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def load_analysis():
    """Load threshold analysis results."""
    analysis_path = ROOT / "archetype_threshold_analysis.json"
    with open(analysis_path, 'r') as f:
        data = json.load(f)
    return data.get("analysis", {})

def load_original_archetypes():
    """Load original refined archetypes."""
    archetypes_path = ROOT / "lakoff_archetypes_refined.json"
    with open(archetypes_path, 'r') as f:
        archetypes = json.load(f)
    return archetypes

def adjust_criteria(original_archetypes, analysis):
    """Create adjusted archetypes with updated thresholds."""
    # Create mapping from feature+archetype+predicate to analysis
    analysis_map = {}
    for key, data in analysis.items():
        feature = data["feature"]
        archetype = data["archetype"]
        predicate = data["predicate"]
        analysis_map[(archetype, feature, predicate)] = data
    
    adjusted_archetypes = []
    
    for arch in original_archetypes:
        arch_name = arch["name"]
        adjusted_arch = {
            "name": arch_name,
            "description": arch["description"],
            "type": arch["type"],
            "grounding_criteria": [],
            "icm": arch.get("icm", {}),
        }
        
        for criterion in arch.get("grounding_criteria", []):
            feature = criterion["feature"]
            predicate = criterion["predicate"]
            current_value = criterion["value"]
            tolerance = criterion.get("tolerance", 0.0)
            rationale = criterion.get("rationale", "")
            layer = criterion.get("layer", "grounded")
            
            # Get analysis data if available
            analysis_key = (arch_name, feature, predicate)
            if analysis_key in analysis_map:
                analysis_data = analysis_map[analysis_key]
                meet_pct = analysis_data["meet_percentage"]
                suggested_value = analysis_data["suggested_value"]
                distribution = analysis_data["distribution"]
                
                # Decision logic for adjustment
                adjusted_value = current_value
                adjusted_tolerance = tolerance
                adjustment_note = "unchanged"
                
                if meet_pct < 25:  # Too strict
                    # Relax towards suggested (p25 for gt, p75 for lt)
                    if predicate == "gt":
                        # For "gt", suggested is p25 (lower bound)
                        # We want to lower threshold to be less strict
                        adjusted_value = suggested_value  # Use p25
                        adjustment_note = f"relaxed (was {current_value:.3f}, {meet_pct:.1f}% meet)"
                    elif predicate == "lt":
                        # For "lt", suggested is p75 (higher bound)  
                        # We want to raise threshold to be less strict
                        adjusted_value = suggested_value  # Use p75
                        adjustment_note = f"relaxed (was {current_value:.3f}, {meet_pct:.1f}% meet)"
                    elif predicate == "between":
                        # For "between", adjust both bounds toward IQR
                        adjusted_value = suggested_value
                        adjusted_tolerance = analysis_data["suggested_tolerance"]
                        adjustment_note = f"adjusted range (was {current_value:.3f}+{tolerance:.3f})"
                
                elif meet_pct > 75:  # Too loose
                    # Tighten towards suggested (opposite direction)
                    if predicate == "gt":
                        # For "gt", we want to raise threshold to be more strict
                        # Use p75 instead of p25
                        adjusted_value = distribution["p75"]
                        adjustment_note = f"tightened (was {current_value:.3f}, {meet_pct:.1f}% meet)"
                    elif predicate == "lt":
                        # For "lt", we want to lower threshold to be more strict
                        # Use p25 instead of p75
                        adjusted_value = distribution["p25"]
                        adjustment_note = f"tightened (was {current_value:.3f}, {meet_pct:.1f}% meet)"
                    elif predicate == "between":
                        # Tighten range (use narrower IQR)
                        iqr = distribution["p75"] - distribution["p25"]
                        adjusted_value = distribution["p50"] - iqr/2
                        adjusted_tolerance = iqr
                        adjustment_note = f"tightened range"
                
                else:  # Moderate (25-75%)
                    # Keep current or slight adjustment toward median
                    if abs(current_value - suggested_value) > 0.1:  # Large difference
                        # Move 25% toward suggested
                        adjustment = 0.25 * (suggested_value - current_value)
                        adjusted_value = current_value + adjustment
                        adjustment_note = f"adjusted toward p{analysis_data['suggested_percentile']}"
                    else:
                        adjusted_value = current_value
                        adjustment_note = "unchanged (moderate satisfaction)"
                
                # Special cases based on biological plausibility
                if feature == "dynamics.ros_amplitude":
                    # ROS amplitude biologically should be < 0.02 typically
                    if predicate == "gt" and adjusted_value > 0.02:
                        adjusted_value = 0.02  # Cap at biologically plausible max
                        adjustment_note += " (capped at 0.02 for biological plausibility)"
                    elif predicate == "lt" and adjusted_value > 0.02:
                        adjusted_value = 0.015  # Conservative upper bound
                        adjustment_note += " (capped at 0.015)"
                
                if feature == "dynamics.nad_slope":
                    # NAD slope often negative due to aging
                    if predicate == "gt" and adjusted_value > 0:
                        adjusted_value = -0.001  # Allow slight negative
                        adjustment_note += " (adjusted for typical NAD decline)"
                
                adjusted_criterion = {
                    "feature": feature,
                    "predicate": predicate,
                    "value": round(adjusted_value, 4),
                    "tolerance": round(adjusted_tolerance, 4),
                    "rationale": rationale,
                    "layer": layer,
                    "original_value": round(current_value, 4),
                    "adjustment_note": adjustment_note,
                    "meet_percentage": round(meet_pct, 1),
                }
                
            else:
                # No analysis data, keep original
                adjusted_criterion = dict(criterion)
                adjusted_criterion["adjustment_note"] = "no analysis data"
                adjusted_criterion["original_value"] = current_value
            
            adjusted_arch["grounding_criteria"].append(adjusted_criterion)
        
        adjusted_archetypes.append(adjusted_arch)
    
    return adjusted_archetypes

def create_adjustment_summary(original_archetypes, adjusted_archetypes, analysis):
    """Create summary of adjustments."""
    summary = []
    
    for orig_arch, adj_arch in zip(original_archetypes, adjusted_archetypes):
        arch_name = orig_arch["name"]
        arch_summary = {
            "archetype": arch_name,
            "description": orig_arch["description"],
            "criteria_changes": [],
        }
        
        for orig_crit, adj_crit in zip(orig_arch.get("grounding_criteria", []), 
                                       adj_arch.get("grounding_criteria", [])):
            feature = orig_crit["feature"]
            predicate = orig_crit["predicate"]
            orig_val = orig_crit["value"]
            adj_val = adj_crit["value"]
            
            if abs(orig_val - adj_val) > 0.001:  # Significant change
                change = {
                    "feature": feature,
                    "predicate": predicate,
                    "original": orig_val,
                    "adjusted": adj_val,
                    "change": adj_val - orig_val,
                    "adjustment_note": adj_crit.get("adjustment_note", ""),
                    "meet_percentage": adj_crit.get("meet_percentage", "unknown"),
                }
                arch_summary["criteria_changes"].append(change)
        
        summary.append(arch_summary)
    
    return summary

def main():
    print("=" * 70)
    print("Create Adjusted Archetypes Based on Threshold Analysis")
    print("=" * 70)
    
    # Load data
    analysis = load_analysis()
    print(f"Loaded analysis for {len(analysis)} criteria")
    
    original_archetypes = load_original_archetypes()
    print(f"Loaded {len(original_archetypes)} original archetypes")
    
    # Create adjusted archetypes
    adjusted_archetypes = adjust_criteria(original_archetypes, analysis)
    
    # Create summary
    summary = create_adjustment_summary(original_archetypes, adjusted_archetypes, analysis)
    
    # Save adjusted archetypes
    output_path = ROOT / "lakoff_archetypes_adjusted.json"
    with open(output_path, 'w') as f:
        json.dump(adjusted_archetypes, f, indent=2)
    print(f"\nAdjusted archetypes saved to {output_path}")
    
    # Save summary
    summary_path = ROOT / "archetype_adjustment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Adjustment summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ADJUSTMENT SUMMARY")
    print("=" * 70)
    
    for arch_summary in summary:
        print(f"\n{arch_summary['archetype'].upper()}: {arch_summary['description']}")
        if arch_summary['criteria_changes']:
            print("  Changes:")
            for change in arch_summary['criteria_changes']:
                direction = "↑" if change['change'] > 0 else "↓" if change['change'] < 0 else "→"
                print(f"    {change['feature']}.{change['predicate']}: {change['original']:.3f} {direction} {change['adjusted']:.3f}")
                print(f"      Note: {change['adjustment_note']}")
                if 'meet_percentage' in change and change['meet_percentage'] != 'unknown':
                    print(f"      Expected meet: {change['meet_percentage']}%")
        else:
            print("  No significant changes")
    
    # Count changes
    total_changes = sum(len(arch['criteria_changes']) for arch in summary)
    print(f"\nTotal criteria adjusted: {total_changes}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()