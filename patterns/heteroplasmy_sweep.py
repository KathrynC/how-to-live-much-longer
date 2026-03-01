#!/usr/bin/env python3
"""
Systematic sweep of heteroplasmy to map archetype transitions.

Fix all patient parameters except heteroplasmy, simulate with a fixed
intervention, classify outcomes with Lakoff archetypes, and identify
thresholds where archetype changes.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.lakoff_classifier import load_adjusted_archetypes, classify_analytics
from simulator import simulate
from analytics import compute_all

# Fixed intervention protocol (moderate cocktail)
FIXED_INTERVENTION = {
    "rapamycin_dose": 0.5,
    "nad_supplement": 0.75,
    "senolytic_dose": 0.25,
    "yamanaka_intensity": 0.0,
    "transplant_rate": 0.0,
    "exercise_level": 0.5,
}

# Base patient (moderate 70-year-old)
BASE_PATIENT = {
    "baseline_age": 70.0,
    "baseline_nad_level": 0.6,
    "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0,
    "inflammation_level": 0.25,
}

def sweep_heteroplasmy(
    het_range: np.ndarray,
    intervention: Dict[str, Any],
    base_patient: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Sweep heteroplasmy values, simulate, classify."""
    library = load_adjusted_archetypes()
    results = []
    
    for i, het in enumerate(het_range):
        patient = base_patient.copy()
        patient["baseline_heteroplasmy"] = float(het)
        
        print(f"Progress {i+1}/{len(het_range)}: het={het:.3f}")
        
        try:
            # Simulate baseline
            baseline = simulate(patient=patient)
            # Simulate intervention
            result = simulate(intervention=intervention, patient=patient)
            analytics = compute_all(result, baseline)
            
            # Classify
            classification = classify_analytics(analytics, library)
            
            # Extract key outcomes
            final_atp = result['states'][-1, 2]
            final_het = result['heteroplasmy'][-1]
            baseline_atp = baseline['states'][-1, 2]
            baseline_het = baseline['heteroplasmy'][-1]
            
            record = {
                "heteroplasmy": float(het),
                "outcomes": {
                    "final_atp": float(final_atp),
                    "final_het": float(final_het),
                    "baseline_atp": float(baseline_atp),
                    "baseline_het": float(baseline_het),
                    "atp_benefit": float(final_atp - baseline_atp),
                    "het_benefit": float(baseline_het - final_het),
                },
                "classification": classification,
            }
            results.append(record)
            
            print(f"  → {classification['best_archetype']} (score: {classification['best_score']:.3f}), ATP: {final_atp:.3f}, het: {final_het:.3f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "heteroplasmy": float(het),
                "error": str(e)
            })
    
    return results

def analyze_sweep(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze sweep results for archetype transitions."""
    successful = [r for r in results if "classification" in r]
    if not successful:
        return {"error": "No successful simulations"}
    
    # Sort by heteroplasmy
    sorted_records = sorted(successful, key=lambda x: x["heteroplasmy"])
    
    # Identify transitions
    transitions = []
    prev_arch = None
    prev_het = None
    for r in sorted_records:
        curr_arch = r["classification"]["best_archetype"]
        curr_het = r["heteroplasmy"]
        if prev_arch is not None and curr_arch != prev_arch:
            transitions.append({
                "heteroplasmy_threshold": curr_het,
                "from_archetype": prev_arch,
                "to_archetype": curr_arch,
                "atp_before": sorted_records[max(0, len(transitions)-1)]["outcomes"]["final_atp"] if len(transitions) > 0 else None,
                "atp_after": r["outcomes"]["final_atp"],
            })
        prev_arch = curr_arch
        prev_het = curr_het
    
    # Compute archetype ranges
    arch_ranges = {}
    for r in sorted_records:
        arch = r["classification"]["best_archetype"]
        if arch not in arch_ranges:
            arch_ranges[arch] = {"min_het": r["heteroplasmy"], "max_het": r["heteroplasmy"]}
        else:
            arch_ranges[arch]["min_het"] = min(arch_ranges[arch]["min_het"], r["heteroplasmy"])
            arch_ranges[arch]["max_het"] = max(arch_ranges[arch]["max_het"], r["heteroplasmy"])
    
    # Compute average outcomes per archetype
    arch_stats = {}
    for arch in arch_ranges.keys():
        arch_records = [r for r in successful if r["classification"]["best_archetype"] == arch]
        avg_het = np.mean([r["heteroplasmy"] for r in arch_records])
        avg_final_atp = np.mean([r["outcomes"]["final_atp"] for r in arch_records])
        avg_atp_benefit = np.mean([r["outcomes"]["atp_benefit"] for r in arch_records])
        arch_stats[arch] = {
            "count": len(arch_records),
            "avg_heteroplasmy": float(avg_het),
            "avg_final_atp": float(avg_final_atp),
            "avg_atp_benefit": float(avg_atp_benefit),
            "het_range": [arch_ranges[arch]["min_het"], arch_ranges[arch]["max_het"]],
        }
    
    return {
        "total_simulations": len(results),
        "successful": len(successful),
        "transitions": transitions,
        "archetype_ranges": arch_ranges,
        "archetype_stats": arch_stats,
        "sorted_results": [{
            "heteroplasmy": r["heteroplasmy"],
            "archetype": r["classification"]["best_archetype"],
            "similarity_score": r["classification"]["best_score"],
            "final_atp": r["outcomes"]["final_atp"],
            "final_het": r["outcomes"]["final_het"],
            "atp_benefit": r["outcomes"]["atp_benefit"],
        } for r in sorted_records],
    }

def save_sweep_results(results: List[Dict[str, Any]], analysis: Dict[str, Any], output_dir: Path):
    """Save sweep results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = output_dir / "heteroplasmy_sweep_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary_path = output_dir / "heteroplasmy_sweep_analysis.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # CSV for easy plotting
    csv_lines = ["heteroplasmy,archetype,similarity_score,final_atp,final_het,atp_benefit"]
    for r in analysis["sorted_results"]:
        csv_lines.append(
            f"{r['heteroplasmy']:.4f},{r['archetype']},{r['similarity_score']:.3f},"
            f"{r['final_atp']:.4f},{r['final_het']:.4f},{r['atp_benefit']:.4f}"
        )
    csv_path = output_dir / "heteroplasmy_sweep.csv"
    csv_path.write_text("\n".join(csv_lines))
    
    print(f"Results saved to {output_dir}/")
    return raw_path, summary_path, csv_path

def plot_sweep(analysis: Dict[str, Any], output_dir: Path):
    """Generate plots for heteroplasmy sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plots.")
        return
    
    sorted_data = analysis["sorted_results"]
    if not sorted_data:
        return
    
    hets = [d["heteroplasmy"] for d in sorted_data]
    archetypes = [d["archetype"] for d in sorted_data]
    scores = [d["similarity_score"] for d in sorted_data]
    atp = [d["final_atp"] for d in sorted_data]
    atp_benefit = [d["atp_benefit"] for d in sorted_data]
    
    # Map archetype to color
    arch_to_color = {
        "conservative": "blue",
        "aggressive": "red",
        "transplant_focused": "green",
        "metabolic_optimizer": "orange",
    }
    colors = [arch_to_color.get(a, "gray") for a in archetypes]
    
    # Plot 1: Archetype similarity scores
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(hets, scores, c=colors, s=80, alpha=0.7, edgecolors='black')
    plt.xlabel("Baseline Heteroplasmy")
    plt.ylabel("Similarity Score to Best Archetype")
    plt.title("Archetype Classification vs Heteroplasmy")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final ATP
    plt.subplot(2, 2, 2)
    plt.plot(hets, atp, 'k-', linewidth=2, alpha=0.7)
    plt.scatter(hets, atp, c=colors, s=60, alpha=0.8, edgecolors='black')
    plt.xlabel("Baseline Heteroplasmy")
    plt.ylabel("Final ATP")
    plt.title("Final ATP vs Heteroplasmy")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: ATP benefit
    plt.subplot(2, 2, 3)
    plt.plot(hets, atp_benefit, 'k-', linewidth=2, alpha=0.7)
    plt.scatter(hets, atp_benefit, c=colors, s=60, alpha=0.8, edgecolors='black')
    plt.xlabel("Baseline Heteroplasmy")
    plt.ylabel("ATP Benefit (vs Baseline)")
    plt.title("Intervention Benefit vs Heteroplasmy")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 4: Archetype regions
    plt.subplot(2, 2, 4)
    # Create horizontal bars for each archetype region
    y_pos = 0
    for arch, color in arch_to_color.items():
        arch_data = [d for d in sorted_data if d["archetype"] == arch]
        if arch_data:
            het_vals = [d["heteroplasmy"] for d in arch_data]
            min_het = min(het_vals)
            max_het = max(het_vals)
            plt.barh(y_pos, max_het - min_het, left=min_het, height=0.8, color=color, alpha=0.7, edgecolor='black')
            plt.text((min_het + max_het)/2, y_pos, arch, ha='center', va='center', fontsize=9)
            y_pos += 1
    plt.xlabel("Heteroplasmy")
    plt.yticks([])
    plt.title("Archetype Regions")
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_path = output_dir / "heteroplasmy_sweep_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to {plot_path}")

def main():
    """Run heteroplasmy sweep."""
    print("=" * 70)
    print("Heteroplasmy Sweep for Archetype Transitions")
    print("=" * 70)
    
    # Define heteroplasmy range
    het_min = 0.05
    het_max = 0.90
    n_points = 15  # Number of points (increase for finer resolution)
    het_range = np.linspace(het_min, het_max, n_points)
    
    print(f"Sweeping heteroplasmy from {het_min} to {het_max} ({n_points} points)")
    print(f"Fixed intervention: {FIXED_INTERVENTION}")
    print(f"Base patient: {BASE_PATIENT}")
    
    # Run sweep
    results = sweep_heteroplasmy(het_range, FIXED_INTERVENTION, BASE_PATIENT)
    
    # Analyze
    analysis = analyze_sweep(results)
    
    # Save results
    output_dir = ROOT / "output" / "heteroplasmy_sweep"
    raw_path, summary_path, csv_path = save_sweep_results(results, analysis, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"Total simulations: {analysis['total_simulations']}")
    print(f"Successful: {analysis['successful']}")
    
    if analysis['transitions']:
        print(f"\nArchetype transitions (heteroplasmy increasing):")
        for t in analysis['transitions']:
            print(f"  at het={t['heteroplasmy_threshold']:.3f}: {t['from_archetype']} → {t['to_archetype']}")
    else:
        print("\nNo archetype transitions detected (single archetype across range).")
    
    print(f"\nArchetype ranges:")
    for arch, ranges in analysis['archetype_ranges'].items():
        print(f"  {arch}: het [{ranges['min_het']:.3f}, {ranges['max_het']:.3f}]")
    
    # Generate plots
    plot_sweep(analysis, output_dir)
    
    print(f"\nDetailed results saved to:")
    print(f"  Raw: {raw_path}")
    print(f"  Analysis: {summary_path}")
    print(f"  CSV: {csv_path}")
    
    print("\n" + "=" * 70)
    print("Sweep complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()