#!/usr/bin/env python3
"""
Archetype transitions across patient continuum.

For a fixed intervention protocol, simulate across a range of patients
(from healthy to near-cliff) and classify the resulting outcomes using
Lakoff archetypes. Map how archetype classification changes as patient
health declines.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
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

def load_edge_patients() -> List[Dict[str, Any]]:
    """Load edge-case patients from sample_patients_edge.json."""
    edge_path = ROOT / "artifacts" / "sample_patients_edge.json"
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge patients file not found: {edge_path}")
    
    with open(edge_path, 'r') as f:
        data = json.load(f)
    
    patients = data.get("patients", [])
    print(f"Loaded {len(patients)} edge patients")
    return patients

def simulate_patient(patient_dict: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate a patient with given intervention, returning analytics."""
    # Extract core patient parameters (6D)
    core_patient = {k: patient_dict[k] for k in [
        "baseline_age", "baseline_heteroplasmy", "baseline_nad_level",
        "genetic_vulnerability", "metabolic_demand", "inflammation_level"
    ] if k in patient_dict}
    
    # Run baseline (no treatment)
    baseline = simulate(patient=core_patient)
    
    # Run intervention
    result = simulate(intervention=intervention, patient=core_patient)
    
    # Compute analytics
    analytics = compute_all(result, baseline)
    
    # Extract key outcomes
    final_atp = result['states'][-1, 2]
    final_het = result['heteroplasmy'][-1]
    baseline_atp = baseline['states'][-1, 2]
    baseline_het = baseline['heteroplasmy'][-1]
    
    return {
        "patient": patient_dict,
        "result": result,
        "baseline": baseline,
        "analytics": analytics,
        "outcomes": {
            "final_atp": float(final_atp),
            "final_het": float(final_het),
            "baseline_atp": float(baseline_atp),
            "baseline_het": float(baseline_het),
            "atp_benefit": float(final_atp - baseline_atp),
            "het_benefit": float(baseline_het - final_het),  # reduction is good
        }
    }

def run_transition_analysis(
    patients: List[Dict[str, Any]],
    intervention: Dict[str, Any],
    max_patients: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run simulation and classification for each patient."""
    if max_patients is not None:
        patients = patients[:max_patients]
    
    library = load_adjusted_archetypes()
    results = []
    
    for i, patient in enumerate(patients):
        label = patient.get("_label", f"patient_{i}")
        category = patient.get("_category", "unknown")
        print(f"Processing {i+1}/{len(patients)}: {label} ({category})")
        
        try:
            # Simulate
            sim_data = simulate_patient(patient, intervention)
            
            # Classify
            classification = classify_analytics(sim_data["analytics"], library)
            
            # Combine results
            record = {
                "patient_index": i,
                "patient_label": label,
                "patient_category": category,
                "patient_params": {
                    "baseline_age": patient["baseline_age"],
                    "baseline_heteroplasmy": patient["baseline_heteroplasmy"],
                    "baseline_nad_level": patient["baseline_nad_level"],
                    "genetic_vulnerability": patient["genetic_vulnerability"],
                    "metabolic_demand": patient["metabolic_demand"],
                    "inflammation_level": patient["inflammation_level"],
                },
                "outcomes": sim_data["outcomes"],
                "classification": classification,
            }
            results.append(record)
            
            print(f"  → Archetype: {classification['best_archetype']} (score: {classification['best_score']:.3f})")
            print(f"  → Final ATP: {sim_data['outcomes']['final_atp']:.3f}, Final het: {sim_data['outcomes']['final_het']:.3f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "patient_index": i,
                "patient_label": label,
                "error": str(e),
            })
    
    return results

def analyze_transitions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze archetype transitions across patient continuum."""
    successful = [r for r in results if "classification" in r]
    if not successful:
        return {"error": "No successful simulations"}
    
    # Group by archetype
    archetype_counts = {}
    for r in successful:
        arch = r["classification"]["best_archetype"]
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
    
    # Sort patients by baseline heteroplasmy (health metric)
    sorted_records = sorted(successful, key=lambda x: x["patient_params"]["baseline_heteroplasmy"])
    
    # Identify transition points where archetype changes
    transitions = []
    prev_arch = None
    for r in sorted_records:
        curr_arch = r["classification"]["best_archetype"]
        if prev_arch is not None and curr_arch != prev_arch:
            het = r["patient_params"]["baseline_heteroplasmy"]
            transitions.append({
                "heteroplasmy_threshold": het,
                "from_archetype": prev_arch,
                "to_archetype": curr_arch,
                "patient_label": r["patient_label"],
            })
        prev_arch = curr_arch
    
    # Compute average outcomes per archetype
    arch_stats = {}
    for arch in archetype_counts.keys():
        arch_records = [r for r in successful if r["classification"]["best_archetype"] == arch]
        avg_het = np.mean([r["patient_params"]["baseline_heteroplasmy"] for r in arch_records])
        avg_final_atp = np.mean([r["outcomes"]["final_atp"] for r in arch_records])
        avg_atp_benefit = np.mean([r["outcomes"]["atp_benefit"] for r in arch_records])
        arch_stats[arch] = {
            "count": len(arch_records),
            "avg_baseline_het": float(avg_het),
            "avg_final_atp": float(avg_final_atp),
            "avg_atp_benefit": float(avg_atp_benefit),
        }
    
    return {
        "total_patients": len(results),
        "successful": len(successful),
        "archetype_distribution": archetype_counts,
        "archetype_stats": arch_stats,
        "transitions": transitions,
        "sorted_by_het": [{
            "patient_label": r["patient_label"],
            "baseline_heteroplasmy": r["patient_params"]["baseline_heteroplasmy"],
            "archetype": r["classification"]["best_archetype"],
            "similarity_score": r["classification"]["best_score"],
            "final_atp": r["outcomes"]["final_atp"],
            "final_het": r["outcomes"]["final_het"],
        } for r in sorted_records],
    }

def save_results(results: List[Dict[str, Any]], analysis: Dict[str, Any], output_dir: Path):
    """Save transition results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Raw results
    raw_path = output_dir / "archetype_transitions_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis summary
    summary_path = output_dir / "archetype_transitions_analysis.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # CSV-like summary for easy plotting
    csv_lines = ["patient_label,baseline_het,archetype,similarity_score,final_atp,final_het"]
    for r in analysis["sorted_by_het"]:
        csv_lines.append(
            f"{r['patient_label']},{r['baseline_heteroplasmy']:.4f},"
            f"{r['archetype']},{r['similarity_score']:.3f},"
            f"{r['final_atp']:.4f},{r['final_het']:.4f}"
        )
    csv_path = output_dir / "archetype_transitions.csv"
    csv_path.write_text("\n".join(csv_lines))
    
    print(f"Results saved to {output_dir}/")
    return raw_path, summary_path, csv_path

def main():
    """Run archetype transition analysis."""
    print("=" * 70)
    print("Archetype Transitions Across Patient Continuum")
    print("=" * 70)
    
    # Load patients
    patients = load_edge_patients()
    
    # Limit for quick test (set to None for full run)
    max_patients = None  # Set to None to process all 82 edge patients
    if max_patients:
        print(f"Limiting to first {max_patients} patients (for speed)")
    else:
        print("Processing all edge patients (full run)")
    
    # Run simulations and classification
    results = run_transition_analysis(patients, FIXED_INTERVENTION, max_patients=max_patients)
    
    # Analyze transitions
    analysis = analyze_transitions(results)
    
    # Save results
    output_dir = ROOT / "output" / "archetype_transitions"
    raw_path, summary_path, csv_path = save_results(results, analysis, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total patients processed: {analysis['total_patients']}")
    print(f"Successful simulations: {analysis['successful']}")
    print(f"\nArchetype distribution:")
    for arch, count in analysis['archetype_distribution'].items():
        stats = analysis['archetype_stats'].get(arch, {})
        avg_het = stats.get('avg_baseline_het', 0)
        print(f"  {arch}: {count} patients (avg baseline het: {avg_het:.3f})")
    
    if analysis['transitions']:
        print(f"\nArchetype transitions (as baseline heteroplasmy increases):")
        for t in analysis['transitions']:
            print(f"  at het={t['heteroplasmy_threshold']:.3f}: {t['from_archetype']} → {t['to_archetype']} ({t['patient_label']})")
    else:
        print("\nNo archetype transitions detected (all patients classified as same archetype).")
    
    print(f"\nDetailed results saved to:")
    print(f"  Raw: {raw_path}")
    print(f"  Analysis: {summary_path}")
    print(f"  CSV: {csv_path}")
    
    # Generate simple plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        generate_plot(analysis, output_dir)
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation.")
    
    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)

def generate_plot(analysis: Dict[str, Any], output_dir: Path):
    """Generate a simple scatter plot of archetype vs baseline heteroplasmy."""
    import matplotlib.pyplot as plt
    sorted_data = analysis["sorted_by_het"]
    if not sorted_data:
        return
    
    hets = [d["baseline_heteroplasmy"] for d in sorted_data]
    archetypes = [d["archetype"] for d in sorted_data]
    scores = [d["similarity_score"] for d in sorted_data]
    
    # Map archetype to color
    arch_to_color = {
        "conservative": "blue",
        "aggressive": "red",
        "transplant_focused": "green",
        "metabolic_optimizer": "orange",
    }
    colors = [arch_to_color.get(a, "gray") for a in archetypes]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(hets, scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add labels
    for i, d in enumerate(sorted_data):
        plt.annotate(d["archetype"][:3], (hets[i], scores[i]), 
                     xytext=(5,5), textcoords='offset points', fontsize=8)
    
    plt.xlabel("Baseline Heteroplasmy (health metric)")
    plt.ylabel("Similarity Score to Best Archetype")
    plt.title("Archetype Classification vs Patient Health")
    plt.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=arch) for arch, color in arch_to_color.items()]
    plt.legend(handles=legend_elements, title="Archetype")
    
    plot_path = output_dir / "archetype_transitions_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()