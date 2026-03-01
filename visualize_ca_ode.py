#!/usr/bin/env python3
"""
Visualizations for CA-ODE bridge validation.

1. Side-by-side trajectories for representative runs.
2. Bin distribution histograms (ODE values per bin).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, '.')

from simulator import simulate
from ca_simulator import run_single_cell
from ca_schema import discretize_state, continuous_exemplar, CA_VAR_ORDER
from constants import PATIENT_NAMES, INTERVENTION_NAMES

# Load updated schema (clamped centers) if available
def load_updated_schema():
    """Load updated BIN_SCHEMA from patch."""
    patch_path = Path("artifacts/ca_ode_validation/schema_patch.py")
    if patch_path.exists():
        namespace = {}
        exec(open(patch_path).read(), namespace)
        return namespace['BIN_SCHEMA']
    # fallback to original
    from ca_schema import BIN_SCHEMA as orig
    return orig

UPDATED_SCHEMA = load_updated_schema()

# Monkey-patch ca_schema module to use updated centers for exemplar mapping
import ca_schema
original_schema = ca_schema.BIN_SCHEMA
ca_schema.BIN_SCHEMA = UPDATED_SCHEMA

def continuous_exemplar_updated(discrete_state):
    """Exemplar using updated centers."""
    return continuous_exemplar(discrete_state)

def plot_side_by_side_trajectories(patient_dict, intervention_dict, title_suffix=""):
    """Plot ODE vs CA reconstructed continuous trajectories (8 variables)."""
    # Run simulations
    ode_result = simulate(patient=patient_dict, intervention=intervention_dict)
    ca_result = run_single_cell(patient=patient_dict, intervention=intervention_dict)
    
    # Times
    t_ode = np.arange(0, 30.0 + 0.01, 0.01)
    t_ca = np.arange(0, 30.0 + 0.25, 0.25)
    
    # CA continuous reconstruction (using updated centers)
    ca_cont = []
    for state in ca_result["trajectory"]:
        ca_cont.append(continuous_exemplar_updated(state))
    ca_cont = np.array(ca_cont)
    
    # Plot
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, var in enumerate(CA_VAR_ORDER):
        ax = axes[idx]
        ax.plot(t_ode, ode_result["states"][:, idx], 'b-', label='ODE', linewidth=1.5, alpha=0.7)
        ax.plot(t_ca, ca_cont[:, idx], 'r--', label='CA reconstructed', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Years')
        ax.set_ylabel(var)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Overall title
    patient_desc = f"age={patient_dict['baseline_age']}, het={patient_dict['baseline_heteroplasmy']:.2f}"
    intervention_desc = list(intervention_dict.values())
    plt.suptitle(f"ODE vs CA trajectories {title_suffix}\nPatient: {patient_desc}", fontsize=12)
    plt.tight_layout()
    return fig, ode_result, ca_result

def plot_bin_distribution_histograms(validation_summary_path):
    """Plot histograms of ODE values per bin for each variable."""
    with open(validation_summary_path, 'r') as f:
        data = json.load(f)
    bin_stats = data['global_bin_stats']
    exemplar_comp = data['global_exemplar_comparison']
    
    # We don't have raw values, so we can't plot histograms.
    # Instead, we can plot bar charts of mean ± std per bin.
    # Let's create a bar chart for each variable.
    for var_name, bins in bin_stats.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(bins.keys())
        means = [bins[lb]['mean'] for lb in labels]
        stds = [bins[lb]['std'] for lb in labels]
        counts = [bins[lb]['count'] for lb in labels]
        
        x = np.arange(len(labels))
        width = 0.6
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, label='Mean ± std')
        ax.set_xlabel('Bin label')
        ax.set_ylabel('ODE value')
        ax.set_title(f'{var_name} — ODE distribution per bin')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add old centers and new centers
        # Old centers from exemplar_comp
        old_centers = [exemplar_comp[var_name][lb]['center'] for lb in labels]
        new_centers = [exemplar_comp[var_name][lb]['mean'] for lb in labels]
        # Plot as points
        ax.scatter(x, old_centers, color='red', zorder=5, label='Old center', s=80, marker='s')
        ax.scatter(x, new_centers, color='green', zorder=5, label='Empirical mean', s=80, marker='^')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        yield fig, var_name

def main():
    output_dir = Path("artifacts/ca_ode_validation/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Side-by-side trajectories for a few representative runs
    # Load normal patients
    normal_path = Path("artifacts/sample_patients_100.json")
    with open(normal_path, 'r') as f:
        patients_all = json.load(f)["patients"]
    # Pick two contrasting patients: young healthy and near cliff
    young = None
    near_cliff = None
    for p in patients_all:
        if p['baseline_age'] < 40 and p['baseline_heteroplasmy'] < 0.1:
            young = p
            break
    for p in patients_all:
        if p['baseline_age'] > 65 and p['baseline_heteroplasmy'] > 0.4:
            near_cliff = p
            break
    if young is None:
        young = patients_all[0]
    if near_cliff is None:
        near_cliff = patients_all[-1]
    
    interventions = {
        "no_treatment": {k: 0.0 for k in INTERVENTION_NAMES},
        "aggressive": {
            "rapamycin_dose": 0.5,
            "nad_supplement": 0.9,
            "senolytic_dose": 0.8,
            "yamanaka_intensity": 0.0,
            "transplant_rate": 0.0,
            "exercise_level": 0.5,
        },
    }
    
    for patient, pname in [(young, "young_healthy"), (near_cliff, "near_cliff")]:
        patient_dict = {k: patient.get(k, 0.0) for k in PATIENT_NAMES}
        for int_name, intervention in interventions.items():
            fig, ode_result, ca_result = plot_side_by_side_trajectories(
                patient_dict, intervention, title_suffix=f"({pname}, {int_name})"
            )
            filename = output_dir / f"trajectories_{pname}_{int_name}.png"
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            print(f"  Saved {filename}")
    
    # 2. Bin distribution bar charts
    summary_path = Path("artifacts/ca_ode_validation/validation_summary.json")
    if summary_path.exists():
        for fig, var_name in plot_bin_distribution_histograms(summary_path):
            filename = output_dir / f"bin_distribution_{var_name}.png"
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            print(f"  Saved {filename}")
    else:
        print("  validation_summary.json not found, skipping bin distribution plots.")
    
    # Restore original schema
    ca_schema.BIN_SCHEMA = original_schema
    
    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()