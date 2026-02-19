#!/usr/bin/env python3
"""Run the 4-scenario APOE4 patient comparison and output results.

Usage:
    python run_scenario_comparison.py [--years 30] [--save-plots]
"""
import sys
import os

# Ensure output directory exists
os.makedirs('output/scenarios', exist_ok=True)

from scenario_definitions import get_example_scenarios
from scenario_runner import run_scenarios
from scenario_analysis import extract_milestones, compare_scenarios, summary_table
from scenario_plot import plot_trajectories, plot_milestone_comparison

def main():
    years = 30
    save = '--save-plots' in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith('--years'):
            years = int(sys.argv[sys.argv.index(arg) + 1])

    print(f"Running 4-scenario comparison for {years} years...")
    scenarios = get_example_scenarios()
    results = run_scenarios(scenarios, years=years)

    # Milestones
    print("\n=== Milestone Comparison ===")
    milestones = compare_scenarios(results)
    for m in milestones:
        print(f"\n{m['scenario_name']}:")
        print(f"  Final het: {m['final_heteroplasmy']:.4f}")
        print(f"  Final ATP: {m['final_atp']:.4f}")
        print(f"  Final memory: {m['final_memory_index']:.4f}")
        print(f"  Final amyloid: {m['final_amyloid']:.4f}")
        print(f"  Het < 50%: {m['het_below_50_age'] or 'Never'}")
        print(f"  ATP > 0.8: {m['atp_above_08_age'] or 'Never'}")
        print(f"  Dementia (MI < 0.5): {m['dementia_age'] or 'Never'}")

    # Summary table
    print("\n=== Summary at Key Ages ===")
    summary = summary_table(results)
    if summary:
        for entry in summary:
            print(f"\n{entry['scenario_name']}:")
            for age, metrics in sorted(entry['metrics'].items()):
                if metrics is not None:
                    print(f"  Age {int(age)}: het={metrics['heteroplasmy']:.4f}, "
                          f"atp={metrics['atp']:.4f}, "
                          f"memory={metrics['memory_index']:.4f}")
                else:
                    print(f"  Age {int(age)}: (out of range)")

    # Plots
    if save:
        fig = plot_trajectories(results, save=True)
        print("\nTrajectory plot saved to output/scenarios/")
        fig2 = plot_milestone_comparison(milestones, save=True)
        print("Milestone comparison saved to output/scenarios/")
    else:
        print("\n(Use --save-plots to generate PNG plots)")

    print("\nDone.")

if __name__ == '__main__':
    main()
