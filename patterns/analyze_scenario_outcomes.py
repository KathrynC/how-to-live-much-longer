#!/usr/bin/env python3
"""Analyze scenario outcomes to understand archetype classification."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scenario_definitions import get_example_scenarios
from scenario_runner import run_scenario
from analytics import compute_all

def main():
    scenarios = get_example_scenarios()
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"{'='*60}")
        
        result = run_scenario(scenario, years=30, include_annotations=False)
        core = result['core']
        analytics = compute_all(core, None)
        
        # Extract key metrics
        energy = analytics.get('energy', {})
        damage = analytics.get('damage', {})
        dynamics = analytics.get('dynamics', {})
        intervention = analytics.get('intervention', {})
        
        print(f"Energy:")
        print(f"  ATP initial: {energy.get('atp_initial', 0):.3f}")
        print(f"  ATP final: {energy.get('atp_final', 0):.3f}")
        print(f"  ATP delta: {energy.get('atp_final', 0) - energy.get('atp_initial', 0):.3f}")
        print(f"  ATP mean: {energy.get('atp_mean', 0):.3f}")
        print(f"  Time to crisis: {energy.get('time_to_crisis_years', 'N/A')}")
        
        print(f"Damage:")
        print(f"  Het initial: {damage.get('het_initial', 0):.3f}")
        print(f"  Het final: {damage.get('het_final', 0):.3f}")
        print(f"  Het delta: {damage.get('delta_het', 0):.3f}")
        print(f"  Deletion het final: {damage.get('deletion_het_final', 0):.3f}")
        print(f"  Time to cliff: {damage.get('time_to_cliff_years', 'N/A')}")
        
        print(f"Dynamics:")
        print(f"  ROS amplitude: {dynamics.get('ros_amplitude', 0):.3f}")
        print(f"  Senescent final: {dynamics.get('senescent_final', 0):.3f}")
        
        print(f"Intervention:")
        print(f"  ATP benefit terminal: {intervention.get('atp_benefit_terminal', 0):.3f}")
        print(f"  Het benefit terminal: {intervention.get('het_benefit_terminal', 0):.3f}")
        print(f"  Crisis delay: {intervention.get('crisis_delay_years', 0):.1f} years")
        
        # Check which archetype criteria these might match
        print(f"\nArchetype indicators:")
        # Conservative: ATP stability, minimal het change, low ROS oscillations, low senescence
        # Aggressive: Significant het reduction, ATP improvement >0.75, hormetic ROS response
        # Transplant focused: Deletion het reduction below 0.5, ATP >0.8
        # Metabolic optimizer: NAD maintenance, stable ATP, moderate ROS amplitude
        
        atp_final = energy.get('atp_final', 0)
        het_final = damage.get('het_final', 0)
        het_delta = damage.get('delta_het', 0)
        ros_amp = dynamics.get('ros_amplitude', 0)
        sen_final = dynamics.get('senescent_final', 0)
        
        if het_delta > -0.1 and atp_final > 0.7 and ros_amp < 0.3 and sen_final < 0.3:
            print("  → Conservative pattern")
        if het_delta < -0.15 and atp_final > 0.75:
            print("  → Aggressive pattern")
        if damage.get('deletion_het_final', 1) < 0.5 and atp_final > 0.8:
            print("  → Transplant focused pattern")
        if ros_amp > 0.0 and ros_amp < 0.2 and het_delta > -0.05:
            print("  → Metabolic optimizer pattern")

if __name__ == "__main__":
    main()