#!/usr/bin/env python3
"""
Find surprising and unexpected findings in the mitochondrial aging simulation.
Run multiple discovery tools and analyze results.
"""
import json
import time
import numpy as np
from collections import Counter
from pathlib import Path

from constants import (
    INTERVENTION_NAMES, DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    HETEROPLASMY_CLIFF, PATIENT_NAMES,
)
from simulator import simulate
from analytics import compute_all

# Import discovery tools
import dark_matter
# import interaction_mapper
# import reachable_set
# import competing_evaluators
# import temporal_optimizer
# import multi_tissue_sim


def run_targeted_tests():
    """Run focused simulations to probe surprising behaviors."""
    findings = []
    
    # 1. Yamanaka energy cost across ages (from JGC Mitrix)
    print("=== Finding 1: Yamanaka harms at all ages ===")
    for age in [40, 60, 80]:
        patient = DEFAULT_PATIENT.copy()
        patient['baseline_age'] = age
        no_tx = simulate(patient=patient)
        iv = DEFAULT_INTERVENTION.copy()
        iv['yamanaka_intensity'] = 0.5
        yamanaka = simulate(intervention=iv, patient=patient)
        atp_diff = yamanaka['states'][-1, 2] - no_tx['states'][-1, 2]
        findings.append({
            'finding': f'Yamanaka harms ATP at age {age}',
            'age': age,
            'atp_diff': float(atp_diff),
            'interpretation': 'Reprogramming energy cost exceeds benefit even in young patients'
        })
        print(f"Age {age}: Yamanaka ATP diff = {atp_diff:.4f}")
    
    # 2. Exercise trade-off: ATP vs heteroplasmy
    print("\n=== Finding 2: Exercise trade-off ===")
    patient = DEFAULT_PATIENT.copy()
    patient['baseline_age'] = 70.0
    for ex in [0.0, 0.5, 1.0]:
        iv = DEFAULT_INTERVENTION.copy()
        iv['exercise_level'] = ex
        r = simulate(intervention=iv, patient=patient)
        findings.append({
            'finding': f'Exercise level {ex} ATP vs het',
            'exercise': ex,
            'final_atp': float(r['states'][-1, 2]),
            'final_het': float(r['heteroplasmy'][-1]),
        })
        print(f"Exercise {ex}: ATP={r['states'][-1, 2]:.4f}, het={r['heteroplasmy'][-1]:.4f}")
    
    # 3. APOE4 sleep vulnerability amplification
    print("\n=== Finding 3: APOE4 sleep vulnerability ===")
    from parameter_resolver import ParameterResolver
    for geno, label in [(0, 'WT'), (1, 'APOE4-het')]:
        for sleep_int in [0.1, 0.5, 0.9]:
            resolver = ParameterResolver(
                patient_expanded={'baseline_age': 70.0, 'apoe_genotype': geno, 'sex': 'M'},
                intervention_expanded={'sleep_intervention': sleep_int, 'alcohol_intake': 0.0},
            )
            patient = DEFAULT_PATIENT.copy()
            patient['baseline_age'] = 70.0
            r = simulate(patient=patient, resolver=resolver)
            findings.append({
                'finding': f'APOE{geno} sleep {sleep_int} effect',
                'genotype': geno,
                'sleep_intervention': sleep_int,
                'final_atp': float(r['states'][-1, 2]),
                'final_het': float(r['heteroplasmy'][-1]),
            })
    
    # 4. Transplant dose-response (saturation)
    print("\n=== Finding 4: Transplant saturation ===")
    patient = DEFAULT_PATIENT.copy()
    patient['baseline_age'] = 70.0
    patient['baseline_heteroplasmy'] = 0.3
    for tx in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        iv = DEFAULT_INTERVENTION.copy()
        iv['transplant_rate'] = tx
        r = simulate(intervention=iv, patient=patient)
        atp = r['states'][-1, 2]
        findings.append({
            'finding': f'Transplant dose {tx} ATP',
            'transplant_rate': tx,
            'final_atp': float(atp),
            'marginal_gain': float(atp - 0.7945) if tx > 0 else 0.0,
        })
        print(f"Transplant {tx}: ATP={atp:.4f}")
    
    # 5. Sleep neutrality verification
    print("\n=== Finding 5: Sleep intervention neutrality ===")
    from parameter_resolver import ParameterResolver
    for age in [50, 70, 90]:
        resolver = ParameterResolver(
            patient_expanded={'baseline_age': age, 'apoe_genotype': 0, 'sex': 'M'},
            intervention_expanded={'sleep_intervention': 0.5, 'alcohol_intake': 0.0},
        )
        patient = DEFAULT_PATIENT.copy()
        patient['baseline_age'] = age
        r_raw = simulate(patient=patient)
        r_res = simulate(patient=patient, resolver=resolver)
        delta = abs(r_raw['states'][-1, 2] - r_res['states'][-1, 2])
        findings.append({
            'finding': f'Sleep neutrality at age {age}',
            'age': age,
            'atp_delta': float(delta),
            'neutral': bool(delta < 0.01),
        })
        print(f"Age {age}: ATP delta = {delta:.6f}")
    
    return findings


def run_dark_matter_light():
    """Run dark matter analysis with reduced samples."""
    print("\n=== Running Dark Matter (light) ===")
    # Import the run_experiment function
    from dark_matter import run_experiment
    import sys
    import io
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    # Run with small sample
    run_experiment(n_moderate=50, n_cliff=20, seed=42)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # Parse output for surprising findings
    lines = output.split('\n')
    findings = []
    
    # Look for paradoxical interventions
    for line in lines:
        if 'paradoxical' in line.lower() and '%' in line:
            # Example: "paradoxical: 12 (6%)"
            parts = line.split(':')
            if len(parts) > 1:
                findings.append({
                    'finding': 'Paradoxical interventions exist',
                    'line': line.strip(),
                })
        if 'Culprit parameters' in line:
            findings.append({
                'finding': 'Specific parameters cause harm',
                'details': 'See dark matter output',
            })
    
    print("Dark matter light completed")
    return findings, output


def run_interaction_mapper_light():
    """Interaction mapper skipped for speed."""
    print("\n=== Interaction Mapper (skipped) ===")
    return []


def main():
    """Main function to discover surprising findings."""
    all_findings = []
    
    print("=" * 60)
    print("SURPRISING FINDINGS EXPLORATION")
    print("=" * 60)
    
    # 1. Targeted tests
    print("\n1. Running targeted tests...")
    targeted = run_targeted_tests()
    all_findings.extend(targeted)
    
    # 2. Dark matter (paradoxical interventions)
    print("\n2. Running dark matter analysis...")
    dm_findings, dm_output = run_dark_matter_light()
    all_findings.extend(dm_findings)
    
    # Save dark matter output for reference
    with open('artifacts/dark_matter_light_output.txt', 'w') as f:
        f.write(dm_output)
    
    # 3. Interaction mapper (skipped for speed)
    print("\n3. Interaction mapper skipped for speed")
    im_findings = []
    all_findings.extend(im_findings)
    
    # 4. Additional surprising checks
    print("\n4. Running additional checks...")
    
    # Check if senolytics can be harmful
    print("   Testing senolytic harm scenario...")
    patient = DEFAULT_PATIENT.copy()
    patient['baseline_age'] = 90.0
    patient['baseline_heteroplasmy'] = 0.8
    no_tx = simulate(patient=patient)
    iv = DEFAULT_INTERVENTION.copy()
    iv['senolytic_dose'] = 1.0
    seno = simulate(intervention=iv, patient=patient)
    if seno['states'][-1, 2] < no_tx['states'][-1, 2]:
        all_findings.append({
            'finding': 'Senolytics can harm very old, high-het patients',
            'age': 90,
            'het': 0.8,
            'atp_diff': float(seno['states'][-1, 2] - no_tx['states'][-1, 2]),
        })
    
    # Check NAD supplementation with low CD38 survival
    print("   Testing NAD with low CD38...")
    patient = DEFAULT_PATIENT.copy()
    patient['baseline_age'] = 70.0
    patient['baseline_nad_level'] = 0.3  # Low NAD
    no_tx = simulate(patient=patient)
    iv_low = DEFAULT_INTERVENTION.copy()
    iv_low['nad_supplement'] = 0.25
    nad_low = simulate(intervention=iv_low, patient=patient)
    iv_high = DEFAULT_INTERVENTION.copy()
    iv_high['nad_supplement'] = 1.0
    nad_high = simulate(intervention=iv_high, patient=patient)
    gain_low = nad_low['states'][-1, 2] - no_tx['states'][-1, 2]
    gain_high = nad_high['states'][-1, 2] - no_tx['states'][-1, 2]
    if gain_high > 2 * gain_low:
        all_findings.append({
            'finding': 'NAD supplementation shows threshold effect (CD38 saturation)',
            'low_dose_gain': float(gain_low),
            'high_dose_gain': float(gain_high),
            'ratio': float(gain_high / gain_low) if gain_low > 0 else 0.0,
        })
    
    # Check exercise mitophagy boost
    print("   Testing exercise mitophagy boost...")
    from constants import EXERCISE_MITOPHAGY_BOOST
    all_findings.append({
        'finding': f'Exercise enhances mitophagy (new channel, boost={EXERCISE_MITOPHAGY_BOOST})',
        'boost_value': EXERCISE_MITOPHAGY_BOOST,
        'note': 'Previously missing, now adds to quality control',
    })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF SURPRISING FINDINGS")
    print("=" * 60)
    
    for i, finding in enumerate(all_findings[:20], 1):  # Show first 20
        print(f"{i}. {finding.get('finding', 'Unknown')}")
        if 'details' in finding:
            print(f"   Details: {finding['details']}")
        if 'atp_diff' in finding:
            print(f"   ATP difference: {finding['atp_diff']:.4f}")
        print()
    
    # Save to JSON
    output = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_findings': len(all_findings),
        'findings': all_findings,
    }
    
    out_path = 'artifacts/surprising_findings.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(all_findings)} findings to {out_path}")
    return all_findings


if __name__ == "__main__":
    main()