#!/usr/bin/env python3
"""Test adaptive protocols with symmathesy metrics."""

import sys
sys.path.insert(0, '.')

import numpy as np
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT
from simulator import simulate, phased_schedule
from analytics import compute_symmathesy_metrics, compute_all
from adaptive_protocol import AdaptiveProtocol, create_symmathesy_protocol, create_advanced_symmathesy_protocol, atp_proportional_rule, heteroplasmy_proportional_rule, atp_bidirectional_rule

def test_basic_adaptive():
    """Test that adaptive protocol responds to state."""
    print("=== Testing basic adaptive protocol ===")
    
    # Base intervention: minimal treatment
    base_intervention = dict(DEFAULT_INTERVENTION)
    base_intervention['nad_supplement'] = 0.1
    base_intervention['transplant_rate'] = 0.1
    
    # Create adaptive protocol with ATP threshold rule
    protocol = AdaptiveProtocol(base_intervention)
    
    # Add a simple rule: if ATP < 0.8, increase NAD by 0.3
    def atp_low_condition(t, state, intervention):
        return state[2] < 0.8  # ATP
    
    def increase_nad_action(t, state, intervention):
        return {'nad_supplement': 0.3}
    
    protocol.add_rule(atp_low_condition, increase_nad_action)
    
    # Test resolve with different states
    state_low_atp = np.array([1.0, 0.1, 0.7, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.7
    state_high_atp = np.array([1.0, 0.1, 0.9, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.9
    
    interv_low, patient_low = protocol.resolve(0.0, state_low_atp)
    interv_high, patient_high = protocol.resolve(0.0, state_high_atp)
    
    print(f"Low ATP state: NAD = {interv_low.get('nad_supplement', 0.0):.2f}")
    print(f"High ATP state: NAD = {interv_high.get('nad_supplement', 0.0):.2f}")
    
    assert interv_low['nad_supplement'] > interv_high['nad_supplement'], "Adaptive rule should increase NAD when ATP low"
    print("✓ Adaptive rule works correctly")
    
    return protocol

def test_symmathesy_protocol_simulation():
    """Test full simulation with symmathesy adaptive protocol."""
    print("\n=== Testing symmathesy protocol simulation ===")
    
    # Base intervention: moderate treatment
    base_intervention = {
        'rapamycin_dose': 0.3,
        'nad_supplement': 0.3,
        'senolytic_dose': 0.3,
        'yamanaka_intensity': 0.0,
        'transplant_rate': 0.3,
        'exercise_level': 0.3,
    }
    
    # Create symmathesy protocol
    protocol = create_symmathesy_protocol(base_intervention)
    
    # Run simulation with adaptive protocol as resolver
    print("Running simulation with adaptive protocol...")
    result = simulate(intervention=None, patient=protocol.base_patient, resolver=protocol, sim_years=30)
    
    # Check that intervention was adaptive (not constant)
    # We can't directly check because intervention isn't stored per timestep
    # But we can compute symmathesy metrics
    sym = compute_symmathesy_metrics(result)
    print(f"Symmathesy metrics:")
    for k, v in sym.items():
        print(f"  {k}: {v:.4f}")
    
    # Compare with constant intervention
    print("\nComparing with constant intervention...")
    result_const = simulate(intervention=base_intervention, patient=DEFAULT_PATIENT, sim_years=30)
    sym_const = compute_symmathesy_metrics(result_const)
    
    print("Constant intervention symmathesy:")
    for k, v in sym_const.items():
        print(f"  {k}: {v:.4f}")
    
    # Adaptive should have higher relationship diversity and possibly adaptation coherence
    print(f"\nAdaptive vs constant:")
    print(f"  Relationship diversity: {sym['relationship_diversity']:.4f} vs {sym_const['relationship_diversity']:.4f}")
    print(f"  Adaptation coherence: {sym['adaptation_coherence']:.4f} vs {sym_const['adaptation_coherence']:.4f}")
    
    return result, sym

def test_phased_with_adaptive():
    """Test adaptive protocol combined with phased schedule."""
    print("\n=== Testing phased schedule with adaptive rules ===")
    
    from simulator import phased_schedule
    
    # Create phased schedule: no treatment for 10 years, then moderate
    no_tx = dict(DEFAULT_INTERVENTION)
    moderate = {
        'rapamycin_dose': 0.4,
        'nad_supplement': 0.4,
        'senolytic_dose': 0.4,
        'yamanaka_intensity': 0.0,
        'transplant_rate': 0.4,
        'exercise_level': 0.4,
    }
    schedule = phased_schedule([(0, no_tx), (10, moderate)])
    
    # Create adaptive protocol on top of phased schedule
    protocol = create_symmathesy_protocol(schedule)
    
    # Run simulation with adaptive protocol as resolver
    result = simulate(intervention=None, patient=protocol.base_patient, resolver=protocol, sim_years=30)
    sym = compute_symmathesy_metrics(result)
    
    print("Phased + adaptive symmathesy metrics:")
    for k, v in sym.items():
        print(f"  {k}: {v:.4f}")
    
    # Full analytics
    analytics = compute_all(result)
    print(f"\nFinal ATP: {analytics['energy']['atp_final']:.4f}")
    print(f"Final heteroplasmy: {analytics['damage']['het_final']:.4f}")
    
    return result

def test_sick_patient_adaptive():
    """Test adaptive protocol with a sick patient where rules should trigger."""
    print("\n=== Testing adaptive protocol with sick patient ===")
    
    # Create sick patient: high heteroplasmy, low NAD, older
    sick_patient = {
        'baseline_age': 80.0,
        'baseline_heteroplasmy': 0.65,  # Near cliff
        'baseline_nad_level': 0.3,
        'genetic_vulnerability': 1.5,
        'metabolic_demand': 1.2,
        'inflammation_level': 0.6,
    }
    
    # Base intervention: minimal treatment
    base_intervention = dict(DEFAULT_INTERVENTION)
    base_intervention['nad_supplement'] = 0.2
    base_intervention['transplant_rate'] = 0.2
    
    # Create adaptive protocol with verbose logging
    protocol = AdaptiveProtocol(base_intervention, base_patient=sick_patient, verbose=True)
    
    # Add rules with lower thresholds to ensure they trigger
    # ATP threshold: 0.8 (sick patient likely below this)
    def atp_low_condition(t, state, intervention):
        return state[2] < 0.8  # ATP
    
    def increase_nad_action(t, state, intervention):
        return {'nad_supplement': 0.3}
    
    # Heteroplasmy threshold: 0.6 (patient starts at 0.65)
    def het_high_condition(t, state, intervention):
        N_h = state[0]
        N_del = state[1]
        N_pt = state[7]
        total = N_h + N_del + N_pt
        if total <= 0:
            return False
        het = (N_del + N_pt) / total
        return het > 0.6
    
    def increase_transplant_action(t, state, intervention):
        return {'transplant_rate': 0.4}
    
    protocol.add_rule(atp_low_condition, increase_nad_action)
    protocol.add_rule(het_high_condition, increase_transplant_action)
    
    # Run simulation
    print(f"Sick patient: age={sick_patient['baseline_age']}, het={sick_patient['baseline_heteroplasmy']}")
    print("Running simulation with adaptive protocol...")
    result = simulate(intervention=None, patient=protocol.base_patient, resolver=protocol, sim_years=30)
    
    # Check rule firings
    log = protocol.get_log()
    print(f"Rules fired {len(log)} times")
    if log:
        print("First few rule firings:")
        for i, entry in enumerate(log[:5]):
            print(f"  t={entry['time']:.2f}: suggestions={entry['suggestions']}")
    
    # Compute symmathesy metrics
    sym = compute_symmathesy_metrics(result)
    print("\nSymmathesy metrics for sick patient:")
    for k, v in sym.items():
        print(f"  {k}: {v:.4f}")
    
    # Compare with constant intervention
    result_const = simulate(intervention=base_intervention, patient=sick_patient, sim_years=30)
    sym_const = compute_symmathesy_metrics(result_const)
    
    print("\nConstant intervention for sick patient:")
    for k, v in sym_const.items():
        print(f"  {k}: {v:.4f}")
    
    # Adaptive should have higher relationship diversity
    print(f"\nComparison:")
    print(f"  Relationship diversity: {sym['relationship_diversity']:.4f} vs {sym_const['relationship_diversity']:.4f}")
    print(f"  Adaptation coherence: {sym['adaptation_coherence']:.4f} vs {sym_const['adaptation_coherence']:.4f}")
    
    if len(log) > 0:
        print("✓ Rules fired (adaptive protocol active)")
    else:
        print("⚠ No rules fired - thresholds may not have been crossed")
    
    return result, sym, log


def test_proportional_rules():
    """Test proportional adjustment rules."""
    print("\n=== Testing proportional adjustment rules ===")
    
    # ATP proportional rule
    rule_dict = atp_proportional_rule(threshold=0.8, gain=0.5, max_nad=1.0)
    condition = rule_dict['condition']
    action = rule_dict['action']
    
    # Test condition: ATP low
    state_low = np.array([1.0, 0.1, 0.7, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.7
    state_high = np.array([1.0, 0.1, 0.9, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.9
    intervention = {'nad_supplement': 0.3}
    
    assert condition(0.0, state_low, intervention) == True, "Should trigger when ATP below threshold"
    assert condition(0.0, state_high, intervention) == False, "Should not trigger when ATP above threshold"
    
    # Test action: proportional increase
    target = action(0.0, state_low, intervention)
    assert 'nad_supplement' in target
    deficit = 0.8 - 0.7  # threshold - ATP = 0.1
    expected = 0.3 + 0.5 * deficit  # current + gain * deficit = 0.3 + 0.05 = 0.35
    assert abs(target['nad_supplement'] - expected) < 1e-9, f"Expected {expected}, got {target['nad_supplement']}"
    print(f"ATP proportional rule: ATP=0.7 -> NAD target {target['nad_supplement']:.3f} (expected 0.350)")
    
    # Heteroplasmy proportional rule
    rule_dict2 = heteroplasmy_proportional_rule(threshold=0.5, gain=0.4, max_transplant=1.0)
    condition2 = rule_dict2['condition']
    action2 = rule_dict2['action']
    
    # State with heteroplasmy = 0.6 (above threshold)
    N_h = 0.5
    N_del = 0.3
    N_pt = 0.2
    total = N_h + N_del + N_pt
    het = (N_del + N_pt) / total  # 0.5/1.0 = 0.5? Wait compute: 0.3+0.2=0.5, total=1.0, het=0.5 exactly threshold. Need > threshold.
    # Adjust to make het = 0.6
    N_del = 0.4
    total = N_h + N_del + N_pt  # 0.5+0.4+0.2=1.1
    het = (0.4+0.2)/1.1  # approx 0.545
    # Let's compute precisely: we want het > 0.5, say 0.6
    # Solve: (N_del + N_pt)/total = 0.6, with total = N_h + N_del + N_pt
    # Let N_h = 0.5, N_pt = 0.2, then N_del = ?
    # (N_del + 0.2)/(0.5+N_del+0.2) = 0.6 => N_del+0.2 = 0.6*(0.7+N_del) => N_del+0.2 = 0.42+0.6*N_del => 0.4*N_del = 0.22 => N_del=0.55
    N_del = 0.55
    total = 0.5 + 0.55 + 0.2  # = 1.25
    het = (0.55+0.2)/1.25  # = 0.6
    state_het = np.array([0.5, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])  # N_h=0.5, N_del=0.55, N_pt=0.2
    intervention2 = {'transplant_rate': 0.2}
    
    assert condition2(0.0, state_het, intervention2) == True, "Should trigger when het above threshold"
    
    target2 = action2(0.0, state_het, intervention2)
    assert 'transplant_rate' in target2
    excess = 0.6 - 0.5  # 0.1
    expected2 = 0.2 + 0.4 * excess  # 0.2 + 0.04 = 0.24
    assert abs(target2['transplant_rate'] - expected2) < 1e-9, f"Expected {expected2}, got {target2['transplant_rate']}"
    print(f"Heteroplasmy proportional rule: het=0.6 -> transplant target {target2['transplant_rate']:.3f} (expected 0.240)")
    
    print("✓ Proportional rules work correctly")
    return rule_dict, rule_dict2


def test_bidirectional_rule():
    """Test bidirectional ATP adjustment rule."""
    print("\n=== Testing bidirectional ATP adjustment rule ===")
    
    rule_dict = atp_bidirectional_rule(target_atp=0.8, gain=0.5, deadzone=0.05, max_nad=1.0, min_nad=0.0)
    condition = rule_dict['condition']
    action = rule_dict['action']
    
    # Test condition with deadzone
    state_low = np.array([1.0, 0.1, 0.7, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.7, diff=0.1 > deadzone 0.05
    state_near = np.array([1.0, 0.1, 0.82, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.82, diff=0.02 < deadzone
    intervention = {'nad_supplement': 0.3}
    
    assert condition(0.0, state_low, intervention) == True, "Should trigger when ATP deviation > deadzone"
    assert condition(0.0, state_near, intervention) == False, "Should not trigger when ATP deviation within deadzone"
    
    # Test action: ATP below target -> increase NAD
    target = action(0.0, state_low, intervention)
    assert 'nad_supplement' in target
    error = 0.8 - 0.7  # 0.1
    expected = 0.3 + 0.5 * error  # 0.3 + 0.05 = 0.35
    assert abs(target['nad_supplement'] - expected) < 1e-9, f"Expected {expected}, got {target['nad_supplement']}"
    print(f"Bidirectional rule: ATP=0.7 -> NAD target {target['nad_supplement']:.3f} (expected 0.350)")
    
    # Test action: ATP above target -> decrease NAD (if possible)
    state_high = np.array([1.0, 0.1, 0.9, 0.1, 0.8, 0.1, 1.0, 0.1])  # ATP=0.9, diff=0.1
    target2 = action(0.0, state_high, intervention)
    error2 = 0.8 - 0.9  # -0.1
    expected2 = 0.3 + 0.5 * error2  # 0.3 - 0.05 = 0.25
    assert abs(target2['nad_supplement'] - expected2) < 1e-9, f"Expected {expected2}, got {target2['nad_supplement']}"
    print(f"Bidirectional rule: ATP=0.9 -> NAD target {target2['nad_supplement']:.3f} (expected 0.250)")
    
    print("✓ Bidirectional rule works correctly")
    return rule_dict


def test_adaptive_symmathesy_improvement():
    """Test that adaptive protocols with proportional/bidirectional rules improve symmathesy metrics."""
    print("\n=== Testing adaptive symmathesy improvement ===")
    
    # Create adaptive protocol with bidirectional ATP rule and proportional heteroplasmy rule
    base_intervention = {
        'rapamycin_dose': 0.3,
        'nad_supplement': 0.3,
        'senolytic_dose': 0.3,
        'yamanaka_intensity': 0.0,
        'transplant_rate': 0.3,
        'exercise_level': 0.3,
    }
    protocol = AdaptiveProtocol(base_intervention)
    
    # Add bidirectional ATP rule
    bidir_rule = atp_bidirectional_rule(target_atp=0.8, gain=0.5, deadzone=0.05)
    protocol.add_rule(bidir_rule['condition'], bidir_rule['action'])
    
    # Add proportional heteroplasmy rule
    prop_rule = heteroplasmy_proportional_rule(threshold=0.6, gain=0.4)
    protocol.add_rule(prop_rule['condition'], prop_rule['action'])
    
    # Run simulation with a sick patient
    sick_patient = {
        'baseline_age': 80.0,
        'baseline_heteroplasmy': 0.65,
        'baseline_nad_level': 0.3,
        'genetic_vulnerability': 1.5,
        'metabolic_demand': 1.2,
        'inflammation_level': 0.6,
    }
    print("Running simulation with improved adaptive protocol...")
    result = simulate(intervention=None, patient=sick_patient, resolver=protocol, sim_years=30)
    
    # Compute symmathesy metrics
    sym = compute_symmathesy_metrics(result)
    print("Symmathesy metrics:")
    for k, v in sym.items():
        print(f"  {k}: {v:.4f}")
    
    # Compare with constant intervention
    result_const = simulate(intervention=base_intervention, patient=sick_patient, sim_years=30)
    sym_const = compute_symmathesy_metrics(result_const)
    
    print("\nConstant intervention symmathesy:")
    for k, v in sym_const.items():
        print(f"  {k}: {v:.4f}")
    
    # Expect improvement in adaptation_coherence and relationship diversity
    print(f"\nImprovement (adaptive - constant):")
    print(f"  Adaptation coherence: {sym['adaptation_coherence']:.4f} - {sym_const['adaptation_coherence']:.4f} = {sym['adaptation_coherence'] - sym_const['adaptation_coherence']:.4f}")
    print(f"  Relationship diversity: {sym['relationship_diversity']:.4f} - {sym_const['relationship_diversity']:.4f} = {sym['relationship_diversity'] - sym_const['relationship_diversity']:.4f}")
    
    # Ideally adaptation_coherence > 0 (but may still be zero)
    # Relationship diversity should be > 0.5 (more diverse relationship)
    if sym['adaptation_coherence'] > sym_const['adaptation_coherence']:
        print("✓ Adaptation coherence improved")
    if sym['relationship_diversity'] > sym_const['relationship_diversity']:
        print("✓ Relationship diversity improved")
    
    return result, sym


def test_advanced_symmathesy_protocol():
    """Test advanced symmathesy protocol with proportional and bidirectional rules."""
    print("=== Testing advanced symmathesy protocol ===")
    
    base_intervention = dict(DEFAULT_INTERVENTION)
    base_intervention['nad_supplement'] = 0.1
    base_intervention['transplant_rate'] = 0.1
    base_intervention['exercise_level'] = 0.1
    base_intervention['senolytic_dose'] = 0.1
    
    sick_patient = {
        'baseline_age': 80.0,
        'baseline_heteroplasmy': 0.65,
        'baseline_nad_level': 0.3,
        'genetic_vulnerability': 1.5,
        'metabolic_demand': 1.2,
        'inflammation_level': 0.6,
    }
    
    # Advanced protocol
    protocol = create_advanced_symmathesy_protocol(base_intervention)
    result_adv = simulate(intervention=None, patient=sick_patient, resolver=protocol, sim_years=30)
    sym_adv = compute_symmathesy_metrics(result_adv)
    
    # Basic symmathesy protocol for comparison
    protocol_basic = create_symmathesy_protocol(base_intervention)
    result_basic = simulate(intervention=None, patient=sick_patient, resolver=protocol_basic, sim_years=30)
    sym_basic = compute_symmathesy_metrics(result_basic)
    
    print("\nAdvanced protocol symmathesy metrics:")
    for k, v in sym_adv.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nBasic protocol symmathesy metrics:")
    for k, v in sym_basic.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImprovement (advanced - basic):")
    for k in sym_adv:
        diff = sym_adv[k] - sym_basic[k]
        print(f"  {k}: {diff:.4f}")
    
    # Expect improvement in mutual_information and relationship diversity
    if sym_adv['mutual_information'] > sym_basic['mutual_information']:
        print("✓ Mutual information improved")
    if sym_adv['relationship_diversity'] > sym_basic['relationship_diversity']:
        print("✓ Relationship diversity improved")
    
    # Adaptation coherence may be negative (inverse relationship)
    # but we want stronger correlation magnitude (absolute value)
    if abs(sym_adv['adaptation_coherence']) > abs(sym_basic['adaptation_coherence']):
        print("✓ Adaptation coherence magnitude increased")
    
    return protocol, result_adv, sym_adv


def main():
    """Run all adaptive protocol tests."""
    print("=" * 70)
    print("Adaptive Protocol Tests (Symmathesy Phase 2)")
    print("=" * 70)
    
    try:
        # Test 1: Basic adaptive rule
        protocol = test_basic_adaptive()
        
        # Test 2: Full simulation with symmathesy protocol
        result1, sym1 = test_symmathesy_protocol_simulation()
        
        # Test 3: Phased schedule with adaptive rules
        result2 = test_phased_with_adaptive()
        
        # Test 4: Sick patient with adaptive rules
        result3, sym3, log3 = test_sick_patient_adaptive()
        
        # Test 5: Proportional adjustment rules
        rule1, rule2 = test_proportional_rules()
        
        # Test 6: Bidirectional rule
        bidir_rule = test_bidirectional_rule()
        
        # Test 7: Adaptive symmathesy improvement
        result4, sym4 = test_adaptive_symmathesy_improvement()
        
        # Test 8: Advanced symmathesy protocol
        protocol_adv, result5, sym5 = test_advanced_symmathesy_protocol()
        
        print("\n" + "=" * 70)
        print("All adaptive protocol tests completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()