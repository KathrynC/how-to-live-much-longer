#!/usr/bin/env python3
import numpy as np
from simulator import simulate
from parameter_resolver import ParameterResolver
from constants import DEFAULT_PATIENT

interventions = [0.0, 0.25, 0.5, 0.75, 1.0]
patient = dict(DEFAULT_PATIENT, baseline_age=70.0)

for si in interventions:
    resolver = ParameterResolver(
        patient_expanded={'baseline_age': 70.0, 'apoe_genotype': 0, 'sex': 'M'},
        intervention_expanded={'sleep_intervention': si},
    )
    result = simulate(patient=patient, resolver=resolver)
    final_atp = result['states'][-1, 2]
    final_het = result['heteroplasmy'][-1]
    print(f"Sleep intervention {si:.2f}: ATP={final_atp:.6f}, Het={final_het:.6f}")
    # Print sleep effects at t=0
    effects = resolver._sleep_trajectory.compute(0.0)
    print(f"  Sleep effects: { {k: round(v, 5) for k, v in effects.items()} }")
    # Get resolved intervention and patient at t=0
    interv, pat = resolver.resolve(0.0)
    print(f"  Resolved rapamycin_dose: {interv['rapamycin_dose']:.5f}")
    print(f"  Resolved inflammation: {pat['inflammation_level']:.5f}")
    print(f"  Sleep ROS boost: {pat.get('_sleep_ros_boost', 0):.5f}")
    print(f"  Sleep ATP boost: {pat.get('_sleep_atp_boost', 0):.5f}")
    print()