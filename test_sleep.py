import numpy as np
from parameter_resolver import ParameterResolver
from simulator import simulate
from analytics import compute_all

def main() -> None:
    # Default patient (70yo)
    patient = {'baseline_age': 70.0, 'baseline_heteroplasmy': 0.3, 'baseline_nad_level': 0.6,
               'genetic_vulnerability': 1.0, 'metabolic_demand': 1.0, 'inflammation_level': 0.25}
    # Vary sleep intervention
    for sleep_int in [0.1, 0.5, 0.9]:
        resolver = ParameterResolver(
            patient_expanded=patient,
            intervention_expanded={'sleep_intervention': sleep_int},
        )
        # Simulate with resolver
        result = simulate(resolver=resolver)
        atp = result['states'][-1, 2]
        het = result['heteroplasmy'][-1]
        print(f'Sleep intervention {sleep_int:.1f}: ATP={atp:.4f}, Het={het:.4f}')


if __name__ == "__main__":
    main()
