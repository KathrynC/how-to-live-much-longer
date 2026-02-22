import numpy as np
from sleep_trajectory import SleepTrajectory
from simulator import simulate
from constants import DEFAULT_PATIENT

# Baseline patient
patient = DEFAULT_PATIENT.copy()
# No intervention
intervention = {'rapamycin_dose': 0.0, 'nad_supplement': 0.0, 'senolytic_dose': 0.0,
                'yamanaka_intensity': 0.0, 'transplant_rate': 0.0, 'exercise_level': 0.0}

results = []
for sleep_int in [0.1, 0.5, 0.9]:
    st = SleepTrajectory(
        sleep_intervention=sleep_int,
        alcohol_trajectory=None,
        time_points=np.array([0.0]),
        baseline_age=patient['baseline_age'],
        genetic_mods={'mitophagy_efficiency': 1.0},
    )
    effects = st.compute(t=0.0)
    # Modify patient with sleep effects
    mod_patient = patient.copy()
    mod_patient['inflammation_level'] += effects['inflammation_delta']
    mod_patient['_sleep_ros_boost'] = effects['ros_boost']
    mod_patient['_sleep_nad_drain'] = effects['nad_drain']
    mod_patient['_sleep_membrane_penalty'] = effects['membrane_penalty']
    mod_patient['_sleep_atp_boost'] = effects['atp_boost']
    # simulate
    result = simulate(patient=mod_patient, intervention=intervention)
    atp = result['states'][-1, 2]
    het = result['heteroplasmy'][-1]
    results.append((sleep_int, atp, het, effects))
    print(f'Sleep intervention {sleep_int:.1f}: ATP={atp:.4f}, Het={het:.4f}')
    print(f'  Effects: infl_delta={effects["inflammation_delta"]:.4f}, ros_boost={effects["ros_boost"]:.4f}, nad_drain={effects["nad_drain"]:.4f}, atp_boost={effects["atp_boost"]:.4f}')

# Compute difference between good and neutral
atp_neutral = results[1][1]
atp_good = results[2][1]
atp_poor = results[0][1]
print(f'\nATP change poor→neutral: {atp_neutral - atp_poor:.4f} ({((atp_neutral - atp_poor)/atp_neutral*100):.1f}%)')
print(f'ATP change neutral→good: {atp_good - atp_neutral:.4f} ({((atp_good - atp_neutral)/atp_neutral*100):.1f}%)')
