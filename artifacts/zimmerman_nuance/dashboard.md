# Zimmerman Toolkit Analysis — JGC Mitochondrial Simulator

Generated: 2026-02-17 17:03:29

## Sobol Global Sensitivity
- Base samples: 16
- Total sims: 448
- Parameters: ['grief_slp_int', 'grief_act_int', 'grief_nut_int', 'grief_alc_int', 'grief_br_int', 'grief_med_int', 'grief_soc_int', 'grief_B', 'grief_M', 'grief_D_circ', 'grief_V_prior', 'grief_age', 'grief_E_ctx', 'grief_infl_0', 'rapamycin_dose', 'nad_supplement', 'senolytic_dose', 'yamanaka_intensity', 'transplant_rate', 'exercise_level', 'baseline_age', 'baseline_heteroplasmy', 'baseline_nad_level', 'genetic_vulnerability', 'metabolic_demand', 'inflammation_level']
- **energy_atp_initial** top S1: baseline_nad_level=2.096, baseline_age=-0.953, baseline_heteroplasmy=0.690
- **energy_atp_final** top S1: yamanaka_intensity=0.433, nad_supplement=-0.202, exercise_level=0.133
- **energy_atp_min** top S1: baseline_nad_level=1.087, nad_supplement=-0.279, yamanaka_intensity=0.268

## Falsification
- Tests run: 200
- Violations: 0
- Violation rate: 0.0%

## Contrastive Analysis
- Flip pairs found: 1
- Most flip-prone params: ['baseline_age', 'grief_age', 'grief_soc_int', 'transplant_rate', 'grief_slp_int']

## POSIWID Alignment
- Mean overall alignment: 0.830
- Direction accuracy: 1.000
- Magnitude accuracy: 0.661

## PDS Mapping
- energy_atp_initial variance explained: 0.995
- energy_atp_final variance explained: 0.999
- energy_atp_min variance explained: 0.999
- energy_atp_max variance explained: 0.995
- energy_atp_mean variance explained: 0.999
- energy_atp_cv variance explained: 0.985
- energy_reserve_ratio variance explained: 0.995
- energy_atp_slope variance explained: 0.998
- energy_terminal_slope variance explained: 0.977
- energy_time_to_crisis_years variance explained: 1.000
- damage_het_initial variance explained: 1.000
- damage_het_final variance explained: 0.992
- damage_het_max variance explained: 1.000
- damage_delta_het variance explained: 0.997
- damage_het_slope variance explained: 0.994
- damage_het_acceleration variance explained: 0.989
- damage_cliff_distance_initial variance explained: 1.000
- damage_cliff_distance_final variance explained: 0.992
- damage_time_to_cliff_years variance explained: 1.000
- damage_frac_above_cliff variance explained: 1.000
- damage_deletion_het_initial variance explained: 0.999
- damage_deletion_het_final variance explained: 0.978
- damage_deletion_het_max variance explained: 0.999
- dynamics_ros_dominant_freq variance explained: 1.000
- dynamics_ros_amplitude variance explained: 0.994
- dynamics_membrane_potential_cv variance explained: 0.975
- dynamics_membrane_potential_slope variance explained: 0.997
- dynamics_nad_slope variance explained: 0.998
- dynamics_ros_het_correlation variance explained: 0.969
- dynamics_ros_atp_correlation variance explained: 0.955
- dynamics_senescent_final variance explained: 0.998
- dynamics_senescent_slope variance explained: 0.999
- intervention_atp_benefit_terminal variance explained: 0.999
- intervention_atp_benefit_mean variance explained: 0.998
- intervention_het_benefit_terminal variance explained: 0.999
- intervention_energy_cost_per_year variance explained: 0.999
- intervention_benefit_cost_ratio variance explained: 0.972
- intervention_total_dose variance explained: 1.000
- intervention_crisis_delay_years variance explained: 1.000
- final_heteroplasmy variance explained: 0.992
- final_atp variance explained: 0.999
- final_ros variance explained: 0.994
- final_nad variance explained: 0.998
- final_senescent variance explained: 0.998
- final_membrane_potential variance explained: 0.998
- grief_pe_decay_rate variance explained: 1.000
- grief_pe_initial variance explained: 1.000
- grief_pe_terminal variance explained: 1.000
- grief_g_peak variance explained: 1.000
- grief_g_terminal variance explained: 1.000
- grief_grief_half_life_years variance explained: 1.000
- grief_pgd_risk_score variance explained: 1.000
- grief_sns_peak variance explained: 1.000
- grief_sns_terminal variance explained: 1.000
- grief_sns_mean_first_year variance explained: 1.000
- grief_cort_peak variance explained: 1.000
- grief_cort_mean variance explained: 1.000
- grief_cort_terminal variance explained: 1.000
- grief_hrv_min variance explained: 1.000
- grief_hrv_terminal variance explained: 1.000
- grief_sleep_mean variance explained: 1.000
- grief_sleep_nadir variance explained: 1.000
- grief_sleep_terminal variance explained: 1.000
- grief_infl_peak variance explained: 1.000
- grief_infl_mean variance explained: 1.000
- grief_infl_terminal variance explained: 1.000
- grief_ctra_max variance explained: 1.000
- grief_ctra_terminal variance explained: 1.000
- grief_five_ht_nadir variance explained: 1.000
- grief_five_ht_terminal variance explained: 1.000
- grief_neuroinfl_peak variance explained: 1.000
- grief_neuroinfl_terminal variance explained: 1.000
- grief_cvd_risk_terminal variance explained: 1.000
- grief_cvd_risk_max_rate variance explained: 1.000
- grief_mortality_hazard_ratio variance explained: 1.000

## Dashboard Summary
- Coverage: 0/0 tools (0%)
### Recommendations
- Most influential parameters: baseline_nad_level, baseline_heteroplasmy, grief_slp_int. Focus LLM prompts on these for maximum impact.
- Interaction strength is 0.113 -- parameters interact non-additively. Consider joint parameter prompts rather than independent per-parameter generation.
- Worst-aligned output keys: final_heteroplasmy, final_atp. These diverge most from intended outcomes.
- Most causal parameters: most_causal_params, most_connected_outputs.
