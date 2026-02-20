"""Parameter resolver — maps ~50D expanded inputs to effective 12D core.

Implements the 10-step modifier chain:

1. Baseline (from DEFAULT_INTERVENTION, DEFAULT_PATIENT)
2. Genetics (APOE4, FOXO3, CD38 → vulnerability, inflammation, NAD)
3. Sex (menopause → inflammation, heteroplasmy acceleration)
4. Grief (time-varying decay → inflammation, ROS stress)
5. Sleep (Oura/intervention → repair efficiency)
6. Lifestyle (alcohol, coffee, diet → inflammation, NAD, demand)
7. Supplements (11 nutraceuticals → NAD, inflammation, mitophagy)
8. Probiotics (gut health ODE → NAD conversion efficiency)
9. Core schedule (passthrough of rapamycin, transplant, etc.)
10. Clamp (all values to valid ranges)

Pre-computes time-varying trajectories at construction:
- Grief: G(t) = G0 * exp(-(base_decay + therapy*COPING_DECAY_RATE) * t)
- Alcohol taper: linear from start to end over taper_years (if scheduled)
- Gut health: simple Euler integration of dM/dt = probiotic*0.1*(1-M) - M*0.02

resolve(t) interpolates pre-computed curves via np.interp().
"""
from __future__ import annotations
import numpy as np

from constants import (
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    INTERVENTION_NAMES, PATIENT_NAMES,
    GRIEF_ROS_FACTOR, GRIEF_NAD_DECAY, GRIEF_SENESCENCE_FACTOR,
    COPING_DECAY_RATE, LOVE_BUFFER_FACTOR, SOCIAL_SUPPORT_BUFFER,
    ALCOHOL_APOE4_SYNERGY,
    PROBIOTIC_GROWTH_RATE, GUT_DECAY_RATE, MAX_GUT_HEALTH,
    SLEEP_DISRUPTION_IMPACT,
    ALCOHOL_SLEEP_DISRUPTION,
)
from genetics_module import compute_genetic_modifiers, compute_sex_modifiers
from lifestyle_module import compute_alcohol_effects, compute_coffee_effects, compute_diet_effects
from supplement_module import compute_supplement_effects


class ParameterResolver:
    """Maps expanded patient+intervention params to effective 12D core params."""

    def __init__(
        self,
        patient_expanded: dict,
        intervention_expanded: dict,
        schedules: dict | None = None,
        duration_years: float = 30.0,
    ):
        self._patient_exp = patient_expanded
        self._intervention_exp = intervention_expanded
        self._schedules = schedules or {}
        self._duration = duration_years

        # Pre-compute genetic modifiers (time-invariant)
        self._genetic_mods = compute_genetic_modifiers(
            apoe_genotype=patient_expanded.get('apoe_genotype', 0),
            foxo3_protective=patient_expanded.get('foxo3_protective', 0),
            cd38_risk=patient_expanded.get('cd38_risk', 0),
        )
        self._sex_mods = compute_sex_modifiers(
            sex=patient_expanded.get('sex', 'M'),
            menopause_status=patient_expanded.get('menopause_status', 'pre'),
            estrogen_therapy=patient_expanded.get('estrogen_therapy', 0),
        )

        # Pre-compute time-varying trajectories
        self._time_points = np.linspace(0, duration_years, int(duration_years * 100) + 1)
        self._grief_trajectory = self._precompute_grief()
        self._alcohol_trajectory = self._precompute_alcohol()
        self._gut_health_trajectory = self._precompute_gut_health()

    def _precompute_grief(self) -> np.ndarray:
        """Pre-compute grief intensity decay curve."""
        G0 = self._patient_exp.get('grief_intensity', 0.0)
        therapy = self._patient_exp.get('therapy_intensity', 0.0)
        support = self._patient_exp.get('social_support', 0.0)
        love = self._patient_exp.get('love_presence', 0.0)

        base_decay = 0.05  # natural grief decay
        therapy_decay = therapy * COPING_DECAY_RATE
        support_decay = support * 0.2
        love_decay = love * LOVE_BUFFER_FACTOR
        total_decay = base_decay + therapy_decay + support_decay + love_decay

        return G0 * np.exp(-total_decay * self._time_points)

    def _precompute_alcohol(self) -> np.ndarray:
        """Pre-compute alcohol intake trajectory (with optional taper)."""
        base_alcohol = self._intervention_exp.get('alcohol_intake', 0.0)

        if 'alcohol_taper' in self._schedules:
            taper = self._schedules['alcohol_taper']
            start = taper.get('start', base_alcohol)
            end = taper.get('end', 0.0)
            taper_years = taper.get('taper_years', 2.0)
            trajectory = np.where(
                self._time_points < taper_years,
                start + (end - start) * self._time_points / taper_years,
                end,
            )
            return np.clip(trajectory, 0.0, 1.0)
        else:
            return np.full_like(self._time_points, base_alcohol)

    def _precompute_gut_health(self) -> np.ndarray:
        """Pre-compute gut health via Euler integration."""
        probiotic = self._intervention_exp.get('probiotic_intensity', 0.0)
        diet_effects = compute_diet_effects(
            diet_type=self._intervention_exp.get('diet_type', 'standard'),
            fasting_regimen=self._intervention_exp.get('fasting_regimen', 0.0),
        )
        diet_boost = diet_effects['gut_health_boost']

        M = self._patient_exp.get('gut_health', 0.5)
        trajectory = np.zeros_like(self._time_points)
        trajectory[0] = M

        dt = self._time_points[1] - self._time_points[0] if len(self._time_points) > 1 else 0.01
        for i in range(1, len(self._time_points)):
            alcohol_t = self._alcohol_trajectory[i - 1]
            dM = (probiotic * PROBIOTIC_GROWTH_RATE + diet_boost) * (MAX_GUT_HEALTH - M) - M * GUT_DECAY_RATE - alcohol_t * 0.125
            M = np.clip(M + dM * dt, 0.0, MAX_GUT_HEALTH)
            trajectory[i] = M

        return trajectory

    def resolve(self, t: float, state: np.ndarray | None = None) -> tuple[dict, dict]:
        """Resolve expanded params to effective 12D core at time t.

        Args:
            t: Current simulation time (years from start).
            state: Current ODE state vector (optional, for future state-dependent resolution).

        Returns:
            (intervention_dict, patient_dict) — both with standard 12D keys.
        """
        # Step 1: Start from defaults
        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)

        # Override with any explicit core params from expanded dicts
        for k in PATIENT_NAMES:
            if k in self._patient_exp:
                patient[k] = self._patient_exp[k]

        # Step 2: Genetics (multiplicative on vulnerability, inflammation, NAD)
        patient['genetic_vulnerability'] *= self._genetic_mods['vulnerability']
        patient['baseline_nad_level'] *= self._genetic_mods['nad_efficiency']

        # Step 3: Sex (additive inflammation, multiplicative heteroplasmy)
        patient['inflammation_level'] = (
            patient['inflammation_level'] * self._genetic_mods['inflammation']
            + self._sex_mods['inflammation_delta']
        )
        patient['baseline_heteroplasmy'] *= self._sex_mods['heteroplasmy_multiplier']

        # Step 4: Grief (time-varying)
        grief_t = float(np.interp(t, self._time_points, self._grief_trajectory))
        grief_sensitivity = self._genetic_mods.get('grief_sensitivity', 1.0)
        patient['inflammation_level'] += grief_t * GRIEF_ROS_FACTOR * grief_sensitivity

        # Step 5: Sleep — independent efficacy modifier
        # Sleep quality modulates repair (mitophagy) via SLEEP_DISRUPTION_IMPACT.
        # Alcohol has a secondary interaction: it degrades sleep quality.
        # These are distinct pathways — alcohol also affects NAD/inflammation
        # independently in Step 6.
        sleep_quality = self._intervention_exp.get('sleep_intervention', 0.5)
        alcohol_t = float(np.interp(t, self._time_points, self._alcohol_trajectory))
        sleep_quality = max(0.0, sleep_quality - alcohol_t * ALCOHOL_SLEEP_DISRUPTION)
        # Poor sleep increases inflammation
        patient['inflammation_level'] += (1.0 - sleep_quality) * 0.05
        # Poor sleep reduces repair efficiency (mitophagy) —
        # scales from 1.0 (perfect sleep) to (1 - SLEEP_DISRUPTION_IMPACT)
        # at sleep_quality=0. Default: 0.7 impact → floor of 0.3 efficacy.
        sleep_repair_factor = 1.0 - SLEEP_DISRUPTION_IMPACT * (1.0 - sleep_quality)
        intervention['rapamycin_dose'] *= sleep_repair_factor

        # Step 6: Lifestyle (alcohol, coffee, diet)
        alcohol_effects = compute_alcohol_effects(
            alcohol_intake=alcohol_t,
            apoe_sensitivity=self._genetic_mods.get('alcohol_sensitivity', 1.0),
        )
        patient['inflammation_level'] += alcohol_effects['inflammation_delta']
        patient['baseline_nad_level'] *= alcohol_effects['nad_multiplier']

        coffee_effects = compute_coffee_effects(
            cups=self._intervention_exp.get('coffee_intake', 0),
            coffee_type=self._intervention_exp.get('coffee_type', 'filtered'),
            sex=self._patient_exp.get('sex', 'M'),
            apoe_genotype=self._patient_exp.get('apoe_genotype', 0),
        )
        patient['baseline_nad_level'] += coffee_effects['nad_boost']
        patient['inflammation_level'] -= coffee_effects['inflammation_reduction']

        diet_effects = compute_diet_effects(
            diet_type=self._intervention_exp.get('diet_type', 'standard'),
            fasting_regimen=self._intervention_exp.get('fasting_regimen', 0.0),
        )
        patient['metabolic_demand'] *= diet_effects['demand_multiplier']

        # Step 7: Supplements
        gut_health_t = float(np.interp(t, self._time_points, self._gut_health_trajectory))
        supplement_dict = {k: v for k, v in self._intervention_exp.items()
                          if k.endswith('_dose') and k not in INTERVENTION_NAMES}
        supp_effects = compute_supplement_effects(supplement_dict, gut_health=gut_health_t)

        intervention['nad_supplement'] += supp_effects['nad_boost']
        patient['inflammation_level'] -= supp_effects['inflammation_reduction']
        intervention['rapamycin_dose'] += supp_effects['mitophagy_boost']
        patient['metabolic_demand'] -= supp_effects['demand_reduction']

        # Step 8: Probiotics (already integrated via gut_health_trajectory affecting supplements)

        # Step 9: Core schedule passthrough
        for k in INTERVENTION_NAMES:
            if k in self._intervention_exp:
                intervention[k] = max(intervention[k], self._intervention_exp[k])

        # Also pass through diet mitophagy boost to exercise/rapamycin
        intervention['exercise_level'] = max(
            intervention['exercise_level'],
            self._intervention_exp.get('exercise_level', 0.0),
        )

        # Step 10: Clamp all values
        patient['inflammation_level'] = np.clip(patient['inflammation_level'], 0.0, 1.0)
        patient['baseline_heteroplasmy'] = np.clip(patient['baseline_heteroplasmy'], 0.0, 0.95)
        patient['baseline_nad_level'] = np.clip(patient['baseline_nad_level'], 0.1, 1.5)
        patient['genetic_vulnerability'] = np.clip(patient['genetic_vulnerability'], 0.5, 3.0)
        patient['metabolic_demand'] = np.clip(patient['metabolic_demand'], 0.3, 2.0)

        for k in INTERVENTION_NAMES:
            intervention[k] = float(np.clip(intervention[k], 0.0, 1.0))

        # Ensure all values are plain floats (not numpy scalars)
        for d in (intervention, patient):
            for k, v in d.items():
                d[k] = float(v)

        return intervention, patient
