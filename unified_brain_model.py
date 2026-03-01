"""unified_brain_model.py — Definitive 37-State "Surgical-Skeletal" Simulator

The absolute peak of the Digital Twin project. Integrates the 8-state Cramer 
core with 29 extensions, including Bone Health (Osteopenia) and Neural 
Recovery (Post-Meningioma near Broca's Area).

The 37-state vector:
  [0-7]   Cramer Core: [N_h, N_del, ATP, ROS, NAD, Sen, ΔΨ, N_pt]
  [8-13]  Cognitive Chain: MEF2, HA, SS, CR, Ab, Tau
  [14-16] Ion Channels: KCNQ, Kv4, Nav
  [17-18] Psychosocial: Gut (M), Grief (G)
  [19-20] Microglia: M1, M2
  [21]    Cerebrovascular: CBF
  [22-23] Metabolic: Liver (L), GSH
  [24-25] Vascular: Kidney (K), BP
  [26-27] Growth: Muscle, BDNF
  [28-29] Respiratory: Lung (O2), SpO2
  [30-31] Mechanical: Heart (H_pump), Stiffness (Art_S)
  [32]    Governor: Insulin_Sensitivity (IS)
  [33]    Endocrine: Hormone_Shield (E_T)
  [34]    Bioactive: Trigonelline_Load (Trig)
  [35]    Skeletal: Bone_Density (B)
  [36]    Neural: Recovery_Zone (NRZ)
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt

from constants import (
    N_STATES as CORE_N_STATES,
    BASELINE_ATP, BASELINE_ROS, BASELINE_NAD, BASELINE_SENESCENT,
    BASELINE_MEMBRANE_POTENTIAL, BASELINE_MITOPHAGY_RATE,
    # Brain-Specific Refinement Constants
    PATHOLOGY_ATP_INHIBITION, PATHOLOGY_ROS_IMPACT,
    SLEEP_CLEARANCE_MULTIPLIER, SLEEP_ROS_PENALTY, DEFAULT_SLEEP_QUALITY,
    BBB_BASE_LEAKAGE, BBB_AGE_SENSITIVITY, BBB_ROS_SENSITIVITY, MICROGLIA_SASP_COEFF,
    # Downstream Constants
    MEF2_INDUCTION_RATE, MEF2_DECAY_RATE, MEF2_MEMORY_BOOST,
    HA_INDUCTION_RATE, HA_DECAY_RATE,
    PLASTICITY_FACTOR_BASE, PLASTICITY_FACTOR_HA_MAX,
    LEARNING_RATE_BASE, SYNAPTIC_DECAY_RATE, MAX_SYNAPTIC_STRENGTH,
    SYNAPSES_TO_MEMORY, BASELINE_MEMORY,
    CR_GROWTH_RATE_BY_ACTIVITY,
    AMYLOID_PRODUCTION_BASE, AMYLOID_PRODUCTION_AGE_FACTOR,
    AMYLOID_CLEARANCE_BASE, AMYLOID_INFLAMMATION_SYNERGY,
    TAU_SEEDING_RATE, TAU_SEEDING_FACTOR, TAU_INFLAMMATION_FACTOR,
    TAU_CLEARANCE_BASE,
    AMYLOID_TOXICITY, TAU_TOXICITY,
    RESILIENCE_WEIGHTS,
    TISSUE_PROFILES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    GRIEF_ROS_FACTOR, GRIEF_NAD_DECAY, GRIEF_SENESCENCE_FACTOR, COPING_DECAY_RATE, GRIEF_REDUCTION_FROM_MEF2,
    PROBIOTIC_GROWTH_RATE, GUT_DECAY_RATE,
    # Liver Constants
    LIVER_REGEN_RATE, LIVER_DECAY_RATE, GSH_PRODUCTION_RATE, GSH_BRAIN_BUFFER_COEFF, FRUCTOSE_LIVER_PENALTY, ALCOHOL_GSH_DRAIN,
    ALCOHOL_INFLAMMATION_FACTOR, ALCOHOL_NAD_FACTOR,
    # Kidney Constants
    RENAL_DECAY_RATE, RENAL_BP_SENSITIVITY, BP_BBB_DAMAGE_COEFF, RENAL_INFLAMMATION_COEFF,
    # Muscle Constants
    MUSCLE_DECAY_RATE, MYOKINE_PRODUCTION_RATE, MYOKINE_SYNAPTIC_BOOST, MYOKINE_BIOGENESIS_BOOST, PROTEIN_ANABOLIC_EFFICIENCY,
    # Respiratory Constants
    LUNG_DECAY_RATE, O2_ATP_GATING_COEFF, HYPOXIA_ROS_PENALTY, VO2MAX_EXERCISE_GAIN,
    # Cardiac Constants
    CARDIAC_DECAY_RATE, STIFFNESS_BP_COEFF, CARDIAC_O2_COEFF, HYPERTROPHY_ATP_DRAIN,
    # Insulin Constants
    INSULIN_DECAY_RATE, GLUCOSE_ROS_SPIKE, IS_ATP_EFFICIENCY, FASTING_IS_BOOST,
    # Endocrine & Bioactive (Phase 7)
    HORMONE_DECAY_RATE, HORMONE_SHIELD_COEFF, ESTROGEN_REPLACEMENT_GAIN,
    TRIGONELLINE_NAD_BOOST, CAFFEINE_P27_MODIFIER,
    # Bone & Neural Recovery (Phase 8)
    BONE_DECAY_RATE, INFL_BONE_LOSS_COEFF, VIT_D_BONE_BOOST, RESISTANCE_BONE_BOOST,
    RECOVERY_AREA_VULN, BROCA_REPAIR_RATE, SYNAPTIC_REPAIR_BOOST,
    # Physical Interventions (Red Light, Sauna, Yoga)
    RED_LIGHT_ATP_BOOST, RED_LIGHT_INFL_REDUCTION,
    SAUNA_MITOPHAGY_BOOST, SAUNA_CBF_GAIN, SAUNA_BP_REDUCTION,
    YOGA_INFL_REDUCTION, YOGA_STIFFNESS_REDUCTION, YOGA_EDS_STABILITY_BOOST,
    # Phase 11 Precision Interventions
    UROLITHIN_A_MITOPHAGY_BOOST, SPERMIDINE_BIOGENESIS_BOOST,
    SPERMIDINE_MITOPHAGY_BOOST, MOLECULAR_H2_ROS_REDUCTION,
    VNS_AUTONOMIC_STABILIZER, AKKERMANSIA_GUT_BOOST,
    AKKERMANSIA_IS_BOOST, SIDE_SLEEPING_CLEARANCE_BOOST
)

from simulator import derivatives as core_derivatives, initial_state as core_initial_state
from downstream_chain import resilience, memory_index, EDUCATION_BASELINE
from supplement_module import compute_supplement_effects

N_UNIFIED_STATES = 37

def unified_derivatives(
    state: npt.NDArray[np.float64],
    t: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    sleep_quality: float = DEFAULT_SLEEP_QUALITY,
) -> npt.NDArray[np.float64]:
    """37-state Surgical-Skeletal coupled derivative function."""
    
    # ── 1. Unpack State ────────────────────────────────────────────────────
    core_state = state[:CORE_N_STATES]
    (mef2, ha, ss, cr, ab, tau, kcnq, kv4, nav, gut_m, grief_g, 
     m1_mic, m2_mic, cbf, liver_l, gsh_pool, renal_k, blood_pressure, 
     m_mass, bdnf_pool, lung_o2, spo2, h_pump, art_s, is_sens, hormone_e, 
     trig_load, bone_b, nrz_area) = state[CORE_N_STATES:]
    
    n_h, n_del, atp, ros, nad, sen, psi, n_pt = core_state
    
    # ── 2. Context & Modifiers ─────────────────────────────────────────────
    age = patient["baseline_age"] + t
    profile = patient.get("profile", "scholar")
    
    # ── Subject Profile Modifiers ──
    # Profiles:
    # - scholar/artist: Mild drag (1.2x)
    # - eds: Severe drag (1.4x) from structural metabolic variants
    # Support for explicit clinical gradient override
    base_drag = 1.4 if profile == "eds" else 1.2
    structural_drag = patient.get("structural_drag_override", base_drag)
    processing_speed_cost = 1.5 if structural_drag > 1.3 else 1.0
    
    # -- Post-Viral Factor (COVID-19 History) --
    post_viral = patient.get("post_viral_load", 0.0) # 0.0 = Never, 1.0 = Severe/Long-COVID
    viral_infl_boost = 0.15 * post_viral
    viral_demand_mult = 1.0 + (0.1 * post_viral)
    if profile == "eds":
        processing_speed_cost *= (1.0 + 0.1 * post_viral) # COVID exacerbates EDS/dysautonomia interaction
    
    exercise = intervention.get("exercise_level", 0.0)
    alcohol = intervention.get("alcohol_intake", 0.0)
    fructose = patient.get("fructose_intake", 0.2)
    salt_intake = patient.get("salt_intake", 0.5)
    protein_intake = patient.get("protein_intake", 0.5)
    fasting = intervention.get("fasting_regimen", 0.0)
    pollution = patient.get("pollution_exposure", 0.1)
    engagement = patient.get("intellectual_engagement", 0.5)
    
    diet = intervention.get("diet_type", "standard") 
    ex_type = intervention.get("exercise_type", "balanced")
    soc_type = intervention.get("social_protocol", "standard")
    ani_type = intervention.get("animal_protocol", "none")
    
    # ── Intensive Cannabis Modifier ──
    cannabis_lvl = intervention.get("cannabis_use", 0.0) 
    cannabis_ss_drag = 1.0 - (0.15 * cannabis_lvl)
    cannabis_sleep_drag = 1.0 - (0.2 * cannabis_lvl)
    
    # ── SUPPLEMENT EFFECTS ──
    supp_effects = compute_supplement_effects(intervention, gut_health=gut_m)
    genetic_sleep_mod = patient.get("deep_sleep_genetic_penalty", 1.0)
    eff_sleep_quality = min(1.0, (sleep_quality * genetic_sleep_mod * cannabis_sleep_drag) + supp_effects['sleep_boost'])
    
    from constants import GENOTYPE_MULTIPLIERS
    geno = GENOTYPE_MULTIPLIERS.get(patient.get("apoe_genotype"), {})
    apoe_vuln = geno.get('vulnerability', 1.0)
    apoe_alc = geno.get('alcohol_sensitivity', 1.0)
    apoe_mef2 = geno.get('mef2_induction', 1.0)
    apoe_clearance = geno.get('amyloid_clearance', 1.0)
    apoe_tau_mult = geno.get('tau_pathology_sensitivity', 1.0)
    apoe_synaptic_mult = geno.get('synaptic_function', 1.0)
    
    # ── Dietary Profiles ──
    diet_infl_mod = 1.0; diet_is_mod = 1.0; diet_atp_mod = 1.0; diet_ros_mod = 1.0
    if diet == "mediterranean": diet_infl_mod = 0.7; diet_ros_mod = 0.8; diet_is_mod = 1.2
    elif diet == "keto": diet_is_mod = 1.5; diet_atp_mod = 1.1; diet_ros_mod = 0.9
    elif diet == "western": diet_infl_mod = 1.3; diet_is_mod = 0.6; diet_ros_mod = 1.2
    
    # ── Exercise Profiles ──
    ex_biogen_mod = 1.0; ex_cbf_mod = 1.0; ex_bp_mod = 1.0; ex_muscle_mod = 1.0; ex_bdnf_mod = 1.0; ex_lung_mod = 1.0
    if ex_type == "aerobic": ex_biogen_mod = 1.5; ex_lung_mod = 1.5
    elif ex_type == "hiit": ex_cbf_mod = 1.5; ex_bp_mod = 1.5; ex_lung_mod = 1.2
    elif ex_type == "resistance": ex_muscle_mod = 1.5; ex_bdnf_mod = 1.5
    elif ex_type == "balanced": ex_biogen_mod = ex_cbf_mod = ex_bp_mod = ex_muscle_mod = ex_bdnf_mod = ex_lung_mod = 1.1
    
    # ── Social Profiles ──
    soc_mef2_mod = 1.0; soc_grief_mod = 1.0; soc_cr_mod = 1.0
    if soc_type == "collaborative": soc_mef2_mod = 1.3; soc_cr_mod = 1.5
    elif soc_type == "emotional": soc_grief_mod = 1.5; soc_mef2_mod = 1.1
    elif soc_type == "teaching": soc_mef2_mod = 1.2; soc_cr_mod = 1.2; soc_grief_mod = 1.2
    elif soc_type == "integrated": soc_mef2_mod = 1.4; soc_cr_mod = 1.6; soc_grief_mod = 1.8
    
    # ── Animal Profiles ──
    ani_gut_mod = 1.0; ani_grief_mod = 1.0; ani_ex_mod = 1.0; ani_bdnf_mod = 1.0; ani_mitophagy_boost = 0.0
    if ani_type == "livestock": ani_gut_mod = 1.4; ani_grief_mod = 1.2
    elif ani_type == "active": ani_ex_mod = 1.3; ani_bdnf_mod = 1.2
    elif ani_type == "emotional": ani_grief_mod = 1.5
    elif ani_type == "full_farm": 
        ani_gut_mod = 1.8
        ani_grief_mod = 2.0
        ani_ex_mod = 1.5
        ani_bdnf_mod = 1.8
        ani_mitophagy_boost = 0.15  # Large effect size for turnover

    # -- Physical Interventions --
    red_light = intervention.get("red_light_therapy", 0.0)
    sauna = intervention.get("sauna_use", 0.0)
    yoga = intervention.get("restorative_yoga", 0.0)
    
    # -- Phase 11 Precision Interventions --
    urolithin_a = intervention.get("urolithin_a", 0.0)
    spermidine = intervention.get("spermidine", 0.0)
    molecular_h2 = intervention.get("molecular_hydrogen", 0.0)
    vns_intensity = intervention.get("vns_intensity", 0.0)
    akkermansia = intervention.get("akkermansia_probiotic", 0.0)
    side_sleeping = intervention.get("side_sleeping", 0.0) # 0 or 1
    
    # VNS: Autonomic Stabilization reduces drag and systemic inflammation
    vns_mod = 1.0 - (VNS_AUTONOMIC_STABILIZER * vns_intensity)
    structural_drag *= vns_mod
    
    # Red Light (PBM) Effects
    pbm_atp_boost = RED_LIGHT_ATP_BOOST * red_light
    pbm_infl_reduction = RED_LIGHT_INFL_REDUCTION * red_light
    
    # Sauna Effects
    sauna_mitophagy = SAUNA_MITOPHAGY_BOOST * sauna
    sauna_cbf = SAUNA_CBF_GAIN * sauna
    sauna_bp = SAUNA_BP_REDUCTION * sauna
    
    # Restorative Yoga Effects
    yoga_infl = YOGA_INFL_REDUCTION * yoga
    yoga_stiff = YOGA_STIFFNESS_REDUCTION * yoga
    if profile == "eds":
        yoga_infl *= YOGA_EDS_STABILITY_BOOST # Yoga is high-value for EDS parasympathetic/joint stability
    
    # ── 3. Bone Health & Neural Recovery ──────────────────────────────────
    # Bone Density: Decay vs. Resistance training and Vitamin D/K2
    vit_d = intervention.get("vitamin_d_dose", 0.0)
    d_bone = (VIT_D_BONE_BOOST * vit_d + RESISTANCE_BONE_BOOST * exercise * ex_muscle_mod) * (1.0 - bone_b) - (BONE_DECAY_RATE + INFL_BONE_LOSS_COEFF * sen) * bone_b
    
    # Neural Recovery Zone (NRZ): Higher vulnerability to ROS, plasticity driven
    # NRZ_area represents the functional integrity of the post-surgical zone
    d_nrz = (BROCA_REPAIR_RATE * engagement * (1.0 + SYNAPTIC_REPAIR_BOOST * bdnf_pool)) * (1.0 - nrz_area) - (RECOVERY_AREA_VULN * ros) * nrz_area

    # ── 4. Endocrine, Bioactive, Insulin ──────────────────────────────────
    hrt = intervention.get("hrt_therapy", 0.0)
    d_hormone = (ESTROGEN_REPLACEMENT_GAIN * hrt) * (1.0 - hormone_e) - (HORMONE_DECAY_RATE + 0.01 * sen) * hormone_e
    
    coffee_intake = intervention.get("coffee_intake", 0.0)
    d_trig = 0.5 * coffee_intake * (1.0 - trig_load) - 0.2 * trig_load
    d_is = (FASTING_IS_BOOST * fasting + 0.05 * exercise + AKKERMANSIA_IS_BOOST * akkermansia) * diet_is_mod * (1.0 - is_sens) - (INSULIN_DECAY_RATE + 0.2 * fructose) * (1.0/diet_is_mod) * is_sens

    # ── 5. Systemic Pumps (Cardiac, Lung, SpO2) ─────────────────────────────
    d_art_s = 0.02 * max(blood_pressure - 1.0, 0) + 0.01 * (age/100) - 0.03 * exercise * ex_cbf_mod * ani_ex_mod * (art_s) - yoga_stiff * art_s
    energy_mult = min(atp / BASELINE_ATP, 1.0)
    d_h_pump = 0.1 * (exercise + 0.5 * sauna) * energy_mult * ex_biogen_mod * ani_ex_mod * (1.0 - h_pump) - (CARDIAC_DECAY_RATE + 0.05 * art_s) * h_pump
    d_lung = VO2MAX_EXERCISE_GAIN * exercise * ex_lung_mod * ani_ex_mod * (1.0 - lung_o2) - (LUNG_DECAY_RATE + 0.2 * pollution) * lung_o2
    spo2_target = lung_o2 * (0.6 + 0.2 * cbf + 0.2 * h_pump)
    d_spo2 = 2.0 * (spo2_target - spo2)

    # ── 6. Muscle & Growth ──────────────────────────────────────────────────
    muscle_energy_mult = energy_mult * spo2 * (0.8 + 0.2 * is_sens)
    d_m_mass = (0.05 * exercise * ex_muscle_mod * ani_ex_mod * muscle_energy_mult + PROTEIN_ANABOLIC_EFFICIENCY * protein_intake) * (1.0 - m_mass) - MUSCLE_DECAY_RATE * m_mass
    d_bdnf = MYOKINE_PRODUCTION_RATE * exercise * ex_bdnf_mod * ani_bdnf_mod * m_mass * muscle_energy_mult - 0.2 * bdnf_pool

    # ── 7. Vascular, Kidney, Liver ──────────────────────────────────────────
    d_renal = 0.02 * muscle_energy_mult * (1.0 - renal_k) - (RENAL_DECAY_RATE + 0.1 * max(blood_pressure - 1.0, 0)) * renal_k
    bp_target = 1.0 + (salt_intake * 0.2) + (grief_g * 0.15) + RENAL_BP_SENSITIVITY * (1.0 - renal_k) + STIFFNESS_BP_COEFF * art_s - sauna_bp
    d_bp = 0.5 * (bp_target - blood_pressure - 0.1 * exercise * ex_bp_mod * ani_ex_mod)
    d_liver = LIVER_REGEN_RATE * (1.0 - liver_l) - (LIVER_DECAY_RATE + 0.1 * alcohol * apoe_alc + FRUCTOSE_LIVER_PENALTY * fructose) * liver_l
    gsh_production = GSH_PRODUCTION_RATE * liver_l * atp * spo2
    gsh_drain = (0.05 + ALCOHOL_GSH_DRAIN * alcohol * apoe_alc + 0.1 * ros) * gsh_pool
    d_gsh = gsh_production - gsh_drain

    # ── 8. Brain Coupling ──────────────────────────────────────────────────
    neuronal_excitability = (nav / max(kcnq * kv4, 0.1)) * (1.2 - 0.2 * is_sens)
    excitability_demand = 1.0 + 0.2 * (neuronal_excitability - 1.0)
    bbb_leakage = min(BBB_BASE_LEAKAGE + BBB_AGE_SENSITIVITY * max(age - 50, 0) + BBB_ROS_SENSITIVITY * ros + BP_BBB_DAMAGE_COEFF * max(blood_pressure - 1.0, 0), 1.0)
    systemic_infl = (patient.get("inflammation_level", 0.25) * (1.0 + alcohol * ALCOHOL_INFLAMMATION_FACTOR * apoe_alc) + RENAL_INFLAMMATION_COEFF * (1.0 - renal_k) + 0.2 * (1.0 - is_sens)) * diet_infl_mod * vns_mod
    brain_infl = min(1.0, systemic_infl * (1.0 + bbb_leakage) + MICROGLIA_SASP_COEFF * sen + 0.3 * m1_mic + viral_infl_boost - pbm_infl_reduction - yoga_infl)
    
    # ── 9. Core Mitochondrial Engine (Wrapped) ─────────────────────────────
    pathology_burden = min(1.0, (ab * AMYLOID_TOXICITY + tau * TAU_TOXICITY))
    mod_patient = dict(patient)
    mod_patient["inflammation_level"] = brain_infl
    mod_patient["metabolic_demand"] = max(0.1, (TISSUE_PROFILES["brain"]["metabolic_demand"] * excitability_demand * structural_drag * viral_demand_mult) - supp_effects['demand_reduction'])
    tissue_mods = dict(TISSUE_PROFILES["brain"])
    tissue_mods["biogenesis_rate"] *= (1.0 + MYOKINE_BIOGENESIS_BOOST * bdnf_pool) * ex_biogen_mod + (SPERMIDINE_BIOGENESIS_BOOST * spermidine)
    gut_nad_efficiency = (0.7 + 0.3 * gut_m) * (0.8 + 0.2 * liver_l)
    mod_intervention = dict(intervention)
    mod_intervention["nad_supplement"] = (intervention.get("nad_supplement", 0.0) + supp_effects['nad_boost'] + TRIGONELLINE_NAD_BOOST * trig_load) * gut_nad_efficiency
    mod_intervention["rapamycin_dose"] = intervention.get("rapamycin_dose", 0.0) + supp_effects['mitophagy_boost'] + CAFFEINE_P27_MODIFIER * coffee_intake + ani_mitophagy_boost + (UROLITHIN_A_MITOPHAGY_BOOST * urolithin_a) + (SPERMIDINE_MITOPHAGY_BOOST * spermidine)
    d_core = core_derivatives(core_state, t, mod_intervention, mod_patient, tissue_mods)
    
    # Feedbacks on Core
    d_core[2] -= (1.0 - spo2) * O2_ATP_GATING_COEFF * atp
    d_core[2] -= 0.1 * pathology_burden * atp
    d_core[2] -= 0.1 * (1.0 - cbf) * atp
    d_core[2] += IS_ATP_EFFICIENCY * (is_sens - 0.5) * atp * diet_atp_mod
    d_core[2] -= HYPERTROPHY_ATP_DRAIN * art_s * atp
    d_core[3] -= HORMONE_SHIELD_COEFF * hormone_e * ros
    d_core[3] += GLUCOSE_ROS_SPIKE * (1.0 - is_sens) * fructose * diet_ros_mod
    d_core[3] += PATHOLOGY_ROS_IMPACT * pathology_burden
    d_core[3] += SLEEP_ROS_PENALTY * (1.0 - eff_sleep_quality)
    d_core[3] += GRIEF_ROS_FACTOR * grief_g * apoe_vuln
    d_core[3] += HYPOXIA_ROS_PENALTY * (1.0 - spo2)
    d_core[3] -= GSH_BRAIN_BUFFER_COEFF * gsh_pool * ros 
    d_core[3] += 0.2 * engagement * processing_speed_cost
    d_core[3] -= MOLECULAR_H2_ROS_REDUCTION * molecular_h2 * ros # H2 selective scavenging
    d_core[4] -= GRIEF_NAD_DECAY * grief_g * nad
    d_core[4] -= ALCOHOL_NAD_FACTOR * alcohol * apoe_alc
    d_core[5] += GRIEF_SENESCENCE_FACTOR * grief_g * (1.0 - sen)
    
    # ── 10. Cognitive Downstream ─────────────────────────────
    d_mef2 = engagement * MEF2_INDUCTION_RATE * apoe_mef2 * soc_mef2_mod * (1.0 - mef2) - mef2 * MEF2_DECAY_RATE * (1.0 - engagement * 0.5)
    d_ha = mef2 * HA_INDUCTION_RATE * (1.0 - ha) - ha * HA_DECAY_RATE
    d_kcnq = mef2 * 0.3 * (2.0 - kcnq) - (kcnq - 1.0) * 0.2
    d_kv4 = mef2 * 0.25 * (1.8 - kv4) - (kv4 - 1.0) * 0.2
    d_nav = mef2 * (-0.2) * (nav - 0.6) - (nav - 1.0) * 0.2
    plasticity = PLASTICITY_FACTOR_BASE + ha * (PLASTICITY_FACTOR_HA_MAX - PLASTICITY_FACTOR_BASE)
    d_ss = (LEARNING_RATE_BASE * engagement * plasticity * apoe_synaptic_mult * muscle_energy_mult * (0.8 + 0.2 * is_sens) * (1.0 + MYOKINE_SYNAPTIC_BOOST * bdnf_pool) * (1.0 - ss / MAX_SYNAPTIC_STRENGTH) * cannabis_ss_drag) - SYNAPTIC_DECAY_RATE * (ss - 1.0)
    d_cr = engagement * CR_GROWTH_RATE_BY_ACTIVITY.get(patient.get("activity_type", "solitary_routine"), 0.03) * soc_cr_mod * (1.0 - cr)
    m2_activation = 0.2 * ab * energy_mult * (1.0 - m2_mic)
    m2_decay = 0.1 * m2_mic + 0.2 * ros * m2_mic
    d_m2 = m2_activation - m2_decay
    m1_activation = 0.1 * (ros + sen) * (1.0 - m1_mic) + 0.05 * m2_mic * ros
    m1_decay = 0.05 * m1_mic
    d_m1 = m1_activation - m1_decay
    sleep_clearance_factor = 1.0 + (SLEEP_CLEARANCE_MULTIPLIER - 1.0 + SIDE_SLEEPING_CLEARANCE_BOOST * side_sleeping) * eff_sleep_quality
    ab_production = (AMYLOID_PRODUCTION_BASE + AMYLOID_PRODUCTION_AGE_FACTOR * max(age - 63.0, 0)) * (1.0 + brain_infl * AMYLOID_INFLAMMATION_SYNERGY)
    ab_clearance = AMYLOID_CLEARANCE_BASE * apoe_clearance * ab * sleep_clearance_factor * (1.0 + m2_mic)
    d_ab = ab_production - ab_clearance
    tau_production = TAU_SEEDING_RATE * ab * TAU_SEEDING_FACTOR * apoe_tau_mult + brain_infl * TAU_INFLAMMATION_FACTOR * apoe_tau_mult
    tau_clearance = TAU_CLEARANCE_BASE * tau * sleep_clearance_factor
    d_tau = tau_production - tau_clearance
    
    # ── 11. Systemic & Psychosocial ─────────────────────────────────────────
    d_gut = (intervention.get("probiotic_intensity", 0.0) * PROBIOTIC_GROWTH_RATE * ani_gut_mod + (0.1 if intervention.get("diet_type") == "keto" else 0.0) + AKKERMANSIA_GUT_BOOST * akkermansia) * (1.0 - gut_m) - (GUT_DECAY_RATE + 0.1 * alcohol) * gut_m
    d_grief = - (0.05 + intervention.get("therapy_intensity", 0.0) * COPING_DECAY_RATE + patient.get("social_support", 0.0) * 0.2 * soc_grief_mod * ani_grief_mod + mef2 * GRIEF_REDUCTION_FROM_MEF2) * grief_g
    d_cbf = (0.05 * exercise * ex_cbf_mod * ani_ex_mod + 0.05 * h_pump) * (1.0 - cbf) - 0.02 * (ros + brain_infl + max(blood_pressure - 1.0, 0)) * cbf
    
    return np.concatenate([d_core, [d_mef2, d_ha, d_ss, d_cr, d_ab, d_tau, d_kcnq, d_kv4, d_nav, d_gut, d_grief, d_m1, d_m2, d_cbf, d_liver, d_gsh, d_renal, d_bp, d_m_mass, d_bdnf, d_lung, d_spo2, d_h_pump, d_art_s, d_is, d_hormone, d_trig, d_bone, d_nrz]])

def initial_unified_state(patient: dict[str, float]) -> npt.NDArray[np.float64]:
    p = dict(patient)
    p["metabolic_demand"] = TISSUE_PROFILES["brain"]["metabolic_demand"]
    core_state = core_initial_state(p)
    ed_baseline = EDUCATION_BASELINE.get(p.get('education_level', 'bachelors'), 0.4)
    ab = max(0.0, 0.02 * (p["baseline_age"] - 40.0))
    gut = 0.3 if p.get("apoe_genotype", "").startswith("apoe4") else 0.5
    hormone_init = max(0.1, 0.8 - 0.01 * max(p["baseline_age"] - 20, 0))
    
    # Phase 8: Bone Health (Osteopenia proxy = 0.4) and Neural Recovery Zone (Initial integrity = 0.8)
    bone_init = 0.4 if p.get("osteopenia", False) else 0.8
    nrz_init = 0.8 if p.get("previous_surgery", False) else 1.0
    
    downstream = [0.2, 0.2, 1.0, ed_baseline, ab, 0.0, 1.0, 1.0, 1.0, gut, p.get("grief_intensity", 0.0), 0.0, 0.1, 1.0, 0.8, 0.5, 0.8, 1.0, 0.8, 0.1, 0.8, 0.95, 0.8, 0.2, 0.8, hormone_init, 0.1, bone_init, nrz_init]
    return np.concatenate([core_state, downstream])

def unified_simulate(
    intervention: dict[str, float] | None = None,
    patient: dict[str, float] | None = None,
    sim_years: float = 30.0,
    dt: float = 0.01,
    sleep_quality: float = DEFAULT_SLEEP_QUALITY,
    transplant_protocol: str = "none",
) -> dict:
    if intervention is None: intervention = dict(DEFAULT_INTERVENTION)
    if patient is None: patient = dict(DEFAULT_PATIENT)
    n_steps = int(sim_years / dt)
    state = initial_unified_state(patient)
    time_arr = np.linspace(0, sim_years, n_steps + 1)
    states = np.zeros((n_steps + 1, N_UNIFIED_STATES))
    states[0] = state
    mi_trace = np.zeros(n_steps + 1)
    mi_trace[0] = memory_index(state[10], state[8], state[11], state[12], state[13])
    for i in range(n_steps):
        t = time_arr[i]
        mod_intervention = dict(intervention)
        if transplant_protocol == "early":
            if 1.0 <= t < 2.0: mod_intervention["transplant_rate"] = 0.8
            else: mod_intervention["transplant_rate"] = 0.0
        elif transplant_protocol == "rescue":
            total_copies = state[0] + state[1] + state[7]
            het = (state[1] + state[7]) / max(total_copies, 1e-12)
            if het > 0.60: mod_intervention["transplant_rate"] = 0.9
            else: mod_intervention["transplant_rate"] = 0.0
        k1 = unified_derivatives(state, t, mod_intervention, patient, sleep_quality)
        k2 = unified_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, mod_intervention, patient, sleep_quality)
        k3 = unified_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, mod_intervention, patient, sleep_quality)
        k4 = unified_derivatives(state + dt * k3, t + dt, mod_intervention, patient, sleep_quality)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state = np.maximum(state, 0.0); state[5] = min(state[5], 1.0); state[8:12] = np.clip(state[8:12], 0.0, 1.0)
        state[10] = np.clip(state[10], 0.0, MAX_SYNAPTIC_STRENGTH); state[14:17] = np.clip(state[14:17], 0.6, 2.0)
        state[17:37] = np.clip(state[17:37], 0.0, 1.0)
        states[i + 1] = state
        mi_trace[i + 1] = memory_index(state[10], state[8], state[11], state[12], state[13])
    return { "time": time_arr, "states": states, "memory_index": mi_trace, "intervention": intervention, "patient": patient }

if __name__ == "__main__":
    print("Definitive 37-State Omni-Twin Simulation Test...")
    test_p = dict(DEFAULT_PATIENT); test_p.update({ "baseline_age": 63.83, "osteopenia": True, "previous_surgery": True })
    res = unified_simulate(patient=test_p)
    print(f"Final Memory Index: {res['memory_index'][-1]:.4f}")
    print(f"Final Bone Density: {res['states'][-1, 35]:.4f}")
    print(f"Final NRZ Integrity: {res['states'][-1, 36]:.4f}")
