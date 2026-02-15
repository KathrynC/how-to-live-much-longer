"""Central configuration for the mitochondrial aging simulation.

All simulation parameters, biological constants (from Cramer 2025),
12-dimensional parameter space definitions, and Ollama model config.

Reference:
    Cramer, J.G. (2025). *How to Live Much Longer: The Mitochondrial
    DNA Connection*. ISBN 979-8-9928220-0-4.
"""

import numpy as np

# ── Simulation parameters ────────────────────────────────────────────────────

SIM_YEARS = 30          # total simulation horizon
DT = 0.01               # timestep in years (~3.65 days)
N_STEPS = int(SIM_YEARS / DT)  # 3000 steps

# ── Biological constants (Cramer 2025) ───────────────────────────────────────

# mtDNA copy number per cell (typical range 100-10,000; we use ~1000 as
# a normalized baseline where 1.0 = full complement)
BASELINE_MTDNA_COPIES = 1000.0

# Heteroplasmy cliff: the fraction of damaged mtDNA at which ATP production
# collapses nonlinearly. Cramer identifies ~70% as the critical threshold.
HETEROPLASMY_CLIFF = 0.7

# Cliff steepness: controls the sigmoid sharpness at the cliff edge.
# Higher = sharper transition. Calibrated so 60%→90% het spans ~10% ATP drop
# to ~80% ATP drop.
CLIFF_STEEPNESS = 15.0

# mtDNA deletion doubling times (Cramer 2025, Ch. 4)
# - Before age ~40: deletions double every 11.8 years (slow accumulation)
# - After age ~40: deletions double every 3.06 years (accelerating damage)
DOUBLING_TIME_YOUNG = 11.8   # years
DOUBLING_TIME_OLD = 3.06     # years
AGE_TRANSITION = 40.0        # age at which doubling rate accelerates

# Baseline ATP production (metabolic units per day)
# A healthy cell with full mtDNA complement produces ~1.0 MU/day
BASELINE_ATP = 1.0

# ROS generation: damaged mitochondria produce more ROS (vicious cycle)
BASELINE_ROS = 0.1           # normalized ROS at zero damage
ROS_PER_DAMAGED = 0.3        # additional ROS per unit damaged fraction

# NAD+ baseline and age-dependent decline
BASELINE_NAD = 1.0
NAD_DECLINE_RATE = 0.01      # per year natural decline

# Membrane potential (ΔΨ in normalized units, 1.0 = healthy ~180mV)
BASELINE_MEMBRANE_POTENTIAL = 1.0

# Senescence: fraction of cells that become senescent
BASELINE_SENESCENT = 0.0
SENESCENCE_RATE = 0.005      # per year base rate

# Energy costs (MU/day) for Yamanaka reprogramming
YAMANAKA_ENERGY_COST_MIN = 3.0
YAMANAKA_ENERGY_COST_MAX = 5.0

# Mitophagy: baseline rate of damaged mtDNA clearance
BASELINE_MITOPHAGY_RATE = 0.02  # fraction per year

# Replication advantage: damaged (shorter) mtDNA replicates slightly faster
DAMAGED_REPLICATION_ADVANTAGE = 1.05

# ── 12-Dimensional parameter space ──────────────────────────────────────────
# 6 intervention parameters + 6 patient parameters

# --- Intervention parameters (what we control) ---

INTERVENTION_PARAMS = {
    "rapamycin_dose": {
        "description": "Rapamycin dose (mTOR inhibition → enhanced mitophagy)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "mitophagy_boost",
    },
    "nad_supplement": {
        "description": "NAD+ precursor supplementation (NMN/NR)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "nad_restoration",
    },
    "senolytic_dose": {
        "description": "Senolytic drug dose (dasatinib+quercetin or navitoclax)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "senescent_clearance",
    },
    "yamanaka_intensity": {
        "description": "Partial Yamanaka reprogramming intensity (OSKM)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "damage_repair (costs ATP)",
    },
    "transplant_rate": {
        "description": "Mitochondrial transplant rate (healthy mtDNA infusion)",
        "unit": "copies/year normalized",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "adds_healthy_copies",
    },
    "exercise_level": {
        "description": "Exercise intensity (hormesis: moderate ROS → adaptation)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "hormetic_adaptation",
    },
}

# --- Patient parameters (who we're treating) ---

PATIENT_PARAMS = {
    "baseline_age": {
        "description": "Patient starting age",
        "unit": "years",
        "range": (20.0, 90.0),
        "grid": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    },
    "baseline_heteroplasmy": {
        "description": "Starting fraction of damaged mtDNA",
        "unit": "fraction 0-1",
        "range": (0.0, 0.95),
        "grid": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "baseline_nad_level": {
        "description": "Starting NAD+ level (age-dependent decline)",
        "unit": "normalized 0-1",
        "range": (0.2, 1.0),
        "grid": [0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "genetic_vulnerability": {
        "description": "Susceptibility to mtDNA damage (haplogroup-dependent)",
        "unit": "multiplier",
        "range": (0.5, 2.0),
        "grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    },
    "metabolic_demand": {
        "description": "Tissue metabolic demand (brain=high, skin=low)",
        "unit": "multiplier",
        "range": (0.5, 2.0),
        "grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    },
    "inflammation_level": {
        "description": "Baseline chronic inflammation (inflammaging)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    },
}

# Combined parameter names for LLM prompt construction
INTERVENTION_NAMES = list(INTERVENTION_PARAMS.keys())
PATIENT_NAMES = list(PATIENT_PARAMS.keys())
ALL_PARAM_NAMES = INTERVENTION_NAMES + PATIENT_NAMES

# ── Discrete grid snapping ───────────────────────────────────────────────────

def snap_param(name, value):
    """Snap a parameter value to its nearest grid point.

    Args:
        name: Parameter name (must be in INTERVENTION_PARAMS or PATIENT_PARAMS).
        value: Raw float value from LLM.

    Returns:
        Nearest grid value after clamping to valid range.
    """
    params = INTERVENTION_PARAMS if name in INTERVENTION_PARAMS else PATIENT_PARAMS
    spec = params[name]
    lo, hi = spec["range"]
    value = max(lo, min(hi, float(value)))
    return min(spec["grid"], key=lambda g: abs(g - value))


def snap_all(raw_dict):
    """Snap all 12 parameters in a dict to their grid points."""
    snapped = {}
    for name in ALL_PARAM_NAMES:
        if name in raw_dict:
            snapped[name] = snap_param(name, raw_dict[name])
    return snapped

# ── Default intervention (no treatment) ──────────────────────────────────────

DEFAULT_INTERVENTION = {
    "rapamycin_dose": 0.0,
    "nad_supplement": 0.0,
    "senolytic_dose": 0.0,
    "yamanaka_intensity": 0.0,
    "transplant_rate": 0.0,
    "exercise_level": 0.0,
}

DEFAULT_PATIENT = {
    "baseline_age": 70.0,
    "baseline_heteroplasmy": 0.3,
    "baseline_nad_level": 0.6,
    "genetic_vulnerability": 1.0,
    "metabolic_demand": 1.0,
    "inflammation_level": 0.3,
}

# ── Ollama configuration ────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"

# Offer wave model: generates intervention protocols from clinical scenarios
OFFER_MODEL = "qwen3-coder:30b"

# Confirmation wave model: evaluates trajectories (different to prevent
# self-confirmation bias)
CONFIRMATION_MODEL = "llama3.1:latest"

# Models that emit <think>...</think> reasoning tokens
REASONING_MODELS = {"deepseek-r1:8b", "qwen3-coder:30b"}

# ── State variable indices ───────────────────────────────────────────────────

STATE_NAMES = [
    "N_healthy",          # 0: healthy mtDNA copies (normalized)
    "N_damaged",          # 1: damaged mtDNA copies (normalized)
    "ATP",                # 2: ATP production rate (MU/day)
    "ROS",                # 3: reactive oxygen species level
    "NAD",                # 4: NAD+ availability
    "Senescent_fraction", # 5: fraction of senescent cells
    "Membrane_potential", # 6: mitochondrial membrane potential ΔΨ
]

N_STATES = len(STATE_NAMES)

# ── Clinical scenario seeds for TIQM experiments ────────────────────────────

CLINICAL_SEEDS = [
    {
        "id": "cognitive_decline_70",
        "description": "70-year-old with early cognitive decline, suspected "
                       "mitochondrial dysfunction in neurons. Family history "
                       "of Alzheimer's. High metabolic demand tissue (brain).",
    },
    {
        "id": "runner_parkinson_family_45",
        "description": "45-year-old marathon runner with family history of "
                       "Parkinson's disease. Currently healthy but concerned "
                       "about substantia nigra mitochondrial vulnerability.",
    },
    {
        "id": "near_cliff_80",
        "description": "80-year-old with heteroplasmy measured at 65%, "
                       "approaching the 70% cliff. Rapid recent decline in "
                       "energy. Can intervention pull back from the edge?",
    },
    {
        "id": "young_prevention_25",
        "description": "25-year-old biohacker seeking optimal prevention "
                       "strategy. Low current damage, interested in NMN and "
                       "rapamycin for longevity.",
    },
    {
        "id": "post_chemo_55",
        "description": "55-year-old cancer survivor post-chemotherapy with "
                       "significant mitochondrial damage from treatment. "
                       "Elevated ROS, depleted NAD+.",
    },
    {
        "id": "diabetic_cardiomyopathy_65",
        "description": "65-year-old with type 2 diabetes and early "
                       "cardiomyopathy. High inflammation, metabolic "
                       "dysfunction in cardiac tissue.",
    },
    {
        "id": "melas_syndrome_35",
        "description": "35-year-old with MELAS syndrome (mitochondrial "
                       "encephalomyopathy). Genetic vulnerability is very "
                       "high, baseline heteroplasmy already at 50%.",
    },
    {
        "id": "sarcopenia_75",
        "description": "75-year-old with progressive sarcopenia (muscle "
                       "wasting). Declining membrane potential in skeletal "
                       "muscle mitochondria.",
    },
    {
        "id": "transplant_candidate_60",
        "description": "60-year-old identified as ideal candidate for "
                       "mitochondrial transplant via platelet-derived mitlets. "
                       "Haplogroup-matched donor available.",
    },
    {
        "id": "centenarian_genetics_50",
        "description": "50-year-old with centenarian family genetics "
                       "(low genetic vulnerability). Interested in whether "
                       "intervention can extend the genetic advantage further.",
    },
]
