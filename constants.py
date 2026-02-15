"""Central configuration for the mitochondrial aging simulation.

All simulation parameters, biological constants (from Cramer 2025),
12-dimensional parameter space definitions, and Ollama model config.

Reference:
    Cramer, J.G. (2025). *How to Live Much Longer: The Mitochondrial
    DNA Connection*. ISBN 979-8-9928220-0-4.

Type aliases:
    ParamDict = dict[str, float]
    InterventionDict = dict[str, float]
    PatientDict = dict[str, float]

Citation key (Cramer 2025 chapter/page references):
    HETEROPLASMY_CLIFF = 0.7
        Not given as an explicit number in the book. Cramer discusses a
        ~25% overall MitoClock damage threshold (Ch. V.K, p.66) which is
        a different metric (total deletions + point mutations). The 70%
        heteroplasmy cliff is a standard value from the mitochondrial
        genetics literature (e.g., Rossignol et al. 2003).
    CLIFF_STEEPNESS = 15.0
        Simulation calibration. Not from the book.
    DOUBLING_TIME_YOUNG = 11.8, DOUBLING_TIME_OLD = 3.06
        Appendix 2, p.155, Fig. 23 (data from Va23: Vandiver et al.,
        *Aging Cell* 22(6), 2023). "The fits indicate that the doubling
        time (DT) before age 65 is 11.81 years and the doubling time
        after age 65 is 3.06 years." Also Ch. II.H, p.15.
        Corrected 2026-02-15: AGE_TRANSITION restored to 65 per Cramer
        email ("the data puts it at 65").
    AGE_TRANSITION = 40.0
        Book says 65 (Appendix 2, p.155). Simulation uses 40.
    BASELINE_ATP = 1.0
        Ch. VIII.A, Table 3, p.100: "Normal Somatic Cell Operation:
        ~1 MU/day" where 1 MU ≡ 10^8 ATP energy releases.
    BASELINE_ROS = 0.1
        Ch. IV.B Stop 8, p.53 and Ch. II.H, p.14: ROS is a byproduct
        of electron transport chain ATP production. Normalized coupling
        strength is a simulation parameter.
    ROS_PER_DAMAGED = 0.3
        Ch. II.H, p.14: damaged mitochondria with defective ETC leak
        electrons → superoxide; also Appendix 2, pp.152-153 (ROS is
        "a rather minor direct contributor" to damage, but the vicious
        cycle coupling is real). Coupling strength is a simulation param.
    BASELINE_NAD = 1.0, NAD_DECLINE_RATE = 0.01
        Ch. VI.A.3, pp.72-73: NAD+ Boosting (NMN/NR). The age-dependent
        decline references Ca16 (Camacho-Pereira et al. 2016, listed in
        Ch. VI refs, p.87). Specific rate is a simulation parameter.
    BASELINE_MEMBRANE_POTENTIAL = 1.0
        Ch. IV, pp.46-47: proton gradient across inner membrane (~180mV
        in healthy mitochondria). Ch. VI.B, p.75: low ΔΨ = damaged →
        triggers PINK1 accumulation → mitophagy. Normalized to 1.0.
    SENESCENCE_RATE = 0.005
        Ch. VII.A, pp.89-92 and Ch. V.I, p.64: senescence mechanisms.
        Ch. VIII.F, p.103: senescent cells use ~2x energy, emit SASP.
        Rate is a simulation parameter.
    YAMANAKA_ENERGY_COST_MIN/MAX = 3.0/5.0
        Ch. VIII.A, Table 3, p.100: "Yamanaka Reprogramming: ~3-5 MU".
        Ch. VII.B, p.95: "demanding an estimated 3 to 10 times as much
        ATP energy as is required for normal somatic cell operation"
        (citing Ci24, Fo18). Code uses the lower end of the range.
    BASELINE_MITOPHAGY_RATE = 0.02
        Ch. VI.B, p.75: PINK1/Parkin pathway — damaged mitochondria with
        low membrane potential are tagged for removal. The specific rate
        is a simulation parameter; no quantitative rate in the book.
    DAMAGED_REPLICATION_ADVANTAGE = 1.05
        Appendix 2, pp.154-155: deleted mtDNA rings (>3kbp, >18% of
        length) replicate "at least 21% faster" than full-length (p.155,
        citing Va23). Also Ch. II.H, p.15 ("deletion damage builds
        exponentially"). The code's 1.05 (5%) is conservative relative
        to Cramer's data suggesting ≥1.21.
"""
from __future__ import annotations

# ── Type aliases ─────────────────────────────────────────────────────────────

ParamDict = dict[str, float]
InterventionDict = dict[str, float]
PatientDict = dict[str, float]

# ── Simulation parameters ────────────────────────────────────────────────────

SIM_YEARS = 30          # total simulation horizon
DT = 0.01               # timestep in years (~3.65 days)
N_STEPS = int(SIM_YEARS / DT)  # 3000 steps

# ── Biological constants (Cramer 2025) ───────────────────────────────────────

# mtDNA copy number per cell (typical range 100-10,000; we use ~1000 as
# a normalized baseline where 1.0 = full complement).
# Cramer Ch. V.J p.65: varies by tissue type; Appendix 3 p.157: PGCs have
# 10-100 copies, mature embryonic cells have 3,000-4,000.
BASELINE_MTDNA_COPIES = 1000.0

# Heteroplasmy cliff: the fraction of damaged mtDNA at which ATP production
# collapses nonlinearly. From mitochondrial genetics literature (Rossignol
# et al. 2003). Cramer's MitoClock uses a different metric (~25% total
# damage score, Ch. V.K p.66).
HETEROPLASMY_CLIFF = 0.7

# Cliff steepness: controls the sigmoid sharpness at the cliff edge.
# Calibrated so 60%→90% het spans ~10% ATP drop to ~80% ATP drop.
# Simulation parameter, not from book.
CLIFF_STEEPNESS = 15.0

# mtDNA deletion doubling times
# Cramer Appendix 2, p.155, Fig. 23 (data from Va23: Vandiver et al.,
# Aging Cell 22(6), 2023): DT = 11.81 yr before age 65, 3.06 yr after.
# Also Ch. II.H, p.15: "deletion damage builds exponentially."
# NOTE: Book uses age 65 as transition; simulation uses 40 (deliberate).
DOUBLING_TIME_YOUNG = 11.8   # years (Cramer Appendix 2, p.155)
DOUBLING_TIME_OLD = 3.06     # years (Cramer Appendix 2, p.155)
AGE_TRANSITION = 65.0        # Cramer Appendix 2 p.155: "DT before age 65" (corrected per Cramer email 2026-02-15)
                             # NOTE (C10): This is now the NOMINAL transition age. The effective
                             # transition is dynamically shifted by ATP and mitophagy in simulator.py
                             # _deletion_rate(). See Cramer email 2026-02-15 (third).

# Baseline ATP production (metabolic units per day)
# Cramer Ch. VIII.A, Table 3, p.100: "Normal Somatic Cell Operation:
# ~1 MU/day" where 1 MU ≡ 10^8 ATP energy releases.
BASELINE_ATP = 1.0

# ATP crisis threshold: fraction of initial ATP below which we consider
# the cell to be in energy crisis. Used by analytics.py for time-to-crisis.
ATP_CRISIS_FRACTION = 0.5

# ROS generation: damaged mitochondria produce more ROS (vicious cycle)
# Cramer Ch. IV.B p.53: ROS as byproduct of ETC; Ch. II.H p.14: damaged
# mitochondria leak electrons → superoxide. Coupling strengths are sim params.
BASELINE_ROS = 0.1           # normalized ROS at zero damage
ROS_PER_DAMAGED = 0.3        # additional ROS per unit damaged fraction

# NAD+ baseline and age-dependent decline
# Cramer Ch. VI.A.3, pp.72-73: NAD+ Boosting (NMN/NR supplementation).
# Ca16 = Camacho-Pereira et al. 2016, cited in Ch. VI refs p.87.
BASELINE_NAD = 1.0
NAD_DECLINE_RATE = 0.01      # per year natural decline (sim param)

# Membrane potential (ΔΨ in normalized units, 1.0 = healthy ~180mV)
# Cramer Ch. IV pp.46-47: proton gradient across inner membrane.
# Ch. VI.B p.75: low ΔΨ = damaged → PINK1 accumulates → mitophagy.
BASELINE_MEMBRANE_POTENTIAL = 1.0

# Senescence: fraction of cells that become senescent
# Cramer Ch. VII.A pp.89-92: senescence mechanisms, p16 marker.
# Ch. VIII.F p.103: senescent cells use ~2x energy, emit SASP.
BASELINE_SENESCENT = 0.0
SENESCENCE_RATE = 0.005      # per year base rate (sim param)

# Energy costs (MU/day) for Yamanaka reprogramming
# Cramer Ch. VIII.A, Table 3, p.100: "Yamanaka Reprogramming: ~3-5 MU".
# Ch. VII.B p.95: "3 to 10 times" normal cell energy (citing Ci24, Fo18).
YAMANAKA_ENERGY_COST_MIN = 3.0
YAMANAKA_ENERGY_COST_MAX = 5.0

# Mitophagy: baseline rate of damaged mtDNA clearance
# Cramer Ch. VI.B p.75: PINK1/Parkin pathway. No quantitative rate in book.
BASELINE_MITOPHAGY_RATE = 0.02  # fraction per year (sim param)

# Replication advantage: damaged (shorter) mtDNA replicates faster
# Cramer Appendix 2 pp.154-155: deletions >3kbp replicate "at least 21%
# faster" (Va23). 1.05 is conservative; book data suggests ≥1.21.
DAMAGED_REPLICATION_ADVANTAGE = 1.05

# CD38 degradation of NMN/NR supplements
# Cramer Ch. VI.A.3 p.73: CD38 enzyme destroys NMN and NR before they can
# boost NAD+ (Mayo Clinic data). A better strategy is CD38 suppression with
# apigenin. At low supplementation, CD38 destroys most precursor; at high
# doses the protocol includes CD38 suppression (apigenin + NMN/NR combo).
CD38_BASE_SURVIVAL = 0.4     # fraction of NMN/NR surviving CD38 at min dose
CD38_SUPPRESSION_GAIN = 0.6  # additional survival from CD38 inhibitor (apigenin)

# Transplant effectiveness (primary rejuvenation mechanism)
# Cramer Ch. VIII.G pp.104-107: bioreactor-grown stem cells → mitlet
# encapsulation. The ONLY method for reversing accumulated mtDNA damage
# at a scale that could be called rejuvenation (Cramer email, 2026-02-15).
TRANSPLANT_ADDITION_RATE = 0.30     # healthy copy addition (was 0.15)
TRANSPLANT_DISPLACEMENT_RATE = 0.12 # competitive displacement of damaged copies
TRANSPLANT_HEADROOM = 1.5           # max total copies allowed with transplant (was 1.2)

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
        "description": "NAD+ restoration strategy (NMN/NR + CD38 suppression via apigenin)",
        "unit": "normalized 0-1",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "nad_restoration (CD38-gated: low dose mostly destroyed, high dose includes apigenin)",
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
        "description": "Mitochondrial transplant rate (healthy mtDNA infusion via bioreactor mitlets)",
        "unit": "copies/year normalized",
        "range": (0.0, 1.0),
        "grid": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "effect": "adds_healthy_copies + displaces_damaged (primary rejuvenation mechanism)",
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

def snap_param(name: str, value: float) -> float:
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


def snap_all(raw_dict: ParamDict) -> ParamDict:
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
    "inflammation_level": 0.25,
}

# ── Tissue-specific profiles ──────────────────────────────────────────────
# Each tissue has different mtDNA vulnerability characteristics.
# Cramer Ch. V.J p.65: copy number varies by tissue; brain and muscle
# are high-demand tissues most affected by mitochondrial dysfunction.
# These profiles modify the patient's metabolic_demand, ROS sensitivity,
# and biogenesis capacity when passed to simulate().

TISSUE_PROFILES = {
    "default": {
        "metabolic_demand": 1.0,
        "ros_sensitivity": 1.0,    # ROS damage multiplier
        "biogenesis_rate": 1.0,    # exercise biogenesis multiplier
        "description": "Generic somatic cell",
    },
    "brain": {
        "metabolic_demand": 2.0,
        "ros_sensitivity": 1.5,    # neurons are ROS-sensitive
        "biogenesis_rate": 0.3,    # low mitochondrial biogenesis in post-mitotic neurons
        "description": "Neuronal tissue (Cramer Ch. V.J p.65, high demand)",
    },
    "muscle": {
        "metabolic_demand": 1.5,
        "ros_sensitivity": 0.8,    # muscle has some antioxidant defense
        "biogenesis_rate": 1.5,    # exercise-responsive PGC-1alpha
        "description": "Skeletal muscle (sarcopenia, Cramer Ch. VII)",
    },
    "cardiac": {
        "metabolic_demand": 1.8,
        "ros_sensitivity": 1.2,    # heart is vulnerable
        "biogenesis_rate": 0.5,    # limited regenerative capacity
        "description": "Cardiac muscle (cardiomyopathy, Cramer Ch. VII)",
    },
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
