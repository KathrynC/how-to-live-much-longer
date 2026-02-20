"""Central configuration for the mitochondrial aging simulation.

All simulation parameters, biological constants (from Cramer),
12-dimensional parameter space definitions, and Ollama model config.

Reference:
    Cramer, J.G. (forthcoming from Springer Verlag in 2026).
    *How to Live Much Longer: The Mitochondrial DNA Connection*.

Type aliases:
    ParamDict = dict[str, float]
    InterventionDict = dict[str, float]
    PatientDict = dict[str, float]

Citation key (Cramer chapter/page references):
    HETEROPLASMY_CLIFF = 0.50
        Post-C11 recalibration. Literature value was 0.70 total het
        (Rossignol et al. 2003), but C11 split damage into deletions
        (~60%) and point mutations (~40%). Cliff uses deletion het only,
        so threshold lowered to 0.50 to maintain biological equivalence.
        Cramer discusses ~25% MitoClock damage (Ch. V.K, p.66).
    CLIFF_STEEPNESS = 15.0
        Simulation calibration. Not from the book.
    DOUBLING_TIME_YOUNG = 11.8, DOUBLING_TIME_OLD = 3.06
        Appendix 2, p.155, Fig. 23 (data from Va23: Vandiver et al.,
        *Aging Cell* 22(6), 2023). "The fits indicate that the doubling
        time (DT) before age 65 is 11.81 years and the doubling time
        after age 65 is 3.06 years." Also Ch. II.H, p.15.
        Corrected 2026-02-15: AGE_TRANSITION restored to 65 per Cramer
        email ("the data puts it at 65").
    AGE_TRANSITION = 65.0
        Appendix 2 (p.155): age-65 regime transition.
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
    DELETION_REPLICATION_ADVANTAGE = 1.21
        Appendix 2, pp.154-155: deleted mtDNA rings (>3kbp, >18% of
        length) replicate "at least 21% faster" than full-length (p.155,
        citing Va23). Also Ch. II.H, p.15 ("deletion damage builds
        exponentially"). Compliance update (2026-02-17) enforces the
        Appendix-2 minimum of 1.21 (>=21%).
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

# ── Biological constants (Cramer) ────────────────────────────────────────────

# mtDNA copy number per cell (typical range 100-10,000; we use ~1000 as
# a normalized baseline where 1.0 = full complement).
# Cramer Ch. V.J p.65: varies by tissue type; Appendix 3 p.157: PGCs have
# 10-100 copies, mature embryonic cells have 3,000-4,000.
BASELINE_MTDNA_COPIES = 1000.0

# Heteroplasmy cliff: the fraction of DELETION-bearing mtDNA at which ATP
# production collapses nonlinearly. Post-C11 recalibration: the literature
# value of 0.70 (Rossignol et al. 2003) refers to TOTAL heteroplasmy
# (all damaged copies). After C11 split into deletions + point mutations,
# the cliff uses deletion het only. Since deletions are ~60% of total
# damage (DELETION_FRACTION at age 70), the equivalent deletion cliff is
# ~0.42 (= 0.70 * 0.60). We use 0.50 as a conservative estimate,
# accounting for the fact that deletion fraction increases with age
# (DELETION_FRACTION_OLD = 0.80 → cliff at 0.56). This preserves the
# cliff's biological meaning: ATP collapse when most ETC complexes lack
# critical subunits due to large-deletion mtDNA.
HETEROPLASMY_CLIFF = 0.50

# Cliff steepness: controls the sigmoid sharpness at the cliff edge.
# Calibrated so 60%→90% het spans ~10% ATP drop to ~80% ATP drop.
# Simulation parameter, not from book.
CLIFF_STEEPNESS = 15.0

# mtDNA deletion doubling times
# Cramer Appendix 2, p.155, Fig. 23 (data from Va23: Vandiver et al.,
# Aging Cell 22(6), 2023): DT = 11.81 yr before age 65, 3.06 yr after.
# Also Ch. II.H, p.15: "deletion damage builds exponentially."
# NOTE: AGE_TRANSITION is 65 per Appendix 2 p.155. C10 can shift the
# effective transition age dynamically in _deletion_rate().
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

# DEPRECATED (C11): Use DELETION_REPLICATION_ADVANTAGE instead.
# Kept for reference; deletion advantage progressed from 1.05 to 1.21.
# DAMAGED_REPLICATION_ADVANTAGE = 1.05

# CD38 degradation of NMN/NR supplements
# Cramer Ch. VI.A.3 p.73: CD38 enzyme destroys NMN and NR before they can
# boost NAD+ (Mayo Clinic data). A better strategy is CD38 suppression with
# apigenin. At low supplementation, CD38 destroys most precursor; at high
# doses the protocol includes CD38 suppression (apigenin + NMN/NR combo).
CD38_BASE_SURVIVAL = 0.4     # fraction of NMN/NR surviving CD38 at min dose
CD38_SUPPRESSION_GAIN = 0.6  # additional survival from CD38 inhibitor (apigenin)

# NAD-dependent ATP and antioxidant coefficients
# These control how much NAD+ contributes to ATP production and ROS defense.
# NOT directly from Cramer — modeling assumptions. See NAD audit
# (artifacts/finding_nad_audit_hype_guard_2026-02-19.md) for discussion.
# ATP formula: (1 - NAD_ATP_DEPENDENCE) + NAD_ATP_DEPENDENCE * NAD
#   At 0.4: 40% of ATP depends on NAD (large effect, hype-vulnerable)
#   At 0.2: 20% of ATP depends on NAD (conservative estimate)
# Defense formula: 1.0 + NAD_DEFENSE_BOOST * NAD
#   At 0.4: 40% ROS defense boost per unit NAD
#   At 0.2: 20% ROS defense boost (conservative)
NAD_ATP_DEPENDENCE = 0.2     # reduced from 0.4 — see NAD audit
NAD_DEFENSE_BOOST = 0.2      # reduced from 0.4 — see NAD audit

# Transplant effectiveness (primary rejuvenation mechanism)
# Cramer Ch. VIII.G pp.104-107: bioreactor-grown stem cells → mitlet
# encapsulation. The ONLY method for reversing accumulated mtDNA damage
# at a scale that could be called rejuvenation (Cramer email, 2026-02-15).
TRANSPLANT_ADDITION_RATE = 0.30     # healthy copy addition (was 0.15)
TRANSPLANT_DISPLACEMENT_RATE = 0.12 # competitive displacement of damaged copies
TRANSPLANT_HEADROOM = 1.5           # max total copies allowed with transplant (was 1.2)

# Transplant efficacy degradation at high heteroplasmy
# When deletion het is above this threshold, transplant addition and displacement
# are scaled down by a sigmoid factor. Rationale: in a severely damaged cell,
# the transplanted healthy mitochondria face a hostile intracellular environment
# (low ATP, high ROS, disrupted fission/fusion dynamics, SASP inflammation)
# that impairs their engraftment and competitive ability.
# This creates the "point of no return" that Cramer's theory requires:
# above ~85-90% deletion het, transplant alone cannot overcome the damage.
TRANSPLANT_HET_PENALTY_MIDPOINT = 0.75  # total het where transplant efficacy halves
TRANSPLANT_HET_PENALTY_STEEPNESS = 25.0  # sigmoid steepness (higher = sharper cutoff)

# Mitophagy ATP-gating: autophagy is energy-dependent. The cell needs ATP
# to form autophagosomes, process cargo, and fuse with lysosomes. Below
# this ATP midpoint, mitophagy efficiency drops sharply (sigmoid).
# Creates bistable trap: low ATP → impaired mitophagy → damaged mitos
# persist → ATP stays low = "point of no return."
MITOPHAGY_ATP_MIDPOINT = 0.6   # ATP level where mitophagy efficiency halves
MITOPHAGY_ATP_STEEPNESS = 8.0  # sigmoid steepness

# ── C11: Split mutation type constants (Cramer email 2026-02-17) ────────────
#
# C11 splits the single "N_damaged" compartment into two biologically distinct
# mutation types: DELETIONS and POINT MUTATIONS. This matters because:
#
#   1. Deletions (>3kbp removed from the 16.5kb ring) replicate FASTER than
#      wild-type — shorter rings complete replication sooner. This is the
#      exponential expansion mechanism that drives the heteroplasmy cliff
#      (Cramer Appendix 2 pp.154-155, Va23: Vandiver et al. 2023).
#
#   2. Point mutations (single-base substitutions) produce SAME-LENGTH mtDNA
#      rings with no replication advantage. They accumulate linearly via
#      polymerase gamma errors and ROS-induced base oxidation. They impair
#      function (defective ETC subunits) but don't drive cliff dynamics.
#
#   3. Mitophagy preferentially removes deletion-bearing mitochondria because
#      large deletions → defective ETC → low membrane potential → PINK1
#      accumulates → mitophagy tag. Point-mutated mitos often maintain near-
#      normal membrane potential, evading quality control (Cramer Ch. VI.B p.75).
#
# The cliff is now driven by DELETION heteroplasmy (N_del / total), not total
# heteroplasmy. Point mutations add background dysfunction but don't trigger
# the catastrophic nonlinear ATP collapse.
#
# Reference: Cramer Appendix 2 pp.152-155, Va23 (Vandiver et al. 2023);
#            Ch. II.H p.15 (exponential deletion growth);
#            Ch. VI.B p.75 (PINK1/Parkin selective mitophagy).

# ── Point mutation dynamics ──

# Pol gamma error rate per replication event (dimensionless probability).
# Biology: Pol gamma (the dedicated mtDNA polymerase) has 3'→5' exonuclease
# proofreading, giving an error rate of ~1e-7 per base pair per replication.
# Integrated over the 16,569 bp genome, that's ~0.002 mutations/replication.
# We use 0.001 as a conservative estimate after accounting for mismatch repair.
# Too high (>0.01): unrealistic mutation load; point mutations would dominate
# the damage landscape even in young individuals.
# Too low (<0.0001): point mutations become negligible; the model collapses
# back to deletions-only, losing the C11 split's explanatory power.
# Reference: Cramer Ch. II.H p.15 (mtDNA polymerase); general mtDNA
# mutagenesis literature (Zheng et al. 2006, Kennedy et al. 2013).
POINT_ERROR_RATE = 0.001

# ROS-induced point mutation coefficient (mutations per unit ROS per year).
# Biology: ROS (primarily superoxide and hydroxyl radicals) oxidize guanine
# to 8-oxo-guanine, causing G→T transversion point mutations. This is the
# dominant mechanism of oxidative mtDNA damage distinct from strand breaks.
# The value 0.05 is ~33% of the old unified damage_rate coefficient (0.15),
# reflecting that point mutations are one of several ROS damage pathways
# (the others being strand breaks → deletions and lipid peroxidation).
# Too high (>0.2): ROS-driven point mutations overwhelm deletion dynamics,
# making the cliff unreachable (too much damage is non-cliff-driving).
# Too low (<0.01): ROS has no effect on point mutation load, eliminating
# the vicious cycle feedback through the point mutation channel.
# Reference: Cramer Ch. II.H p.14 (ROS-damage vicious cycle);
#            Appendix 2 pp.152-153 ("ROS is a rather minor direct contributor").
ROS_POINT_COEFF = 0.05

# Mitophagy selectivity for point-mutated mitochondria (0 = invisible, 1 = same
# as deletions).
# Biology: The PINK1/Parkin mitophagy pathway detects damaged mitochondria via
# membrane potential collapse. Deletion-bearing mitos have severely defective
# ETC → low ΔΨ → strong PINK1 signal → efficiently cleared. Point-mutated
# mitos often retain near-normal ΔΨ because single amino acid substitutions
# may only partially impair ETC complex function. Selectivity 0.3 means
# mitophagy clears point-mutated mitos at 30% the rate of deletion-bearing ones.
# Too high (>0.7): point mutations cleared almost as efficiently as deletions;
# they never accumulate, negating the purpose of tracking them separately.
# Too low (<0.1): point-mutated mitos are effectively invisible to quality
# control; they accumulate without bound, creating unrealistic dysfunction.
# Reference: Cramer Ch. VI.B p.75 (PINK1 accumulates on low-ΔΨ mitochondria).
POINT_MITOPHAGY_SELECTIVITY = 0.3

# ── Deletion replication dynamics ──

# Deletion replication advantage (REVISED from DAMAGED_REPLICATION_ADVANTAGE).
# Biology: mtDNA rings with large deletions (>3kbp, >18% of the 16.5kb genome)
# complete replication faster because the polymerase has less DNA to copy.
# Cramer Appendix 2 pp.154-155: deleted rings replicate "at least 21% faster"
# than full-length (citing Va23: Vandiver et al. 2023). The advantage is
# size-dependent — a 5kb deletion (~30% of genome) gives ~30% speedup.
# COMPLIANCE UPDATE (2026-02-17): set to 1.21 to satisfy strict conformance
# with Appendix 2 wording ("at least 21% faster" for large deletions >3kbp).
# This update was made for book-compliance traceability, not as a new
# calibration pass against external datasets.
# RAISED from the pre-C11 value of 1.05 because the old constant lumped
# deletions and point mutations together — with point mutations split out,
# the deletion-specific advantage can be more accurately modeled.
# Too high (>1.25): deletions expand so fast that no intervention can
# overcome the replication advantage; the cliff becomes inevitable even with
# aggressive treatment, which contradicts Cramer's transplant rescue data.
# Too low (<1.02): deletion expansion is too slow; the cliff takes >100 years
# to reach, making the aging model biologically implausible.
# Reference: Cramer Appendix 2 pp.154-155 (Va23 data);
#            Ch. II.H p.15 ("deletion damage builds exponentially").
DELETION_REPLICATION_ADVANTAGE = 1.21

# ── Age-dependent deletion fraction of total damage (for initial state) ──

# Young adults: most accumulated mtDNA damage is point mutations because
# deletions haven't had time to exponentially expand yet. Pol gamma errors
# and ROS-oxidized bases accumulate linearly from birth.
# Biology: In 20-year-olds, deep sequencing shows a mix of point mutations
# (dominant by count) and rare deletions (dominant by functional impact).
# The 40% deletion fraction reflects that even in youth, some deletions
# have begun expanding due to their replication advantage.
# Too high (>0.7): implies young adults already have significant deletion
# load, which contradicts the observed rarity of mtDNA deletions in youth.
# Too low (<0.2): implies almost no deletions in young adults; the model
# would need unrealistically fast deletion expansion to reach the cliff by
# age 65-80.
# Reference: Cramer Appendix 2 p.155 (age-dependent damage composition);
#            Ch. II.H p.15 (exponential deletion accumulation).
DELETION_FRACTION_YOUNG = 0.4

# Older adults: deletions dominate because their exponential growth has
# compounded over decades, while point mutations grew only linearly.
# Biology: In 90-year-olds, deep sequencing of post-mitotic tissues (brain,
# muscle) shows that large deletions account for the majority of mtDNA damage
# by functional impact. Total mtDNA deletion burden (multiple clonally
# expanded deletion types, not just the common 4,977 bp deletion) averages
# ~43% in aged controls and ~52% in PD substantia nigra neurons (Bender
# et al. 2006, Nature Genetics 38:515).
# The 80% deletion fraction reflects this late-life dominance.
# Too high (>0.95): leaves almost no room for point mutations in the elderly,
# which contradicts sequencing data showing substantial point mutation load.
# Too low (<0.5): implies deletions haven't overtaken point mutations even
# by age 90, which contradicts the exponential growth model.
# Reference: Cramer Appendix 2 p.155 (age-dependent damage composition);
#            Ch. V.K p.66 (MitoClock damage metric).
DELETION_FRACTION_OLD = 0.8

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
#
# C11 expanded the state vector from 7D to 8D by splitting "N_damaged" into
# two compartments: N_deletion (index 1) and N_point (index 7).
#
# Index 1 was RENAMED from "N_damaged" to "N_deletion". This preserves
# backwards compatibility because the old N_damaged was effectively tracking
# deletion-dominated damage (the cliff-driving population). All existing code
# that reads state[1] continues to get the cliff-relevant quantity.
#
# Index 7 (N_point) was APPENDED at the end of the state vector to minimize
# blast radius. This means:
#   - All existing code indexing state[0:7] continues to work without changes.
#   - Only code that explicitly checks state.shape or iterates over all state
#     variables needs updating (simulator.py, analytics.py, disturbances.py).
#   - The heteroplasmy calculation changes from N_d/(N_h+N_d) to
#     (N_del+N_pt)/(N_h+N_del+N_pt) for total, and N_del/(N_h+N_del+N_pt)
#     for the cliff-driving deletion heteroplasmy.
#
# The full 8D state vector:
#   [0] N_healthy          — wild-type mtDNA copies (normalized to ~1.0)
#   [1] N_deletion         — deletion-mutated mtDNA: shorter rings, replication
#                            advantage, drives the heteroplasmy cliff at ~70%
#   [2] ATP                — energy production rate (MU/day)
#   [3] ROS                — reactive oxygen species level (normalized)
#   [4] NAD                — NAD+ cofactor availability (normalized)
#   [5] Senescent_fraction — fraction of cells in irreversible growth arrest (0..1)
#   [6] Membrane_potential — inner membrane electrochemical gradient ΔΨ (normalized)
#   [7] N_point            — point-mutated mtDNA: same-length rings, no replication
#                            advantage, linear accumulation, evades mitophagy

STATE_NAMES = [
    "N_healthy",          # 0: healthy mtDNA copies (normalized)
    "N_deletion",         # 1: deletion-mutated mtDNA (exponential growth, drives cliff)
    "ATP",                # 2: ATP production rate (MU/day)
    "ROS",                # 3: reactive oxygen species level
    "NAD",                # 4: NAD+ availability
    "Senescent_fraction", # 5: fraction of senescent cells
    "Membrane_potential", # 6: mitochondrial membrane potential ΔΨ
    "N_point",            # 7: point-mutated mtDNA (linear growth, C11)
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

# ============================================================================
# EXPANSION CONSTANTS — Precision Medicine Upgrade (2026-02-19)
# ============================================================================
# These constants support the expanded parameter space (genetics, lifestyle,
# neuroplasticity, Alzheimer's pathology) described in the handoff docs at
# artifacts/handoff_batch{1,2,3,4}_*_2026-02-19.md.
#
# IMPORTANT: Nothing below modifies the Cramer core ODE. These constants are
# consumed by the parameter resolver (parameter_resolver.py), downstream chain
# (downstream_chain.py), and scenario framework — never by simulator.py or
# derivatives().
# ============================================================================

# ── Grief module (O'Connor 2022/2025 bereavement studies) ───────────────────
GRIEF_ROS_FACTOR = 0.3
GRIEF_NAD_DECAY = 0.15
GRIEF_SENESCENCE_FACTOR = 0.1
SLEEP_DISRUPTION_IMPACT = 0.5  # literature-approximated; awaiting LEMURS data from Dodds & Danforth
SOCIAL_SUPPORT_BUFFER = 0.5
COPING_DECAY_RATE = 0.3
LOVE_BUFFER_FACTOR = 0.2
GRIEF_REDUCTION_FROM_MEF2 = 0.1

# ── Genetic multipliers (qualitative estimates; see provenance notes below) ─
GENOTYPE_MULTIPLIERS = {
    'apoe4_het': {
        'mitophagy_efficiency': 0.65,
        'inflammation': 1.2,
        'vulnerability': 1.3,
        'grief_sensitivity': 1.3,
        'alcohol_sensitivity': 1.3,
        'mef2_induction': 1.0,       # Neutralized: no literature supports APOE4-specific
                                      # MEF2 effect (DeepSeek audit 2026-02-20, status D)
        'amyloid_clearance': 0.7,
        'tau_pathology_sensitivity': 1.25,  # Shi et al. 2017 (Nature); Therriault et al. 2020
                                            # (JAMA Neurol) — APOE4 exacerbates tau independently
        'synaptic_function': 0.8,    # Dumanis et al. 2009 (J Neuroscience 29:15317) — reduced
                                      # dendritic spine density in APOE4 mice at all ages
    },
    'apoe4_hom': {
        'mitophagy_efficiency': 0.45,
        'inflammation': 1.4,
        'vulnerability': 1.6,
        'grief_sensitivity': 1.5,
        'alcohol_sensitivity': 1.5,
        'mef2_induction': 1.0,       # Neutralized: see het note above
        'amyloid_clearance': 0.5,
        'tau_pathology_sensitivity': 1.4,   # Hom scaling ~1.6x het, consistent with Cai et al. 2025
        'synaptic_function': 0.65,   # Hom scaling consistent with structural deficit data
    },
    'foxo3_protective': {
        'mitophagy_efficiency': 1.3,
        'inflammation': 0.9,
        'vulnerability': 0.9,
    },
    'cd38_risk': {
        'nad_efficiency': 0.7,
        'baseline_nad': 0.8,
    },
}
# APOE4 multiplier provenance (DeepSeek audit 2026-02-20):
#   mitophagy_efficiency: qualitative estimate (C). General literature supports
#     APOE4-associated lysosomal dysfunction. Defensible range 20-50%.
#   inflammation: qualitative estimate (C). Friday 2025 confirms heightened
#     inflammatory responses but no specific baseline percentage reported.
#   vulnerability: model-specific composite (C). Aggregates multiple risk pathways.
#   grief_sensitivity: model-specific estimate (C). Indirect link via stress biology.
#   alcohol_sensitivity: conservative estimate (A/B). Anttila 2004 (BMJ 329:539)
#     reports OR 2.3-3.6x for clinical dementia (not HR as previously noted);
#     1.3x applied to intermediate biological variables, not clinical endpoints.
#   mef2_induction: NEUTRALIZED (D). No literature supports APOE4-specific MEF2 effect.
#   amyloid_clearance: qualitative estimate (C). Castellano et al. 2011
#     (Sci Transl Med 3:89ra57) demonstrates isoform-specific clearance.
#   tau_pathology_sensitivity: quantitative anchor (B/C). Therriault et al. 2020
#     (JAMA Neurol) shows ~0.33 SD higher tau-PET independently of amyloid.
#   synaptic_function: qualitative estimate (C). Dumanis et al. 2009
#     (J Neurosci 29:15317) shows significantly fewer dendritic spines at all ages.
#
# Original citations: O'Shea et al. 2024 (Alzh. Dement. 20:8062) confirms
# APOE4 x lifestyle interaction effects but does NOT provide these specific
# numerical multiplier values. See artifacts/apoe4_integration_analysis_2026-02-20.md.

# ── Sex-specific modifiers ─────────────────────────────────────────────────
# Note: Ivanich et al. 2025 (J Neurochem PMID 40890565) covers keto diet x
# gut microbiota x APOE4 in a sex-specific manner, NOT neuroinflammation
# directly. These sex-specific constants are qualitative modeling estimates.
FEMALE_APOE4_INFLAMMATION_BOOST = 1.1  # qualitative estimate (C)
MENOPAUSE_HETEROPLASMY_ACCELERATION = 1.05
ESTROGEN_PROTECTION_LOSS_FACTOR = 1.0

# ── Alcohol ───────────────────────────────────────────────────────────────
# Anttila et al. 2004 (BMJ 329:539, doi:10.1136/bmj.38181.418958.BE)
# Downer et al. 2014 (Alcohol Alcohol. 49:17, doi:10.1093/alcalc/agt144)
ALCOHOL_INFLAMMATION_FACTOR = 0.25
ALCOHOL_NAD_FACTOR = 0.15
ALCOHOL_APOE4_SYNERGY = 1.3  # conservative (A/B); Anttila OR 2.3-3.6x for endpoints
ALCOHOL_SLEEP_DISRUPTION = 0.4

# ── Coffee ────────────────────────────────────────────────────────────────
# Membrez et al. 2024 (Nature Metabolism 6:433, doi:10.1038/s42255-024-00997-x)
# Note: Membrez paper covers trigonelline/NAD+ in aging/sarcopenia.
# It contains NO APOE4 data — the former APOE4-specific coffee benefit was unsupported.
COFFEE_TRIGONELLINE_NAD_EFFECT = 0.05
COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY = 0.05
COFFEE_CAFFEINE_MITOCHONDRIAL_BOOST = 0.03
# DEPRECATED: Membrez et al. 2024 contains no APOE4 data. Set to 1.0 (neutral).
# DeepSeek audit 2026-02-20, status D. See apoe4_integration_analysis_2026-02-20.md.
COFFEE_APOE4_BENEFIT_MULTIPLIER = 1.0
COFFEE_FEMALE_BENEFIT_MULTIPLIER = 1.3
COFFEE_MAX_BENEFICIAL_CUPS = 3
COFFEE_SLEEP_DISRUPTION_THRESHOLD_HOURS = 12
COFFEE_PREPARATION_MULTIPLIERS = {
    'filtered': 1.0,
    'unfiltered': 0.8,
    'instant': 0.5,
}

# ── Diet ───────────────────────────────────────────────────────────────────
KETONE_ATP_FACTOR = 0.1
IF_MITOPHAGY_FACTOR = 0.2
KETO_FEMALE_APOE4_MULTIPLIER = 1.3

# ── Probiotics ─────────────────────────────────────────────────────────────
PROBIOTIC_GROWTH_RATE = 0.1
GUT_DECAY_RATE = 0.02
MAX_GUT_HEALTH = 1.0
MIN_NAD_CONVERSION_EFFICIENCY = 0.7
MAX_NAD_CONVERSION_EFFICIENCY = 1.0

# ── Supplement dose-response (simulation estimates; no single literature
# source). Hill-function parameters are modeling constructs. Norwitz et al.
# 2021 (Nutrients 13:1362) discusses qualitative supplement recommendations
# for APOE4 carriers but does not provide these quantitative values. ────────
MAX_NR_EFFECT = 2.0;           NR_HALF_MAX = 0.5
MAX_DHA_EFFECT = 0.2;          DHA_HALF_MAX = 0.5
MAX_COQ10_EFFECT = 0.15;       COQ10_HALF_MAX = 0.5
MAX_RESVERATROL_EFFECT = 0.15; RESVERATROL_HALF_MAX = 0.5
MAX_PQQ_EFFECT = 0.15;         PQQ_HALF_MAX = 0.5
MAX_ALA_EFFECT = 0.12;         ALA_HALF_MAX = 0.5
MAX_VITAMIN_D_EFFECT = 0.10;   VITAMIN_D_HALF_MAX = 0.5
MAX_B_COMPLEX_EFFECT = 0.15;   B_COMPLEX_HALF_MAX = 0.5
MAX_MAGNESIUM_EFFECT = 0.10;   MAGNESIUM_HALF_MAX = 0.5
MAX_ZINC_EFFECT = 0.08;        ZINC_HALF_MAX = 0.5
MAX_SELENIUM_EFFECT = 0.08;    SELENIUM_HALF_MAX = 0.5

# ── MEF2 pathway (Barker et al. 2021, Science Translational Medicine) ──────
MEF2_INDUCTION_RATE = 0.15
MEF2_DECAY_RATE = 0.08
EXCITABILITY_SUPPRESSION_FACTOR = 0.7
MEF2_RESILIENCE_BOOST = 0.4
MEF2_MEMORY_BOOST = 0.2

# ── Epigenetics (histone acetylation) ──────────────────────────────────────
HA_INDUCTION_RATE = 0.2
HA_DECAY_RATE = 0.05
PLASTICITY_FACTOR_BASE = 0.5
PLASTICITY_FACTOR_HA_MAX = 1.0

# ── Synaptic plasticity ───────────────────────────────────────────────────
LEARNING_RATE_BASE = 0.3
SYNAPTIC_DECAY_RATE = 0.1
MAX_SYNAPTIC_STRENGTH = 2.0
SYNAPSES_TO_MEMORY = 0.3
BASELINE_MEMORY = 0.5

# ── Cognitive reserve (Nature 2025) ────────────────────────────────────────
CR_GROWTH_RATE_BY_ACTIVITY = {
    'collaborative_novel': 0.10,
    'solitary_novel': 0.08,
    'collaborative_routine': 0.05,
    'solitary_routine': 0.03,
}

# ── Amyloid and tau pathology ──────────────────────────────────────────────
AMYLOID_PRODUCTION_BASE = 0.05
AMYLOID_PRODUCTION_AGE_FACTOR = 0.001
AMYLOID_CLEARANCE_BASE = 0.12
AMYLOID_CLEARANCE_APOE4_FACTOR = 0.7  # Castellano et al. 2011 (Sci Transl Med 3:89ra57); estimate (C)
AMYLOID_INFLAMMATION_SYNERGY = 0.2
TAU_SEEDING_RATE = 0.1
TAU_SEEDING_FACTOR = 0.5
TAU_INFLAMMATION_FACTOR = 0.1
TAU_CLEARANCE_BASE = 0.05
AMYLOID_TOXICITY = 0.3
TAU_TOXICITY = 0.5
RESILIENCE_WEIGHTS = {'MEF2': 0.3, 'synaptic_gain': 0.3, 'CR': 0.4}
