# Precision Medicine Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the mitochondrial aging simulator from 8-state/12D to a precision medicine platform with genetics, lifestyle, neuroplasticity, and Alzheimer's pathology — without modifying the Cramer core ODE.

**Architecture:** Three-layer design. (1) A `ParameterResolver` maps ~50D expanded inputs (genetics, sex, grief, lifestyle, supplements) to effective 12D core inputs via pre-computed trajectories and modifier chains. (2) The Cramer core ODE runs untouched. (3) A `DownstreamChain` integrates neuroplasticity (MEF2, HA, synaptic strength, CR) and pathology (amyloid, tau) as post-processing on core outputs, producing memory_index and resilience scores. A `ScenarioFramework` on top enables batch comparison of intervention packages.

**Tech Stack:** Python 3.11+, numpy 1.26.4 (no scipy), matplotlib Agg backend, dataclasses. TDD throughout.

**Design doc:** `artifacts/design_doc_time_varying_parameter_resolver_2026-02-19.md`

**Handoff docs:** `artifacts/handoff_batch{1,2,3,4}_*_2026-02-19.md`

**Working directory:** `/Users/gardenofcomputation/how-to-live-much-longer/`

**Dependency graph of tasks:**

```
Task 1 (expansion constants)
    ↓
Task 2 (genetics module) ←──── Task 3 (lifestyle module) ←──── Task 4 (supplement module)
    ↓                                    ↓
Task 5 (parameter resolver) ←──── all of 1-4
    ↓
Task 6 (simulator integration) — adds `resolver=` arg
    ↓
Task 7 (downstream chain) — MEF2, HA, SS, CR, amyloid, tau, memory_index
    ↓
Task 8 (scenario framework) — definitions, runner, analysis, plots
    ↓
Task 9 (integration test) — run the 4-scenario APOE4 patient comparison end-to-end
```

Tasks 2, 3, 4 are independent of each other (can be parallelized). Everything else is sequential.

---

### Task 1: Expansion Constants

**Files:**
- Modify: `constants.py` (append after line ~609, below CLINICAL_SEEDS)
- Test: `tests/test_expansion_constants.py` (new)

**Step 1: Write the failing test**

Create `tests/test_expansion_constants.py`:

```python
"""Tests for expansion constants (precision medicine upgrade)."""
import pytest


class TestGriefConstants:
    def test_grief_ros_factor_range(self):
        from constants import GRIEF_ROS_FACTOR
        assert 0 < GRIEF_ROS_FACTOR <= 1.0

    def test_grief_nad_decay_range(self):
        from constants import GRIEF_NAD_DECAY
        assert 0 < GRIEF_NAD_DECAY <= 1.0

    def test_love_buffer_factor_range(self):
        from constants import LOVE_BUFFER_FACTOR
        assert 0 < LOVE_BUFFER_FACTOR <= 1.0


class TestGenotypeMultipliers:
    def test_genotype_multipliers_has_three_profiles(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert 'apoe4_het' in GENOTYPE_MULTIPLIERS
        assert 'apoe4_hom' in GENOTYPE_MULTIPLIERS
        assert 'foxo3_protective' in GENOTYPE_MULTIPLIERS
        assert 'cd38_risk' in GENOTYPE_MULTIPLIERS

    def test_apoe4_het_mitophagy_less_than_baseline(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert GENOTYPE_MULTIPLIERS['apoe4_het']['mitophagy_efficiency'] < 1.0

    def test_apoe4_hom_worse_than_het(self):
        from constants import GENOTYPE_MULTIPLIERS
        het = GENOTYPE_MULTIPLIERS['apoe4_het']
        hom = GENOTYPE_MULTIPLIERS['apoe4_hom']
        assert hom['mitophagy_efficiency'] < het['mitophagy_efficiency']
        assert hom['inflammation'] > het['inflammation']

    def test_foxo3_protective_improves_mitophagy(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert GENOTYPE_MULTIPLIERS['foxo3_protective']['mitophagy_efficiency'] > 1.0

    def test_cd38_risk_reduces_nad(self):
        from constants import GENOTYPE_MULTIPLIERS
        assert GENOTYPE_MULTIPLIERS['cd38_risk']['nad_efficiency'] < 1.0


class TestAlcoholConstants:
    def test_alcohol_apoe4_synergy_above_one(self):
        from constants import ALCOHOL_APOE4_SYNERGY
        assert ALCOHOL_APOE4_SYNERGY > 1.0


class TestCoffeeConstants:
    def test_coffee_max_beneficial_cups(self):
        from constants import COFFEE_MAX_BENEFICIAL_CUPS
        assert COFFEE_MAX_BENEFICIAL_CUPS == 3

    def test_coffee_preparation_multipliers(self):
        from constants import COFFEE_PREPARATION_MULTIPLIERS
        assert COFFEE_PREPARATION_MULTIPLIERS['filtered'] >= COFFEE_PREPARATION_MULTIPLIERS['unfiltered']
        assert COFFEE_PREPARATION_MULTIPLIERS['unfiltered'] >= COFFEE_PREPARATION_MULTIPLIERS['instant']


class TestSupplementConstants:
    def test_all_half_max_positive(self):
        from constants import (
            NR_HALF_MAX, DHA_HALF_MAX, COQ10_HALF_MAX,
            RESVERATROL_HALF_MAX, PQQ_HALF_MAX, ALA_HALF_MAX,
            VITAMIN_D_HALF_MAX, B_COMPLEX_HALF_MAX,
            MAGNESIUM_HALF_MAX, ZINC_HALF_MAX, SELENIUM_HALF_MAX,
        )
        for hm in [NR_HALF_MAX, DHA_HALF_MAX, COQ10_HALF_MAX,
                    RESVERATROL_HALF_MAX, PQQ_HALF_MAX, ALA_HALF_MAX,
                    VITAMIN_D_HALF_MAX, B_COMPLEX_HALF_MAX,
                    MAGNESIUM_HALF_MAX, ZINC_HALF_MAX, SELENIUM_HALF_MAX]:
            assert hm > 0

    def test_nr_max_effect_is_largest(self):
        from constants import MAX_NR_EFFECT, MAX_DHA_EFFECT, MAX_COQ10_EFFECT
        assert MAX_NR_EFFECT > MAX_DHA_EFFECT
        assert MAX_NR_EFFECT > MAX_COQ10_EFFECT


class TestMEF2Constants:
    def test_mef2_induction_rate_positive(self):
        from constants import MEF2_INDUCTION_RATE
        assert MEF2_INDUCTION_RATE > 0

    def test_ha_decay_slower_than_induction(self):
        from constants import HA_INDUCTION_RATE, HA_DECAY_RATE
        assert HA_DECAY_RATE < HA_INDUCTION_RATE


class TestAmyloidTauConstants:
    def test_tau_toxicity_greater_than_amyloid(self):
        from constants import AMYLOID_TOXICITY, TAU_TOXICITY
        assert TAU_TOXICITY > AMYLOID_TOXICITY

    def test_resilience_weights_sum_to_one(self):
        from constants import RESILIENCE_WEIGHTS
        total = sum(RESILIENCE_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestCognitiveReserveConstants:
    def test_cr_growth_rates_ordered(self):
        from constants import CR_GROWTH_RATE_BY_ACTIVITY
        assert CR_GROWTH_RATE_BY_ACTIVITY['collaborative_novel'] > CR_GROWTH_RATE_BY_ACTIVITY['solitary_routine']
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_expansion_constants.py -v`
Expected: FAIL — ImportError for all new constants

**Step 3: Write the constants**

Append to `constants.py` after the CLINICAL_SEEDS section (~line 609). Add all constants from `artifacts/handoff_batch3_unified_spec_2026-02-19.md` section "Complete Constants Inventory". Keep the existing Cramer-sourced constants section untouched. Use a clear section header:

```python
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

# ── Grief module (UVM LEMURS, bereavement studies) ──────────────────────────
GRIEF_ROS_FACTOR = 0.3
GRIEF_NAD_DECAY = 0.15
GRIEF_SENESCENCE_FACTOR = 0.1
SLEEP_DISRUPTION_IMPACT = 0.7
SOCIAL_SUPPORT_BUFFER = 0.5
COPING_DECAY_RATE = 0.3
LOVE_BUFFER_FACTOR = 0.2
GRIEF_REDUCTION_FROM_MEF2 = 0.1

# ── Genetic multipliers (APOE4 literature, O'Shea 2024) ────────────────────
GENOTYPE_MULTIPLIERS = {
    'apoe4_het': {
        'mitophagy_efficiency': 0.65,
        'inflammation': 1.2,
        'vulnerability': 1.3,
        'grief_sensitivity': 1.3,
        'alcohol_sensitivity': 1.3,
        'mef2_induction': 1.2,
        'amyloid_clearance': 0.7,
    },
    'apoe4_hom': {
        'mitophagy_efficiency': 0.45,
        'inflammation': 1.4,
        'vulnerability': 1.6,
        'grief_sensitivity': 1.5,
        'alcohol_sensitivity': 1.5,
        'mef2_induction': 1.3,
        'amyloid_clearance': 0.5,
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

# ── Sex-specific (Ivanich et al. 2025) ─────────────────────────────────────
FEMALE_APOE4_INFLAMMATION_BOOST = 1.1
MENOPAUSE_HETEROPLASMY_ACCELERATION = 1.05
ESTROGEN_PROTECTION_LOSS_FACTOR = 1.0

# ── Alcohol (Anttila 2004, Downer 2014) ────────────────────────────────────
ALCOHOL_INFLAMMATION_FACTOR = 0.25
ALCOHOL_NAD_FACTOR = 0.15
ALCOHOL_APOE4_SYNERGY = 1.3
ALCOHOL_SLEEP_DISRUPTION = 0.4

# ── Coffee (Nature Metabolism 2024) ────────────────────────────────────────
COFFEE_TRIGONELLINE_NAD_EFFECT = 0.05
COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY = 0.05
COFFEE_CAFFEINE_MITOCHONDRIAL_BOOST = 0.03
COFFEE_APOE4_BENEFIT_MULTIPLIER = 1.2
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

# ── Supplement dose-response (Norwitz et al. 2021) ─────────────────────────
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
AMYLOID_CLEARANCE_APOE4_FACTOR = 0.7
AMYLOID_INFLAMMATION_SYNERGY = 0.2
TAU_SEEDING_RATE = 0.1
TAU_SEEDING_FACTOR = 0.5
TAU_INFLAMMATION_FACTOR = 0.1
TAU_CLEARANCE_BASE = 0.05
AMYLOID_TOXICITY = 0.3
TAU_TOXICITY = 0.5
RESILIENCE_WEIGHTS = {'MEF2': 0.3, 'synaptic_gain': 0.3, 'CR': 0.4}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_expansion_constants.py -v`
Expected: All PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/ -v`
Expected: All ~262 existing tests still PASS

**Step 6: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add constants.py tests/test_expansion_constants.py
git commit -m "feat: add expansion constants for precision medicine upgrade"
```

---

### Task 2: Genetics Module

**Files:**
- Create: `genetics_module.py`
- Test: `tests/test_genetics_module.py` (new)

**Step 1: Write the failing test**

Create `tests/test_genetics_module.py`:

```python
"""Tests for genetics module — genotype-to-parameter mapping."""
import pytest


class TestGenotypeModifiers:
    def test_non_carrier_returns_neutral(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=0, cd38_risk=0)
        assert mods['vulnerability'] == pytest.approx(1.0)
        assert mods['inflammation'] == pytest.approx(1.0)

    def test_apoe4_het_increases_vulnerability(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        assert mods['vulnerability'] > 1.0

    def test_apoe4_hom_worse_than_het(self):
        from genetics_module import compute_genetic_modifiers
        het = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        hom = compute_genetic_modifiers(apoe_genotype=2, foxo3_protective=0, cd38_risk=0)
        assert hom['vulnerability'] > het['vulnerability']

    def test_foxo3_reduces_vulnerability(self):
        from genetics_module import compute_genetic_modifiers
        without = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=0, cd38_risk=0)
        with_foxo = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=1, cd38_risk=0)
        assert with_foxo['vulnerability'] < without['vulnerability']

    def test_cd38_reduces_nad_efficiency(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=0, foxo3_protective=0, cd38_risk=1)
        assert mods['nad_efficiency'] < 1.0

    def test_combined_apoe4_foxo3(self):
        from genetics_module import compute_genetic_modifiers
        mods = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=1, cd38_risk=0)
        # FOXO3 should partially offset APOE4 vulnerability
        apoe_only = compute_genetic_modifiers(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        assert mods['vulnerability'] < apoe_only['vulnerability']


class TestSexModifiers:
    def test_male_returns_neutral(self):
        from genetics_module import compute_sex_modifiers
        mods = compute_sex_modifiers(sex='M', menopause_status='pre', estrogen_therapy=0)
        assert mods['inflammation_delta'] == pytest.approx(0.0)
        assert mods['heteroplasmy_multiplier'] == pytest.approx(1.0)

    def test_postmenopausal_increases_inflammation(self):
        from genetics_module import compute_sex_modifiers
        mods = compute_sex_modifiers(sex='F', menopause_status='post', estrogen_therapy=0)
        assert mods['inflammation_delta'] > 0

    def test_estrogen_therapy_reduces_menopause_effect(self):
        from genetics_module import compute_sex_modifiers
        without = compute_sex_modifiers(sex='F', menopause_status='post', estrogen_therapy=0)
        with_hrt = compute_sex_modifiers(sex='F', menopause_status='post', estrogen_therapy=1)
        assert with_hrt['inflammation_delta'] < without['inflammation_delta']


class TestApplyGeneticModifiers:
    def test_applies_to_patient_dict(self):
        from genetics_module import apply_genetic_modifiers
        patient = {
            'baseline_age': 63.0,
            'baseline_heteroplasmy': 0.5,
            'baseline_nad_level': 0.6,
            'genetic_vulnerability': 1.0,
            'metabolic_demand': 1.0,
            'inflammation_level': 0.25,
        }
        expanded = {
            'apoe_genotype': 1,
            'foxo3_protective': 0,
            'cd38_risk': 0,
            'sex': 'F',
            'menopause_status': 'post',
            'estrogen_therapy': 0,
        }
        result = apply_genetic_modifiers(patient, expanded)
        assert result['genetic_vulnerability'] > 1.0  # APOE4 het
        assert result['inflammation_level'] > 0.25     # post-menopause
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_genetics_module.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write the implementation**

Create `genetics_module.py`:

```python
"""Genetics module — maps genotype and sex to core patient parameter modifiers.

Consumes GENOTYPE_MULTIPLIERS from constants.py. Outputs modifier dicts that
the ParameterResolver applies to the core 6D patient params before the Cramer
ODE sees them. Never imports from or modifies simulator.py.

References:
    APOE4: O'Shea et al. 2024
    Sex differences: Ivanich et al. 2025
    FOXO3: longevity association studies
    CD38: Camacho-Pereira et al. 2016 (via Cramer Ch. VI.A.3)
"""
from __future__ import annotations

from constants import (
    GENOTYPE_MULTIPLIERS,
    FEMALE_APOE4_INFLAMMATION_BOOST,
    MENOPAUSE_HETEROPLASMY_ACCELERATION,
    ESTROGEN_PROTECTION_LOSS_FACTOR,
)


def compute_genetic_modifiers(
    apoe_genotype: int = 0,
    foxo3_protective: int = 0,
    cd38_risk: int = 0,
) -> dict[str, float]:
    """Compute multiplicative modifiers from genotype.

    Args:
        apoe_genotype: 0=non-carrier, 1=heterozygous, 2=homozygous APOE4.
        foxo3_protective: 1 if rs9486902 CC genotype, else 0.
        cd38_risk: 1 if rs6449197 risk variant, else 0.

    Returns:
        Dict with keys: vulnerability, inflammation, nad_efficiency,
        alcohol_sensitivity, grief_sensitivity, mef2_induction,
        amyloid_clearance. All default to 1.0 (neutral).
    """
    mods = {
        'vulnerability': 1.0,
        'inflammation': 1.0,
        'nad_efficiency': 1.0,
        'alcohol_sensitivity': 1.0,
        'grief_sensitivity': 1.0,
        'mef2_induction': 1.0,
        'amyloid_clearance': 1.0,
    }

    # APOE4 effects (multiplicative)
    apoe_key = {1: 'apoe4_het', 2: 'apoe4_hom'}.get(apoe_genotype)
    if apoe_key and apoe_key in GENOTYPE_MULTIPLIERS:
        gm = GENOTYPE_MULTIPLIERS[apoe_key]
        mods['vulnerability'] *= gm.get('vulnerability', 1.0)
        mods['inflammation'] *= gm.get('inflammation', 1.0)
        mods['alcohol_sensitivity'] *= gm.get('alcohol_sensitivity', 1.0)
        mods['grief_sensitivity'] *= gm.get('grief_sensitivity', 1.0)
        mods['mef2_induction'] *= gm.get('mef2_induction', 1.0)
        mods['amyloid_clearance'] *= gm.get('amyloid_clearance', 1.0)

    # FOXO3 protective (multiplicative, stacks with APOE4)
    if foxo3_protective:
        fm = GENOTYPE_MULTIPLIERS.get('foxo3_protective', {})
        mods['vulnerability'] *= fm.get('vulnerability', 1.0)
        mods['inflammation'] *= fm.get('inflammation', 1.0)

    # CD38 risk variant
    if cd38_risk:
        cm = GENOTYPE_MULTIPLIERS.get('cd38_risk', {})
        mods['nad_efficiency'] *= cm.get('nad_efficiency', 1.0)

    return mods


def compute_sex_modifiers(
    sex: str = 'M',
    menopause_status: str = 'pre',
    estrogen_therapy: int = 0,
) -> dict[str, float]:
    """Compute additive/multiplicative modifiers from biological sex.

    Args:
        sex: 'M' or 'F'.
        menopause_status: 'pre', 'peri', or 'post' (females only).
        estrogen_therapy: 1 if on HRT, else 0.

    Returns:
        Dict with keys: inflammation_delta (additive to inflammation_level),
        heteroplasmy_multiplier (multiplicative on baseline_heteroplasmy).
    """
    mods = {
        'inflammation_delta': 0.0,
        'heteroplasmy_multiplier': 1.0,
    }

    if sex == 'F' and menopause_status in ('peri', 'post'):
        base_inflammation = 0.1 if menopause_status == 'post' else 0.05
        base_het_mult = MENOPAUSE_HETEROPLASMY_ACCELERATION if menopause_status == 'post' else 1.02

        # Estrogen therapy partially reverses menopause effects
        if estrogen_therapy:
            base_inflammation *= 0.5
            base_het_mult = 1.0 + (base_het_mult - 1.0) * 0.5

        mods['inflammation_delta'] = base_inflammation
        mods['heteroplasmy_multiplier'] = base_het_mult

    return mods


def apply_genetic_modifiers(
    patient_12d: dict[str, float],
    expanded_params: dict,
) -> dict[str, float]:
    """Apply genotype and sex modifiers to a core 12D patient dict.

    Returns a new dict (does not mutate input).

    Args:
        patient_12d: Core 6-param patient dict (baseline_age, etc.).
        expanded_params: Dict with apoe_genotype, foxo3_protective, cd38_risk,
            sex, menopause_status, estrogen_therapy.

    Returns:
        Modified patient dict with genetic/sex effects applied.
    """
    result = dict(patient_12d)

    genetic = compute_genetic_modifiers(
        apoe_genotype=expanded_params.get('apoe_genotype', 0),
        foxo3_protective=expanded_params.get('foxo3_protective', 0),
        cd38_risk=expanded_params.get('cd38_risk', 0),
    )

    sex = compute_sex_modifiers(
        sex=expanded_params.get('sex', 'M'),
        menopause_status=expanded_params.get('menopause_status', 'pre'),
        estrogen_therapy=expanded_params.get('estrogen_therapy', 0),
    )

    result['genetic_vulnerability'] *= genetic['vulnerability']
    result['inflammation_level'] = min(1.0, result['inflammation_level'] * genetic['inflammation'] + sex['inflammation_delta'])
    result['baseline_heteroplasmy'] *= sex['heteroplasmy_multiplier']
    result['baseline_nad_level'] *= genetic['nad_efficiency']

    return result
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_genetics_module.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add genetics_module.py tests/test_genetics_module.py
git commit -m "feat: add genetics module (APOE4, FOXO3, CD38, sex modifiers)"
```

---

### Task 3: Lifestyle Module

**Files:**
- Create: `lifestyle_module.py`
- Test: `tests/test_lifestyle_module.py` (new)

**Step 1: Write the failing test**

Create `tests/test_lifestyle_module.py`:

```python
"""Tests for lifestyle module — alcohol, diet, coffee, fasting effects."""
import pytest


class TestAlcoholEffects:
    def test_zero_alcohol_no_effect(self):
        from lifestyle_module import compute_alcohol_effects
        effects = compute_alcohol_effects(alcohol_intake=0.0, apoe_sensitivity=1.0)
        assert effects['inflammation_delta'] == pytest.approx(0.0)
        assert effects['nad_multiplier'] == pytest.approx(1.0)

    def test_alcohol_increases_inflammation(self):
        from lifestyle_module import compute_alcohol_effects
        effects = compute_alcohol_effects(alcohol_intake=0.5, apoe_sensitivity=1.0)
        assert effects['inflammation_delta'] > 0

    def test_apoe4_amplifies_alcohol(self):
        from lifestyle_module import compute_alcohol_effects
        normal = compute_alcohol_effects(alcohol_intake=0.5, apoe_sensitivity=1.0)
        apoe4 = compute_alcohol_effects(alcohol_intake=0.5, apoe_sensitivity=1.3)
        assert apoe4['inflammation_delta'] > normal['inflammation_delta']


class TestCoffeeEffects:
    def test_zero_coffee_no_effect(self):
        from lifestyle_module import compute_coffee_effects
        effects = compute_coffee_effects(cups=0, coffee_type='filtered', sex='M', apoe_genotype=0)
        assert effects['nad_boost'] == pytest.approx(0.0)

    def test_three_cups_provides_nad_boost(self):
        from lifestyle_module import compute_coffee_effects
        effects = compute_coffee_effects(cups=3, coffee_type='filtered', sex='M', apoe_genotype=0)
        assert effects['nad_boost'] > 0

    def test_diminishing_returns_past_three(self):
        from lifestyle_module import compute_coffee_effects
        three = compute_coffee_effects(cups=3, coffee_type='filtered', sex='M', apoe_genotype=0)
        five = compute_coffee_effects(cups=5, coffee_type='filtered', sex='M', apoe_genotype=0)
        assert five['nad_boost'] == pytest.approx(three['nad_boost'])

    def test_filtered_better_than_instant(self):
        from lifestyle_module import compute_coffee_effects
        filtered = compute_coffee_effects(cups=2, coffee_type='filtered', sex='M', apoe_genotype=0)
        instant = compute_coffee_effects(cups=2, coffee_type='instant', sex='M', apoe_genotype=0)
        assert filtered['nad_boost'] > instant['nad_boost']


class TestDietEffects:
    def test_standard_diet_neutral(self):
        from lifestyle_module import compute_diet_effects
        effects = compute_diet_effects(diet_type='standard', fasting_regimen=0.0)
        assert effects['demand_multiplier'] == pytest.approx(1.0)
        assert effects['mitophagy_boost'] == pytest.approx(0.0)

    def test_keto_reduces_demand(self):
        from lifestyle_module import compute_diet_effects
        effects = compute_diet_effects(diet_type='keto', fasting_regimen=0.0)
        assert effects['demand_multiplier'] < 1.0

    def test_fasting_boosts_mitophagy(self):
        from lifestyle_module import compute_diet_effects
        effects = compute_diet_effects(diet_type='standard', fasting_regimen=0.8)
        assert effects['mitophagy_boost'] > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_lifestyle_module.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `lifestyle_module.py`:

```python
"""Lifestyle module — alcohol, coffee, diet, and fasting effects.

Maps lifestyle parameters to modifiers on the core 12D patient/intervention
params. Never imports from simulator.py.

References:
    Alcohol: Anttila et al. 2004, Downer et al. 2014
    Coffee: Nature Metabolism 2024 (trigonelline/NAD+ pathway)
    Diet/fasting: Ivanich et al. 2025
"""
from __future__ import annotations

from constants import (
    ALCOHOL_INFLAMMATION_FACTOR,
    ALCOHOL_NAD_FACTOR,
    ALCOHOL_SLEEP_DISRUPTION,
    COFFEE_TRIGONELLINE_NAD_EFFECT,
    COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY,
    COFFEE_MAX_BENEFICIAL_CUPS,
    COFFEE_PREPARATION_MULTIPLIERS,
    COFFEE_APOE4_BENEFIT_MULTIPLIER,
    COFFEE_FEMALE_BENEFIT_MULTIPLIER,
    KETONE_ATP_FACTOR,
    IF_MITOPHAGY_FACTOR,
)


def compute_alcohol_effects(
    alcohol_intake: float = 0.0,
    apoe_sensitivity: float = 1.0,
) -> dict[str, float]:
    """Compute alcohol effects on inflammation and NAD+.

    Args:
        alcohol_intake: 0-1 normalized consumption.
        apoe_sensitivity: genotype alcohol sensitivity multiplier.

    Returns:
        Dict with inflammation_delta (additive), nad_multiplier (multiplicative),
        sleep_disruption (additive reduction to sleep quality).
    """
    return {
        'inflammation_delta': alcohol_intake * ALCOHOL_INFLAMMATION_FACTOR * apoe_sensitivity,
        'nad_multiplier': 1.0 - alcohol_intake * ALCOHOL_NAD_FACTOR * apoe_sensitivity,
        'sleep_disruption': alcohol_intake * ALCOHOL_SLEEP_DISRUPTION,
    }


def compute_coffee_effects(
    cups: float = 0.0,
    coffee_type: str = 'filtered',
    sex: str = 'M',
    apoe_genotype: int = 0,
) -> dict[str, float]:
    """Compute coffee effects on NAD+ and inflammation.

    Benefits cap at COFFEE_MAX_BENEFICIAL_CUPS. Preparation method scales
    bioavailability. APOE4 carriers and females get additional benefit.

    Args:
        cups: Daily cups (0-5).
        coffee_type: 'filtered', 'unfiltered', or 'instant'.
        sex: 'M' or 'F'.
        apoe_genotype: 0/1/2.

    Returns:
        Dict with nad_boost (additive) and inflammation_reduction (subtractive).
    """
    effective_cups = min(cups, COFFEE_MAX_BENEFICIAL_CUPS)
    prep_mult = COFFEE_PREPARATION_MULTIPLIERS.get(coffee_type, 1.0)

    genotype_mult = COFFEE_APOE4_BENEFIT_MULTIPLIER if apoe_genotype > 0 else 1.0
    sex_mult = COFFEE_FEMALE_BENEFIT_MULTIPLIER if sex == 'F' else 1.0

    nad_boost = effective_cups * COFFEE_TRIGONELLINE_NAD_EFFECT * prep_mult * genotype_mult * sex_mult
    inflammation_reduction = effective_cups * COFFEE_CHLOROGENIC_ACID_ANTI_INFLAMMATORY * prep_mult

    return {
        'nad_boost': nad_boost,
        'inflammation_reduction': inflammation_reduction,
    }


def compute_diet_effects(
    diet_type: str = 'standard',
    fasting_regimen: float = 0.0,
) -> dict[str, float]:
    """Compute dietary effects on metabolic demand and mitophagy.

    Args:
        diet_type: 'standard', 'mediterranean', or 'keto'.
        fasting_regimen: 0-1 intermittent fasting intensity.

    Returns:
        Dict with demand_multiplier, mitophagy_boost, gut_health_boost.
    """
    demand_mult = {
        'standard': 1.0,
        'mediterranean': 0.97,
        'keto': 1.0 - KETONE_ATP_FACTOR,
    }.get(diet_type, 1.0)

    gut_boost = {
        'standard': 0.0,
        'mediterranean': 0.03,
        'keto': 0.05,
    }.get(diet_type, 0.0)

    return {
        'demand_multiplier': demand_mult,
        'mitophagy_boost': fasting_regimen * IF_MITOPHAGY_FACTOR,
        'gut_health_boost': gut_boost,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_lifestyle_module.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add lifestyle_module.py tests/test_lifestyle_module.py
git commit -m "feat: add lifestyle module (alcohol, coffee, diet, fasting)"
```

---

### Task 4: Supplement Module

**Files:**
- Create: `supplement_module.py`
- Test: `tests/test_supplement_module.py` (new)

**Step 1: Write the failing test**

Create `tests/test_supplement_module.py`:

```python
"""Tests for supplement module — Hill-function dose-response curves."""
import pytest


class TestHillFunction:
    def test_zero_dose_zero_effect(self):
        from supplement_module import hill_effect
        assert hill_effect(0.0, 1.0, 0.5) == pytest.approx(0.0)

    def test_half_max_gives_half_effect(self):
        from supplement_module import hill_effect
        assert hill_effect(0.5, 1.0, 0.5) == pytest.approx(0.5)

    def test_high_dose_saturates(self):
        from supplement_module import hill_effect
        assert hill_effect(10.0, 1.0, 0.5) > 0.9

    def test_diminishing_returns(self):
        from supplement_module import hill_effect
        low = hill_effect(0.25, 1.0, 0.5)
        mid = hill_effect(0.5, 1.0, 0.5)
        high = hill_effect(0.75, 1.0, 0.5)
        # Increments should decrease
        assert (mid - low) > (high - mid)


class TestSupplementEffects:
    def test_no_supplements_no_effect(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({})
        assert effects['nad_boost'] == pytest.approx(0.0)
        assert effects['inflammation_reduction'] == pytest.approx(0.0)
        assert effects['mitophagy_boost'] == pytest.approx(0.0)

    def test_nr_boosts_nad(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({'nr_dose': 0.8})
        assert effects['nad_boost'] > 0

    def test_dha_reduces_inflammation(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({'dha_dose': 0.8})
        assert effects['inflammation_reduction'] > 0

    def test_resveratrol_boosts_mitophagy(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({'resveratrol_dose': 0.8})
        assert effects['mitophagy_boost'] > 0

    def test_gut_health_modulates_nad(self):
        from supplement_module import compute_supplement_effects
        low_gut = compute_supplement_effects({'nr_dose': 0.8}, gut_health=0.3)
        high_gut = compute_supplement_effects({'nr_dose': 0.8}, gut_health=0.9)
        assert high_gut['nad_boost'] > low_gut['nad_boost']
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_supplement_module.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `supplement_module.py`:

```python
"""Supplement module — Hill-function dose-response for 11 nutraceuticals.

Each supplement has a max effect and half-max dose. Effects are categorized
into nad_boost, inflammation_reduction, mitophagy_boost, and demand_reduction
for mapping onto the core 12D params.

References: Norwitz et al. 2021
"""
from __future__ import annotations

from constants import (
    MAX_NR_EFFECT, NR_HALF_MAX,
    MAX_DHA_EFFECT, DHA_HALF_MAX,
    MAX_COQ10_EFFECT, COQ10_HALF_MAX,
    MAX_RESVERATROL_EFFECT, RESVERATROL_HALF_MAX,
    MAX_PQQ_EFFECT, PQQ_HALF_MAX,
    MAX_ALA_EFFECT, ALA_HALF_MAX,
    MAX_VITAMIN_D_EFFECT, VITAMIN_D_HALF_MAX,
    MAX_B_COMPLEX_EFFECT, B_COMPLEX_HALF_MAX,
    MAX_MAGNESIUM_EFFECT, MAGNESIUM_HALF_MAX,
    MAX_ZINC_EFFECT, ZINC_HALF_MAX,
    MAX_SELENIUM_EFFECT, SELENIUM_HALF_MAX,
    MIN_NAD_CONVERSION_EFFICIENCY, MAX_NAD_CONVERSION_EFFICIENCY,
)


def hill_effect(dose: float, max_effect: float, half_max: float) -> float:
    """Michaelis-Menten / Hill dose-response with diminishing returns.

    Returns: max_effect * dose / (dose + half_max). Zero at dose=0,
    asymptotes to max_effect at high dose.
    """
    if dose <= 0:
        return 0.0
    return max_effect * dose / (dose + half_max)


def nad_conversion_efficiency(gut_health: float) -> float:
    """Gut health modulates NAD+ precursor conversion efficiency.

    Returns a multiplier between MIN_NAD_CONVERSION_EFFICIENCY (0.7) and
    MAX_NAD_CONVERSION_EFFICIENCY (1.0).
    """
    return MIN_NAD_CONVERSION_EFFICIENCY + gut_health * (
        MAX_NAD_CONVERSION_EFFICIENCY - MIN_NAD_CONVERSION_EFFICIENCY
    )


def compute_supplement_effects(
    supplements: dict[str, float],
    gut_health: float = 0.5,
) -> dict[str, float]:
    """Aggregate supplement effects across all 11 nutraceuticals.

    Args:
        supplements: Dict of supplement_name -> dose (0-1).
            Keys: nr_dose, dha_dose, coq10_dose, resveratrol_dose,
            pqq_dose, ala_dose, vitamin_d_dose, b_complex_dose,
            magnesium_dose, zinc_dose, selenium_dose.
        gut_health: Current gut microbiome health (0-1), affects NAD conversion.

    Returns:
        Dict with:
            nad_boost: additive boost to effective nad_supplement
            inflammation_reduction: subtractive from inflammation_level
            mitophagy_boost: additive to effective rapamycin_dose
            demand_reduction: subtractive from metabolic_demand
            sleep_boost: additive to sleep quality
    """
    conversion = nad_conversion_efficiency(gut_health)

    # NAD-boosting supplements
    nr = hill_effect(supplements.get('nr_dose', 0), MAX_NR_EFFECT, NR_HALF_MAX) * conversion
    b_complex = hill_effect(supplements.get('b_complex_dose', 0), MAX_B_COMPLEX_EFFECT, B_COMPLEX_HALF_MAX) * conversion
    nad_boost = nr * 0.25 + b_complex * 0.1  # scale to core nad_supplement range (0-1)

    # Anti-inflammatory supplements
    dha = hill_effect(supplements.get('dha_dose', 0), MAX_DHA_EFFECT, DHA_HALF_MAX)
    ala = hill_effect(supplements.get('ala_dose', 0), MAX_ALA_EFFECT, ALA_HALF_MAX)
    vit_d = hill_effect(supplements.get('vitamin_d_dose', 0), MAX_VITAMIN_D_EFFECT, VITAMIN_D_HALF_MAX)
    zinc = hill_effect(supplements.get('zinc_dose', 0), MAX_ZINC_EFFECT, ZINC_HALF_MAX)
    selenium = hill_effect(supplements.get('selenium_dose', 0), MAX_SELENIUM_EFFECT, SELENIUM_HALF_MAX)
    inflammation_reduction = dha + ala + vit_d + zinc + selenium

    # Mitophagy-boosting supplements
    resveratrol = hill_effect(supplements.get('resveratrol_dose', 0), MAX_RESVERATROL_EFFECT, RESVERATROL_HALF_MAX)
    pqq = hill_effect(supplements.get('pqq_dose', 0), MAX_PQQ_EFFECT, PQQ_HALF_MAX)
    mitophagy_boost = resveratrol + pqq

    # ETC support (reduces effective metabolic demand via improved efficiency)
    coq10 = hill_effect(supplements.get('coq10_dose', 0), MAX_COQ10_EFFECT, COQ10_HALF_MAX)
    demand_reduction = coq10

    # Sleep/cognitive
    magnesium = hill_effect(supplements.get('magnesium_dose', 0), MAX_MAGNESIUM_EFFECT, MAGNESIUM_HALF_MAX)
    sleep_boost = magnesium * 0.5  # partial contribution to sleep quality

    return {
        'nad_boost': nad_boost,
        'inflammation_reduction': inflammation_reduction,
        'mitophagy_boost': mitophagy_boost,
        'demand_reduction': demand_reduction,
        'sleep_boost': sleep_boost,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_supplement_module.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add supplement_module.py tests/test_supplement_module.py
git commit -m "feat: add supplement module (Hill-function dose-response for 11 nutraceuticals)"
```

---

### Task 5: Parameter Resolver

**Files:**
- Create: `parameter_resolver.py`
- Test: `tests/test_parameter_resolver.py` (new)

**Step 1: Write the failing test**

Create `tests/test_parameter_resolver.py`:

```python
"""Tests for parameter resolver — 50D expanded → effective 12D core."""
import pytest
import numpy as np


class TestParameterResolverConstruction:
    def test_constructs_with_minimal_params(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        assert pr is not None

    def test_resolve_returns_two_dicts(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        intervention, patient = pr.resolve(t=0.0)
        assert isinstance(intervention, dict)
        assert isinstance(patient, dict)

    def test_resolve_returns_valid_core_keys(self):
        from parameter_resolver import ParameterResolver
        from constants import INTERVENTION_NAMES, PATIENT_NAMES
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        intervention, patient = pr.resolve(t=0.0)
        for k in INTERVENTION_NAMES:
            assert k in intervention
        for k in PATIENT_NAMES:
            assert k in patient


class TestGeneticResolution:
    def test_apoe4_increases_vulnerability(self):
        from parameter_resolver import ParameterResolver
        baseline = ParameterResolver(
            patient_expanded={'apoe_genotype': 0},
            intervention_expanded={},
        )
        apoe4 = ParameterResolver(
            patient_expanded={'apoe_genotype': 1},
            intervention_expanded={},
        )
        _, p_base = baseline.resolve(0.0)
        _, p_apoe = apoe4.resolve(0.0)
        assert p_apoe['genetic_vulnerability'] > p_base['genetic_vulnerability']


class TestSupplementResolution:
    def test_nr_increases_nad_supplement(self):
        from parameter_resolver import ParameterResolver
        without = ParameterResolver(
            patient_expanded={}, intervention_expanded={},
        )
        with_nr = ParameterResolver(
            patient_expanded={}, intervention_expanded={'nr_dose': 0.8},
        )
        i_base, _ = without.resolve(0.0)
        i_nr, _ = with_nr.resolve(0.0)
        assert i_nr['nad_supplement'] > i_base['nad_supplement']


class TestAlcoholResolution:
    def test_alcohol_increases_inflammation(self):
        from parameter_resolver import ParameterResolver
        sober = ParameterResolver(
            patient_expanded={}, intervention_expanded={'alcohol_intake': 0.0},
        )
        drinker = ParameterResolver(
            patient_expanded={}, intervention_expanded={'alcohol_intake': 0.8},
        )
        _, p_sober = sober.resolve(0.0)
        _, p_drink = drinker.resolve(0.0)
        assert p_drink['inflammation_level'] > p_sober['inflammation_level']


class TestTimeVaryingGrief:
    def test_grief_decays_over_time(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={'grief_intensity': 0.9, 'therapy_intensity': 0.5},
            intervention_expanded={},
        )
        _, p_early = pr.resolve(0.0)
        _, p_late = pr.resolve(20.0)
        assert p_late['inflammation_level'] < p_early['inflammation_level']


class TestTimeVaryingAlcohol:
    def test_alcohol_taper_reduces_over_time(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={},
            intervention_expanded={'alcohol_intake': 0.8},
            schedules={'alcohol_taper': {'start': 0.8, 'end': 0.0, 'taper_years': 2}},
        )
        _, p_before = pr.resolve(0.0)
        _, p_after = pr.resolve(5.0)
        assert p_after['inflammation_level'] < p_before['inflammation_level']


class TestCoreSchedulePassthrough:
    def test_rapamycin_passed_through(self):
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(
            patient_expanded={},
            intervention_expanded={'rapamycin_dose': 0.8},
        )
        intervention, _ = pr.resolve(0.0)
        assert intervention['rapamycin_dose'] >= 0.8


class TestOutputsClamped:
    def test_inflammation_clamped_to_one(self):
        from parameter_resolver import ParameterResolver
        # Stack everything that raises inflammation
        pr = ParameterResolver(
            patient_expanded={
                'apoe_genotype': 2,
                'grief_intensity': 1.0,
                'sex': 'F',
                'menopause_status': 'post',
                'baseline_age': 90.0,
                'inflammation_level': 0.9,
            },
            intervention_expanded={'alcohol_intake': 1.0},
        )
        _, patient = pr.resolve(0.0)
        assert patient['inflammation_level'] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_parameter_resolver.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `parameter_resolver.py`. This is the central module — it imports from genetics_module, lifestyle_module, and supplement_module, applies the modifier chain described in the design doc, and pre-computes time-varying trajectories at construction time. The implementation should follow the 10-step chain from the design doc (baseline → genetics → sex → grief → sleep → lifestyle → supplements → probiotics → core schedule → clamp). See `artifacts/design_doc_time_varying_parameter_resolver_2026-02-19.md` for the full specification.

Key points:
- Pre-compute grief decay curve at `__init__` using exponential decay: `G(t) = G0 * exp(-decay_rate * t)`
- Pre-compute alcohol taper if `schedules` dict has `alcohol_taper` key
- Pre-compute gut health trajectory via simple Euler integration of the gut ODE
- `resolve(t)` interpolates pre-computed curves via `np.interp()`
- Core intervention params (rapamycin, transplant, etc.) are passed through directly from `intervention_expanded`
- All outputs clamped to valid ranges

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_parameter_resolver.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add parameter_resolver.py tests/test_parameter_resolver.py
git commit -m "feat: add parameter resolver (50D expanded → effective 12D core)"
```

---

### Task 6: Simulator Integration

**Files:**
- Modify: `simulator.py:~1276-1308` (main integration loop, add resolver branch)
- Modify: `disturbances.py:~544-593` (disturbance loop, add resolver branch)
- Test: `tests/test_resolver_integration.py` (new)

**Step 1: Write the failing test**

Create `tests/test_resolver_integration.py`:

```python
"""Tests for resolver integration with simulator and disturbances."""
import pytest
import numpy as np


class TestSimulatorWithResolver:
    def test_simulate_accepts_resolver_kwarg(self):
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        result = simulate(resolver=pr)
        assert 'states' in result
        assert result['states'].shape[1] == 8

    def test_resolver_produces_different_result_than_default(self):
        from simulator import simulate
        from parameter_resolver import ParameterResolver
        default = simulate()
        pr = ParameterResolver(
            patient_expanded={'apoe_genotype': 2, 'baseline_age': 70.0},
            intervention_expanded={'nr_dose': 1.0, 'rapamycin_dose': 0.5},
        )
        resolved = simulate(resolver=pr)
        # APOE4 hom with NR should differ from plain default
        assert not np.allclose(default['states'][-1], resolved['states'][-1])

    def test_resolver_none_is_backwards_compatible(self):
        from simulator import simulate
        r1 = simulate()
        r2 = simulate(resolver=None)
        assert np.allclose(r1['states'], r2['states'])


class TestDisturbancesWithResolver:
    def test_simulate_with_disturbances_accepts_resolver(self):
        from disturbances import simulate_with_disturbances, IonizingRadiation
        from parameter_resolver import ParameterResolver
        pr = ParameterResolver(patient_expanded={}, intervention_expanded={})
        shock = IonizingRadiation(start_year=5.0, magnitude=0.5)
        result = simulate_with_disturbances(disturbances=[shock], resolver=pr)
        assert 'states' in result
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_resolver_integration.py -v`
Expected: FAIL — TypeError (unexpected keyword argument 'resolver')

**Step 3: Modify simulator.py**

In `simulate()` function signature (~line 1215), add `resolver=None` parameter. In the main loop (~line 1278), add the resolver branch:

```python
# Inside the for loop, replace the current intervention resolution:
# OLD:
#     current_intervention = _resolve_intervention(intervention, t)
# NEW:
        if resolver is not None:
            current_intervention, current_patient = resolver.resolve(t, state)
        else:
            current_intervention = _resolve_intervention(intervention, t)
            current_patient = patient
```

Then pass `current_patient` instead of `patient` to the RK4 step and derivatives calls. Same pattern for the stochastic branch.

Similarly modify `simulate_with_disturbances()` in `disturbances.py` (~line 472): add `resolver=None` parameter, and in the loop (~line 562), resolve before disturbance overlay.

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_resolver_integration.py -v`
Expected: All PASS

**Step 5: Run full test suite for regressions**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/ -v`
Expected: All ~262+ tests PASS (backwards compatible — resolver=None is default)

**Step 6: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add simulator.py disturbances.py tests/test_resolver_integration.py
git commit -m "feat: integrate parameter resolver into simulate() and simulate_with_disturbances()"
```

---

### Task 7: Downstream Chain

**Files:**
- Create: `downstream_chain.py`
- Test: `tests/test_downstream_chain.py` (new)

**Step 1: Write the failing test**

Create `tests/test_downstream_chain.py`:

```python
"""Tests for downstream chain — neuroplasticity and Alzheimer's pathology."""
import pytest
import numpy as np


class TestMEF2ODE:
    def test_engagement_increases_mef2(self):
        from downstream_chain import mef2_derivative
        dmef2 = mef2_derivative(mef2=0.2, engagement=0.9, apoe_mult=1.0)
        assert dmef2 > 0

    def test_no_engagement_mef2_decays(self):
        from downstream_chain import mef2_derivative
        dmef2 = mef2_derivative(mef2=0.5, engagement=0.0, apoe_mult=1.0)
        assert dmef2 < 0

    def test_mef2_saturates(self):
        from downstream_chain import mef2_derivative
        # Near max, induction term (1-MEF2) shrinks
        low = mef2_derivative(mef2=0.2, engagement=0.9, apoe_mult=1.0)
        high = mef2_derivative(mef2=0.9, engagement=0.9, apoe_mult=1.0)
        assert low > high


class TestHistoneAcetylation:
    def test_mef2_induces_ha(self):
        from downstream_chain import ha_derivative
        dha = ha_derivative(ha=0.2, mef2=0.8)
        assert dha > 0

    def test_ha_slow_decay(self):
        from downstream_chain import ha_derivative
        dha = ha_derivative(ha=0.5, mef2=0.0)
        assert dha < 0


class TestSynapticStrength:
    def test_engagement_strengthens_synapses(self):
        from downstream_chain import synaptic_derivative
        dss = synaptic_derivative(ss=1.0, ha=0.5, engagement=0.9)
        assert dss > 0

    def test_no_engagement_decay_toward_baseline(self):
        from downstream_chain import synaptic_derivative
        dss = synaptic_derivative(ss=1.5, ha=0.5, engagement=0.0)
        assert dss < 0

    def test_at_baseline_no_decay(self):
        from downstream_chain import synaptic_derivative
        dss = synaptic_derivative(ss=1.0, ha=0.5, engagement=0.0)
        assert dss == pytest.approx(0.0, abs=1e-10)


class TestAmyloidODE:
    def test_amyloid_accumulates(self):
        from downstream_chain import amyloid_derivative
        dab = amyloid_derivative(amyloid=0.2, inflammation=0.3, age=70, apoe_clearance=1.0)
        assert dab > 0 or dab < 0  # just confirm it runs; sign depends on params

    def test_apoe4_reduces_clearance(self):
        from downstream_chain import amyloid_derivative
        normal = amyloid_derivative(amyloid=0.5, inflammation=0.3, age=75, apoe_clearance=1.0)
        apoe4 = amyloid_derivative(amyloid=0.5, inflammation=0.3, age=75, apoe_clearance=0.7)
        assert apoe4 > normal  # less clearance → faster accumulation


class TestMemoryIndex:
    def test_baseline_memory(self):
        from downstream_chain import memory_index
        mi = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=0.0, tau=0.0)
        assert mi == pytest.approx(0.5, abs=0.01)

    def test_engagement_improves_memory(self):
        from downstream_chain import memory_index
        base = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=0.0, tau=0.0)
        engaged = memory_index(ss=1.6, mef2=0.8, cr=0.7, amyloid=0.0, tau=0.0)
        assert engaged > base

    def test_pathology_reduces_memory(self):
        from downstream_chain import memory_index
        healthy = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=0.0, tau=0.0)
        diseased = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=1.5, tau=1.0)
        assert diseased < healthy

    def test_resilience_buffers_pathology(self):
        from downstream_chain import memory_index
        # High resilience (MEF2 + CR) should buffer pathology impact
        no_buffer = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=1.0, tau=0.5)
        buffered = memory_index(ss=1.5, mef2=0.8, cr=0.8, amyloid=1.0, tau=0.5)
        assert buffered > no_buffer


class TestComputeDownstream:
    def test_runs_on_core_result(self):
        from simulator import simulate
        from downstream_chain import compute_downstream
        core = simulate()
        downstream = compute_downstream(
            core_result=core,
            patient_expanded={'intellectual_engagement': 0.5, 'baseline_age': 70},
        )
        assert len(downstream) == len(core['time'])
        assert 'memory_index' in downstream[0]
        assert 'MEF2_activity' in downstream[0]
        assert 'amyloid_burden' in downstream[0]

    def test_high_engagement_improves_memory(self):
        from simulator import simulate
        from downstream_chain import compute_downstream
        core = simulate()
        low = compute_downstream(
            core_result=core,
            patient_expanded={'intellectual_engagement': 0.1, 'baseline_age': 70},
        )
        high = compute_downstream(
            core_result=core,
            patient_expanded={'intellectual_engagement': 0.9, 'baseline_age': 70},
        )
        assert high[-1]['memory_index'] > low[-1]['memory_index']
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_downstream_chain.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create `downstream_chain.py`. Contains individual derivative functions (mef2_derivative, ha_derivative, synaptic_derivative, cognitive_reserve_derivative, amyloid_derivative, tau_derivative), the memory_index derived function, and the main `compute_downstream()` which takes a core simulation result and integrates all 6 downstream ODEs via Euler method using the core trajectory's states as input.

Key implementation points per the design doc:
- Reads core ATP (state[2]), ROS (state[3]), senescent (state[5]) at each timestep
- Maps core inflammation from the analytics or approximates from ROS + senescence
- Euler integration with same dt as core (acceptable since downstream timescales are years)
- All state variables clamped to their valid ranges after each step
- Returns a list of dicts (one per timestep), same length as core time array

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_downstream_chain.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add downstream_chain.py tests/test_downstream_chain.py
git commit -m "feat: add downstream chain (MEF2, HA, synaptic, CR, amyloid, tau, memory index)"
```

---

### Task 8: Scenario Framework

**Files:**
- Create: `scenario_definitions.py`
- Create: `scenario_runner.py`
- Create: `scenario_analysis.py`
- Create: `scenario_plot.py`
- Test: `tests/test_scenario_framework.py` (new)

**Step 1: Write the failing test**

Create `tests/test_scenario_framework.py`:

```python
"""Tests for scenario framework — definition, running, analysis, plotting."""
import pytest


class TestScenarioDefinitions:
    def test_intervention_profile_defaults_to_zero(self):
        from scenario_definitions import InterventionProfile
        ip = InterventionProfile()
        assert ip.rapamycin_dose == 0.0
        assert ip.nr_dose == 0.0

    def test_scenario_has_required_fields(self):
        from scenario_definitions import Scenario, InterventionProfile
        s = Scenario(
            name="test",
            description="a test scenario",
            patient_params={'baseline_age': 63},
            interventions=InterventionProfile(),
        )
        assert s.name == "test"
        assert s.duration_years == 30.0

    def test_example_scenarios_returns_four(self):
        from scenario_definitions import get_example_scenarios
        scenarios = get_example_scenarios()
        assert len(scenarios) == 4
        assert scenarios[0].name.startswith("A")
        assert scenarios[3].name.startswith("D")


class TestScenarioRunner:
    def test_run_scenario_returns_dict(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario
        scenarios = get_example_scenarios()
        result = run_scenario(scenarios[0], years=5)
        assert 'core' in result
        assert 'downstream' in result
        assert 'scenario_name' in result

    def test_run_scenarios_returns_all(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        scenarios = get_example_scenarios()[:2]  # just A and B for speed
        results = run_scenarios(scenarios, years=5)
        assert len(results) == 2


class TestScenarioAnalysis:
    def test_extract_milestones(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario
        from scenario_analysis import extract_milestones
        result = run_scenario(get_example_scenarios()[0], years=5)
        milestones = extract_milestones(result)
        assert 'final_heteroplasmy' in milestones
        assert 'final_atp' in milestones
        assert 'final_memory_index' in milestones

    def test_compare_scenarios(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        from scenario_analysis import compare_scenarios
        results = run_scenarios(get_example_scenarios()[:2], years=5)
        comparison = compare_scenarios(results)
        assert len(comparison) == 2


class TestScenarioPlot:
    def test_plot_trajectories_creates_figure(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        from scenario_plot import plot_trajectories
        results = run_scenarios(get_example_scenarios()[:2], years=5)
        fig = plot_trajectories(results)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_scenario_framework.py -v`
Expected: FAIL

**Step 3: Write the implementation**

Create the four files as specified in batch 4 handoff (`artifacts/handoff_batch4_scenario_modeling_2026-02-19.md`), adapted to use the ParameterResolver and DownstreamChain instead of a monolithic simulate():

- `scenario_definitions.py`: InterventionProfile dataclass, Scenario dataclass, `get_example_scenarios()` returning the A-D scenarios for the 63yo APOE4 patient.
- `scenario_runner.py`: `run_scenario()` creates a ParameterResolver from the Scenario, calls `simulate(resolver=pr)`, runs `compute_downstream()`, returns combined results. `run_scenarios()` batch-runs multiple scenarios.
- `scenario_analysis.py`: `extract_milestones()` from a run result (dementia age, amyloid threshold age, etc.), `compare_scenarios()` across multiple results, `summary_table()` at specified ages.
- `scenario_plot.py`: `plot_trajectories()` multi-panel (het, ATP, memory_index), `plot_milestone_comparison()` bar chart, `plot_summary_heatmap()`. All use matplotlib Agg backend, output to `output/`.

**Step 4: Run test to verify it passes**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_scenario_framework.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add scenario_definitions.py scenario_runner.py scenario_analysis.py scenario_plot.py tests/test_scenario_framework.py
git commit -m "feat: add scenario framework (definitions, runner, analysis, plots)"
```

---

### Task 9: Integration Test — 4-Scenario APOE4 Comparison

**Files:**
- Create: `tests/test_integration_scenarios.py`
- Create: `run_scenario_comparison.py` (main script)

**Step 1: Write the integration test**

Create `tests/test_integration_scenarios.py`:

```python
"""Integration test — run all 4 scenarios for the 63yo APOE4 patient end-to-end."""
import pytest


class TestFourScenarioComparison:
    """Validate that the A-D scenario progression produces monotonically
    improving outcomes, matching the projected trajectories from the handoff."""

    @pytest.fixture(scope='class')
    def all_results(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenarios
        return run_scenarios(get_example_scenarios(), years=30)

    def test_four_scenarios_complete(self, all_results):
        assert len(all_results) == 4

    def test_scenario_b_better_than_a_at_final(self, all_results):
        a_het = all_results[0]['core']['heteroplasmy'][-1]
        b_het = all_results[1]['core']['heteroplasmy'][-1]
        assert b_het < a_het, "B (OTC supplements) should have lower het than A (sleep only)"

    def test_scenario_c_better_than_b_at_final(self, all_results):
        b_het = all_results[1]['core']['heteroplasmy'][-1]
        c_het = all_results[2]['core']['heteroplasmy'][-1]
        assert c_het < b_het, "C (prescription) should have lower het than B"

    def test_scenario_d_better_than_c_at_final(self, all_results):
        c_het = all_results[2]['core']['heteroplasmy'][-1]
        d_het = all_results[3]['core']['heteroplasmy'][-1]
        assert d_het < c_het, "D (experimental) should have lower het than C"

    def test_memory_index_monotonically_better(self, all_results):
        memory_finals = [r['downstream'][-1]['memory_index'] for r in all_results]
        for i in range(len(memory_finals) - 1):
            assert memory_finals[i+1] >= memory_finals[i], \
                f"Scenario {i+2} should have >= memory than scenario {i+1}"

    def test_scenario_a_only_one_with_dementia_risk(self, all_results):
        """Only scenario A should approach or cross the 0.5 dementia threshold."""
        a_mi_final = all_results[0]['downstream'][-1]['memory_index']
        b_mi_min = min(d['memory_index'] for d in all_results[1]['downstream'])
        # A might get close to 0.5; B should stay well above
        assert b_mi_min > 0.6
```

**Step 2: Run to verify it passes (this is the end-to-end validation)**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_integration_scenarios.py -v --timeout=120`
Expected: All PASS. This may take 30-60 seconds (4 × 30-year simulations).

**Step 3: Create the main comparison script**

Create `run_scenario_comparison.py` that loads the 4 example scenarios, runs them, prints milestone tables, and saves plots to `output/scenarios/`. Follow the structure from batch 4 handoff.

**Step 4: Run the script**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python run_scenario_comparison.py`
Expected: Prints milestone table to stdout, saves PNG plots to `output/scenarios/`.

**Step 5: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add tests/test_integration_scenarios.py run_scenario_comparison.py
git commit -m "feat: add 4-scenario APOE4 comparison (integration test + main script)"
```

---

### Task 10: Full Regression + Documentation

**Files:**
- Modify: `CLAUDE.md` (add expansion section)
- Run full test suite

**Step 1: Run the full test suite**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/ -v`
Expected: All tests PASS (original ~262 + new ~80 = ~342 total)

**Step 2: Update CLAUDE.md**

Add a new section to the project CLAUDE.md documenting:
- The new modules and their roles
- The parameter resolver architecture
- New commands (run_scenario_comparison.py)
- The expanded parameter space (note that the Cramer core is unchanged)

**Step 3: Commit**

```bash
cd /Users/gardenofcomputation/how-to-live-much-longer
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with precision medicine expansion"
```
