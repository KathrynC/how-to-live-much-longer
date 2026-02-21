# Split Mutation Types (C11) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `N_damaged` into `N_deletion` (exponential growth, replication advantage, drives cliff) and `N_point` (linear growth, no advantage, ~33% from ROS) per John Cramer's email correction (C11), based on Appendix 2 pp.152-155 and Va23 data.

**Architecture:** Append `N_point` at state index 7, rename `N_damaged` (index 1) to `N_deletion`. Indices 0-6 unchanged, minimizing blast radius. Cliff driven by deletion heteroplasmy only. ROS coupling weakened to ~33% of previous (point mutations only). Total heteroplasmy reported for backwards compatibility.

**Tech Stack:** Python 3.11+, numpy (no scipy), matplotlib Agg backend. TDD throughout.

**Design doc:** `docs/plans/2026-02-17-split-mutation-types-design.md`

**Working directory:** `/Users/gardenofcomputation/how-to-live-much-longer/`

---

### Task 1: Constants & Heteroplasmy Functions

**Files:**
- Modify: `constants.py:374-384` (STATE_NAMES, N_STATES)
- Modify: `constants.py:87-163` (add new biological constants)
- Modify: `simulator.py:150-165` (heteroplasmy functions)
- Modify: `tests/test_simulator.py` (add new tests)

**Step 1: Write failing tests for new state vector and heteroplasmy functions**

Add to `tests/test_simulator.py` at the bottom:

```python
from simulator import _total_heteroplasmy, _deletion_heteroplasmy


class TestMutationTypeSplit:
    """Tests for C11: split N_damaged into N_point + N_deletion."""

    def test_state_vector_is_8d(self):
        """State vector should have 8 variables after C11 split."""
        from constants import N_STATES
        assert N_STATES == 8

    def test_state_names_has_n_deletion(self):
        from constants import STATE_NAMES
        assert "N_deletion" in STATE_NAMES
        assert STATE_NAMES[1] == "N_deletion"

    def test_state_names_has_n_point(self):
        from constants import STATE_NAMES
        assert "N_point" in STATE_NAMES
        assert STATE_NAMES[7] == "N_point"

    def test_total_heteroplasmy_sums_both(self):
        """Total het = (N_del + N_pt) / (N_h + N_del + N_pt)."""
        het = _total_heteroplasmy(0.7, 0.2, 0.1)
        assert het == pytest.approx(0.3, abs=1e-10)

    def test_deletion_heteroplasmy_ignores_point(self):
        """Deletion het = N_del / total (point mutations don't drive cliff)."""
        het = _deletion_heteroplasmy(0.7, 0.2, 0.1)
        assert het == pytest.approx(0.2, abs=1e-10)

    def test_total_het_greater_than_deletion_het(self):
        het_total = _total_heteroplasmy(0.7, 0.2, 0.1)
        het_del = _deletion_heteroplasmy(0.7, 0.2, 0.1)
        assert het_total > het_del

    def test_heteroplasmy_edge_case_zero_copies(self):
        assert _total_heteroplasmy(0.0, 0.0, 0.0) == 1.0
        assert _deletion_heteroplasmy(0.0, 0.0, 0.0) == 1.0

    def test_heteroplasmy_no_damage(self):
        assert _total_heteroplasmy(1.0, 0.0, 0.0) == 0.0
        assert _deletion_heteroplasmy(1.0, 0.0, 0.0) == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_simulator.py::TestMutationTypeSplit -v 2>&1 | head -30`

Expected: FAIL — `ImportError: cannot import name '_total_heteroplasmy'`

**Step 3: Update constants.py**

In `constants.py`, after the existing biological constants block (around line 179), add:

```python
# ── C11: Split mutation type constants (Cramer email 2026-02-17) ────────────
# Point mutations: Pol gamma errors + ROS-induced transitions. Linear growth,
# no replication advantage (same-length mtDNA as wild-type).
# Deletion mutations: Pol gamma slippage + DSB misrepair. Exponential growth,
# size-dependent replication advantage (shorter rings replicate faster).
# Reference: Cramer Appendix 2 pp.152-155, Va23 (Vandiver et al. 2023).

# Point mutation dynamics
POINT_ERROR_RATE = 0.001           # Pol gamma error rate per replication event
ROS_POINT_COEFF = 0.05             # ~33% of old 0.15 damage_rate coefficient
POINT_MITOPHAGY_SELECTIVITY = 0.3  # low: point-mutated mitos often have normal delta-psi

# Deletion replication advantage (REVISED from DAMAGED_REPLICATION_ADVANTAGE)
# Appendix 2 pp.154-155: "at least 21% faster" for >3kbp deletions (Va23).
# Raised from 1.05 to 1.21 — still compliance-aligned with book minimum 1.21.
DELETION_REPLICATION_ADVANTAGE = 1.21

# Age-dependent deletion fraction of total damage (for initial state)
# Young adults: mostly point mutations (deletions haven't accumulated yet).
# Older adults: dominated by deletions (exponential growth catches up).
DELETION_FRACTION_YOUNG = 0.4      # age 20: 40% of damage is deletions
DELETION_FRACTION_OLD = 0.8        # age 90: 80% of damage is deletions
```

Then update `STATE_NAMES` (around line 374):

```python
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
```

Keep the old `DAMAGED_REPLICATION_ADVANTAGE = 1.05` line but add a deprecation comment:

```python
# DEPRECATED (C11): Use DELETION_REPLICATION_ADVANTAGE instead.
# Kept for reference; deletion advantage raised from 1.05 to 1.21.
# DAMAGED_REPLICATION_ADVANTAGE = 1.05
```

**Step 4: Add heteroplasmy functions to simulator.py**

Replace the old `_heteroplasmy_fraction` function (lines 150-155) with three functions:

```python
def _heteroplasmy_fraction(n_healthy: float, n_damaged: float) -> float:
    """DEPRECATED: Use _total_heteroplasmy or _deletion_heteroplasmy instead.

    Kept for backwards compatibility with external callers.
    """
    total = n_healthy + n_damaged
    if total < 1e-12:
        return 1.0
    return n_damaged / total


def _total_heteroplasmy(n_healthy: float, n_deletion: float, n_point: float) -> float:
    """Total heteroplasmy: (N_del + N_pt) / total. For reporting."""
    total = n_healthy + n_deletion + n_point
    if total < 1e-12:
        return 1.0
    return (n_deletion + n_point) / total


def _deletion_heteroplasmy(n_healthy: float, n_deletion: float, n_point: float) -> float:
    """Deletion heteroplasmy: N_del / total. Drives the cliff factor."""
    total = n_healthy + n_deletion + n_point
    if total < 1e-12:
        return 1.0
    return n_deletion / total
```

Also update the imports at the top of `simulator.py` to include the new constants:

```python
from constants import (
    SIM_YEARS, DT, N_STATES,
    HETEROPLASMY_CLIFF, CLIFF_STEEPNESS,
    DOUBLING_TIME_YOUNG, DOUBLING_TIME_OLD, AGE_TRANSITION,
    BASELINE_ATP, BASELINE_ROS, ROS_PER_DAMAGED,
    BASELINE_NAD, NAD_DECLINE_RATE,
    BASELINE_MEMBRANE_POTENTIAL, BASELINE_SENESCENT, SENESCENCE_RATE,
    BASELINE_MITOPHAGY_RATE,
    DELETION_REPLICATION_ADVANTAGE,
    CD38_BASE_SURVIVAL, CD38_SUPPRESSION_GAIN,
    TRANSPLANT_ADDITION_RATE, TRANSPLANT_DISPLACEMENT_RATE, TRANSPLANT_HEADROOM,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    TISSUE_PROFILES,
    POINT_ERROR_RATE, ROS_POINT_COEFF, POINT_MITOPHAGY_SELECTIVITY,
    DELETION_FRACTION_YOUNG, DELETION_FRACTION_OLD,
)
```

Remove `DAMAGED_REPLICATION_ADVANTAGE` from the import (it's replaced by `DELETION_REPLICATION_ADVANTAGE`).

**Step 5: Run tests**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_simulator.py::TestMutationTypeSplit -v`

Expected: All 8 new tests PASS.

**Step 6: Commit**

```bash
git add constants.py simulator.py tests/test_simulator.py
git commit -m "feat(C11): add split mutation constants and heteroplasmy functions

New STATE_NAMES with N_deletion (idx 1) and N_point (idx 7).
_total_heteroplasmy() and _deletion_heteroplasmy() for the split.
DELETION_REPLICATION_ADVANTAGE raised from 1.05 to 1.21 (book: >=1.21 (strict compliance enforced)).
ROS_POINT_COEFF = 0.05 (~33% of old 0.15 damage_rate).

Per John Cramer email 2026-02-17, Appendix 2 pp.152-155."
```

---

### Task 2: Core ODE — derivatives() and initial_state()

**Files:**
- Modify: `simulator.py:205-469` (derivatives function — FULL REWRITE)
- Modify: `simulator.py:488-525` (initial_state)
- Modify: `tests/test_simulator.py` (add C11 dynamics tests)

**Step 1: Write failing tests for split dynamics**

Add to `tests/test_simulator.py` inside `TestMutationTypeSplit`:

```python
    def test_initial_state_8d(self):
        """initial_state should return 8D vector."""
        state = initial_state(DEFAULT_PATIENT)
        assert state.shape == (8,)

    def test_initial_state_split_copies(self):
        """N_h + N_del + N_pt should sum to ~1.0."""
        state = initial_state(DEFAULT_PATIENT)
        total = state[0] + state[1] + state[7]
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_initial_deletion_fraction_age_dependent(self):
        """Older patients should have higher deletion fraction."""
        young_p = dict(DEFAULT_PATIENT, baseline_age=30.0, baseline_heteroplasmy=0.3)
        old_p = dict(DEFAULT_PATIENT, baseline_age=80.0, baseline_heteroplasmy=0.3)
        young_state = initial_state(young_p)
        old_state = initial_state(old_p)
        # Old should have more deletions relative to point
        young_del_frac = young_state[1] / (young_state[1] + young_state[7])
        old_del_frac = old_state[1] / (old_state[1] + old_state[7])
        assert old_del_frac > young_del_frac

    def test_derivatives_returns_8d(self):
        """derivatives() should return 8-element array."""
        state = initial_state(DEFAULT_PATIENT)
        deriv = derivatives(state, 0.0, DEFAULT_INTERVENTION, DEFAULT_PATIENT)
        assert deriv.shape == (8,)

    def test_simulate_returns_8d_states(self):
        """simulate() should produce (n_steps+1, 8) states array."""
        result = simulate(sim_years=1)
        assert result["states"].shape[1] == 8

    def test_simulate_has_deletion_heteroplasmy(self):
        """Result should include both total and deletion heteroplasmy."""
        result = simulate(sim_years=1)
        assert "heteroplasmy" in result
        assert "deletion_heteroplasmy" in result
        # Deletion het should be <= total het at all times
        assert np.all(result["deletion_heteroplasmy"] <= result["heteroplasmy"] + 1e-10)

    def test_deletions_grow_faster_than_points(self):
        """Over 30 years, deletion fraction should increase (exponential advantage)."""
        result = simulate(sim_years=30)
        n_del_0 = result["states"][0, 1]
        n_pt_0 = result["states"][0, 7]
        n_del_f = result["states"][-1, 1]
        n_pt_f = result["states"][-1, 7]
        if n_pt_0 > 1e-6 and n_del_0 > 1e-6:
            del_growth = n_del_f / n_del_0
            pt_growth = n_pt_f / max(n_pt_0, 1e-6)
            assert del_growth > pt_growth

    def test_cliff_driven_by_deletions_not_points(self):
        """High deletion het should collapse ATP; high point het alone should not."""
        # Patient with all damage as deletions (high cliff risk)
        p_del = dict(DEFAULT_PATIENT, baseline_heteroplasmy=0.75)
        r_del = simulate(patient=p_del, sim_years=5)
        # Same total het but check that cliff behavior exists
        assert r_del["deletion_heteroplasmy"][-1] > 0.6
        # ATP should be low (cliff crossed)
        assert r_del["states"][-1, 2] < 0.3

    def test_point_mutations_no_replication_advantage(self):
        """Point mutations should not have the exponential growth deletions show."""
        result = simulate(sim_years=30)
        # Over time, deletion fraction of total damage should increase
        n_del_mid = result["states"][1500, 1]  # year 15
        n_pt_mid = result["states"][1500, 7]
        n_del_end = result["states"][-1, 1]
        n_pt_end = result["states"][-1, 7]
        if n_del_mid > 1e-6 and n_pt_mid > 1e-6:
            del_frac_mid = n_del_mid / (n_del_mid + n_pt_mid)
            del_frac_end = n_del_end / (n_del_end + n_pt_end)
            assert del_frac_end >= del_frac_mid - 0.05  # deletion fraction grows or holds
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_simulator.py::TestMutationTypeSplit::test_initial_state_8d -v 2>&1 | head -20`

Expected: FAIL — initial_state returns 7D not 8D

**Step 3: Rewrite `initial_state()` in simulator.py**

Replace the existing `initial_state` function (lines ~488-525):

```python
def initial_state(patient: dict[str, float]) -> npt.NDArray[np.float64]:
    """Compute initial state vector from patient parameters.

    Total copy number N_h + N_del + N_pt = 1.0 (normalized). The split between
    deletion and point mutations is age-dependent: older patients have a higher
    fraction of deletions (exponential growth catches up over decades).

    State vector (8D):
        [0] N_healthy, [1] N_deletion, [2] ATP, [3] ROS, [4] NAD,
        [5] Senescent_fraction, [6] Membrane_potential, [7] N_point

    Args:
        patient: Dict with patient parameter values.

    Returns:
        np.array of shape (8,) — initial state.
    """
    het0 = patient["baseline_heteroplasmy"]
    nad0 = patient["baseline_nad_level"]
    met_demand = patient["metabolic_demand"]
    age = patient["baseline_age"]

    # Age-dependent deletion fraction (C11: Cramer Appendix 2)
    # Young adults: mostly point mutations; old adults: dominated by deletions.
    age_frac = min(max(age - 20.0, 0.0) / 70.0, 1.0)
    deletion_frac = (DELETION_FRACTION_YOUNG
                     + (DELETION_FRACTION_OLD - DELETION_FRACTION_YOUNG) * age_frac)

    n_h0 = 1.0 - het0
    n_del0 = het0 * deletion_frac
    n_pt0 = het0 * (1.0 - deletion_frac)

    # Deletion heteroplasmy for cliff factor
    het_del0 = _deletion_heteroplasmy(n_h0, n_del0, n_pt0)
    cliff0 = _cliff_factor(het_del0)

    # Senescence: accumulates with age
    sen0 = BASELINE_SENESCENT + 0.005 * max(age - 40, 0)
    sen0 = min(sen0, 0.5)

    # ATP: consistent with dynamics equilibrium formula
    atp0 = (BASELINE_ATP * cliff0
            * (0.6 + 0.4 * min(nad0, 1.0))
            * (1.0 - 0.15 * sen0))

    # ROS: uses total het (both damage types produce ROS from defective ETC)
    ros0 = (BASELINE_ROS * met_demand
            + ROS_PER_DAMAGED * het0 * het0) / (1.0 + 0.4 * min(nad0, 1.0))

    # Membrane potential
    psi0 = cliff0 * min(nad0, 1.0) * (1.0 - 0.3 * sen0)
    psi0 = min(psi0, BASELINE_MEMBRANE_POTENTIAL)

    return np.array([n_h0, n_del0, atp0, ros0, nad0, min(sen0, 1.0), psi0, n_pt0])
```

**Step 4: Rewrite `derivatives()` in simulator.py**

Replace the existing `derivatives` function (lines ~205-469). The full replacement:

```python
def derivatives(
    state: npt.NDArray[np.float64],
    t: float,
    intervention: dict[str, float],
    patient: dict[str, float],
    tissue_mods: dict[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Compute time derivatives of all 8 state variables.

    C11 split (Cramer email 2026-02-17, Appendix 2 pp.152-155):
      - N_damaged split into N_deletion (index 1) and N_point (index 7)
      - Deletions: exponential growth with replication advantage (drives cliff)
      - Point mutations: linear growth, no replication advantage
      - ROS coupling weakened: ~33% of old coefficient, feeds only point mutations
      - Cliff factor based on deletion heteroplasmy only

    Previous fixes preserved:
      C1: Cliff feeds back into replication and apoptosis
      C2: Total copy number regulated (N_h + N_del + N_pt → 1.0)
      C3: NAD selectively benefits healthy mitochondria
      C4: Deletion replication advantage creates bistability past cliff
      C7: CD38 degrades NMN/NR
      C8: Transplant is primary rejuvenation (displaces deletions)
      M1: Yamanaka gated by ATP

    Args:
        state: np.array of shape (8,) — current state.
        t: Current time in years from simulation start.
        intervention: Dict with 6 intervention parameter values.
        patient: Dict with 6 patient parameter values.
        tissue_mods: Optional tissue-specific modifiers.

    Returns:
        np.array of shape (8,) — derivatives (dstate/dt).
    """
    if tissue_mods is None:
        tissue_mods = _DEFAULT_TISSUE_MODS
    n_h, n_del, atp, ros, nad, sen, psi, n_pt = state

    # Prevent negative values in derivative computation
    n_h = max(n_h, 1e-6)
    n_del = max(n_del, 1e-6)
    n_pt = max(n_pt, 0.0)
    atp = max(atp, 0.0)
    ros = max(ros, 0.0)
    nad = max(nad, 0.0)
    sen = max(sen, 0.0)
    psi = max(psi, 0.0)

    # Unpack intervention
    rapa = intervention["rapamycin_dose"]
    nad_supp = intervention["nad_supplement"]
    seno = intervention["senolytic_dose"]
    yama = intervention["yamanaka_intensity"]
    transplant = intervention["transplant_rate"]
    exercise = intervention["exercise_level"]

    # CD38 survival factor (C7)
    cd38_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * nad_supp

    # Unpack patient
    age = patient["baseline_age"] + t
    gen_vuln = patient["genetic_vulnerability"]
    met_demand = patient["metabolic_demand"]
    inflammation = patient["inflammation_level"]

    # ── Derived quantities ────────────────────────────────────────────────
    total = n_h + n_del + n_pt
    het_del = n_del / total                    # deletion heteroplasmy (drives cliff)
    het_total = (n_del + n_pt) / total         # total heteroplasmy (for ROS equation)
    cliff = _cliff_factor(het_del)             # C11: cliff from DELETION het only
    atp_norm = min(atp / BASELINE_ATP, 1.0)
    energy_available = max(atp_norm, 0.05)

    # ── Copy number regulation (C2, now across 3 pools) ──────────────────
    copy_number_pressure = max(1.0 - total, -0.5)

    # ── Age- and health-dependent deletion rate (C10) ────────────────────
    _current_mitophagy_rate = (BASELINE_MITOPHAGY_RATE
                               + rapa * 0.08
                               + nad_supp * 0.03 * cd38_survival)
    del_rate = _deletion_rate(age, gen_vuln, atp_norm=energy_available,
                              mitophagy_rate=_current_mitophagy_rate)

    # ── 1. dN_healthy/dt ─────────────────────────────────────────────────
    base_replication_rate = 0.1
    replication_h = (base_replication_rate * n_h * nad
                     * energy_available * max(copy_number_pressure, 0.0))

    # C11: ROS-induced damage creates POINT mutations only (~33% of old rate).
    # ROS does NOT cause deletions — deletions arise from replication errors.
    ros_point_damage = (ROS_POINT_COEFF * ros * gen_vuln * n_h
                        * tissue_mods["ros_sensitivity"])

    # Transplant: adds healthy AND displaces deletions (C8)
    transplant_headroom = max(TRANSPLANT_HEADROOM - total, 0.0)
    transplant_add = (transplant * TRANSPLANT_ADDITION_RATE
                      * min(transplant_headroom, 1.0))
    transplant_displace = (transplant * TRANSPLANT_DISPLACEMENT_RATE
                           * n_del * energy_available)

    # Yamanaka repair: converts damaged → healthy (M1: gated by ATP)
    # Deletions harder to repair than point mutations.
    repair_deletion = yama * 0.05 * n_del * energy_available
    repair_point = yama * 0.02 * n_pt * energy_available

    # Exercise biogenesis (M5)
    exercise_biogenesis = (exercise * 0.03 * energy_available
                           * max(copy_number_pressure, 0.0)
                           * tissue_mods["biogenesis_rate"])

    # Apoptosis (C1: cliff feedback)
    apoptosis_h = 0.02 * max(1.0 - energy_available, 0.0) * n_h * (1.0 - cliff)

    dn_h = (replication_h - ros_point_damage + transplant_add
            + repair_deletion + repair_point + exercise_biogenesis - apoptosis_h)

    # ── 2. dN_deletion/dt (C11: REVISED from dN_damaged) ────────────────
    # Deletions replicate FASTER (exponential growth, C4 bistability).
    # Cramer Appendix 2 pp.154-155: deleted mtDNA rings replicate "at least
    # 21% faster" (Va23). Using conservative 1.21 (raised from 1.05).
    replication_del = (base_replication_rate * DELETION_REPLICATION_ADVANTAGE
                       * n_del * nad * energy_available
                       * max(copy_number_pressure, 0.0))

    # De novo deletions from Pol γ replication slippage and DSB misrepair.
    # C11: NOT from ROS. Deletions arise during mtDNA replication, proportional
    # to the healthy pool (errors in copying healthy copies).
    age_deletions = del_rate * 0.05 * n_h * energy_available

    # Mitophagy: selective for deletions (low ΔΨ → PINK1 pathway, C3)
    mitophagy_del = _current_mitophagy_rate * n_del

    # Apoptosis
    apoptosis_del = (0.02 * max(1.0 - energy_available, 0.0)
                     * n_del * (1.0 - cliff))

    dn_del = (replication_del + age_deletions
              - mitophagy_del - repair_deletion - apoptosis_del
              - transplant_displace)

    # ── 3. dN_point/dt (C11: NEW) ───────────────────────────────────────
    # Point mutations replicate at SAME rate as healthy (no advantage).
    replication_pt = (base_replication_rate * n_pt * nad
                      * energy_available * max(copy_number_pressure, 0.0))

    # New point mutations from Pol γ replication errors during healthy copying.
    point_from_replication = POINT_ERROR_RATE * replication_h

    # ROS-induced point mutations (ros_point_damage subtracted from N_h above)
    # — this IS the 33% of old damage_rate, already computed.

    # Mitophagy: LOW selectivity for point mutations (they often have normal ΔΨ)
    mitophagy_pt = (_current_mitophagy_rate * POINT_MITOPHAGY_SELECTIVITY
                    * n_pt)

    # Apoptosis
    apoptosis_pt = (0.02 * max(1.0 - energy_available, 0.0)
                    * n_pt * (1.0 - cliff))

    dn_pt = (replication_pt + point_from_replication + ros_point_damage
             - mitophagy_pt - repair_point - apoptosis_pt)

    # ── 4. dATP/dt (unchanged logic, uses DELETION cliff) ───────────────
    atp_target = (BASELINE_ATP * cliff
                  * (0.6 + 0.4 * min(nad, 1.0))
                  * (1.0 - 0.15 * sen))
    yama_cost = yama * (0.15 + 0.2 * yama)
    exercise_cost = exercise * 0.03
    atp_target = max(atp_target - yama_cost - exercise_cost, 0.0)
    datp = 1.0 * (atp_target - atp)

    # ── 5. dROS/dt (C11: uses total het, both damage types produce ROS) ─
    ros_baseline = BASELINE_ROS * met_demand
    ros_from_damage = (ROS_PER_DAMAGED * het_total * het_total
                       * (1.0 + inflammation))
    defense_factor = 1.0 + 0.4 * min(nad, 1.0)
    defense_factor += exercise * 0.2
    exercise_ros = exercise * 0.03
    ros_eq = (ros_baseline + ros_from_damage + exercise_ros) / defense_factor
    dros = 1.0 * (ros_eq - ros)

    # ── 6. dNAD/dt (unchanged) ──────────────────────────────────────────
    age_factor = max(1.0 - NAD_DECLINE_RATE * max(age - 30, 0), 0.2)
    nad_boost = nad_supp * 0.25 * cd38_survival
    nad_target = BASELINE_NAD * age_factor + nad_boost
    nad_target = min(nad_target, 1.2)
    ros_drain = 0.03 * ros
    yama_drain = yama * 0.03
    dnad = 0.3 * (nad_target - nad) - ros_drain - yama_drain

    # ── 7. dSenescent/dt (unchanged) ───────────────────────────────────
    energy_stress = max(1.0 - energy_available, 0.0)
    new_sen = (SENESCENCE_RATE * (1.0 + 2.0 * ros + energy_stress)
               * (1.0 + 0.01 * max(age - 40, 0)))
    clearance = seno * 0.2 * sen
    immune_clear = 0.01 * sen * max(1.0 - 0.01 * max(age - 50, 0), 0.1)
    if sen >= 1.0:
        new_sen = 0.0
    dsen = new_sen - clearance - immune_clear

    # ── 8. dΔΨ/dt (unchanged) ──────────────────────────────────────────
    psi_eq = cliff * min(nad, 1.0) * (1.0 - 0.3 * sen)
    psi_eq = min(psi_eq, BASELINE_MEMBRANE_POTENTIAL)
    dpsi = 0.5 * (psi_eq - psi)

    return np.array([dn_h, dn_del, datp, dros, dnad, dsen, dpsi, dn_pt])
```

**Step 5: Update simulate() in simulator.py**

Update the main integration loop in `simulate()` to compute both heteroplasmy arrays.

In the array allocation section (around line 600), change:

```python
    het_arr = np.zeros(n_steps + 1)
```

to:

```python
    het_arr = np.zeros(n_steps + 1)
    del_het_arr = np.zeros(n_steps + 1)
```

Update the initial recording (around line 606):

```python
    states[0] = state
    het_arr[0] = _total_heteroplasmy(state[0], state[1], state[7])
    del_het_arr[0] = _deletion_heteroplasmy(state[0], state[1], state[7])
```

Update the per-step recording (around line 633):

```python
        het_arr[i + 1] = _total_heteroplasmy(state[0], state[1], state[7])
        del_het_arr[i + 1] = _deletion_heteroplasmy(state[0], state[1], state[7])
```

Add `deletion_heteroplasmy` to the return dict:

```python
    return {
        "time": time_arr,
        "states": states,
        "heteroplasmy": het_arr,
        "deletion_heteroplasmy": del_het_arr,
        "intervention": intervention,
        "patient": patient,
        "tissue_type": tissue_type,
    }
```

Also update the stochastic mode. In `_simulate_stochastic()`:

- Add `all_del_het = np.zeros((n_trajectories, n_steps + 1))` to pre-allocation
- Update initial recording: `all_het[traj, 0] = _total_heteroplasmy(...)` and `all_del_het[traj, 0] = _deletion_heteroplasmy(...)`
- Update per-step: same pattern
- Add noise for N_point: `noise[7] = noise_scale * state[7] * dW[7]` (in addition to existing noise[1] and noise[3])
- Update return dict to include `"deletion_heteroplasmy": all_del_het`

In single stochastic mode within `simulate()`, add the same noise line for index 7.

**Step 6: Run tests**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_simulator.py::TestMutationTypeSplit -v`

Expected: All 18 tests PASS (8 from Task 1 + 10 from this step).

**Step 7: Commit**

```bash
git add simulator.py tests/test_simulator.py
git commit -m "feat(C11): split derivatives into N_deletion + N_point dynamics

Deletions: exponential growth (1.21x advantage), de novo from Pol gamma
slippage. Point: linear growth, no advantage, 33% from ROS.
Cliff driven by deletion heteroplasmy only. ROS coupling weakened.
Result dict includes both 'heteroplasmy' (total) and
'deletion_heteroplasmy' (cliff-driving).

Per John Cramer email 2026-02-17, Appendix 2 pp.152-155."
```

---

### Task 3: Update Disturbances

**Files:**
- Modify: `disturbances.py:134-198` (IonizingRadiation), `285-312` (ChemotherapyBurst)
- Modify: `disturbances.py:400-552` (simulate_with_disturbances)
- Modify: `tests/test_resilience.py:58-98` (impulse assertions)

**Step 1: Write failing tests**

Update the hardcoded 7D state arrays in `tests/test_resilience.py` to 8D, and add split-damage assertions. In `TestDisturbanceClasses`:

Replace the state array in `test_ionizing_radiation_modifies_state`:

```python
    def test_ionizing_radiation_modifies_state(self):
        shock = IonizingRadiation(start_year=0.0, magnitude=1.0)
        #                        N_h  N_del ATP  ROS  NAD  Sen  ΔΨ   N_pt
        state = np.array([       0.8, 0.15, 1.0, 0.1, 0.8, 0.05, 0.9, 0.05])
        modified = shock.modify_state(state, 0.0)
        # Should transfer healthy → deletion + point
        assert modified[0] < state[0]
        assert modified[1] > state[1]  # deletions increase
        assert modified[7] > state[7]  # point mutations increase
        # Conservation: total mtDNA unchanged
        total_before = state[0] + state[1] + state[7]
        total_after = modified[0] + modified[1] + modified[7]
        assert abs(total_after - total_before) < 1e-10
        # ROS should increase
        assert modified[3] > state[3]
```

Similarly update all other test methods that create 7-element state arrays to use 8 elements (add `0.05` at the end for N_point).

**Step 2: Update IonizingRadiation.modify_state() in disturbances.py**

```python
    def modify_state(self, state, t):
        state = state.copy()
        # IMPULSE: Direct mtDNA damage from ionizing radiation.
        # Radiation causes both deletion mutations (from double-strand breaks,
        # ~70%) and point mutations (from oxidative base damage, ~30%).
        damage_fraction = 0.05 * self.magnitude
        transfer = damage_fraction * state[0]
        state[0] -= transfer
        state[1] += transfer * 0.7   # 70% → deletions (DSBs)
        state[7] += transfer * 0.3   # 30% → point mutations (base damage)
        # IMPULSE: Acute ROS burst
        state[3] += 0.15 * self.magnitude
        return state
```

**Step 3: Update ChemotherapyBurst.modify_state()**

```python
    def modify_state(self, state, t):
        state = state.copy()
        # IMPULSE: Direct mtDNA damage from cytotoxic agents.
        # Chemo causes deletion mutations (from crosslinks/intercalation, ~70%)
        # and point mutations (from alkylation/adducts, ~30%).
        damage_fraction = 0.1 * self.magnitude
        transfer = damage_fraction * state[0]
        state[0] -= transfer
        state[1] += transfer * 0.7   # 70% → deletions
        state[7] += transfer * 0.3   # 30% → point mutations
        # IMPULSE: Massive ROS burst
        state[3] += 0.3 * self.magnitude
        # IMPULSE: NAD+ crash
        state[4] *= (1.0 - 0.25 * self.magnitude)
        # IMPULSE: Membrane potential collapse
        state[6] *= (1.0 - 0.15 * self.magnitude)
        return state
```

**Step 4: Update simulate_with_disturbances()**

Update the imports at the top of `disturbances.py`:

```python
from simulator import (
    derivatives, initial_state, _resolve_intervention,
    _total_heteroplasmy, _deletion_heteroplasmy,
)
```

In `simulate_with_disturbances()`, add `del_het_arr`:

```python
    het_arr = np.zeros(n_steps + 1)
    del_het_arr = np.zeros(n_steps + 1)
```

Update heteroplasmy recording (2 places):

```python
    het_arr[0] = _total_heteroplasmy(state[0], state[1], state[7])
    del_het_arr[0] = _deletion_heteroplasmy(state[0], state[1], state[7])
```

and:

```python
        het_arr[i + 1] = _total_heteroplasmy(state[0], state[1], state[7])
        del_het_arr[i + 1] = _deletion_heteroplasmy(state[0], state[1], state[7])
```

Add to return dict:

```python
        "deletion_heteroplasmy": del_het_arr,
```

Also update the docstring to say shape `(n_steps+1, 8)` instead of 7.

**Step 5: Run resilience tests**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_resilience.py -v`

Expected: All tests PASS (some existing tests may need shape assertions updated — see Task 5).

**Step 6: Commit**

```bash
git add disturbances.py tests/test_resilience.py
git commit -m "feat(C11): split damage transfer in disturbances (70% del, 30% pt)

IonizingRadiation and ChemotherapyBurst now split damage between
deletion mutations (DSBs, crosslinks) and point mutations (base
damage, adducts). simulate_with_disturbances reports both
total and deletion heteroplasmy."
```

---

### Task 4: Update Analytics

**Files:**
- Modify: `analytics.py:130-192` (compute_damage)
- Modify: `tests/test_simulator.py` or create new test section

**Step 1: Write failing test**

Add to `TestMutationTypeSplit` in `tests/test_simulator.py`:

```python
    def test_analytics_includes_deletion_metrics(self):
        from analytics import compute_all
        result = simulate(sim_years=10)
        baseline = simulate(sim_years=10)
        analytics = compute_all(result, baseline)
        damage = analytics["damage"]
        assert "deletion_het_final" in damage
        assert "deletion_het_initial" in damage
        assert damage["deletion_het_final"] <= damage["het_final"]
```

**Step 2: Update compute_damage() in analytics.py**

Add deletion heteroplasmy metrics. After the existing `het` variable assignment, add:

```python
    # C11: Deletion-specific heteroplasmy (drives the cliff)
    del_het = result.get("deletion_heteroplasmy", het)  # fallback for old results
```

Then add to the return dict:

```python
        "deletion_het_initial": float(del_het[0]),
        "deletion_het_final": float(del_het[-1]),
        "deletion_het_max": float(np.max(del_het)),
```

**Step 3: Run test**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/test_simulator.py::TestMutationTypeSplit::test_analytics_includes_deletion_metrics -v`

Expected: PASS.

**Step 4: Commit**

```bash
git add analytics.py tests/test_simulator.py
git commit -m "feat(C11): add deletion-specific damage metrics to analytics"
```

---

### Task 5: Fix Existing Tests & Downstream Files

**Files:**
- Modify: `tests/test_simulator.py` (shape assertions)
- Modify: `tests/test_resilience.py` (all 7-element state arrays → 8)
- Modify: `tests/test_grief_bridge.py` (state array size, ROS index)
- Modify: `causal_surgery.py` (het callsites)
- Modify: `multi_tissue_sim.py` (state index access)

**Step 1: Fix test_simulator.py shape assertions**

Update all `(3001, 7)` shape checks to `(3001, 8)`:

- `TestBasicSimulation.test_no_intervention_aging`: `assert result["states"].shape == (3001, 8)`
- `TestTissueTypes.test_tissue_runs`: `assert result["states"].shape == (3001, 8)`
- `TestStochastic.test_multi_trajectory`: `assert r["states"].shape == (10, 3001, 8)`

Update `TestInitialState.test_total_copies_normalized`:

```python
    def test_total_copies_normalized(self, default_patient):
        """N_healthy + N_deletion + N_point should sum to ~1.0."""
        state = initial_state(default_patient)
        assert state[0] + state[1] + state[7] == pytest.approx(1.0, abs=1e-10)
```

**Step 2: Fix test_resilience.py state arrays**

Every hardcoded 7-element `np.array([...])` in the test file needs an 8th element (N_point). Find and replace each one:

```python
# Old:
state = np.array([0.8, 0.2, 1.0, 0.1, 0.8, 0.05, 0.9])
# New (added N_point=0.05 at end):
state = np.array([0.8, 0.15, 1.0, 0.1, 0.8, 0.05, 0.9, 0.05])
```

Note: N_deletion (index 1) reduced from 0.2 to 0.15, and N_point (index 7) added as 0.05, so total damage = 0.2 (same as before).

Update `test_senescence_bounded`:

```python
    def test_senescence_bounded(self, radiation_result):
        assert np.all(radiation_result["states"][:, 5] <= 1.0)  # unchanged index
```

This should already work since index 5 is still senescence.

**Step 3: Fix test_grief_bridge.py**

In `TestGriefDisturbance.test_modify_state_adds_ros`, update the state vector:

```python
    def test_modify_state_adds_ros(self):
        d = GriefDisturbance()
        state = np.array([0.5, 0.25, 0.8, 0.1, 0.6, 0.05, 0.9, 0.05])
        new_state = d.modify_state(state, 0.5)
        # SNS-driven ROS should increase state[3]
        assert new_state[3] >= state[3]
```

**Step 4: Update causal_surgery.py heteroplasmy callsites**

Find all `_heteroplasmy_fraction(state[0], state[1])` calls and replace with `_total_heteroplasmy(state[0], state[1], state[7])`. Update the import.

**Step 5: Update multi_tissue_sim.py**

Update all hardcoded index patterns:

- `tissue_states[tissue][0] + tissue_states[tissue][1]` → add `+ tissue_states[tissue][7]`
- `tissue_states[tissue][1] / max(total, 1e-12)` → `(tissue_states[tissue][1] + tissue_states[tissue][7]) / max(total, 1e-12)` (for total het)
- `state[5] = min(state[5], 1.0)` → still correct (index 5 is senescence)

**Step 6: Run full test suite**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/ -v 2>&1 | tail -30`

Expected: All 204+ tests PASS.

**Step 7: Commit**

```bash
git add tests/test_simulator.py tests/test_resilience.py tests/test_grief_bridge.py causal_surgery.py multi_tissue_sim.py
git commit -m "fix(C11): update all downstream files for 8D state vector

Shape assertions 7->8, state arrays extended with N_point,
heteroplasmy callsites updated in causal_surgery and multi_tissue_sim."
```

---

### Task 6: Calibration, Standalone Tests & Documentation

**Files:**
- Modify: `simulator.py` (standalone test section at bottom — update state name list)
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Run standalone simulator test**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python simulator.py 2>&1 | head -60`

Update the `if __name__ == "__main__"` block in `simulator.py` to print all 8 state names:

```python
    for j, name in enumerate(["N_healthy", "N_deletion", "ATP", "ROS",
                               "NAD", "Senescent", "ΔΨ", "N_point"]):
```

**Step 2: Verify calibration targets**

Run the simulator standalone and check:

1. **Natural aging (70yo, 30yr):** Total het should reach ~0.85-0.95. Most growth should be deletions.
2. **Young healthy (30yo):** Total het < 0.20 at 30 years. Point mutations dominate early.
3. **Cliff behavior:** Deletion het > 0.70 should trigger ATP collapse.
4. **Transplant:** Should still outperform NAD supplementation.
5. **Cocktail intervention:** Should still reduce heteroplasmy vs baseline.

If calibration is off (e.g., het doesn't rise enough or rises too fast), adjust these constants:
- `POINT_ERROR_RATE` (point mutation generation rate)
- `ROS_POINT_COEFF` (ROS → point coupling)
- `DELETION_REPLICATION_ADVANTAGE` (deletion exponential growth speed)
- `del_rate * 0.05` coefficient in `age_deletions` (de novo deletion rate)

**Step 3: Run full test suite one more time**

Run: `cd /Users/gardenofcomputation/how-to-live-much-longer && python -m pytest tests/ -v`

Verify all tests pass including the grief-simulator tests:

Run: `cd /Users/gardenofcomputation/grief-simulator && python -m pytest tests/ -v`

**Step 4: Update CLAUDE.md**

Add after the existing C10 correction entry in the Action Items section:

```markdown
- **CRAMER CORRECTION APPLIED (2026-02-17):** Per John Cramer's email:
  - **C11: Split mutation types** — ROS is NOT the main mtDNA mutation driver (1980s Free Radical Theory is outdated). Two distinct mutation types with different dynamics:
    - **Point mutations** (N_point, state[7]): Linear growth, no replication advantage. Sources: ~67% Pol γ errors + ~33% ROS-induced transitions. Functionally mild.
    - **Deletion mutations** (N_deletion, state[1]): Exponential growth, size-dependent replication advantage (1.21x, book says ≥1.21). These drive the heteroplasmy cliff. Source: Pol γ slippage, NOT ROS.
  - State vector expanded from 7D to 8D. Cliff factor uses deletion heteroplasmy only. ROS→damage coupling weakened to ~33% of previous (point mutations only).
  - Reference: Appendix 2 pp.152-155, Va23 (Vandiver et al. 2023).
```

Update the State Variables table in the docstring/documentation to list 8 variables.

Update test count in Quick Commands section.

**Step 5: Update README.md**

Add C11 to the corrections table. Update the state variable documentation.

**Step 6: Commit**

```bash
git add simulator.py CLAUDE.md README.md
git commit -m "docs(C11): calibration verification and documentation updates

Standalone test updated for 8D state. CLAUDE.md and README.md
updated with C11 correction details, 8D state vector, and
split mutation type biology."
```

---

### Summary

| Task | Description | Files | New Tests |
|------|-------------|-------|-----------|
| 1 | Constants + heteroplasmy functions | constants.py, simulator.py, test_simulator.py | 8 |
| 2 | Core ODE split (derivatives, initial_state) | simulator.py, test_simulator.py | 10 |
| 3 | Disturbances (impulse split) | disturbances.py, test_resilience.py | ~5 updated |
| 4 | Analytics (deletion metrics) | analytics.py, test_simulator.py | 1 |
| 5 | Fix downstream (tests, campaign scripts) | 5 files | ~15 updated |
| 6 | Calibration + documentation | simulator.py, CLAUDE.md, README.md | 0 |

**Total: ~19 new tests + ~20 updated tests across 6 tasks.**

## Assumptions and Scientific Grounding

- **Primary theoretical grounding:** This codebase operationalizes John G. Cramer's mitochondrial-aging theory as presented in *How to Live Much Longer* (**forthcoming from Springer in 2026; ISBN 978-3-032-17740-7**). In this repository, that work is treated as the model-level ground truth for mechanism selection and parameterization.
- **Model-form assumption:** Biological mechanisms are represented in reduced-form computational structures (ODE-style dynamics, scenario perturbations, and optimization surfaces) to make hypotheses testable and comparable.
- **Parameter assumption:** Constants and intervention ranges are interpreted as theory-informed approximations for simulation and stress-testing, not as universal physiological truths for all populations.
- **Evidence and scope assumption:** Outputs are hypothesis-generating research artifacts for mechanism exploration, sensitivity analysis, and scenario comparison. They are not clinical prescriptions, medical advice, or proof of efficacy.
- **Validation assumption:** Scientific confidence depends on empirical triangulation (literature checks, mechanistic plausibility, sensitivity behavior, and experimental/clinical follow-up), not simulator output alone.
