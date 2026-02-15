# Cramer Corrections — 2026-02-15

## Source

Email from John G. Cramer (author, *How to Live Much Longer*, 2025) reviewing the
simulation, received 2026-02-15. Accompanied by two reference PDFs:

- Cramer & Mead (2020). "Symmetry, Transactions, and the Mechanism of Wave Function
  Collapse." *Symmetry* 12(1543).
- Cramer, J.G. (2016). *The Quantum Handshake: Entanglement, Nonlocality and
  Transactions.* Springer. ISBN 978-3-319-24642-0.

## Cramer's Two Critiques

### Critique 1: NAD+ Supplementation Overemphasized

> "The simulation overemphasizes NAD+ supplementation. A group at the Mayo Clinic
> has shown that the enzyme CD38 destroys NMN and NR supplements before they can do
> much for NAD+ enhancement. A better strategy is to suppress CD38 with supplements
> like apigenin. See p. 73 of How to Live Much Longer. Moreover, such intervention
> at best is likely to produce a temporary boost in ATP levels, and the focus should
> be on more permanent interventions."

**Key biology:** CD38 is a NADase enzyme that degrades NAD+ precursors (NMN, NR)
before they reach the mitochondria. At typical supplementation doses, CD38 destroys
the majority of the precursor. Apigenin (a flavonoid found in parsley, celery,
chamomile) inhibits CD38, allowing more NMN/NR to reach its target.

**Book reference:** Cramer Ch. VI.A.3, p.73.

### Critique 2: Transplantation Underemphasized

> "The simulation underemphasized the value of transplantation with new
> externally-produced mitochondria containing unmutated mtDNA. This is the only
> available method of reversing the accumulated damage to mtDNA at a scale that
> could be called rejuvenation."

**Key biology:** Mitochondrial transplant via bioreactor-grown stem cells and mitlet
encapsulation (Cramer Ch. VIII.G pp.104-107) is the only intervention that can
actually *reverse* accumulated heteroplasmy, not merely slow its progression. All
other interventions (rapamycin, NAD+, senolytics, exercise) can only slow the rate
of damage accumulation. Transplant adds healthy copies AND, through competitive
displacement, the healthy mitochondria with intact ETC and high membrane potential
outcompete damaged copies for cellular resources.

**Book reference:** Cramer Ch. VIII.G pp.104-107.

---

## Corrections Applied

### C7: CD38 Degrades NMN/NR

**Files modified:** `constants.py`, `simulator.py`, `prompt_templates.py`

**New constants:**
```python
CD38_BASE_SURVIVAL = 0.4      # fraction surviving CD38 at minimal supplementation
CD38_SUPPRESSION_GAIN = 0.6   # additional survival from CD38 inhibitor (apigenin)
```

**ODE change in `derivatives()`:**

The NAD+ boost is now gated by a CD38 survival factor:
```python
cd38_survival = CD38_BASE_SURVIVAL + CD38_SUPPRESSION_GAIN * nad_supp
# 0.4 at nad_supp=0, rising to 1.0 at nad_supp=1.0

nad_boost = nad_supp * 0.25 * cd38_survival   # was: nad_supp * 0.35
nad_target = BASELINE_NAD * age_factor + nad_boost
```

The mitophagy quality control boost from NAD is also CD38-gated:
```python
mitophagy_rate = (BASELINE_MITOPHAGY_RATE
                  + rapa * 0.08
                  + nad_supp * 0.03 * cd38_survival)  # was: nad_supp * 0.03
```

**Effect on NAD benefit by dose level:**

| nad_supplement | cd38_survival | NAD boost | Old boost | Reduction |
|---|---|---|---|---|
| 0.25 | 0.55 | 0.034 | 0.088 | -61% |
| 0.50 | 0.70 | 0.088 | 0.175 | -50% |
| 0.75 | 0.85 | 0.159 | 0.263 | -39% |
| 1.00 | 1.00 | 0.250 | 0.350 | -29% |

**Interpretation:** Low-dose NMN/NR supplementation is largely futile — CD38
destroys most of it. High-dose supplementation (interpreted as NMN/NR + apigenin
CD38 suppression) retains ~71% of the original benefit. This creates a strong
nonlinearity: high-dose NAD gives 6.7x the heteroplasmy benefit of low-dose
(verified in Test 10a).

### C8: Transplant as Primary Rejuvenation

**Files modified:** `constants.py`, `simulator.py`, `prompt_templates.py`

**New constants:**
```python
TRANSPLANT_ADDITION_RATE = 0.30      # was 0.15 (doubled)
TRANSPLANT_DISPLACEMENT_RATE = 0.12  # NEW: competitive displacement
TRANSPLANT_HEADROOM = 1.5            # was 1.2 (raised ceiling)
```

**ODE change in `derivatives()`:**

Healthy copy addition rate doubled, headroom raised:
```python
transplant_headroom = max(TRANSPLANT_HEADROOM - total, 0.0)  # was 1.2
transplant_add = transplant * TRANSPLANT_ADDITION_RATE * min(transplant_headroom, 1.0)
```

New competitive displacement mechanism — transplanted healthy mitochondria with
intact ETC and high membrane potential outcompete damaged copies:
```python
transplant_displace = transplant * TRANSPLANT_DISPLACEMENT_RATE * n_d * energy_available
```

This displacement term is subtracted from dN_damaged/dt, meaning transplant now
actively *removes* damaged copies, not just adds healthy ones.

**Quantitative results (default 70yo patient, 30-year simulation):**

| Intervention | Final het | Het benefit vs untreated |
|---|---|---|
| No treatment | 0.590 | — |
| NAD=1.0 only | 0.408 | 0.183 |
| Transplant=1.0 only | 0.174 | 0.416 |
| Transplant+rapamycin | 0.139 (near-cliff) | 0.731 (from 0.870) |

Transplant benefit (0.416) is 2.3x NAD benefit (0.183), confirming Cramer's
assertion that transplant is the primary rejuvenation mechanism.

**Near-cliff rescue:** An 80-year-old with 65% heteroplasmy (approaching the cliff)
receiving transplant + rapamycin: heteroplasmy drops from 0.870 to 0.139, ATP
recovers from 0.049 to 0.642. This demonstrates the unique ability of transplant
to *reverse* accumulated damage — no other intervention achieves this.

---

## Verification

### Standalone tests (10 scenarios)
```
python simulator.py
# Test 10: Cramer corrections (CD38 + transplant)
#   [10a] NAD low=0.25: het_benefit=0.0271  NAD high=1.0: het_benefit=0.1827
#         ratio=6.7x (PASS: >2x)
#   [10b] Transplant=1.0: final_het=0.1740  het_benefit=0.4163 (PASS: strong)
#   [10c] Transplant benefit=0.4163 vs NAD benefit=0.1827
#         (PASS: transplant > NAD)
#   [10d] Near-cliff + transplant: final_het=0.1385  final_ATP=0.6422
#         (vs untreated: het=0.8698  ATP=0.0485)
```

### Pytest suite (85 tests)
```
pytest tests/ -v
# tests/test_simulator.py::TestCramerCorrections::test_cd38_nonlinearity PASSED
# tests/test_simulator.py::TestCramerCorrections::test_transplant_strong_rejuvenation PASSED
# tests/test_simulator.py::TestCramerCorrections::test_transplant_beats_nad PASSED
# tests/test_simulator.py::TestCramerCorrections::test_transplant_rescues_near_cliff PASSED
# 85 passed in 4.75s
```

### All existing tests unaffected
- All 81 pre-existing tests continue to pass
- All 9 pre-existing standalone tests continue to pass
- NAD still reduces heteroplasmy (fix C3 preserved), just less dramatically
- Past-cliff bistability preserved (fix C4)
- Yamanaka ATP gating preserved (fix M1)

---

## Prompt Template Updates

Both OFFER_NUMERIC and OFFER_DIEGETIC templates updated:

1. `nad_supplement` description now mentions CD38 and apigenin
2. `transplant_rate` description now says "ONLY method for true rejuvenation"
3. "Think carefully" section emphasizes transplant over NAD+
4. Near-cliff Example 2: transplant raised from 0.5 to 0.75, NAD lowered from 0.75 to 0.5
5. Diegetic reasoning for Example 2 now explains CD38 limitation and transplant primacy

---

## TI Context from PDFs

The two PDFs provided by Cramer establish the theoretical physics foundation for
the TIQM pipeline analogy:

**The Quantum Handshake (2016):** The Transactional Interpretation of QM. Key
mappings to our simulation:

| TI Concept | Simulation Mapping |
|---|---|
| Offer wave (ψ, retarded) | LLM generates 12D intervention+patient vector |
| Confirmation wave (ψ*, advanced) | Second LLM evaluates trajectory |
| Transaction formation | Protocol validation (offer + simulation + confirmation agree) |
| Different emitter/absorber | Different models for offer vs confirmation (no self-confirmation) |
| Hierarchy and selection (Ch. 5.5) | Fitness selection among competing protocols |
| Time symmetry | Bidirectional information flow (forward simulation, backward evaluation) |

**Cramer-Mead 2020:** Technical paper on TI mechanism. Establishes that transactions
require (1) an emitter sending an offer wave, (2) an absorber sending a confirmation
wave, and (3) the two waves forming a standing wave (the transaction). The transaction
is the quantum event — it's not the wave function that's "real," it's the completed
transaction. This maps to our pipeline where neither the LLM's protocol nor the
simulation's trajectory is the "answer" — the completed transaction (validated
protocol with confirmed trajectory) is.
