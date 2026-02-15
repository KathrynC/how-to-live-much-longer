"""Prompt templates for LLM-mediated intervention design.

Provides both numeric (original) and diegetic (narrative) prompt styles
for A/B comparison. Informed by Zimmerman (2025) dissertation findings
on LLM meaning construction:

- Diegetic prompts embed parameters in clinical narrative (Ch. 2-3:
  LLMs handle semantic content better than structural/numeric)
- Contrastive prompts generate opposing protocols to exploit
  TALOT/OTTITT meaning-from-contrast (Ch. 5)
- Anti-flattening measures separate qualitatively different parameters
  with semantic context (Ch. 3: LLMs collapse qualitative distinctions)

Reference:
    Zimmerman, J.W. (2025). "Locality, Relation, and Meaning
    Construction in Language, as Implemented in Humans and Large
    Language Models (LLMs)." PhD dissertation, University of Vermont.
"""


# ── Numeric prompts (original style) ────────────────────────────────────────

OFFER_NUMERIC = """\
You are a mitochondrial medicine specialist designing a personalized \
intervention protocol. You must choose BOTH intervention parameters AND \
characterize the patient based on the clinical scenario.

INTERVENTION PARAMETERS (6 params, each 0.0 to 1.0):
  rapamycin_dose: mTOR inhibition -> enhanced mitophagy (0=none, 1=maximum)
  nad_supplement: NAD+ precursor (NMN/NR) dose (0=none, 1=maximum)
  senolytic_dose: Senolytic drug dose (dasatinib+quercetin) (0=none, 1=maximum)
  yamanaka_intensity: Partial reprogramming (OSKM) intensity (0=none, 1=max) \
WARNING: costs 3-5 MU of ATP -- only use if patient can afford the energy
  transplant_rate: Mitochondrial transplant rate via mitlets (0=none, 1=maximum)
  exercise_level: Exercise intensity for hormetic adaptation (0=sedentary, 1=intense)

PATIENT PARAMETERS (6 params):
  baseline_age: Starting age in years (20-90)
  baseline_heteroplasmy: Fraction of damaged mtDNA (0.0-0.95). \
CRITICAL: the heteroplasmy cliff is at ~0.7 -- above this, ATP collapses
  baseline_nad_level: NAD+ level (0.2-1.0, declines with age)
  genetic_vulnerability: Susceptibility to mtDNA damage (0.5-2.0, 1.0=normal)
  metabolic_demand: Tissue energy need (0.5=skin, 1.0=normal, 2.0=brain)
  inflammation_level: Chronic inflammation (0.0-1.0)

CLINICAL SCENARIO:
{scenario}

Think carefully about this patient:
- How close are they to the heteroplasmy cliff?
- What is the most urgent intervention?
- Can they afford Yamanaka's energy cost?
- Would transplant help (adding healthy copies)?
- Is exercise safe given their current energy reserves?

Choose intervention values from: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
Choose patient values within the ranges described above.

Output a single JSON object with ALL 12 keys:
{{"rapamycin_dose":_, "nad_supplement":_, "senolytic_dose":_, \
"yamanaka_intensity":_, "transplant_rate":_, "exercise_level":_, \
"baseline_age":_, "baseline_heteroplasmy":_, "baseline_nad_level":_, \
"genetic_vulnerability":_, "metabolic_demand":_, "inflammation_level":_}}"""


# ── Diegetic prompts (Zimmerman-informed) ───────────────────────────────────
# Key insight: LLMs construct meaning from distributional semantics (diegetic
# content), not from structural form (supradiegetic). By embedding parameter
# choices in narrative rather than numeric form, we align with the LLM's
# natural meaning-making process.

OFFER_DIEGETIC = """\
You are a mitochondrial medicine specialist seeing a patient. Read the \
clinical scenario below, then prescribe a treatment and characterize \
the patient.

CLINICAL SCENARIO:
{scenario}

=== TREATMENT DECISIONS ===

Think through each treatment decision as a clinical judgment:

1. MITOPHAGY ENHANCEMENT (rapamycin/mTOR inhibition)
   How aggressively should we clear damaged mitochondria?
   Options: none / minimal / moderate / substantial / aggressive / maximum

2. NAD+ RESTORATION (NMN or NR supplementation)
   How much cofactor support does this patient need?
   Options: none / minimal / moderate / substantial / aggressive / maximum

3. SENESCENT CELL CLEARANCE (dasatinib + quercetin + fisetin)
   How much senolytic therapy? Frees energy budget from zombie cells.
   Options: none / minimal / moderate / substantial / aggressive / maximum

4. EPIGENETIC REPROGRAMMING (partial Yamanaka / OSKM factors)
   CAUTION: This costs 3-5x the cell's normal daily energy budget.
   Only prescribe if the patient has enough energy reserves.
   Options: none / minimal / moderate / substantial / aggressive / maximum

5. MITOCHONDRIAL TRANSPLANT (healthy mtDNA infusion via platelet-derived mitlets)
   How much fresh healthy mitochondrial DNA should we inject?
   Options: none / minimal / moderate / substantial / aggressive / maximum

6. EXERCISE PRESCRIPTION (hormetic adaptation)
   Moderate stress triggers antioxidant defense. But is the patient strong enough?
   Options: sedentary / light / moderate / vigorous / intense / extreme

=== PATIENT ASSESSMENT ===

Based on the scenario, characterize this patient:

7. AGE: How old is this patient? (young adult / thirties / middle-aged / \
fifties / sixties / seventies / eighties / elderly)

8. MITOCHONDRIAL DAMAGE: What fraction of their mtDNA is damaged?
   The cliff is at ~70% -- above this, cellular energy collapses.
   (minimal / low / moderate / elevated / high / very high / \
near-cliff / past-cliff)

9. NAD+ STATUS: How depleted is their NAD+ cofactor?
   Declines naturally with age; worse with chronic disease.
   (severely depleted / low / somewhat low / adequate / good)

10. GENETIC VULNERABILITY: How susceptible is this patient to mtDNA damage?
    Varies by mitochondrial haplogroup and family history.
    (resilient / below-average / normal / above-average / high / very high)

11. TISSUE ENERGY DEMAND: Which tissue type is most affected?
    (low-demand like skin / below-average / normal / above-average / \
high-demand / very-high like brain)

12. CHRONIC INFLAMMATION: How inflamed is this patient?
    (none / mild / moderate / substantial / high / severe)

In 1-2 sentences, explain your clinical reasoning. Then output a JSON \
object mapping each decision to its value. Use the EXACT key names below:
{{"rapamycin_dose":_, "nad_supplement":_, "senolytic_dose":_, \
"yamanaka_intensity":_, "transplant_rate":_, "exercise_level":_, \
"baseline_age":_, "baseline_heteroplasmy":_, "baseline_nad_level":_, \
"genetic_vulnerability":_, "metabolic_demand":_, "inflammation_level":_}}

For intervention keys, map your word choice to a number:
  none=0.0, minimal=0.1, moderate=0.25, substantial=0.5, \
aggressive=0.75, maximum=1.0
For patient keys, use the numeric ranges: age (20-90), \
heteroplasmy (0.0-0.95), NAD (0.2-1.0), vulnerability (0.5-2.0), \
demand (0.5-2.0), inflammation (0.0-1.0)."""


# ── Contrastive prompts (TALOT/OTTITT) ──────────────────────────────────────
# "Things Are Like Other Things" vs "Only The Thing Is The Thing"
# By generating opposing protocols and articulating the contrast,
# the LLM is forced to think more carefully about parameter choices.

OFFER_CONTRASTIVE = """\
You are a mitochondrial medicine specialist. Two colleagues disagree \
about this patient's treatment. You must present BOTH positions.

CLINICAL SCENARIO:
{scenario}

=== DR. CAUTIOUS (conservative approach) ===
Prioritizes safety, minimal intervention, preserving energy reserves. \
Would never risk Yamanaka reprogramming unless ATP is abundant. \
Prefers gentle, sustainable protocols.

=== DR. BOLD (aggressive approach) ===
Believes aging is an emergency requiring maximum intervention. Willing \
to use Yamanaka even at energy cost. Pushes all interventions hard. \
Time is the enemy.

For EACH doctor, provide a 12-parameter protocol as a JSON object.

Parameters:
  Intervention (0.0-1.0 each): rapamycin_dose, nad_supplement, \
senolytic_dose, yamanaka_intensity, transplant_rate, exercise_level
  Patient (characterize the same patient for both):
    baseline_age (20-90), baseline_heteroplasmy (0.0-0.95), \
baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0), \
metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)

In 2-3 sentences, explain what each doctor would argue and WHY they \
disagree. Then output TWO JSON objects labeled "cautious" and "bold":
{{"cautious": {{...12 keys...}}, "bold": {{...12 keys...}}}}"""


# ── Character seed prompts ──────────────────────────────────────────────────

CHARACTER_NUMERIC = (
    "You are a mitochondrial medicine specialist who sees patients through "
    "the lens of fictional archetypes.\n\n"
    "A patient presents who reminds you strongly of '{character}' from "
    "'{story}'. Think about this character's:\n"
    "  - Age and life stage\n"
    "  - Energy level and physical activity\n"
    "  - Stress level and inflammation (villains, warriors -> high; "
    "peaceful characters -> low)\n"
    "  - Resilience and genetic robustness\n"
    "  - Tissue vulnerability (intellectual characters -> brain/high demand; "
    "physical characters -> muscle; sedentary -> low demand)\n\n"
    "Design a personalized mitochondrial intervention protocol AND "
    "characterize the patient based on this archetype.\n\n"
    "PARAMETERS (output a JSON object with ALL 12 keys):\n"
    "  Intervention (0.0-1.0 each):\n"
    "    rapamycin_dose: mTOR inhibition -> enhanced mitophagy\n"
    "    nad_supplement: NAD+ precursor (NMN/NR)\n"
    "    senolytic_dose: Senescent cell clearance\n"
    "    yamanaka_intensity: Partial reprogramming (HIGH energy cost!)\n"
    "    transplant_rate: Mitochondrial transplant via mitlets\n"
    "    exercise_level: Hormetic exercise\n"
    "  Patient:\n"
    "    baseline_age (20-90), baseline_heteroplasmy (0.0-0.95),\n"
    "    baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0),\n"
    "    metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)\n\n"
    "In 1-2 sentences, describe why this character-archetype maps to "
    "this protocol. Then output ONLY the JSON object. Keep reasoning SHORT."
)


CHARACTER_DIEGETIC = (
    "You are a mitochondrial medicine specialist. A patient walks in who "
    "instantly reminds you of '{character}' from '{story}'.\n\n"
    "Think about this character as a PERSON:\n"
    "  - How old are they? What stage of life?\n"
    "  - Are they energetic or exhausted? Active or sedentary?\n"
    "  - Are they under chronic stress? Fighting, struggling, traumatized?\n"
    "  - Are they genetically robust or fragile?\n"
    "  - What part of their body takes the most punishment -- their brain "
    "(thinkers, strategists), their muscles (fighters, athletes), or are "
    "they relatively low-stress (peaceful, domestic characters)?\n\n"
    "Now prescribe their mitochondrial treatment:\n"
    "  - How aggressively should you clear their damaged mitochondria? "
    "(none / minimal / moderate / substantial / aggressive / maximum)\n"
    "  - How much NAD+ restoration do they need?\n"
    "  - Should you clear their senescent cells?\n"
    "  - Can they afford the enormous energy cost of Yamanaka reprogramming "
    "(3-5x daily energy budget)?\n"
    "  - Would transplanting fresh healthy mitochondria help?\n"
    "  - What exercise level is safe for them?\n\n"
    "Brief reasoning (1-2 sentences), then a JSON object with these keys:\n"
    "  rapamycin_dose, nad_supplement, senolytic_dose, yamanaka_intensity, "
    "transplant_rate, exercise_level (each 0.0-1.0)\n"
    "  baseline_age (20-90), baseline_heteroplasmy (0.0-0.95), "
    "baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0), "
    "metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)"
)


# ── OEIS seed prompts ───────────────────────────────────────────────────────

OEIS_NUMERIC = (
    "You are a mitochondrial medicine specialist who finds deep patterns "
    "in mathematical sequences.\n\n"
    "Consider integer sequence {seq_id}: '{seq_name}'\n"
    "First 16 terms: {terms}\n\n"
    "Let this sequence's mathematical character inspire a mitochondrial "
    "intervention protocol. Think about:\n"
    "  - Growth rate: exponential sequences -> aggressive intervention\n"
    "  - Oscillation: periodic sequences -> cycling protocols\n"
    "  - Convergence: converging sequences -> maintenance therapy\n"
    "  - Chaos: erratic sequences -> emergency intervention\n"
    "  - Primes/divisibility: selective sequences -> targeted therapy\n\n"
    "PARAMETERS (JSON object with ALL 12 keys):\n"
    "  Intervention (0.0-1.0): rapamycin_dose, nad_supplement, "
    "senolytic_dose, yamanaka_intensity, transplant_rate, exercise_level\n"
    "  Patient: baseline_age (20-90), baseline_heteroplasmy (0.0-0.95), "
    "baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0), "
    "metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)\n\n"
    "In 1-2 sentences, explain how the sequence maps to the protocol. "
    "Then output ONLY the JSON."
)


OEIS_DIEGETIC = (
    "You are a mitochondrial medicine specialist with a deep love of "
    "mathematics. A patient arrives whose medical chart reminds you "
    "strikingly of {seq_id}: '{seq_name}'.\n\n"
    "The sequence: {terms}\n\n"
    "What is it about this sequence that maps to a patient?\n"
    "  - Does it grow explosively? (aging accelerating, needs aggressive help)\n"
    "  - Does it oscillate? (symptoms come and go, needs cycling protocol)\n"
    "  - Is it gentle and regular? (stable patient, maintenance only)\n"
    "  - Is it chaotic? (unpredictable decline, emergency measures)\n"
    "  - Is it selective/prime? (specific targeted intervention)\n\n"
    "Prescribe a protocol that matches the sequence's character.\n"
    "  Treatment intensity: none / minimal / moderate / substantial / "
    "aggressive / maximum\n"
    "  For each of: mitophagy enhancement, NAD+ restoration, senolytic "
    "clearance, Yamanaka reprogramming (WARNING: enormous energy cost!), "
    "mitochondrial transplant, exercise\n\n"
    "Also characterize the patient this sequence represents.\n\n"
    "Brief reasoning, then JSON with keys: rapamycin_dose, nad_supplement, "
    "senolytic_dose, yamanaka_intensity, transplant_rate, exercise_level "
    "(0.0-1.0), baseline_age (20-90), baseline_heteroplasmy (0.0-0.95), "
    "baseline_nad_level (0.2-1.0), genetic_vulnerability (0.5-2.0), "
    "metabolic_demand (0.5-2.0), inflammation_level (0.0-1.0)"
)


# ── Confirmation wave prompt (shared across styles) ─────────────────────────

CONFIRMATION_PROMPT = """\
A mitochondrial aging simulation was run with these results:

  Simulation: {sim_years} years, starting age {age}
  Initial heteroplasmy: {het_initial:.2f} (cliff at {cliff})
  Final heteroplasmy: {het_final:.2f}
  Time to cliff: {time_to_cliff}
  Initial ATP: {atp_initial:.3f} MU/day
  Final ATP: {atp_final:.3f} MU/day
  ATP slope: {atp_slope:+.4f} MU/day/year
  Time to energy crisis: {time_to_crisis}
  ROS-heteroplasmy correlation: {ros_het_corr:.3f}
  Final senescent fraction: {sen_final:.3f}
  Membrane potential CV: {psi_cv:.3f}
  Intervention benefit (ATP): {atp_benefit:+.3f} vs no treatment
  Intervention benefit (heteroplasmy): {het_benefit:+.3f} vs no treatment

  Intervention used:
    Rapamycin: {rapamycin_dose}
    NAD+ supplement: {nad_supplement}
    Senolytics: {senolytic_dose}
    Yamanaka: {yamanaka_intensity}
    Transplant: {transplant_rate}
    Exercise: {exercise_level}

The clinical scenario was: "{scenario}"

Questions:
1. Describe the trajectory in 1-2 sentences (as if watching a patient's \
cellular health over decades).
2. Does this intervention protocol MATCH the clinical scenario? Rate the \
clinical resonance from 0.0 (no connection) to 1.0 (perfect match).
3. Does the simulation trajectory look physiologically plausible? Rate \
trajectory resonance from 0.0 (unrealistic) to 1.0 (highly plausible).
4. What would you change to better serve this patient?

Output a JSON object:
{{"trajectory_description": "...", "resonance_behavior": 0.X, \
"resonance_trajectory": 0.X, "suggestion": "..."}}"""


# ── Template selection ──────────────────────────────────────────────────────

PROMPT_STYLES = {
    "numeric": {
        "offer": OFFER_NUMERIC,
        "character": CHARACTER_NUMERIC,
        "oeis": OEIS_NUMERIC,
        "confirmation": CONFIRMATION_PROMPT,
    },
    "diegetic": {
        "offer": OFFER_DIEGETIC,
        "character": CHARACTER_DIEGETIC,
        "oeis": OEIS_DIEGETIC,
        "confirmation": CONFIRMATION_PROMPT,
    },
    "contrastive": {
        "offer": OFFER_CONTRASTIVE,
        "confirmation": CONFIRMATION_PROMPT,
    },
}


def get_prompt(style, prompt_type):
    """Get a prompt template by style and type.

    Args:
        style: "numeric", "diegetic", or "contrastive"
        prompt_type: "offer", "character", "oeis", or "confirmation"

    Returns:
        Prompt template string.

    Raises:
        KeyError: If style or prompt_type not found.
    """
    return PROMPT_STYLES[style][prompt_type]
