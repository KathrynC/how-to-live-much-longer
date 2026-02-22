"""Literature search via local LLMs for falsification of 5 surprising model findings.

Queries local Ollama models to generate literature evidence summaries for each
surprising finding from the mitochondrial aging simulation. Uses multiple models
(offer + confirmation pattern) to reduce single-model bias.

Output: artifacts/lit_search/finding_{1-5}_{model}.md + compiled summary

Usage:
    python lit_search_falsification.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = Path("artifacts/lit_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to query (largest first for primary evidence, smaller for cross-check)
MODELS = [
    "qwen3-coder:30b",
    "deepseek-r1:8b",
    "gpt-oss:20b",
]

REASONING_MODELS = {"qwen3-coder:30b", "deepseek-r1:8b"}

# ── The 5 Surprising Findings ──────────────────────────────────────────────

FINDINGS = {
    1: {
        "title": "Exercise Is Harmful at Every Dose and Age",
        "model_behavior": (
            "In our mitochondrial aging ODE model, exercise_level > 0 always worsens "
            "outcomes (lower final ATP, higher heteroplasmy) compared to exercise=0, "
            "across all ages (30-80) and all dose levels (0.1-1.0). The model implements "
            "exercise as: (1) metabolic demand increase via EXERCISE_METABOLIC_COST=0.03, "
            "and (2) mitochondrial biogenesis stimulation via EXERCISE_BIOGENESIS_FACTOR=0.03. "
            "These exactly cancel, but the increased metabolic demand raises ROS production "
            "quadratically while biogenesis is only linear, making net effect always negative."
        ),
        "prompt": (
            "I am building a computational model of mitochondrial aging where exercise "
            "always harms outcomes. I need to check this against published literature. "
            "Please provide a thorough literature review covering:\n\n"
            "1. **Exercise and mitochondrial biogenesis**: What is the magnitude of exercise-induced "
            "mitochondrial biogenesis? Cite specific studies (PGC-1alpha activation, mtDNA copy "
            "number changes). How large is the biogenesis effect relative to metabolic cost?\n\n"
            "2. **Exercise and ROS**: Does exercise increase or decrease net oxidative stress? "
            "What is hormesis — does moderate exercise upregulate antioxidant defenses (SOD2, "
            "catalase, GPx) enough to more than compensate for increased ROS production? "
            "Cite specific studies quantifying the ROS/antioxidant balance.\n\n"
            "3. **Exercise and mtDNA damage**: Does exercise increase or decrease mtDNA mutations "
            "and deletions? What does the literature say about exercise effects on heteroplasmy?\n\n"
            "4. **Exercise and mitophagy**: Does exercise stimulate PINK1/Parkin-mediated "
            "mitophagy (selective removal of damaged mitochondria)? How significant is this?\n\n"
            "5. **Net effect on aging**: What do longitudinal studies show about exercise and "
            "cellular/mitochondrial aging markers? Include: telomere length, mtDNA copy number, "
            "respiratory chain function in elderly exercisers vs sedentary controls.\n\n"
            "6. **Dose-response**: Is there a U-shaped curve? Does extreme exercise become "
            "harmful? What is the optimal dose range?\n\n"
            "For each point, provide specific author names, publication years, journal names, "
            "and quantitative findings where possible. I need concrete numbers to calibrate "
            "my model parameters."
        ),
    },
    2: {
        "title": "Sleep Effect Is 26x Weaker Than NAD Supplementation",
        "model_behavior": (
            "In our model, improving sleep from intervention=0.3 to 0.9 at age 70 only "
            "increases final ATP by +0.0024 (0.36% gain), while NAD supplementation at "
            "dose=0.75 gives +0.0633 (9.5% gain) — a 26x difference. Sleep enters via "
            "5 coupling channels (inflammation, repair, ROS, NAD drain, membrane potential) "
            "with coefficients: SLEEP_INFLAMMATION_COEFF=0.08, SLEEP_REPAIR_COEFF=0.5, "
            "SLEEP_ROS_COEFF=0.04, SLEEP_NAD_DRAIN_COEFF=0.02, SLEEP_MEMBRANE_COEFF=0.03. "
            "The question is whether these coefficients are too conservative."
        ),
        "prompt": (
            "I am modeling the impact of sleep on mitochondrial function and aging. My model "
            "shows sleep having a 26x smaller effect than NAD+ supplementation on cellular "
            "energy (ATP). I need literature evidence to check if this is realistic or if "
            "sleep effects are being underestimated. Please review:\n\n"
            "1. **Sleep and mitochondrial function**: What happens to mitochondrial function "
            "during sleep deprivation? Cite studies measuring ATP production, respiratory chain "
            "activity, membrane potential in sleep-deprived subjects. Include Everson et al. "
            "rodent studies and any human data.\n\n"
            "2. **Sleep and oxidative stress**: Quantify the ROS/oxidative damage increase "
            "from poor sleep. How much does 8-oxodG increase? How much do antioxidant defenses "
            "drop? Cite Villafuerte et al. 2015 and related work.\n\n"
            "3. **Sleep and NAD+ metabolism**: Does sleep deprivation affect NAD+ levels "
            "directly? What about NAMPT expression (the rate-limiting NAD+ salvage enzyme)? "
            "What about circadian regulation of NAD+ via BMAL1/CLOCK/SIRT1? Cite Ramsey et al. "
            "2009 and Nakahata et al. 2009.\n\n"
            "4. **Sleep and inflammation (inflammaging)**: Quantify CRP, IL-6, TNF-alpha "
            "changes from chronic poor sleep. Cite Irwin et al. 2016 meta-analysis. How does "
            "this compare to other inflammation sources?\n\n"
            "5. **Sleep and autophagy/mitophagy**: Does sleep regulate autophagy? What about "
            "the glymphatic system (Xie et al. 2013)? Does this extend to mitochondrial "
            "quality control?\n\n"
            "6. **Comparative magnitude**: How does the effect of chronic poor sleep on "
            "cellular aging compare to NAD+ supplementation? Are there studies comparing "
            "these interventions head-to-head or providing effect sizes that allow comparison?\n\n"
            "7. **Sleep and mtDNA**: Any evidence that poor sleep accelerates mtDNA damage "
            "accumulation or affects heteroplasmy levels?\n\n"
            "Provide specific author names, years, journals, and quantitative effect sizes. "
            "I need concrete numbers to determine whether my sleep coupling coefficients "
            "(0.02-0.08) are too low."
        ),
    },
    3: {
        "title": "APOE4 Carriers Show LESS Sleep Vulnerability (Reversed Direction)",
        "model_behavior": (
            "In our model, APOE4 homozygotes show LESS ATP loss from poor sleep than "
            "wild-type individuals (delta=0.0024 for WT vs 0.0021 for APOE4-hom). This "
            "is because APOE4 only enters our model through reduced mitophagy_efficiency "
            "(0.65 for hom vs 1.0 for WT), which appears in the sleep repair channel as "
            "a DIVISOR: sleep_repair_factor = 1.0 - (SLEEP_REPAIR_COEFF / mitophagy_eff) * deficit. "
            "Lower mitophagy_eff makes repair worse, but this operates on a different "
            "timescale than direct sleep sensitivity. The model lacks a direct APOE4→sleep "
            "vulnerability pathway."
        ),
        "prompt": (
            "I am modeling the interaction between APOE4 genotype and sleep vulnerability "
            "in mitochondrial aging. My model paradoxically shows APOE4 carriers being LESS "
            "affected by poor sleep than wild-type. I need literature evidence on:\n\n"
            "1. **APOE4 and sleep architecture**: How does APOE4 affect sleep quality, "
            "deep sleep (N3/SWS), sleep fragmentation? Cite Lim et al. 2013 and other "
            "large cohort studies. Do APOE4 carriers have worse baseline sleep?\n\n"
            "2. **APOE4 and sleep-dependent clearance**: What is the relationship between "
            "APOE4, glymphatic clearance during sleep, and amyloid-beta accumulation? "
            "Cite Xie et al. 2013 (glymphatic), Shokri-Kojori et al. 2018 (one night "
            "sleep deprivation → amyloid). Is APOE4's effect on clearance sleep-dependent?\n\n"
            "3. **APOE4 × sleep interaction on cognitive decline**: Are there studies showing "
            "that poor sleep is MORE harmful for APOE4 carriers than non-carriers? Cite "
            "Lim et al. 2013 (sleep consolidation × APOE4 interaction), Osorio et al. 2014, "
            "and any meta-analyses.\n\n"
            "4. **APOE4 and mitochondrial function**: How does APOE4 directly affect "
            "mitochondrial dynamics? Cite studies on APOE4 and: mitochondrial fission/fusion, "
            "respiratory chain efficiency, membrane potential, ROS production. Is there "
            "a direct APOE4→mitochondrial vulnerability pathway beyond amyloid?\n\n"
            "5. **APOE4 and neuroinflammation**: Does APOE4 amplify inflammation from "
            "sleep deprivation specifically? Are there interaction effects?\n\n"
            "6. **APOE4 and oxidative stress vulnerability**: Does APOE4 reduce antioxidant "
            "defense capacity, making sleep-induced ROS more damaging?\n\n"
            "I need to determine: (a) Does the literature support APOE4 carriers being MORE "
            "vulnerable to poor sleep (opposite of my model)? (b) Through what mechanisms "
            "does APOE4 increase sleep vulnerability? (c) What is the effect size of the "
            "APOE4 × sleep interaction?\n\n"
            "Provide specific citations with authors, years, journals, and quantitative "
            "findings."
        ),
    },
    4: {
        "title": "Parameter Resolver Degrades Outcomes vs Raw Defaults",
        "model_behavior": (
            "Using the ParameterResolver (precision medicine expansion) with default settings "
            "produces WORSE outcomes than running the raw simulator with DEFAULT_INTERVENTION "
            "and DEFAULT_PATIENT. The resolver with sleep_intervention=0.5 (moderate) at age 70 "
            "gives lower ATP than the raw simulator because the sleep trajectory model applies "
            "hidden penalties: at age 70, sleep_quality=0.78 creates a deficit of 0.22, which "
            "feeds into inflammation (+0.029), repair factor (0.89), ROS boost (+0.011), NAD drain "
            "(+0.006), and membrane penalty (+0.008). These penalties don't exist in the raw "
            "simulator where sleep is not modeled. The question is: is it correct that sleep "
            "at age 70 with moderate intervention should be net-negative vs no sleep modeling?"
        ),
        "prompt": (
            "I am building a mitochondrial aging model where adding a sleep module makes "
            "outcomes WORSE than having no sleep modeling at all. This is because even moderate "
            "sleep intervention (50th percentile) at age 70 imposes net negative effects: "
            "the age-dependent sleep quality decline creates inflammation, reduces repair "
            "efficiency, increases ROS, drains NAD+, and penalizes membrane potential. "
            "I need literature evidence on:\n\n"
            "1. **Sleep quality in healthy aging**: What is the actual sleep quality trajectory "
            "for healthy agers vs pathological agers? Cite Ohayon et al. 2004 meta-analysis "
            "and Mander et al. 2017 (Neuron). Is age-related sleep decline inevitable, or "
            "do healthy agers maintain better sleep?\n\n"
            "2. **Sleep intervention efficacy in elderly**: How effective are sleep interventions "
            "(CBT-I, sleep hygiene, behavioral programs) in older adults? Cite Irwin et al. 2006, "
            "Trauer et al. 2015. Can interventions restore sleep quality to near-youthful levels "
            "or only partially recover?\n\n"
            "3. **Baseline mitochondrial state at age 70**: What is the expected mitochondrial "
            "function at age 70 in healthy individuals? Is the decline significant enough that "
            "sleep-related stressors would be meaningful?\n\n"
            "4. **Should sleep modeling be net-positive or net-negative?**: In the real world, "
            "does having normal (not optimal) sleep at age 70 represent a stress on mitochondria, "
            "or is it neutral/protective? The key question: should the model treat age-70 "
            "normal sleep as imposing a penalty relative to a hypothetical perfect-sleep "
            "baseline, or should the baseline already incorporate normal aging sleep?\n\n"
            "5. **Sleep as protective vs stressor**: Does sleep provide active mitochondrial "
            "benefits (autophagy activation, repair, clearance) during sleep itself? Or is "
            "poor sleep merely the absence of protection? This distinction matters for whether "
            "a sleep model should add benefits (when sleep is good) or add penalties (when "
            "sleep is poor).\n\n"
            "6. **Comparison of sleep impact magnitude to other age-related factors**: "
            "How does the mitochondrial impact of normal age-70 sleep compare to other "
            "age-related stressors (inflammaging, NAD decline, senescent cell accumulation)?\n\n"
            "Provide specific citations and quantitative data. I need to decide whether "
            "my model architecture is correct (sleep adds penalties) or needs restructuring "
            "(sleep at baseline should be neutral, only deviations should matter)."
        ),
    },
    5: {
        "title": "Mitochondrial Transplant Saturates at 10% Dose",
        "model_behavior": (
            "In our model, transplant_rate=0.1 (10% of maximum) achieves 62% of the benefit "
            "of transplant_rate=1.0 (100%). The model implements transplant as: healthy mtDNA "
            "addition (rate=0.30 * dose), competitive displacement of damaged copies (0.12 * dose "
            "* N_deletion), and headroom expansion (up to 1.5 total copies). The sharp "
            "saturation occurs because even small transplant doses establish a positive "
            "feedback loop: more healthy mitos → more ATP → better mitophagy → less damage → "
            "less need for transplant. The question is whether real mitochondrial transplant "
            "shows similar dose saturation."
        ),
        "prompt": (
            "I am modeling mitochondrial transplantation (healthy mtDNA infusion) for aging. "
            "My model shows extreme dose saturation: 10% of maximum dose gives 62% of full "
            "benefit. I need literature evidence on:\n\n"
            "1. **Mitochondrial transplant mechanisms**: How does mitochondrial transplantation "
            "actually work? Cite McCully et al. (cardiac surgery), Emani et al. (pediatric), "
            "and Cowan et al. What are the actual delivery methods — isolated mitochondria, "
            "platelet-derived mitlets (as in Cramer's book), exosomes, direct injection?\n\n"
            "2. **Dose-response in transplant studies**: What dose-response curves have been "
            "observed? Is there evidence of saturation? What is the relationship between "
            "number of transplanted mitochondria and functional improvement? Cite specific "
            "studies with quantitative dose-response data.\n\n"
            "3. **Engraftment efficiency**: What fraction of transplanted mitochondria actually "
            "engraft and become functional? Does this change with recipient cell health? "
            "Is there a saturation point for uptake capacity?\n\n"
            "4. **Competitive dynamics**: Do transplanted healthy mitochondria actually "
            "displace damaged ones? What is the mechanism — selective mitophagy of damaged "
            "copies, replication advantage of healthy copies, or passive displacement? "
            "Cite evidence from heteroplasmy studies.\n\n"
            "5. **Feedback loops**: Is there evidence for positive feedback loops in "
            "mitochondrial transplant? Specifically: does improving ATP via transplant "
            "enhance the cell's own quality control (mitophagy), creating a self-reinforcing "
            "improvement cycle?\n\n"
            "6. **Clinical dose ranges**: In human trials, what doses are used? How many "
            "mitochondria per treatment? How does this scale relative to the ~1000-2000 "
            "mitochondria per cell?\n\n"
            "7. **Long-term dynamics**: Do transplanted mitochondria persist? Do they "
            "replicate? Does the benefit persist after a single treatment or require "
            "repeated dosing?\n\n"
            "8. **Cramer's platelet-derived mitlets**: John Cramer's book (How to Live Much "
            "Longer, 2026, Springer) proposes platelet-derived mitochondrial transfer as the "
            "primary rejuvenation strategy. What is the evidence for platelet-derived "
            "mitochondrial transfer specifically? Cite any relevant literature.\n\n"
            "Provide specific citations, quantitative data, and any dose-response curves "
            "or saturation effects observed in the literature."
        ),
    },
}


def query_ollama(model: str, prompt: str, max_tokens: int = 4000, timeout: int = 300) -> str | None:
    """Query Ollama and return raw response."""
    effective_max = 6000 if model in REASONING_MODELS else max_tokens
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": effective_max},
    })
    try:
        r = subprocess.run(
            ["curl", "-s", OLLAMA_URL, "-d", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        if "error" in data:
            print(f"  Error from {model}: {data['error']}")
            return None
        response = data["response"]
        # Strip think tags from reasoning models
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        return response
    except subprocess.TimeoutExpired:
        print(f"  Timeout querying {model}")
        return None
    except Exception as e:
        print(f"  Exception querying {model}: {e}")
        return None


def run_finding(finding_num: int, finding: dict) -> None:
    """Query all models for a single finding and save results."""
    print(f"\n{'='*70}")
    print(f"Finding {finding_num}: {finding['title']}")
    print(f"{'='*70}")

    preamble = (
        "You are a biomedical research assistant. I need a thorough literature review "
        "to help calibrate a computational model of mitochondrial aging.\n\n"
        f"**Context**: {finding['model_behavior']}\n\n"
        "**Task**: Provide a comprehensive literature review with specific citations. "
        "For each cited paper, include: author(s), year, journal, and key quantitative "
        "finding. Organize your response with clear headers. Focus on empirical data "
        "and quantitative results, not speculation.\n\n"
    )

    for model in MODELS:
        print(f"\n  Querying {model}...")
        t0 = time.time()
        response = query_ollama(model, preamble + finding["prompt"])
        elapsed = time.time() - t0

        if response:
            # Save individual model response
            model_slug = model.replace(":", "_").replace(".", "_")
            out_path = OUTPUT_DIR / f"finding_{finding_num}_{model_slug}.md"
            with open(out_path, "w") as f:
                f.write(f"# Finding {finding_num}: {finding['title']}\n\n")
                f.write(f"**Model**: {model}\n")
                f.write(f"**Query time**: {elapsed:.1f}s\n\n")
                f.write("---\n\n")
                f.write(response)
                f.write("\n")
            print(f"  Saved to {out_path} ({len(response)} chars, {elapsed:.1f}s)")
        else:
            print(f"  No response from {model}")


def compile_summary() -> None:
    """Compile all individual results into a single summary document."""
    summary_path = OUTPUT_DIR / "compiled_lit_search_2026-02-22.md"
    with open(summary_path, "w") as f:
        f.write("# Literature Search for Model Falsification\n\n")
        f.write("**Date**: 2026-02-22\n")
        f.write(f"**Models queried**: {', '.join(MODELS)}\n")
        f.write("**Purpose**: Gather published evidence to falsify/validate 5 surprising "
                "findings from the mitochondrial aging simulation\n\n")
        f.write("---\n\n")

        for finding_num, finding in sorted(FINDINGS.items()):
            f.write(f"## Finding {finding_num}: {finding['title']}\n\n")
            f.write(f"**Model behavior**: {finding['model_behavior']}\n\n")

            for model in MODELS:
                model_slug = model.replace(":", "_").replace(".", "_")
                path = OUTPUT_DIR / f"finding_{finding_num}_{model_slug}.md"
                if path.exists():
                    content = path.read_text()
                    # Extract just the response (after the --- separator)
                    if "---\n\n" in content:
                        content = content.split("---\n\n", 1)[1]
                    f.write(f"### {model}\n\n")
                    f.write(content)
                    f.write("\n\n")

            f.write("---\n\n")

    print(f"\nCompiled summary saved to {summary_path}")


if __name__ == "__main__":
    print("Literature Search for Model Falsification")
    print("Using local Ollama LLMs to generate evidence summaries")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Output: {OUTPUT_DIR}/")

    for finding_num, finding in sorted(FINDINGS.items()):
        run_finding(finding_num, finding)

    compile_summary()
    print("\nDone! All evidence written to artifacts/lit_search/")
