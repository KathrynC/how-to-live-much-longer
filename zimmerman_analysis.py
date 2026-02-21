"""Zimmerman toolkit full analysis for the mitochondrial aging simulator.

Runs all 14 Zimmerman interrogation tools against MitoSimulator and saves
results to artifacts/zimmerman/. Generates a unified dashboard report.

Usage:
    python zimmerman_analysis.py                           # all tools
    python zimmerman_analysis.py --tools sobol             # single tool
    python zimmerman_analysis.py --tools sobol,falsifier   # multiple tools
    python zimmerman_analysis.py --patient near_cliff_80   # different patient
    python zimmerman_analysis.py --viz                     # also generate plots
    python zimmerman_analysis.py --n-base 128              # Sobol sample size

Requires:
    zimmerman-toolkit (at ~/zimmerman-toolkit)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent
ZIMMERMAN_PATH = PROJECT.parent / "zimmerman-toolkit"
if str(ZIMMERMAN_PATH) not in sys.path:
    sys.path.insert(0, str(ZIMMERMAN_PATH))

# Project imports
from constants import (
    INTERVENTION_PARAMS, PATIENT_PARAMS,
    INTERVENTION_NAMES, PATIENT_NAMES,
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    HETEROPLASMY_CLIFF,
)
from analytics import NumpyEncoder
from zimmerman_bridge import MitoSimulator

# Zimmerman imports
from zimmerman.sobol import sobol_sensitivity
from zimmerman.falsifier import Falsifier
from zimmerman.contrastive import ContrastiveGenerator
from zimmerman.contrast_set_generator import ContrastSetGenerator
from zimmerman.pds import PDSMapper
from zimmerman.posiwid import POSIWIDAuditor
from zimmerman.prompts import PromptBuilder
from zimmerman.locality_profiler import LocalityProfiler
from zimmerman.relation_graph_extractor import RelationGraphExtractor
from zimmerman.diegeticizer import Diegeticizer
from zimmerman.token_extispicy import TokenExtispicyWorkbench
from zimmerman.prompt_receptive_field import PromptReceptiveField
from zimmerman.supradiegetic_benchmark import SuperdiegeticBenchmark
from zimmerman.meaning_construction_dashboard import MeaningConstructionDashboard

# ── Configuration ────────────────────────────────────────────────────────────

ARTIFACTS_DIR = PROJECT / "artifacts" / "zimmerman"

PATIENT_PROFILES = {
    "default": DEFAULT_PATIENT,
    "near_cliff_80": {
        "baseline_age": 80.0, "baseline_heteroplasmy": 0.65,
        "baseline_nad_level": 0.4, "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.5,
    },
    "young_prevention_25": {
        "baseline_age": 25.0, "baseline_heteroplasmy": 0.05,
        "baseline_nad_level": 0.95, "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.0,
    },
    "melas_35": {
        "baseline_age": 35.0, "baseline_heteroplasmy": 0.50,
        "baseline_nad_level": 0.6, "genetic_vulnerability": 2.0,
        "metabolic_demand": 1.0, "inflammation_level": 0.25,
    },
}

# Tool names in execution order
ALL_TOOLS = [
    "sobol", "falsifier", "contrastive", "contrast_sets",
    "pds", "posiwid", "prompts",
    "locality", "relation_graph", "diegeticizer",
    "token_extispicy", "receptive_field", "supradiegetic_benchmark",
    "dashboard",
]

# ── Helper: outcome function for contrastive/contrast set tools ───────────────

def _cliff_outcome(result: dict) -> str:
    """Classify simulation outcome relative to the heteroplasmy cliff."""
    het = result.get("final_heteroplasmy", result.get("damage_het_final", 0.5))
    if het >= HETEROPLASMY_CLIFF:
        return "collapsed"
    elif het >= HETEROPLASMY_CLIFF - 0.1:
        return "near_cliff"
    else:
        return "healthy"


# ── Helper: default midpoint params ──────────────────────────────────────────

def _midpoint_params(sim: MitoSimulator) -> dict[str, float]:
    """Return parameter values at the midpoint of each range."""
    spec = sim.param_spec()
    return {k: (lo + hi) / 2 for k, (lo, hi) in spec.items()}


def _default_full_params(sim: MitoSimulator | None = None) -> dict[str, float]:
    """Return default parameter dict aligned to a simulator's param spec.

    If `sim` is None, returns the canonical 12D mito defaults.
    If `sim` is provided (e.g., GriefMitoSimulator with 26D), returns a
    full parameter dict matching `sim.param_spec()`:
      - known mito keys use project defaults
      - unknown keys are initialized to range midpoints
    """
    base = {**DEFAULT_INTERVENTION, **DEFAULT_PATIENT}
    if sim is None:
        return base

    spec = sim.param_spec()
    out = {}
    for k, (lo, hi) in spec.items():
        if k in base:
            out[k] = float(base[k])
        else:
            out[k] = float((lo + hi) / 2.0)
    return out


# ── Tool Runners ──────────────────────────────────────────────────────────────


def run_sobol(sim: MitoSimulator, n_base: int = 256, seed: int = 42) -> dict:
    """Run Sobol global sensitivity analysis.

    Total sims: n_base * (2*D + 2) where D = number of params.
    For 12D with n_base=256: 256 * 26 = 6,656 sims.
    For 6D with n_base=256: 256 * 14 = 3,584 sims.
    """
    print(f"  Sobol: n_base={n_base}, D={len(sim.param_spec())}")
    result = sobol_sensitivity(sim, n_base=n_base, seed=seed)
    print(f"  Sobol: {result.get('n_total_sims', '?')} sims completed")
    return result


def run_falsifier(sim: MitoSimulator, seed: int = 42) -> dict:
    """Run systematic falsification with JGC-specific assertions."""

    def het_in_range(result):
        het = result.get("final_heteroplasmy", 0.5)
        return 0.0 <= het <= 1.0

    def atp_non_negative(result):
        return result.get("final_atp", 0.0) >= 0.0

    def no_nan(result):
        return all(
            np.isfinite(v) for v in result.values()
            if isinstance(v, (int, float))
        )

    def ros_non_negative(result):
        return result.get("final_ros", 0.0) >= 0.0

    def senescent_in_range(result):
        sen = result.get("final_senescent", 0.0)
        return 0.0 <= sen <= 1.0

    falsifier = Falsifier(
        sim,
        assertions=[het_in_range, atp_non_negative, no_nan,
                     ros_non_negative, senescent_in_range],
    )
    result = falsifier.falsify(n_random=100, n_boundary=50, seed=seed)
    n_violations = result.get("summary", {}).get("violations_found", 0)
    n_tests = result.get("summary", {}).get("total_tests", 0)
    print(f"  Falsifier: {n_violations}/{n_tests} violations")
    return result


def run_contrastive(sim: MitoSimulator, seed: int = 42) -> dict:
    """Find minimal parameter changes that flip cliff outcome."""
    dim = len(sim.param_spec())
    spec = sim.param_spec()

    # Use a few starting points
    starts = [
        _default_full_params(sim),
        _midpoint_params(sim),
    ]
    # Add a near-cliff starting point
    near_cliff = _default_full_params(sim)
    if "baseline_heteroplasmy" in near_cliff:
        near_cliff["baseline_heteroplasmy"] = 0.60
    starts.append(near_cliff)

    # Nuance-specific fallback for high-dimensional spaces.
    # The toolkit's generic contrastive search can become very expensive in 26D.
    # This approximate mode still produces actionable flip pairs and sensitivity
    # rankings by random one-parameter edits around representative starts.
    if dim > 20:
        rng = np.random.default_rng(seed)
        keys = list(spec.keys())
        pairs = []
        sensitivity_counts = {}
        n_trials_per_start = 80

        for base in starts:
            base_full = _midpoint_params(sim)
            base_full.update(base)
            base_outcome = _cliff_outcome(sim.run(base_full))

            for _ in range(n_trials_per_start):
                cand = dict(base_full)
                key = str(rng.choice(keys))
                lo, hi = spec[key]
                cand[key] = float(rng.uniform(lo, hi))
                cand_outcome = _cliff_outcome(sim.run(cand))
                if cand_outcome != base_outcome:
                    pairs.append({
                        "base_outcome": base_outcome,
                        "counterfactual_outcome": cand_outcome,
                        "edited_params": [key],
                        "base": base_full,
                        "counterfactual": cand,
                    })
                    sensitivity_counts[key] = sensitivity_counts.get(key, 0) + 1

        total = max(sum(sensitivity_counts.values()), 1)
        sensitivity = {
            "rankings": sorted(
                [
                    (k, v / total)
                    for k, v in sensitivity_counts.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
            "method": "approx_random_single_edit",
            "n_trials_per_start": n_trials_per_start,
        }
        result = {
            "n_pairs": len(pairs),
            "pairs": pairs,
            "sensitivity": sensitivity,
        }
        print(f"  Contrastive: {len(pairs)} flip pairs found "
              f"(approx mode, D={dim})")
        return result

    gen = ContrastiveGenerator(sim, outcome_fn=_cliff_outcome)
    # Nuance compatibility: high-dimensional spaces (e.g., 26D grief+mito)
    # become combinatorially expensive with the 12D default probe count.
    n_per_point = 30 if dim <= 12 else 8
    pairs = gen.contrastive_pairs(starts, n_per_point=n_per_point, seed=seed)
    sensitivity = gen.sensitivity_from_contrastives(pairs) if pairs else {}

    result = {
        "n_pairs": len(pairs),
        "pairs": pairs,
        "sensitivity": sensitivity,
    }
    print(f"  Contrastive: {len(pairs)} flip pairs found "
          f"(n_per_point={n_per_point}, D={dim})")
    return result


def run_contrast_sets(sim: MitoSimulator, seed: int = 42) -> dict:
    """Find minimal ordered edit sequences that flip cliff outcome."""
    gen = ContrastSetGenerator(sim, outcome_fn=_cliff_outcome)
    base = _default_full_params(sim)
    result = gen.batch_contrast_sets(base, n_paths=10, n_edits=20, seed=seed)
    n_tips = len(result.get("pairs", []))
    print(f"  ContrastSets: {n_tips} tipping points found, "
          f"mean flip size: {result.get('mean_flip_size', 'N/A')}")
    return result


def run_pds(sim: MitoSimulator, seed: int = 42) -> dict:
    """Map Power/Danger/Structure dimensions to mitochondrial parameters.

    PDS mapping for JGC:
      Power    → rapamycin, transplant, exercise (protective mechanisms)
      Danger   → yamanaka_intensity, inflammation_level, genetic_vulnerability
      Structure → nad_supplement, senolytic_dose, baseline_age, metabolic_demand
    """
    mapping = {
        "power": {
            "rapamycin_dose": 0.4,
            "transplant_rate": 0.4,
            "exercise_level": 0.2,
        },
        "danger": {
            "yamanaka_intensity": 0.3,
            "inflammation_level": 0.4,
            "genetic_vulnerability": 0.3,
        },
        "structure": {
            "nad_supplement": 0.3,
            "senolytic_dose": 0.2,
            "baseline_age": 0.2,
            "metabolic_demand": 0.15,
            "baseline_heteroplasmy": 0.15,
        },
    }
    pds = PDSMapper(sim, dimension_names=["power", "danger", "structure"],
                     dimension_to_param_mapping=mapping)
    audit = pds.audit_mapping(n_samples=100, seed=seed)
    sensitivity = pds.sensitivity_per_dimension(n_samples=100, seed=seed)
    result = {
        "mapping": mapping,
        "audit": audit,
        "sensitivity": sensitivity,
    }
    print(f"  PDS: variance explained: "
          f"{', '.join(f'{k}={v:.3f}' for k, v in audit.get('variance_explained', {}).items())}"
          [:80])
    return result


def run_posiwid(sim: MitoSimulator, seed: int = 42) -> dict:
    """Audit alignment between intended and actual outcomes.

    Tests clinical intention scenarios (e.g., "I intend to reduce heteroplasmy")
    against what the simulator actually produces.
    """
    auditor = POSIWIDAuditor(sim)

    scenarios = [
        {
            "label": "Mild rapamycin for mitophagy",
            "intended": {
                "final_heteroplasmy": 0.25,
                "final_atp": 0.85,
            },
            "params": {
                "rapamycin_dose": 0.25, "nad_supplement": 0.0,
                "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
                "transplant_rate": 0.0, "exercise_level": 0.0,
                **DEFAULT_PATIENT,
            },
        },
        {
            "label": "Full cocktail for near-cliff patient",
            "intended": {
                "final_heteroplasmy": 0.30,
                "final_atp": 0.80,
            },
            "params": {
                "rapamycin_dose": 0.5, "nad_supplement": 0.75,
                "senolytic_dose": 0.5, "yamanaka_intensity": 0.0,
                "transplant_rate": 0.5, "exercise_level": 0.5,
                "baseline_age": 80.0, "baseline_heteroplasmy": 0.65,
                "baseline_nad_level": 0.4, "genetic_vulnerability": 1.0,
                "metabolic_demand": 1.0, "inflammation_level": 0.5,
            },
        },
        {
            "label": "Transplant-only rejuvenation",
            "intended": {
                "final_heteroplasmy": 0.15,
                "final_atp": 0.90,
            },
            "params": {
                "rapamycin_dose": 0.0, "nad_supplement": 0.0,
                "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
                "transplant_rate": 1.0, "exercise_level": 0.0,
                **DEFAULT_PATIENT,
            },
        },
        {
            "label": "Exercise + NAD for young prevention",
            "intended": {
                "final_heteroplasmy": 0.05,
                "final_atp": 0.95,
            },
            "params": {
                "rapamycin_dose": 0.0, "nad_supplement": 0.5,
                "senolytic_dose": 0.0, "yamanaka_intensity": 0.0,
                "transplant_rate": 0.0, "exercise_level": 0.75,
                "baseline_age": 25.0, "baseline_heteroplasmy": 0.05,
                "baseline_nad_level": 0.95, "genetic_vulnerability": 1.0,
                "metabolic_demand": 1.0, "inflammation_level": 0.0,
            },
        },
        {
            "label": "Yamanaka high intensity (energy risk)",
            "intended": {
                "final_heteroplasmy": 0.20,
                "final_atp": 0.50,
            },
            "params": {
                "rapamycin_dose": 0.0, "nad_supplement": 0.0,
                "senolytic_dose": 0.0, "yamanaka_intensity": 0.75,
                "transplant_rate": 0.0, "exercise_level": 0.0,
                **DEFAULT_PATIENT,
            },
        },
    ]

    result = auditor.batch_audit(scenarios)
    overall = result.get("aggregate", {}).get("mean_overall", 0.0)
    print(f"  POSIWID: {len(scenarios)} scenarios, mean alignment={overall:.3f}")
    return result


def run_prompts(sim: MitoSimulator) -> dict:
    """Build prompt templates for LLM-mediated parameter generation."""
    builder = PromptBuilder(sim, context={
        "domain": "Mitochondrial aging dynamics (Cramer, forthcoming 2026)",
        "goal": "Design intervention protocols to delay heteroplasmy cliff",
    })
    scenario = ("70-year-old patient with 30% heteroplasmy, declining NAD+, "
                "moderate inflammation. Design an intervention protocol.")

    result = {
        "numeric": builder.build_numeric(scenario),
        "diegetic": builder.build_diegetic(
            scenario,
            state_description="Current state: het=0.30, ATP=0.82, NAD=0.60"),
        "contrastive": builder.build_contrastive(
            scenario,
            agent_a="conservative geriatrician",
            agent_b="aggressive biohacker"),
    }
    print(f"  Prompts: 3 styles generated "
          f"(numeric={len(result['numeric'])}ch, "
          f"diegetic={len(result['diegetic'])}ch, "
          f"contrastive={len(result['contrastive'])}ch)")
    return result


def run_locality(sim: MitoSimulator, seed: int = 42) -> dict:
    """Profile perturbation decay: how local are the system's responses?"""
    profiler = LocalityProfiler(sim)
    base = _default_full_params(sim)
    result = profiler.profile(task={"base_params": base}, n_seeds=10, seed=seed)
    n_sims = result.get("n_sims", "?")
    print(f"  Locality: {n_sims} sims, profiled decay curves")
    return result


def run_relation_graph(sim: MitoSimulator, seed: int = 42) -> dict:
    """Build causal relation graph: param → output influence."""
    extractor = RelationGraphExtractor(sim)
    base = _default_full_params(sim)
    result = extractor.extract(base, n_probes=50, seed=seed)
    n_causal = len(result.get("edges", {}).get("causal", []))
    print(f"  RelationGraph: {n_causal} causal edges, "
          f"{result.get('n_sims', '?')} sims")
    return result


def run_diegeticizer(sim: MitoSimulator) -> dict:
    """Roundtrip diegeticization with clinical lexicon."""
    lexicon = {
        "rapamycin_dose": "rapamycin_intensity",
        "nad_supplement": "NAD_restoration",
        "senolytic_dose": "senolytic_clearance",
        "yamanaka_intensity": "reprogramming_strength",
        "transplant_rate": "mitochondrial_infusion",
        "exercise_level": "physical_activity",
        "baseline_age": "patient_age",
        "baseline_heteroplasmy": "damage_level",
        "baseline_nad_level": "cellular_NAD",
        "genetic_vulnerability": "genetic_fragility",
        "metabolic_demand": "tissue_demand",
        "inflammation_level": "chronic_inflammation",
    }
    # Nuance compatibility: include grief and any other simulator-specific keys.
    # Diegeticizer requires a complete lexicon over param_spec keys.
    for key in sim.param_spec().keys():
        lexicon.setdefault(key, key)
    dieg = Diegeticizer(sim, lexicon=lexicon, n_bins=5)
    params = _default_full_params(sim)
    narrative = dieg.diegeticize(params)
    recovered = dieg.re_diegeticize(narrative["narrative"])
    roundtrip = dieg.run(params)
    result = {
        "narrative": narrative,
        "recovered": recovered,
        "roundtrip": roundtrip,
        "lexicon": lexicon,
    }
    error = narrative.get("roundtrip_error", "?")
    print(f"  Diegeticizer: roundtrip error={error}")
    return result


def run_token_extispicy(sim: MitoSimulator, seed: int = 42) -> dict:
    """Quantify tokenization-induced flattening as hazard surface."""
    workbench = TokenExtispicyWorkbench(sim)
    result = workbench.analyze(n_samples=100, seed=seed)
    frag_corr = result.get("fragmentation_output_correlation", "?")
    print(f"  TokenExtispicy: frag-output correlation={frag_corr}, "
          f"{result.get('n_sims', '?')} sims")
    return result


def run_receptive_field(sim: MitoSimulator, seed: int = 42) -> dict:
    """Sobol analysis over parameter segment groupings.

    Segments for JGC:
      pharmacological → rapamycin, nad_supplement, senolytic, yamanaka
      biological       → transplant, exercise
      demographics     → baseline_age, baseline_heteroplasmy, baseline_nad_level
      vulnerability    → genetic_vulnerability, metabolic_demand, inflammation_level
    """
    jgc_segments = [
        {"name": "pharmacological",
         "params": ["rapamycin_dose", "nad_supplement", "senolytic_dose",
                     "yamanaka_intensity"]},
        {"name": "biological",
         "params": ["transplant_rate", "exercise_level"]},
        {"name": "demographics",
         "params": ["baseline_age", "baseline_heteroplasmy",
                     "baseline_nad_level"]},
        {"name": "vulnerability",
         "params": ["genetic_vulnerability", "metabolic_demand",
                     "inflammation_level"]},
    ]

    def jgc_segmenter(spec):
        """Group JGC params into clinical segments."""
        available = set(spec.keys())
        result = []
        for seg in jgc_segments:
            params = [p for p in seg["params"] if p in available]
            if params:
                result.append({"name": seg["name"], "params": params})
        return result

    field = PromptReceptiveField(sim, segmenter=jgc_segmenter)
    base = _default_full_params()
    result = field.analyze(base_params=base, n_base=64, seed=seed)
    rankings = result.get("rankings", [])
    if isinstance(rankings, dict):
        print(f"  ReceptiveField: most influential={rankings.get('most_influential', '?')}")
    else:
        print(f"  ReceptiveField: rankings={rankings[:5] if rankings else '?'}")
    return result


def run_supradiegetic_benchmark(sim: MitoSimulator, seed: int = 42) -> dict:
    """Run form-vs-meaning benchmark."""
    bench = SuperdiegeticBenchmark(sim)
    result = bench.run_benchmark(seed=seed)
    gain = result.get("summary", {}).get("mean_gain", "?")
    print(f"  SuperdiegeticBenchmark: mean diegeticization gain={gain}")
    return result


def run_dashboard(reports: dict, sim: MitoSimulator) -> dict:
    """Compile all reports into unified dashboard."""
    dashboard = MeaningConstructionDashboard(sim)
    result = dashboard.compile(reports)
    coverage = result.get("coverage", {})
    n_recs = len(result.get("recommendations", []))
    print(f"  Dashboard: {coverage.get('tools_present', 0)}/"
          f"{coverage.get('tools_total', 0)} tools, "
          f"{n_recs} recommendations")
    return result


# ── Tool dispatcher ──────────────────────────────────────────────────────────

TOOL_RUNNERS = {
    "sobol": lambda sim, args: run_sobol(sim, n_base=args.n_base),
    "falsifier": lambda sim, args: run_falsifier(sim),
    "contrastive": lambda sim, args: run_contrastive(sim),
    "contrast_sets": lambda sim, args: run_contrast_sets(sim),
    "pds": lambda sim, args: run_pds(sim),
    "posiwid": lambda sim, args: run_posiwid(sim),
    "prompts": lambda sim, args: run_prompts(sim),
    "locality": lambda sim, args: run_locality(sim),
    "relation_graph": lambda sim, args: run_relation_graph(sim),
    "diegeticizer": lambda sim, args: run_diegeticizer(sim),
    "token_extispicy": lambda sim, args: run_token_extispicy(sim),
    "receptive_field": lambda sim, args: run_receptive_field(sim),
    "supradiegetic_benchmark": lambda sim, args: run_supradiegetic_benchmark(sim),
    # dashboard is special — needs all other reports
}


# ── Generate markdown report ─────────────────────────────────────────────────

def _generate_markdown(reports: dict, dashboard: dict | None) -> str:
    """Generate a markdown summary of all Zimmerman analysis results."""
    lines = [
        "# Zimmerman Toolkit Analysis — JGC Mitochondrial Simulator",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Sobol summary
    if "sobol" in reports:
        sobol = reports["sobol"]
        lines.append("## Sobol Global Sensitivity")
        lines.append(f"- Base samples: {sobol.get('n_base', '?')}")
        lines.append(f"- Total sims: {sobol.get('n_total_sims', '?')}")
        lines.append(f"- Parameters: {sobol.get('parameter_names', [])}")
        for key in sobol.get("output_keys", [])[:3]:
            if key in sobol:
                s1 = sobol[key].get("S1", {})
                top3 = sorted(s1.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                lines.append(f"- **{key}** top S1: " +
                             ", ".join(f"{k}={v:.3f}" for k, v in top3))
        lines.append("")

    # Falsifier summary
    if "falsifier" in reports:
        summary = reports["falsifier"].get("summary", {})
        lines.append("## Falsification")
        lines.append(f"- Tests run: {summary.get('total_tests', '?')}")
        lines.append(f"- Violations: {summary.get('violations_found', 0)}")
        lines.append(f"- Violation rate: {summary.get('violation_rate', 0):.1%}")
        lines.append("")

    # Contrastive summary
    if "contrastive" in reports:
        contrastive = reports["contrastive"]
        lines.append("## Contrastive Analysis")
        lines.append(f"- Flip pairs found: {contrastive.get('n_pairs', 0)}")
        sens = contrastive.get("sensitivity", {})
        if "rankings" in sens:
            lines.append(f"- Most flip-prone params: {sens['rankings'][:5]}")
        lines.append("")

    # POSIWID summary
    if "posiwid" in reports:
        agg = reports["posiwid"].get("aggregate", {})
        lines.append("## POSIWID Alignment")
        lines.append(f"- Mean overall alignment: {agg.get('mean_overall', 0):.3f}")
        lines.append(f"- Direction accuracy: {agg.get('mean_direction_accuracy', 0):.3f}")
        lines.append(f"- Magnitude accuracy: {agg.get('mean_magnitude_accuracy', 0):.3f}")
        lines.append("")

    # PDS summary
    if "pds" in reports:
        pds_audit = reports["pds"].get("audit", {})
        lines.append("## PDS Mapping")
        ve = pds_audit.get("variance_explained", {})
        for k, v in ve.items():
            lines.append(f"- {k} variance explained: {v:.3f}")
        lines.append("")

    # Dashboard summary
    if dashboard:
        coverage = dashboard.get("coverage", {})
        lines.append("## Dashboard Summary")
        lines.append(f"- Coverage: {coverage.get('tools_present', 0)}/"
                      f"{coverage.get('tools_total', 0)} tools "
                      f"({coverage.get('coverage_pct', 0):.0f}%)")
        recs = dashboard.get("recommendations", [])
        if recs:
            lines.append("### Recommendations")
            for rec in recs[:10]:
                if isinstance(rec, dict):
                    lines.append(f"- **{rec.get('finding', '')}** → {rec.get('action', '')}")
                else:
                    lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zimmerman toolkit analysis for mitochondrial aging simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tools: {', '.join(ALL_TOOLS)}\n"
               f"Available patients: {', '.join(PATIENT_PROFILES.keys())}",
    )
    parser.add_argument("--tools", type=str, default=None,
                        help="Comma-separated tool names (default: all)")
    parser.add_argument("--patient", type=str, default=None,
                        help="Patient profile for intervention-only mode")
    parser.add_argument("--viz", action="store_true",
                        help="Generate visualizations after analysis")
    parser.add_argument("--n-base", type=int, default=256,
                        help="Sobol base sample count (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Determine which tools to run
    if args.tools:
        tools = [t.strip() for t in args.tools.split(",")]
        for t in tools:
            if t not in ALL_TOOLS:
                print(f"Unknown tool: {t}. Available: {', '.join(ALL_TOOLS)}")
                sys.exit(1)
    else:
        tools = ALL_TOOLS

    # Create simulator
    if args.patient:
        patient = PATIENT_PROFILES.get(args.patient, DEFAULT_PATIENT)
        sim = MitoSimulator(intervention_only=True, patient_override=patient)
        print(f"Mode: intervention-only (patient={args.patient})")
    else:
        sim = MitoSimulator()
        print(f"Mode: full 12D (intervention + patient)")

    print(f"Parameters: {len(sim.param_spec())}D")
    print(f"Tools: {', '.join(tools)}")
    print(f"Sobol n_base: {args.n_base}")
    print()

    # Create output directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run tools
    reports = {}
    total_t0 = time.time()

    for tool_name in tools:
        if tool_name == "dashboard":
            continue  # run last
        if tool_name not in TOOL_RUNNERS:
            print(f"  Skipping unknown tool: {tool_name}")
            continue

        print(f"[{tool_name}]")
        t0 = time.time()
        try:
            result = TOOL_RUNNERS[tool_name](sim, args)
            reports[tool_name] = result
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            # Save individual report
            out_path = ARTIFACTS_DIR / f"{tool_name}_report.json"
            out_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder,
                                            default=str))
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            reports[tool_name] = {"error": str(e)}
        print()

    # Dashboard compilation
    dashboard_result = None
    if "dashboard" in tools:
        print("[dashboard]")
        t0 = time.time()
        try:
            # Map report keys to dashboard expected keys
            dashboard_input = {}
            key_mapping = {
                "sobol": "sobol",
                "falsifier": "falsifier",
                "posiwid": "posiwid",
                "contrast_sets": "contrast_sets",
                "contrastive": "contrastive",
                "locality": "locality",
                "relation_graph": "relation_graph",
                "receptive_field": "receptive_field",
                "diegeticizer": "diegeticizer",
                "supradiegetic_benchmark": "benchmark",
                "token_extispicy": "token_extispicy",
                "pds": "pds",
            }
            for our_key, dash_key in key_mapping.items():
                if our_key in reports and "error" not in reports[our_key]:
                    dashboard_input[dash_key] = reports[our_key]

            dashboard_result = run_dashboard(dashboard_input, sim)
            reports["dashboard"] = dashboard_result
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            out_path = ARTIFACTS_DIR / "dashboard.json"
            out_path.write_text(json.dumps(dashboard_result, indent=2,
                                            cls=NumpyEncoder, default=str))
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Generate markdown report
    md = _generate_markdown(reports, dashboard_result)
    md_path = ARTIFACTS_DIR / "dashboard.md"
    md_path.write_text(md)
    print(f"Markdown report: {md_path}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Reports saved to: {ARTIFACTS_DIR}/")

    # Optionally generate visualizations
    if args.viz:
        print("\nGenerating visualizations...")
        try:
            from zimmerman_viz import generate_all_visualizations
            generate_all_visualizations(reports)
            print("Visualizations complete.")
        except Exception as e:
            print(f"Visualization error: {e}")


if __name__ == "__main__":
    main()
