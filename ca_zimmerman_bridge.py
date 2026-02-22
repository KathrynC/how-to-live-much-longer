"""Zimmerman Simulator protocol adapters for the mitochondrial semantic CA.

Three adapters wrapping the CA simulation modes as Zimmerman-compatible
simulators, enabling all 14 interrogation tools (Sobol, Falsifier,
ContrastiveGenerator, etc.) to operate on the discrete CA dynamics:

1. MitoCASimulator     -- single-cell CA (12D, deterministic)
2. MitoTissueSimulator -- 4-tissue grid CA (13D, deterministic)
3. MitoCAEnsembleSimulator -- stochastic ensemble CA (12D, probabilistic)

The Zimmerman Simulator protocol requires:
    run(params: dict) -> dict   -- flat param dict in, flat metric dict out
    param_spec() -> dict[str, tuple[float, float]]   -- parameter bounds

Usage:
    from ca_zimmerman_bridge import MitoCASimulator

    sim = MitoCASimulator()
    result = sim.run({"rapamycin_dose": 0.5, "baseline_age": 70})
    spec = sim.param_spec()

Mirrors zimmerman_bridge.py architecture for the ODE simulator.
"""
from __future__ import annotations


# ── Flatten helpers ──────────────────────────────────────────────────────


def _flatten_ca_analytics(analytics, ca_result):
    """Flatten CA analytics to flat dict of scalars."""
    out = {}

    # Rule stats
    rs = analytics.get("rule_stats", {})
    out["ca_total_firings"] = float(rs.get("total_firings", 0))
    out["ca_unique_rules"] = float(rs.get("unique_rules", 0))
    out["ca_mean_rules_per_step"] = float(rs.get("mean_rules_per_step", 0))

    # Cascade stats
    cs = analytics.get("cascade_stats", {})
    out["ca_n_cascades"] = float(cs.get("n_cascades", 0))
    out["ca_max_cascade_length"] = float(cs.get("max_cascade_length", 0))

    # Attractor stats
    att = analytics.get("attractor_stats", {})
    out["ca_attractor_transitions"] = float(att.get("attractor_transitions", 0))

    # Encode final attractor as numeric
    attractor_map = {
        "healthy_aging": 0.0,
        "slow_decline": 1.0,
        "cliff_approaching": 2.0,
        "point_of_no_return": 3.0,
    }
    final_att = att.get("final_attractor", "healthy_aging")
    out["ca_final_attractor"] = attractor_map.get(final_att, -1.0)

    # Time in attractor fractions
    time_in = att.get("time_in_attractor", {})
    for att_name in attractor_map:
        out[f"ca_time_{att_name}"] = float(time_in.get(att_name, 0.0))

    # Epoch diagnostic
    ep = analytics.get("epoch_diagnostic", {})
    out["ca_epoch_n_vars_changed"] = float(ep.get("n_variables_changed", 0))
    transition_step = ep.get("transition_step")
    out["ca_epoch_transition_step"] = (
        float(transition_step) if transition_step is not None else -1.0
    )

    # Final state bin indices
    from ca_schema import bin_index, CA_VAR_ORDER

    final = ca_result.get("final_state", {})
    for var_name in CA_VAR_ORDER:
        label = final.get(var_name, "unknown")
        try:
            out[f"ca_final_{var_name}"] = float(bin_index(var_name, label))
        except ValueError:
            out[f"ca_final_{var_name}"] = -1.0

    # NaN/Inf guard
    for k, v in out.items():
        if v != v:  # NaN
            out[k] = 0.0
        elif v == float("inf") or v == float("-inf"):
            out[k] = 999.0

    return out


def _flatten_tissue_summary(tissue_result, tissue_analytics):
    """Flatten tissue grid results to scalars."""
    out = {}

    attractor_map = {
        "healthy_aging": 0.0,
        "slow_decline": 1.0,
        "cliff_approaching": 2.0,
        "point_of_no_return": 3.0,
    }

    # Per-tissue attractor
    tissue_att = tissue_analytics.get("tissue_attractors", {})
    for tissue, att in tissue_att.items():
        out[f"tissue_{tissue}_attractor"] = attractor_map.get(att, -1.0)

    out["tissue_n_distinct_attractors"] = float(
        tissue_analytics.get("n_distinct_attractors", 0)
    )

    # Per-tissue final bin indices
    from ca_schema import bin_index, CA_VAR_ORDER

    final_tissues = tissue_result.get("final_tissues", {})
    for tissue, state in final_tissues.items():
        for var_name in CA_VAR_ORDER:
            label = state.get(var_name, "unknown")
            try:
                out[f"tissue_{tissue}_{var_name}"] = float(
                    bin_index(var_name, label)
                )
            except ValueError:
                out[f"tissue_{tissue}_{var_name}"] = -1.0

    # Population summary
    summary = tissue_result.get("population_summary", {})
    out["tissue_total"] = float(summary.get("total_tissues", 4))

    # NaN/Inf guard
    for k, v in out.items():
        if v != v:
            out[k] = 0.0
        elif v == float("inf") or v == float("-inf"):
            out[k] = 999.0

    return out


def _flatten_ensemble_analytics(analytics):
    """Flatten ensemble analytics to scalars."""
    out = {}

    # Attractor probabilities
    probs = analytics.get("attractor_probabilities", {})
    for att in (
        "healthy_aging",
        "slow_decline",
        "cliff_approaching",
        "point_of_no_return",
    ):
        out[f"ens_{att}_prob"] = float(probs.get(att, 0.0))

    out["ens_cliff_crossing_prob"] = float(
        analytics.get("cliff_crossing_probability", 0.0)
    )

    # Time to crisis
    ttc = analytics.get("time_to_crisis", {})
    out["ens_crisis_fraction"] = float(ttc.get("fraction_reaching_crisis", 0.0))
    mean_step = ttc.get("mean_step")
    out["ens_crisis_mean_step"] = (
        float(mean_step) if mean_step is not None else -1.0
    )

    # Variable distributions (just final ATP and N_deletion)
    vd = analytics.get("variable_distributions", {})
    for var_name in ("ATP", "N_deletion"):
        if var_name in vd:
            for label, prob in vd[var_name].items():
                out[f"ens_{var_name}_{label}"] = float(prob)

    # NaN/Inf guard
    for k, v in out.items():
        if v != v:
            out[k] = 0.0
        elif v == float("inf") or v == float("-inf"):
            out[k] = 999.0

    return out


# ── MitoCASimulator ──────────────────────────────────────────────────────


class MitoCASimulator:
    """Zimmerman adapter for single-cell CA simulation.

    Wraps ca_simulator.run_single_cell() + ca_analytics.compute_ca_analytics()
    into the Zimmerman Simulator protocol (12D input, flat scalar output).

    The parameter space is identical to the ODE simulator: 6 intervention +
    6 patient parameters. The CA discretizes the continuous initial state
    and applies rule-based transitions.
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return 12D parameter specification (same as ODE simulator)."""
        from constants import INTERVENTION_PARAMS, PATIENT_PARAMS

        spec = {}
        for name, info in INTERVENTION_PARAMS.items():
            spec[name] = info["range"]
        for name, info in PATIENT_PARAMS.items():
            spec[name] = info["range"]
        return spec

    def run(self, params: dict) -> dict:
        """Run CA simulation and return flat dict of scalar metrics.

        Args:
            params: Flat dict mapping parameter names to float values.
                Accepts any subset of the 12 params; missing params use
                defaults from constants.py.

        Returns:
            Flat dict of scalar metrics: rule stats, cascade stats,
            attractor classification, epoch diagnostic, final state bins.
        """
        from constants import (
            INTERVENTION_PARAMS,
            PATIENT_PARAMS,
            DEFAULT_INTERVENTION,
            DEFAULT_PATIENT,
        )
        from ca_simulator import run_single_cell
        from ca_analytics import compute_ca_analytics

        # Split params
        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)
        for k, v in params.items():
            if k in INTERVENTION_PARAMS:
                intervention[k] = float(v)
            elif k in PATIENT_PARAMS:
                patient[k] = float(v)

        # Run CA
        result = run_single_cell(patient=patient, intervention=intervention)
        analytics = compute_ca_analytics(result)

        # Flatten to scalars
        return _flatten_ca_analytics(analytics, result)


# ── MitoTissueSimulator ──────────────────────────────────────────────────


class MitoTissueSimulator:
    """Zimmerman adapter for 4-tissue CA grid.

    Wraps ca_simulator.run_tissue_grid() + ca_analytics.compute_tissue_analytics()
    into the Zimmerman Simulator protocol (13D input, flat scalar output).

    Adds one parameter beyond the 12D ODE space: tissue_coupling (0.0-1.0),
    controlling the strength of inter-tissue coupling channels (systemic SASP,
    circulating NAD equilibrium).
    """

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return 13D parameter specification (12 base + tissue_coupling)."""
        from constants import INTERVENTION_PARAMS, PATIENT_PARAMS

        spec = {}
        for name, info in INTERVENTION_PARAMS.items():
            spec[name] = info["range"]
        for name, info in PATIENT_PARAMS.items():
            spec[name] = info["range"]
        spec["tissue_coupling"] = (0.0, 1.0)
        return spec

    def run(self, params: dict) -> dict:
        """Run 4-tissue CA simulation and return flat dict of scalar metrics.

        Args:
            params: Flat dict mapping parameter names to float values.
                Accepts 12 base params + tissue_coupling. Missing params
                use defaults.

        Returns:
            Flat dict of scalar metrics: per-tissue attractors, per-tissue
            final bin indices, tissue divergence, population summary.
        """
        from constants import (
            INTERVENTION_PARAMS,
            PATIENT_PARAMS,
            DEFAULT_INTERVENTION,
            DEFAULT_PATIENT,
        )
        from ca_simulator import run_tissue_grid
        from ca_analytics import compute_tissue_analytics

        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)
        tissue_coupling = params.get("tissue_coupling", 0.5)

        for k, v in params.items():
            if k in INTERVENTION_PARAMS:
                intervention[k] = float(v)
            elif k in PATIENT_PARAMS:
                patient[k] = float(v)

        result = run_tissue_grid(
            patient=patient,
            intervention=intervention,
            tissue_coupling=float(tissue_coupling),
        )
        tissue_analytics = compute_tissue_analytics(result)

        return _flatten_tissue_summary(result, tissue_analytics)


# ── MitoCAEnsembleSimulator ──────────────────────────────────────────────


class MitoCAEnsembleSimulator:
    """Zimmerman adapter for stochastic CA ensemble.

    Wraps ca_stochastic.run_single_cell_stochastic() +
    ca_stochastic.compute_ensemble_analytics() into the Zimmerman Simulator
    protocol (12D input, flat scalar output).

    Each call runs n_trials Monte Carlo trajectories with stochastic rule
    application, returning attractor probabilities, cliff crossing probability,
    time-to-crisis statistics, and terminal variable distributions.

    Args:
        n_trials: Number of Monte Carlo trials per run() call.
            Lower values (5-10) for fast exploratory sweeps; higher values
            (100-1000) for publication-quality sensitivity analysis.
    """

    def __init__(self, n_trials: int = 100) -> None:
        self._n_trials = n_trials

    def param_spec(self) -> dict[str, tuple[float, float]]:
        """Return 12D parameter specification (same as ODE simulator)."""
        from constants import INTERVENTION_PARAMS, PATIENT_PARAMS

        spec = {}
        for name, info in INTERVENTION_PARAMS.items():
            spec[name] = info["range"]
        for name, info in PATIENT_PARAMS.items():
            spec[name] = info["range"]
        return spec

    def run(self, params: dict) -> dict:
        """Run stochastic CA ensemble and return flat dict of scalar metrics.

        Args:
            params: Flat dict mapping parameter names to float values.
                Accepts any subset of the 12 params; missing params use
                defaults from constants.py.

        Returns:
            Flat dict of scalar metrics: attractor probabilities, cliff
            crossing probability, crisis fraction, time-to-crisis mean,
            terminal ATP and N_deletion distributions.
        """
        from constants import (
            INTERVENTION_PARAMS,
            PATIENT_PARAMS,
            DEFAULT_INTERVENTION,
            DEFAULT_PATIENT,
        )
        from ca_stochastic import (
            run_single_cell_stochastic,
            compute_ensemble_analytics,
        )

        intervention = dict(DEFAULT_INTERVENTION)
        patient = dict(DEFAULT_PATIENT)
        for k, v in params.items():
            if k in INTERVENTION_PARAMS:
                intervention[k] = float(v)
            elif k in PATIENT_PARAMS:
                patient[k] = float(v)

        result = run_single_cell_stochastic(
            patient=patient,
            intervention=intervention,
            n_trials=self._n_trials,
        )
        analytics = compute_ensemble_analytics(result)

        return _flatten_ensemble_analytics(analytics)
