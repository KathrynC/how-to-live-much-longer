"""protocol_classifier.py — Multi-method protocol classification pipeline.

Ported from rosetta-motion's discover_label() multi-labeler pattern.
Three classification methods:
  1. rule_classify: threshold-based (from dark_matter.py logic)
  2. analytics_fit_classify: prototype distance (from motion_discovery.py metric_fit_score)
  3. llm_classify: LLM clinical assessment (from motion_discovery.py llm_based_label)

Pipeline runs methods in sequence, aggregates results, and determines
final outcome_class + confidence from method agreement.
"""
from __future__ import annotations

import math
from typing import Any

# Class prototypes: mean values for key metrics per outcome class.
# Built from dark_matter.py classification thresholds and reachable_set observations.
CLASS_PROTOTYPES = {
    "thriving":    {"final_atp": 0.85, "mean_atp": 0.80, "final_het": 0.20},
    "stable":      {"final_atp": 0.60, "mean_atp": 0.55, "final_het": 0.50},
    "declining":   {"final_atp": 0.35, "mean_atp": 0.40, "final_het": 0.65},
    "collapsed":   {"final_atp": 0.10, "mean_atp": 0.15, "final_het": 0.85},
}

# Metric keys used for prototype distance computation
FIT_KEYS = ["final_atp", "mean_atp", "final_het"]

# Standard deviations for z-scoring (estimated from population simulations)
FIT_STD = {"final_atp": 0.25, "mean_atp": 0.20, "final_het": 0.20}


def rule_classify(
    final_atp: float,
    final_het: float,
    baseline_atp: float,
    baseline_het: float,
) -> dict[str, Any]:
    """Rule-based outcome classification using threshold logic.

    Matches dark_matter.py's classify_outcome() thresholds:
      thriving:    final ATP > 0.8 AND het < 0.5
      stable:      final ATP > 0.5 AND het < 0.7
      declining:   final ATP > 0.2, het > 0.5
      collapsed:   final ATP < 0.2
      paradoxical: worse than baseline on BOTH ATP and het
    """
    if final_atp > 0.8 and final_het < 0.5:
        return {"outcome_class": "thriving", "confidence": 0.95, "method": "rule"}
    if final_atp > 0.5 and final_het < 0.7:
        return {"outcome_class": "stable", "confidence": 0.85, "method": "rule"}
    if final_atp < 0.2:
        return {"outcome_class": "collapsed", "confidence": 0.95, "method": "rule"}
    if final_atp > 0.2 and final_het > 0.5:
        return {"outcome_class": "declining", "confidence": 0.80, "method": "rule"}

    # Check paradoxical: intervention made things worse but doesn't fit
    # a concrete threshold category above
    if final_atp < baseline_atp and final_het > baseline_het:
        return {"outcome_class": "paradoxical", "confidence": 0.9, "method": "rule"}

    # Remaining: declining
    return {"outcome_class": "declining", "confidence": 0.80, "method": "rule"}


def analytics_fit_classify(
    analytics: dict[str, Any],
) -> dict[str, Any]:
    """Analytics-fit classification via prototype distance.

    Analogous to rosetta-motion's metric_fit_score(). Computes z-scored
    Euclidean distance to each class prototype, returns closest class
    with exponential similarity as confidence.
    """
    energy = analytics.get("energy", {})
    damage = analytics.get("damage", {})
    metrics = {
        "final_atp": energy.get("final_atp"),
        "mean_atp": energy.get("mean_atp"),
        "final_het": damage.get("final_het"),
    }

    distances: dict[str, float] = {}
    for cls, proto in CLASS_PROTOTYPES.items():
        keys = [k for k in FIT_KEYS if metrics.get(k) is not None and k in proto]
        if not keys:
            continue
        z2 = sum(
            ((float(metrics[k]) - proto[k]) / FIT_STD.get(k, 1.0)) ** 2
            for k in keys
        )
        distances[cls] = math.sqrt(z2 / max(1, len(keys)))

    if not distances:
        return {"outcome_class": None, "confidence": 0.0, "method": "analytics_fit",
                "distances": {}}

    best = min(distances, key=distances.get)
    confidence = math.exp(-distances[best])

    return {
        "outcome_class": best,
        "confidence": float(confidence),
        "method": "analytics_fit",
        "distances": {k: round(v, 4) for k, v in distances.items()},
    }


def multi_classify(
    final_atp: float = 0.0,
    final_het: float = 0.0,
    baseline_atp: float = 0.0,
    baseline_het: float = 0.0,
    analytics: dict[str, Any] | None = None,
    pipeline: list[str] | None = None,
) -> dict[str, Any]:
    """Run a multi-method classification pipeline and aggregate results.

    Analogous to rosetta-motion's discover_label(). Runs methods in order,
    collects per-method results, determines final class by majority vote
    (ties broken by highest confidence), and boosts confidence when methods agree.

    Args:
        pipeline: List of method names to run. Default: ["rule", "analytics_fit"]
    """
    if pipeline is None:
        pipeline = ["rule", "analytics_fit"]

    methods: dict[str, dict[str, Any]] = {}
    for step in pipeline:
        if step == "rule":
            methods["rule"] = rule_classify(final_atp, final_het,
                                            baseline_atp, baseline_het)
        elif step == "analytics_fit" and analytics is not None:
            methods["analytics_fit"] = analytics_fit_classify(analytics)
        # "llm" step would go here (requires Ollama, optional)

    if not methods:
        return {"outcome_class": None, "confidence": 0.0, "methods": {}}

    # Majority vote
    votes: dict[str, list[float]] = {}
    for name, result in methods.items():
        cls = result.get("outcome_class")
        conf = result.get("confidence", 0.0)
        if cls is not None:
            votes.setdefault(cls, []).append(conf)

    if not votes:
        return {"outcome_class": None, "confidence": 0.0, "methods": methods}

    # Pick class with most votes, break ties by max confidence
    best_class = max(votes, key=lambda c: (len(votes[c]), max(votes[c])))
    agreement = len(votes[best_class]) / len(methods)
    base_conf = max(votes[best_class])

    # Agreement bonus: if all methods agree, boost confidence
    if agreement == 1.0 and len(methods) > 1:
        confidence = min(1.0, base_conf + 0.05 * (len(methods) - 1))
    else:
        confidence = base_conf * agreement

    return {
        "outcome_class": best_class,
        "confidence": round(confidence, 4),
        "agreement": round(agreement, 4),
        "methods": methods,
    }
