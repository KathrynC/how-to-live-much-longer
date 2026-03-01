"""protocol_enrichment.py — Computed enrichment fields for protocol records.

Ported from rosetta-motion's controller_simplicity(), sensory_signature(),
prototype_descriptor(), and prototype_strength(). Adapted for the
mitochondrial aging intervention domain.
"""
from __future__ import annotations

import copy
from typing import Any

from protocol_record import ProtocolRecord, protocol_fingerprint

# Interventions considered "active" above this threshold
ACTIVE_THRESHOLD = 0.05

# Expanded intervention schema includes numeric and categorical fields.
# Complexity/prototype metrics should only use numeric dose-like fields.
KNOWN_NUMERIC_INTERVENTION_KEYS = {
    "rapamycin_dose",
    "nad_supplement",
    "senolytic_dose",
    "yamanaka_intensity",
    "transplant_rate",
    "exercise_level",
    "sleep_intervention",
    "fasting_regimen",
    "alcohol_intake",
    "coffee_intake",
    "probiotic_intensity",
    "nr_dose",
    "dha_dose",
    "coq10_dose",
    "resveratrol_dose",
    "pqq_dose",
    "ala_dose",
    "vitamin_d_dose",
    "b_complex_dose",
    "magnesium_dose",
    "zinc_dose",
    "selenium_dose",
    "therapy_intensity",
    "intellectual_engagement_intervention",
    "nmn_dose",
    "apigenin_dose",
    "fisetin_dose",
    "quercetin_dose",
    "pterostilbene_dose",
    "melatonin_dose",
    "curcumin_dose",
    "caffeine_dose",
    "theanine_dose",
}


def coerce_numeric_fields(
    values: dict[str, Any],
    allowed_keys: set[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Return numeric subset of values and list of skipped non-numeric keys."""
    numeric: dict[str, float] = {}
    skipped: list[str] = []

    for key, value in values.items():
        if value is None:
            continue
        if allowed_keys is not None and key not in allowed_keys:
            continue

        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            skipped.append(str(key))

    return numeric, skipped


def protocol_complexity(intervention: dict[str, Any]) -> dict[str, Any]:
    """Compute protocol complexity metrics.

    Analogous to rosetta-motion's controller_simplicity(). Measures how
    "heavy" an intervention protocol is in terms of dose burden.
    """
    numeric_fields, skipped = coerce_numeric_fields(
        intervention,
        allowed_keys=KNOWN_NUMERIC_INTERVENTION_KEYS,
    )
    doses = list(numeric_fields.values())
    if not doses:
        return {"total_dose": 0.0, "active_count": 0, "max_single_dose": 0.0,
                "mean_active_dose": 0.0, "param_count": 0, "skipped_non_numeric": skipped}

    active = [d for d in doses if abs(d) > ACTIVE_THRESHOLD]
    return {
        "total_dose": sum(doses),
        "active_count": len(active),
        "max_single_dose": max(doses) if doses else 0.0,
        "mean_active_dose": (sum(active) / len(active)) if active else 0.0,
        "param_count": len(doses),
        "skipped_non_numeric": skipped,
    }


def clinical_signature(analytics: dict[str, Any]) -> dict[str, Any]:
    """Extract clinical signature from 4-pillar analytics.

    Analogous to rosetta-motion's sensory_signature(). Captures the
    trajectory shape and clinical risk profile.
    """
    energy = analytics.get("energy", {})
    damage = analytics.get("damage", {})
    intervention = analytics.get("intervention", {})

    atp_slope = energy.get("atp_slope", 0.0)
    if atp_slope is None:
        atp_slope = 0.0
    if atp_slope > 0.001:
        energy_trend = "improving"
    elif atp_slope < -0.001:
        energy_trend = "declining"
    else:
        energy_trend = "stable"

    ttc = damage.get("time_to_cliff_years", damage.get("time_to_cliff", 999))
    if ttc is None:
        ttc = 999
    if ttc < 10:
        cliff_risk = "imminent"
    elif ttc < 20:
        cliff_risk = "moderate"
    else:
        cliff_risk = "none"

    return {
        "final_atp": energy.get("atp_final", energy.get("final_atp")),
        "final_het": damage.get("het_final", damage.get("final_het")),
        "energy_trend": energy_trend,
        "cliff_risk": cliff_risk,
        "benefit_cost_ratio": intervention.get("benefit_cost_ratio"),
    }


def prototype_group(intervention: dict[str, Any]) -> dict[str, Any]:
    """Assign protocol to an archetype group.

    Analogous to rosetta-motion's prototype_descriptor(). Groups protocols
    by their dominant intervention mechanism.
    """
    fp = protocol_fingerprint(intervention)
    numeric_fields, skipped = coerce_numeric_fields(
        intervention,
        allowed_keys=KNOWN_NUMERIC_INTERVENTION_KEYS,
    )
    active = {k: v for k, v in numeric_fields.items() if v > ACTIVE_THRESHOLD}

    if not active:
        return {"archetype": "no_treatment", "fingerprint": fp, "skipped_non_numeric": skipped}

    transplant_val = numeric_fields.get("transplant_rate", 0.0)
    yamanaka_val = numeric_fields.get("yamanaka_intensity", 0.0)
    has_transplant = transplant_val > ACTIVE_THRESHOLD
    has_yamanaka = yamanaka_val > ACTIVE_THRESHOLD
    transplant_dominant = transplant_val >= max(active.values()) * 0.8

    if has_yamanaka and has_transplant:
        archetype = "full_experimental"
    elif has_transplant and transplant_dominant:
        archetype = "transplant_focused"
    elif has_yamanaka:
        archetype = "reprogramming"
    elif len(active) >= 3:
        archetype = "cocktail"
    elif len(active) == 2:
        archetype = "dual_therapy"
    else:
        archetype = "monotherapy"

    return {
        "archetype": archetype,
        "fingerprint": fp,
        "dominant": max(active, key=lambda k: active[k]),
        "skipped_non_numeric": skipped,
    }


def lakoff_archetype_classification(analytics: dict[str, Any]) -> dict[str, Any]:
    """Classify protocol using Lakoff archetypes (adjusted grounding criteria).
    
    Returns dict with:
        - lakoff_archetype: best matching archetype name
        - lakoff_score: similarity score (0-1)
        - similarity_vector: dict mapping archetype names to scores
        - grounding_stats: grounded vs linking feature counts
    """
    try:
        from patterns.lakoff_classifier import classify_analytics
        return classify_analytics(analytics)
    except ImportError as e:
        # If Lakoff classifier not available, return empty
        return {
            "lakoff_archetype": None,
            "lakoff_score": 0.0,
            "similarity_vector": {},
            "grounding_stats": {"grounded": 0, "linking": 0, "grounding_ratio": 0.0},
            "error": str(e)
        }


def enrich_record(record: ProtocolRecord) -> ProtocolRecord:
    """Apply all enrichment fields to a protocol record.

    Returns a new record with enrichment dict populated.
    """
    enriched = copy.deepcopy(record)
    enriched.enrichment = {
        "complexity": protocol_complexity(record.intervention),
        "clinical_signature": clinical_signature(record.analytics),
        "prototype": prototype_group(record.intervention),
        "lakoff_archetype": lakoff_archetype_classification(record.analytics),
    }
    return enriched
