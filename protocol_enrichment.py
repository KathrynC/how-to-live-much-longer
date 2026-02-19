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


def protocol_complexity(intervention: dict[str, Any]) -> dict[str, Any]:
    """Compute protocol complexity metrics.

    Analogous to rosetta-motion's controller_simplicity(). Measures how
    "heavy" an intervention protocol is in terms of dose burden.
    """
    doses = [float(v) for v in intervention.values() if v is not None]
    if not doses:
        return {"total_dose": 0.0, "active_count": 0, "max_single_dose": 0.0,
                "mean_active_dose": 0.0, "param_count": 0}

    active = [d for d in doses if abs(d) > ACTIVE_THRESHOLD]
    return {
        "total_dose": sum(doses),
        "active_count": len(active),
        "max_single_dose": max(doses) if doses else 0.0,
        "mean_active_dose": (sum(active) / len(active)) if active else 0.0,
        "param_count": len(doses),
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

    ttc = damage.get("time_to_cliff", 999)
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
    active = {k: v for k, v in intervention.items()
              if v is not None and float(v) > ACTIVE_THRESHOLD}

    if not active:
        return {"archetype": "no_treatment", "fingerprint": fp}

    has_transplant = float(intervention.get("transplant_rate", 0)) > ACTIVE_THRESHOLD
    has_yamanaka = float(intervention.get("yamanaka_intensity", 0)) > ACTIVE_THRESHOLD
    transplant_dominant = (float(intervention.get("transplant_rate", 0))
                          >= max(float(v) for v in active.values()) * 0.8)

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

    return {"archetype": archetype, "fingerprint": fp,
            "dominant": max(active, key=lambda k: float(active[k]))}


def enrich_record(record: ProtocolRecord) -> ProtocolRecord:
    """Apply all enrichment fields to a protocol record.

    Returns a new record with enrichment dict populated.
    """
    enriched = copy.deepcopy(record)
    enriched.enrichment = {
        "complexity": protocol_complexity(record.intervention),
        "clinical_signature": clinical_signature(record.analytics),
        "prototype": prototype_group(record.intervention),
    }
    return enriched
