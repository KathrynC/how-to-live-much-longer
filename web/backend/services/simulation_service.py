"""Simulation orchestration service for the web workbench backend."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Any
from uuid import uuid4

import numpy as np

from analytics import NumpyEncoder, compute_all
from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT
from simulator import simulate
from web.backend.schemas import RunSimulationRequest


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_serializable(value: Any) -> Any:
    """Convert numpy-heavy payloads to JSON-compatible Python objects."""
    decoded = json.loads(json.dumps(value, cls=NumpyEncoder))

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    return _sanitize(decoded)


def _as_float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


@dataclass
class SimulationArtifact:
    run_id: str
    kind: str
    created_at: str
    request: dict[str, Any]
    summary: dict[str, Any]
    analytics: dict[str, Any]
    series: dict[str, Any]
    raw_result: dict[str, Any]
    baseline_result: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": "completed",
            "created_at": self.created_at,
            "request": self.request,
            "summary": self.summary,
            "analytics": self.analytics,
            "series": self.series,
            "raw_result": self.raw_result,
            "baseline_result": self.baseline_result,
        }


class SimulationService:
    """Run simulations and build API-friendly artifacts."""

    def _normalized_intervention(self, intervention: dict[str, Any]) -> dict[str, Any]:
        merged = dict(DEFAULT_INTERVENTION)
        merged.update(intervention or {})
        return merged

    def _normalized_patient(self, patient: dict[str, Any]) -> dict[str, Any]:
        merged = dict(DEFAULT_PATIENT)
        merged.update(patient or {})
        return merged

    def _build_summary(
        self,
        states: np.ndarray,
        het: np.ndarray,
        del_het: np.ndarray,
        analytics: dict[str, Any],
    ) -> dict[str, Any]:
        energy = analytics.get("energy", {})
        damage = analytics.get("damage", {})

        return {
            "final_atp": _as_float_or_none(states[-1, 2]),
            "final_heteroplasmy": _as_float_or_none(het[-1]),
            "final_deletion_heteroplasmy": _as_float_or_none(del_het[-1]),
            "final_nad": _as_float_or_none(states[-1, 4]),
            "final_senescent_fraction": _as_float_or_none(states[-1, 5]),
            "atp_initial": _as_float_or_none(states[0, 2]),
            "heteroplasmy_initial": _as_float_or_none(het[0]),
            "time_to_cliff_years": _as_float_or_none(damage.get("time_to_cliff_years")),
            "time_to_crisis_years": _as_float_or_none(energy.get("time_to_crisis_years")),
        }

    def _extract_series(self, result: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(result["states"])
        het = np.asarray(result["heteroplasmy"])
        del_het = np.asarray(result.get("deletion_heteroplasmy", result["heteroplasmy"]))
        time = np.asarray(result["time"])

        multi_trajectory = states.ndim == 3
        if multi_trajectory:
            states_view = states.mean(axis=0)
            het_view = het.mean(axis=0)
            del_het_view = del_het.mean(axis=0)
            applied_intensity = result.get("applied_intervention_intensity")
            if applied_intensity is not None:
                intensity = np.asarray(applied_intensity)
                if intensity.ndim == 2:
                    intensity = intensity.mean(axis=0)
                else:
                    intensity = np.asarray(intensity)
            else:
                intensity = None
        else:
            states_view = states
            het_view = het
            del_het_view = del_het
            intensity = result.get("applied_intervention_intensity")
            if intensity is not None:
                intensity = np.asarray(intensity)

        series = {
            "time": time,
            "atp": states_view[:, 2],
            "heteroplasmy": het_view,
            "deletion_heteroplasmy": del_het_view,
            "ros": states_view[:, 3],
            "nad": states_view[:, 4],
            "senescent_fraction": states_view[:, 5],
            "membrane_potential": states_view[:, 6],
        }
        if intensity is not None:
            series["applied_intervention_intensity"] = intensity
        if multi_trajectory:
            series["mode"] = "stochastic_mean"
            series["n_trajectories"] = int(states.shape[0])

        return _to_serializable(series), states_view, het_view, del_het_view

    def _collapse_for_analytics(self, result: dict[str, Any]) -> dict[str, Any]:
        """Ensure analytics always receive 2D states / 1D trajectories."""
        states = np.asarray(result["states"])
        if states.ndim != 3:
            return result

        collapsed = dict(result)
        collapsed["states"] = states.mean(axis=0)

        het = np.asarray(result["heteroplasmy"])
        if het.ndim == 2:
            collapsed["heteroplasmy"] = het.mean(axis=0)

        del_het = np.asarray(result.get("deletion_heteroplasmy", result["heteroplasmy"]))
        if del_het.ndim == 2:
            collapsed["deletion_heteroplasmy"] = del_het.mean(axis=0)

        intensity = result.get("applied_intervention_intensity")
        if intensity is not None:
            arr = np.asarray(intensity)
            if arr.ndim == 2:
                collapsed["applied_intervention_intensity"] = arr.mean(axis=0)
        return collapsed

    def run(self, request: RunSimulationRequest, run_id: str | None = None) -> SimulationArtifact:
        intervention = self._normalized_intervention(request.intervention)
        patient = self._normalized_patient(request.patient)
        run_id = run_id or uuid4().hex
        created_at = _utc_now_iso()

        sim_kwargs = {
            "intervention": intervention,
            "patient": patient,
            "sim_years": request.sim_years,
            "dt": request.dt,
            "stochastic": request.stochastic,
            "noise_scale": request.noise_scale,
            "n_trajectories": request.n_trajectories,
            "rng_seed": request.rng_seed,
        }
        if request.tissue_type:
            sim_kwargs["tissue_type"] = request.tissue_type

        result = simulate(**sim_kwargs)

        baseline_kwargs = dict(sim_kwargs)
        baseline_kwargs["intervention"] = dict(DEFAULT_INTERVENTION)
        baseline_result = simulate(**baseline_kwargs)

        analytics_result = self._collapse_for_analytics(result)
        analytics_baseline = self._collapse_for_analytics(baseline_result)
        analytics = compute_all(analytics_result, analytics_baseline)
        series, states, het, del_het = self._extract_series(result)
        summary = self._build_summary(states, het, del_het, analytics)

        request_payload = _to_serializable({
            "intervention": intervention,
            "patient": patient,
            "sim_years": request.sim_years,
            "dt": request.dt,
            "tissue_type": request.tissue_type,
            "stochastic": request.stochastic,
            "noise_scale": request.noise_scale,
            "n_trajectories": request.n_trajectories,
            "rng_seed": request.rng_seed,
        })

        return SimulationArtifact(
            run_id=run_id,
            kind="simulation",
            created_at=created_at,
            request=request_payload,
            summary=_to_serializable(summary),
            analytics=_to_serializable(analytics),
            series=series,
            raw_result=_to_serializable(result),
            baseline_result=_to_serializable(baseline_result),
        )

    def compare(
        self,
        baseline_request: RunSimulationRequest,
        candidate_request: RunSimulationRequest,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or uuid4().hex
        baseline_artifact = self.run(baseline_request)
        candidate_artifact = self.run(candidate_request)

        baseline_summary = baseline_artifact.summary
        candidate_summary = candidate_artifact.summary

        def _delta(key: str) -> float | None:
            b = _as_float_or_none(baseline_summary.get(key))
            c = _as_float_or_none(candidate_summary.get(key))
            if b is None or c is None:
                return None
            return c - b

        delta = {
            "delta_final_atp": _delta("final_atp"),
            "delta_final_heteroplasmy": _delta("final_heteroplasmy"),
            "delta_final_deletion_heteroplasmy": _delta("final_deletion_heteroplasmy"),
            "delta_time_to_cliff_years": _delta("time_to_cliff_years"),
            "delta_time_to_crisis_years": _delta("time_to_crisis_years"),
        }

        payload = {
            "run_id": run_id,
            "kind": "comparison",
            "status": "completed",
            "created_at": _utc_now_iso(),
            "baseline": baseline_artifact.to_payload(),
            "candidate": candidate_artifact.to_payload(),
            "delta": _to_serializable(delta),
            "series_overlay": {
                "time": candidate_artifact.series.get("time"),
                "baseline_atp": baseline_artifact.series.get("atp"),
                "candidate_atp": candidate_artifact.series.get("atp"),
                "baseline_heteroplasmy": baseline_artifact.series.get("heteroplasmy"),
                "candidate_heteroplasmy": candidate_artifact.series.get("heteroplasmy"),
            },
        }
        return _to_serializable(payload)
