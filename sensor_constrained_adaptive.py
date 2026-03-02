"""Sensor-constrained adaptive protocol wrapper.

Wraps AdaptiveProtocol (from adaptive_protocol.py) to replace direct
ODE state access with sensor-derived estimates. This implements the
key insight from the Evolutionary Robotics project: closed-loop control
through sensors (partial observability) vs. God's-eye open-loop control
(full state access).

The wrapper follows the same resolver interface as AdaptiveProtocol:
    resolve(t, state) -> (intervention_dict, patient_dict)

But internally:
    1. observe(true_state) -> noisy sensor readings
    2. estimate_state(readings) -> estimated state
    3. base_protocol.resolve(t, estimated_state) -> intervention

The information loss from steps 1-2 means the controller makes
suboptimal decisions compared to direct state access.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from adaptive_protocol import AdaptiveProtocol
from wearable_sensors import WearableObservationModel


class SensorConstrainedProtocol:
    """Adaptive protocol constrained to observe state through wearable sensors.

    Acts as a drop-in replacement for AdaptiveProtocol (same resolve()
    interface), but the adaptive rules see only sensor-derived estimates
    of the ODE state, not the true state.

    Args:
        base_protocol: An AdaptiveProtocol instance with rules.
        observation_model: A WearableObservationModel instance.
        log_observations: If True, store observation history for
            post-hoc analysis.
    """

    def __init__(
        self,
        base_protocol: AdaptiveProtocol,
        observation_model: WearableObservationModel,
        log_observations: bool = True,
        family_priors: Optional[Dict[str, float]] = None,
    ):
        self.base_protocol = base_protocol
        self.observation_model = observation_model
        self.log_observations = log_observations
        self.family_priors = family_priors
        self.observation_log: List[Dict[str, Any]] = []

    def resolve(
        self, t: float, state: np.ndarray | None = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Resolve intervention at time t using sensor-derived state estimate.

        Args:
            t: Current simulation time (years).
            state: True ODE state vector (8D). The protocol does NOT
                pass this directly to the adaptive rules — it first
                generates noisy sensor readings, then estimates state.

        Returns:
            (intervention_dict, patient_dict) from the base protocol,
            but driven by estimated state instead of true state.
        """
        if state is None:
            # No state: fall through to base protocol without adaptation
            return self.base_protocol.resolve(t, state)

        # Step 1: Observe — generate noisy sensor readings
        # Use the most recent intervention for activity observation
        last_intervention = None
        if self.observation_log:
            last_intervention = self.observation_log[-1].get("intervention")
        observations = self.observation_model.observe(
            state, t, intervention=last_intervention)

        # Step 2: Estimate — algebraic inversion from observations
        estimated_state = self.observation_model.estimate_state(
            observations, family_priors=self.family_priors)

        # Step 3: Resolve — pass estimated state to adaptive rules
        intervention, patient = self.base_protocol.resolve(t, estimated_state)

        # Log for post-hoc analysis
        if self.log_observations:
            self.observation_log.append({
                "time": t,
                "true_state": state.copy(),
                "observations": dict(observations),
                "estimated_state": estimated_state.copy(),
                "intervention": dict(intervention),
            })

        return intervention, patient

    def get_observation_log(self) -> List[Dict[str, Any]]:
        """Return the observation log for post-hoc analysis."""
        return list(self.observation_log)

    def clear_log(self) -> None:
        """Clear both observation log and base protocol rule log."""
        self.observation_log.clear()
        self.base_protocol.clear_log()

    def information_loss_summary(self) -> Dict[str, float]:
        """Compute average information loss across all logged observations.

        Returns:
            Dict with mean per-variable error and mean total RMSE.
        """
        if not self.observation_log:
            return {"mean_total_rmse": 0.0}

        all_losses = []
        for entry in self.observation_log:
            loss = self.observation_model.information_loss(
                entry["true_state"], entry["estimated_state"])
            all_losses.append(loss)

        # Average each key across all entries
        summary: Dict[str, float] = {}
        for key in all_losses[0]:
            values = [d[key] for d in all_losses]
            summary[f"mean_{key}"] = sum(values) / len(values)

        return summary
