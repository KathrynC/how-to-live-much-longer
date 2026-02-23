#!/usr/bin/env python3
"""Adaptive intervention protocols that respond to patient state.

Implements symmathesy Phase 2: interventions that adjust based on current
patient state, enabling mutual learning between intervention and patient.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from simulator import InterventionSchedule

from constants import (
    DEFAULT_INTERVENTION, DEFAULT_PATIENT,
    INTERVENTION_NAMES, PATIENT_NAMES,
)


class AdaptiveProtocol:
    """Intervention protocol that adjusts based on patient state.
    
    Wraps a base intervention (constant dict or InterventionSchedule) and
    applies state-dependent adjustment rules at each timestep.
    
    Implements the resolver interface: resolve(t, state) -> (intervention, patient).
    
    Example rules:
        - If ATP < 0.7, increase NAD supplement by 0.2
        - If heteroplasmy > 0.6, increase transplant rate by 0.3
        - If ROS > 0.5, increase exercise (hormetic adaptation)
    """
    
    def __init__(
        self,
        base_intervention: Dict[str, float] | InterventionSchedule,
        base_patient: Optional[Dict[str, float]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = False,
    ):
        """Initialize adaptive protocol.
        
        Args:
            base_intervention: Constant intervention dict or InterventionSchedule.
            base_patient: Patient parameters (optional, defaults to DEFAULT_PATIENT).
            rules: List of adjustment rules, each with:
                - condition: function(t, state, current_intervention) -> bool
                - action: function(t, state, current_intervention) -> Dict[str, float]
                  returning adjustments to apply (added to current intervention).
            verbose: If True, log rule firings to self.rule_log.
        """
        self.base_intervention = base_intervention
        self.base_patient = base_patient or dict(DEFAULT_PATIENT)
        self.rules = rules or []
        self.verbose = verbose
        self.rule_log: List[Dict[str, Any]] = []  # Log of rule firings
        
        # State indices for easy reference
        self.STATE_INDICES = {
            'N_healthy': 0,
            'N_deletion': 1,
            'ATP': 2,
            'ROS': 3,
            'NAD': 4,
            'Senescent_fraction': 5,
            'Membrane_potential': 6,
            'N_point': 7,
        }
    
    def resolve(self, t: float, state: np.ndarray | None = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Resolve intervention and patient at time t given current state.
        
        Args:
            t: Current simulation time (years).
            state: Current ODE state vector (8 elements).
            
        Returns:
            (intervention_dict, patient_dict) - both with standard keys.
        """
        # Get base intervention at time t
        if isinstance(self.base_intervention, InterventionSchedule):
            intervention = self.base_intervention.at(t)
        else:
            intervention = dict(self.base_intervention)
        
        # Apply state-dependent adjustments if state is provided
        if state is not None:
            intervention = self._apply_rules(t, state, intervention)
        
        # Return patient unchanged (for now)
        patient = dict(self.base_patient)
        
        return intervention, patient
    
    def _apply_rules(self, t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        """Apply all adjustment rules to intervention.
        
        Rules are applied in order; each rule can suggest target values for parameters.
        For each parameter, we take the maximum suggested target across all firing rules
        (or the parameter's current value if no rule suggests a change).
        """
        result = dict(intervention)
        # Track targets suggested by rules
        targets: Dict[str, float] = {}
        
        for i, rule in enumerate(self.rules):
            condition = rule.get('condition')
            action = rule.get('action')
            
            if condition and condition(t, state, result):
                suggestions = action(t, state, result)
                # Log rule firing if verbose
                if self.verbose:
                    self.rule_log.append({
                        'time': t,
                        'rule_index': i,
                        'suggestions': suggestions.copy(),
                        'state': state.copy(),
                    })
                # Collect suggestions: for each parameter, track maximum target
                for key, value in suggestions.items():
                    if key in INTERVENTION_NAMES:
                        # Action returns absolute target values (0-1); clamp to valid range
                        target = max(0.0, min(1.0, value))
                        if key not in targets or target > targets[key]:
                            targets[key] = target
        
        # Apply targets: override intervention parameters with suggested targets
        for key, target in targets.items():
            result[key] = target
        
        return result
    
    def add_rule(
        self,
        condition: Callable[[float, np.ndarray, Dict[str, float]], bool],
        action: Callable[[float, np.ndarray, Dict[str, float]], Dict[str, float]],
    ) -> None:
        """Add a new adjustment rule.
        
        Condition: function(t, state, current_intervention) -> bool.
        Action: function(t, state, current_intervention) -> Dict[str, float]
            returning absolute target values for intervention parameters (0-1).
        """
        self.rules.append({'condition': condition, 'action': action})
    
    def clear_log(self) -> None:
        """Clear the rule firing log."""
        self.rule_log.clear()
    
    def get_log(self) -> List[Dict[str, Any]]:
        """Get the rule firing log."""
        return self.rule_log.copy()


# ── Predefined rule constructors ───────────────────────────────────────────

def atp_threshold_rule(
    threshold: float = 0.7,
    nad_increase: float = 0.2,
    max_nad: float = 1.0,
) -> Dict[str, Any]:
    """Increase NAD supplement when ATP falls below threshold.
    
    Returns absolute target NAD level (current + increase, capped at max_nad).
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        atp = state[2]  # ATP index
        return atp < threshold
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        current_nad = intervention.get('nad_supplement', 0.0)
        # Increase NAD but don't exceed max_nad
        increase = min(nad_increase, max_nad - current_nad)
        target = current_nad + increase
        return {'nad_supplement': target}
    
    return {'condition': condition, 'action': action}


def heteroplasmy_threshold_rule(
    threshold: float = 0.6,
    transplant_increase: float = 0.3,
    max_transplant: float = 1.0,
) -> Dict[str, Any]:
    """Increase transplant rate when heteroplasmy exceeds threshold.
    
    Returns absolute target transplant rate (current + increase, capped at max_transplant).
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        # Compute total heteroplasmy: (N_del + N_pt) / total
        N_h = state[0]
        N_del = state[1]
        N_pt = state[7]
        total = N_h + N_del + N_pt
        if total <= 0:
            return False
        het = (N_del + N_pt) / total
        return het > threshold
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        current_transplant = intervention.get('transplant_rate', 0.0)
        increase = min(transplant_increase, max_transplant - current_transplant)
        target = current_transplant + increase
        return {'transplant_rate': target}
    
    return {'condition': condition, 'action': action}


def ros_oscillation_rule(
    ros_threshold: float = 0.4,
    exercise_increase: float = 0.2,
    max_exercise: float = 1.0,
) -> Dict[str, Any]:
    """Increase exercise when ROS is elevated (hormetic adaptation).
    
    Returns absolute target exercise level (current + increase, capped at max_exercise).
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        ros = state[3]  # ROS index
        return ros > ros_threshold
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        current_exercise = intervention.get('exercise_level', 0.0)
        increase = min(exercise_increase, max_exercise - current_exercise)
        target = current_exercise + increase
        return {'exercise_level': target}
    
    return {'condition': condition, 'action': action}


def senescence_clearance_rule(
    sen_threshold: float = 0.3,
    senolytic_increase: float = 0.4,
    max_senolytic: float = 1.0,
) -> Dict[str, Any]:
    """Increase senolytic dose when senescence exceeds threshold.
    
    Returns absolute target senolytic dose (current + increase, capped at max_senolytic).
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        sen = state[5]  # Senescent_fraction index
        return sen > sen_threshold
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        current_senolytic = intervention.get('senolytic_dose', 0.0)
        increase = min(senolytic_increase, max_senolytic - current_senolytic)
        target = current_senolytic + increase
        return {'senolytic_dose': target}
    
    return {'condition': condition, 'action': action}


# ── Advanced rule constructors ──────────────────────────────────────────

def atp_proportional_rule(
    threshold: float = 0.7,
    gain: float = 0.5,
    max_nad: float = 1.0,
) -> Dict[str, Any]:
    """Adjust NAD supplement proportionally to ATP deficit.
    
    Target NAD = current + gain * (threshold - ATP), clamped to [0, max_nad].
    Applies only when ATP < threshold.
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        atp = state[2]
        return atp < threshold
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        atp = state[2]
        current_nad = intervention.get('nad_supplement', 0.0)
        deficit = threshold - atp  # positive when ATP low
        delta = gain * deficit
        target = current_nad + delta
        target = max(0.0, min(max_nad, target))
        return {'nad_supplement': target}
    
    return {'condition': condition, 'action': action}


def heteroplasmy_proportional_rule(
    threshold: float = 0.6,
    gain: float = 0.5,
    max_transplant: float = 1.0,
) -> Dict[str, Any]:
    """Adjust transplant rate proportionally to heteroplasmy excess.
    
    Target transplant = current + gain * (het - threshold), clamped to [0, max_transplant].
    Applies only when het > threshold.
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        N_h = state[0]
        N_del = state[1]
        N_pt = state[7]
        total = N_h + N_del + N_pt
        if total <= 0:
            return False
        het = (N_del + N_pt) / total
        return het > threshold
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        N_h = state[0]
        N_del = state[1]
        N_pt = state[7]
        total = N_h + N_del + N_pt
        het = (N_del + N_pt) / total if total > 0 else 0.0
        current_transplant = intervention.get('transplant_rate', 0.0)
        excess = het - threshold  # positive when het high
        delta = gain * excess
        target = current_transplant + delta
        target = max(0.0, min(max_transplant, target))
        return {'transplant_rate': target}
    
    return {'condition': condition, 'action': action}


def atp_bidirectional_rule(
    target_atp: float = 0.8,
    gain: float = 0.5,
    deadzone: float = 0.05,
    max_nad: float = 1.0,
    min_nad: float = 0.0,
) -> Dict[str, Any]:
    """Adjust NAD supplement bidirectionally to maintain ATP near target.
    
    Target NAD = current + gain * (target_atp - ATP), clamped to [min_nad, max_nad].
    Only applies when |ATP - target_atp| > deadzone.
    """
    def condition(t: float, state: np.ndarray, intervention: Dict[str, float]) -> bool:
        atp = state[2]
        return abs(atp - target_atp) > deadzone
    
    def action(t: float, state: np.ndarray, intervention: Dict[str, float]) -> Dict[str, float]:
        atp = state[2]
        current_nad = intervention.get('nad_supplement', 0.0)
        error = target_atp - atp  # positive when ATP below target
        delta = gain * error
        target = current_nad + delta
        target = max(min_nad, min(max_nad, target))
        return {'nad_supplement': target}
    
    return {'condition': condition, 'action': action}


# ── Example adaptive protocol constructors ────────────────────────────────

def create_symmathesy_protocol(
    base_intervention: Dict[str, float] | InterventionSchedule,
    base_patient: Optional[Dict[str, float]] = None,
) -> AdaptiveProtocol:
    """Create a symmathesy-optimized adaptive protocol.
    
    Includes rules for mutual adaptation:
    1. ATP support: increase NAD when energy low
    2. Damage control: increase transplant when heteroplasmy high
    3. Stress response: increase exercise when ROS elevated
    4. Senescence clearance: increase senolytics when senescence high
    """
    protocol = AdaptiveProtocol(base_intervention, base_patient)
    
    protocol.add_rule(
        atp_threshold_rule(threshold=0.7, nad_increase=0.2)['condition'],
        atp_threshold_rule(threshold=0.7, nad_increase=0.2)['action'],
    )
    
    protocol.add_rule(
        heteroplasmy_threshold_rule(threshold=0.6, transplant_increase=0.3)['condition'],
        heteroplasmy_threshold_rule(threshold=0.6, transplant_increase=0.3)['action'],
    )
    
    protocol.add_rule(
        ros_oscillation_rule(ros_threshold=0.4, exercise_increase=0.2)['condition'],
        ros_oscillation_rule(ros_threshold=0.4, exercise_increase=0.2)['action'],
    )
    
    protocol.add_rule(
        senescence_clearance_rule(sen_threshold=0.3, senolytic_increase=0.4)['condition'],
        senescence_clearance_rule(sen_threshold=0.3, senolytic_increase=0.4)['action'],
    )
    
    return protocol
def create_advanced_symmathesy_protocol(
    base_intervention: Dict[str, float] | InterventionSchedule,
    base_patient: Optional[Dict[str, float]] = None,
) -> AdaptiveProtocol:
    """Create an advanced symmathesy protocol with proportional and bidirectional rules.
    
    Includes:
    1. ATP bidirectional rule: maintain ATP near target with deadzone.
    2. Heteroplasmy proportional rule: adjust transplant proportionally to het excess.
    3. Senescence clearance threshold rule.
    4. ROS oscillation threshold rule.
    """
    protocol = AdaptiveProtocol(base_intervention, base_patient)
    
    # Bidirectional ATP regulation
    protocol.add_rule(
        atp_bidirectional_rule(target_atp=0.8, gain=0.3, deadzone=0.05, max_nad=1.0, min_nad=0.0)['condition'],
        atp_bidirectional_rule(target_atp=0.8, gain=0.3, deadzone=0.05, max_nad=1.0, min_nad=0.0)['action'],
    )
    
    # Proportional heteroplasmy control
    protocol.add_rule(
        heteroplasmy_proportional_rule(threshold=0.6, gain=0.5, max_transplant=1.0)['condition'],
        heteroplasmy_proportional_rule(threshold=0.6, gain=0.5, max_transplant=1.0)['action'],
    )
    
    # Threshold rules for other interventions
    protocol.add_rule(
        ros_oscillation_rule(ros_threshold=0.4, exercise_increase=0.2)['condition'],
        ros_oscillation_rule(ros_threshold=0.4, exercise_increase=0.2)['action'],
    )
    
    protocol.add_rule(
        senescence_clearance_rule(sen_threshold=0.3, senolytic_increase=0.4)['condition'],
        senescence_clearance_rule(sen_threshold=0.3, senolytic_increase=0.4)['action'],
    )
    
    return protocol
