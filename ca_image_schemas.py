"""Image schema detectors for mitochondrial cellular automaton trajectories.

Adapts Lakoff's image schemas (PATH, CYCLE, CONTAINER, SCALE, BALANCE, FORCE)
to discrete CA state trajectories, mapping bin labels to continuous exemplars
via BIN_SCHEMA centers for detection of pre-conceptual patterns.

Implements Lakoff Maxim 7: ground first in observable bin transitions,
then link to cross-domain abstractions for cognitive accessibility.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ca_schema import BIN_SCHEMA, CA_VAR_ORDER, bin_index, continuous_exemplar
from ca_simulator import run_single_cell


@dataclass
class ImageSchema:
    """A detected image schema with its quantified metrics."""
    name: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CATrajectory:
    """Wrapper for CA trajectory data compatible with image schema detectors.
    
    Converts discrete state dicts to continuous exemplar vectors using
    BIN_SCHEMA centers, preserving both discrete labels and continuous
    representations for schema detection.
    """
    discrete_states: List[Dict[str, str]]  # list of {var: bin_label}
    timestep_years: float = 0.25  # CA_DT = quarterly
    sampling_rate: float = 4.0    # 1 / timestep_years
    
    def __post_init__(self):
        """Compute continuous exemplar matrix and other derived representations."""
        self.n_steps = len(self.discrete_states)
        if self.n_steps == 0:
            self.continuous_matrix = np.zeros((0, len(CA_VAR_ORDER)))
            self.bin_indices = np.zeros((0, len(CA_VAR_ORDER)), dtype=int)
            return
            
        # Convert each discrete state to continuous exemplar vector
        cont_vectors = []
        bin_idx_matrix = []
        for state in self.discrete_states:
            cont_vec = continuous_exemplar(state)  # shape (8,)
            # Map from index position to CA_VAR_ORDER position
            ordered_vec = []
            for var in CA_VAR_ORDER:
                idx = BIN_SCHEMA[var]["index"]
                ordered_vec.append(cont_vec[idx])
            cont_vectors.append(ordered_vec)
            
            # Also store bin indices (0-based) for ordinal analysis
            idx_vec = []
            for var in CA_VAR_ORDER:
                label = state.get(var, BIN_SCHEMA[var]["labels"][0])
                idx = BIN_SCHEMA[var]["labels"].index(label)
                idx_vec.append(idx)
            bin_idx_matrix.append(idx_vec)
        
        self.continuous_matrix = np.array(cont_vectors)  # shape (n_steps, 8)
        self.bin_indices = np.array(bin_idx_matrix, dtype=int)
        self.var_names = CA_VAR_ORDER.copy()
        
    @property
    def timesteps(self) -> List['CATimestep']:
        """Return list of timestep objects for compatibility with motion-analytics interface."""
        # Simple adapter: create dummy objects with required attributes
        return [CATimestep(i, self.continuous_matrix[i], self.bin_indices[i])
                for i in range(self.n_steps)]
    
    @classmethod
    def from_simulation(cls, patient=None, intervention=None, sim_years=30.0, dt=0.25):
        """Create CATrajectory from a CA simulation."""
        result = run_single_cell(patient=patient, intervention=intervention,
                                sim_years=sim_years, dt=dt)
        return cls(result['trajectory'], timestep_years=dt)


@dataclass
class CATimestep:
    """Simplified timestep representation for compatibility."""
    step: int
    continuous: np.ndarray  # shape (8,)
    bin_indices: np.ndarray  # shape (8,)


class CAImageSchemaDetector:
    """Detect Lakoff image schemas in CA trajectories.
    
    Six schemas adapted for mitochondrial semantics:
      PATH   — movement toward/away from cliff (N_deletion, ATP, ROS)
      CYCLE  — periodic oscillations between bins (ROS, NAD, senescence)
      CONTAINER — bounded variables (heteroplasmy [0,1], ATP reserve headroom)
      SCALE  — graded severity progression across ordered bins
      BALANCE — homeostatic regulation (N_healthy copy homeostasis)
      FORCE  — intervention-driven push against damage accumulation
    
    Detection uses both continuous exemplars (for PATH, CYCLE metrics)
    and discrete bin indices (for SCALE, BALANCE ordinal patterns).
    """
    
    def detect_all(self, trajectory: CATrajectory) -> Dict[str, ImageSchema]:
        """Run all six schema detectors."""
        return {
            'PATH': self.detect_path(trajectory),
            'CYCLE': self.detect_cycle(trajectory),
            'CONTAINER': self.detect_container(trajectory),
            'SCALE': self.detect_scale(trajectory),
            'BALANCE': self.detect_balance(trajectory),
            'FORCE': self.detect_force(trajectory),
        }
    
    # ------------------------------------------------------------------
    # PATH — source-path-goal movement
    # ------------------------------------------------------------------
    def detect_path(self, trajectory: CATrajectory) -> ImageSchema:
        """Detect PATH schema: directed movement toward/away from health.
        
        Primary paths:
          - N_deletion: toward cliff (worsening) or away (improving)
          - ATP: toward collapse or toward health
          - ROS: toward pathological or toward basal
        
        Metrics: net displacement, straightness, curvature for each variable.
        """
        if trajectory.n_steps < 2:
            return ImageSchema('PATH', {
                'net_displacement': 0.0,
                'straightness': 0.0,
                'curvature_integral': 0.0,
                'cliff_approach_rate': 0.0,
                'health_recovery_rate': 0.0,
            })
        
        # Focus on key variables for cliff/health movement
        var_idx = {var: i for i, var in enumerate(trajectory.var_names)}
        
        # N_deletion trajectory (toward cliff = positive movement)
        del_idx = var_idx['N_deletion']
        del_vals = trajectory.continuous_matrix[:, del_idx]
        del_start = del_vals[0]
        del_end = del_vals[-1]
        del_displacement = del_end - del_start  # positive = toward cliff
        
        # ATP trajectory (toward collapse = negative movement)
        atp_idx = var_idx['ATP']
        atp_vals = trajectory.continuous_matrix[:, atp_idx]
        atp_displacement = atp_vals[-1] - atp_vals[0]  # negative = toward collapse
        
        # Overall path metrics using first principal component of all vars
        # Simple approach: average displacement across all normalized vars
        norms = np.linalg.norm(trajectory.continuous_matrix, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = trajectory.continuous_matrix / norms
        
        # Net displacement (Euclidean distance between start and end in normalized space)
        start = normalized[0]
        end = normalized[-1]
        net_displacement = float(np.linalg.norm(end - start))
        
        # Path length (sum of segment lengths)
        segments = np.diff(normalized, axis=0)
        path_length = float(np.sum(np.linalg.norm(segments, axis=1)))
        straightness = net_displacement / (path_length + 1e-10)
        
        # Curvature integral (2D projection onto first two variables for simplicity)
        if trajectory.n_steps >= 3:
            # Use first two variables (N_healthy, N_deletion) as projection
            # These capture key dynamics and are orthogonal in meaning
            proj = normalized[:, :2]  # shape (n_steps, 2)
            dx = np.gradient(proj[:, 0])
            dy = np.gradient(proj[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            denom = (dx**2 + dy**2 + 1e-10) ** 1.5
            curvature = np.abs(dx * ddy - dy * ddx) / denom
            dt = trajectory.timestep_years
            try:
                # numpy 2.x
                curvature_integral = float(np.trapezoid(curvature, dx=dt))
            except AttributeError:
                # numpy 1.x
                curvature_integral = float(np.trapz(curvature, dx=dt))
        else:
            curvature_integral = 0.0
        
        return ImageSchema('PATH', {
            'net_displacement': float(net_displacement),
            'straightness': float(straightness),
            'curvature_integral': curvature_integral,
            'cliff_approach_rate': float(del_displacement / (trajectory.n_steps * trajectory.timestep_years + 1e-10)),
            'health_recovery_rate': float(atp_displacement / (trajectory.n_steps * trajectory.timestep_years + 1e-10)),
            'n_deletion_displacement': float(del_displacement),
            'atp_displacement': float(atp_displacement),
        })
    
    # ------------------------------------------------------------------
    # CYCLE — periodic oscillations
    # ------------------------------------------------------------------
    def detect_cycle(self, trajectory: CATrajectory) -> ImageSchema:
        """Detect CYCLE schema: periodic oscillations in ROS, NAD, senescence.
        
        Uses FFT on continuous exemplars to detect dominant frequencies.
        """
        if trajectory.n_steps < 4:
            return ImageSchema('CYCLE', {
                'dominant_frequency': 0.0,
                'cycle_count': 0.0,
                'regularity': 0.0,
                'oscillating_variables': 0.0,
            })
        
        # Test each variable for periodicity
        var_idx = {var: i for i, var in enumerate(trajectory.var_names)}
        cycle_vars = []
        dominant_freqs = []
        regularities = []
        
        # Variables known to oscillate in mitochondrial dynamics
        oscillatory_vars = ['ROS', 'NAD', 'Senescent_fraction', 'Membrane_potential']
        
        for var in oscillatory_vars:
            idx = var_idx[var]
            signal = trajectory.continuous_matrix[:, idx]
            signal = signal - np.mean(signal)
            
            n = len(signal)
            sr = trajectory.sampling_rate  # samples per year
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            spectrum = np.abs(np.fft.rfft(signal))
            
            # Ignore DC
            spectrum[0] = 0.0
            if len(spectrum) < 2:
                continue
            
            peak_idx = int(np.argmax(spectrum))
            dominant_freq = float(freqs[peak_idx])
            if dominant_freq < 1e-6:  # essentially DC
                continue
            
            # Regularity: fraction of spectral energy in dominant peak ±1 bin
            lo = max(1, peak_idx - 1)
            hi = min(len(spectrum), peak_idx + 2)
            peak_energy = float(np.sum(spectrum[lo:hi] ** 2))
            total_energy = float(np.sum(spectrum[1:] ** 2)) + 1e-10
            regularity = peak_energy / total_energy
            
            # Only count if significant regularity
            if regularity > 0.3 and dominant_freq > 0.1:  # at least 0.1 cycles/year
                cycle_vars.append(var)
                dominant_freqs.append(dominant_freq)
                regularities.append(regularity)
        
        if not cycle_vars:
            return ImageSchema('CYCLE', {
                'dominant_frequency': 0.0,
                'cycle_count': 0.0,
                'regularity': 0.0,
                'oscillating_variables': 0.0,
            })
        
        # Aggregate across variables
        duration = trajectory.n_steps / trajectory.sampling_rate  # years
        avg_freq = np.mean(dominant_freqs) if dominant_freqs else 0.0
        avg_regularity = np.mean(regularities) if regularities else 0.0
        cycle_count = avg_freq * duration
        
        return ImageSchema('CYCLE', {
            'dominant_frequency': float(avg_freq),
            'cycle_count': float(cycle_count),
            'regularity': float(avg_regularity),
            'oscillating_variables': float(len(cycle_vars)),
        })
    
    # ------------------------------------------------------------------
    # CONTAINER — bounded variables
    # ------------------------------------------------------------------
    def detect_container(self, trajectory: CATrajectory) -> ImageSchema:
        """Detect CONTAINER schema: variables staying within biological bounds.
        
        Key containers:
          - Heteroplasmy fractions bounded [0, 1]
          - ATP reserve headroom (distance to collapse)
          - Senescence fraction bounded [0, 1]
        
        Metrics: boundary proximity, time near boundaries, containment stability.
        """
        if trajectory.n_steps == 0:
            return ImageSchema('CONTAINER', {
                'boundary_proximity': 0.0,
                'time_near_boundary': 0.0,
                'containment_stability': 0.0,
                'variables_at_boundary': 0.0,
            })
        
        # Define bounds for each variable (normalized [0,1] for exemplars)
        # Using approximate bounds based on bin schema extremes
        bounds = {
            'N_healthy': (0.0, 1.0),
            'N_deletion': (0.0, 1.0),
            'ATP': (0.0, 1.0),
            'ROS': (0.0, 1.0),
            'NAD': (0.0, 1.0),
            'Senescent_fraction': (0.0, 1.0),
            'Membrane_potential': (0.0, 1.0),
            'N_point': (0.0, 1.0),
        }
        
        var_idx = {var: i for i, var in enumerate(trajectory.var_names)}
        boundary_proximities = []
        near_boundary_counts = []
        
        for var, (low, high) in bounds.items():
            idx = var_idx[var]
            vals = trajectory.continuous_matrix[:, idx]
            
            # Normalize to [0,1] within bounds
            vals_norm = (vals - low) / (high - low + 1e-10)
            vals_norm = np.clip(vals_norm, 0.0, 1.0)
            
            # Distance to nearest boundary (0 = at boundary, 0.5 = middle)
            dist_to_low = vals_norm
            dist_to_high = 1.0 - vals_norm
            min_dist = np.minimum(dist_to_low, dist_to_high)
            avg_proximity = 0.5 - np.mean(min_dist)  # 0 = middle, 0.5 = at boundary
            boundary_proximities.append(avg_proximity)
            
            # Fraction of time near boundary (within 10% of boundary)
            near_boundary = np.mean(min_dist < 0.1)
            near_boundary_counts.append(near_boundary)
        
        # Aggregate metrics
        avg_proximity = np.mean(boundary_proximities) if boundary_proximities else 0.0
        avg_near_boundary = np.mean(near_boundary_counts) if near_boundary_counts else 0.0
        
        # Containment stability: inverse of variance in boundary proximity across time
        # Compute for each variable across time steps
        stability_scores = []
        for var, (low, high) in bounds.items():
            idx = var_idx[var]
            vals = trajectory.continuous_matrix[:, idx]
            vals_norm = (vals - low) / (high - low + 1e-10)
            vals_norm = np.clip(vals_norm, 0.0, 1.0)
            # Stability = 1 - (normalized variance)
            var_norm = np.var(vals_norm) / 0.25  # 0.25 is max variance for [0,1]
            stability = 1.0 - min(1.0, var_norm)
            stability_scores.append(stability)
        
        containment_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        return ImageSchema('CONTAINER', {
            'boundary_proximity': float(avg_proximity),
            'time_near_boundary': float(avg_near_boundary),
            'containment_stability': float(containment_stability),
            'variables_at_boundary': float(np.sum(np.array(near_boundary_counts) > 0.5)),
        })
    
    # ------------------------------------------------------------------
    # SCALE — graded severity progression
    # ------------------------------------------------------------------
    def detect_scale(self, trajectory: CATrajectory) -> ImageSchema:
        """Detect SCALE schema: ordered progression through severity bins.
        
        Each variable's bins are ordinally scaled (e.g., minimal→growing→approaching_cliff→past_cliff).
        Metrics: monotonicity, scale consistency, net progression.
        """
        if trajectory.n_steps < 2:
            return ImageSchema('SCALE', {
                'monotonicity': 0.0,
                'scale_consistency': 0.0,
                'net_progression': 0.0,
                'variables_with_scale': 0.0,
            })
        
        # Variables with clear ordinal scaling (worse → better direction varies)
        # For each variable, define direction: +1 = worsening, -1 = improving
        scale_vars = {
            'N_deletion': +1,    # higher bin index = worse (toward cliff)
            'ATP': -1,           # higher bin index = better (away from collapse)
            'ROS': +1,           # higher bin index = worse (toward pathological)
            'NAD': -1,           # higher bin index = better (toward robust)
            'Senescent_fraction': +1,  # higher = worse
            'Membrane_potential': -1,  # higher = better (intact)
            'N_point': +1,        # higher = worse
            'N_healthy': -1,      # higher = better (adequate)
        }
        
        var_idx = {var: i for i, var in enumerate(trajectory.var_names)}
        monotonicities = []
        net_progressions = []
        scale_consistencies = []
        
        for var, direction in scale_vars.items():
            idx = var_idx[var]
            bin_indices = trajectory.bin_indices[:, idx]
            
            if len(np.unique(bin_indices)) < 2:
                # No movement along scale
                monotonicities.append(0.0)
                net_progressions.append(0.0)
                scale_consistencies.append(1.0)  # perfectly consistent (no change)
                continue
            
            # Compute monotonicity (fraction of steps that move in consistent direction)
            diffs = np.diff(bin_indices) * direction  # positive = worsening, negative = improving
            pos_steps = np.sum(diffs > 0)
            neg_steps = np.sum(diffs < 0)
            zero_steps = np.sum(diffs == 0)
            total_steps = len(diffs)
            
            if total_steps == 0:
                monotonicity = 0.0
            else:
                # Monotonicity = max(fraction positive, fraction negative)
                monotonicity = max(pos_steps, neg_steps) / total_steps
            
            # Net progression (final - initial) × direction
            net_prog = (bin_indices[-1] - bin_indices[0]) * direction
            
            # Scale consistency: variance of bin indices (lower = more stable on scale)
            var_norm = np.var(bin_indices) / (len(BIN_SCHEMA[var]["labels"]) - 1) ** 2
            consistency = 1.0 - min(1.0, var_norm)
            
            monotonicities.append(monotonicity)
            net_progressions.append(net_prog)
            scale_consistencies.append(consistency)
        
        # Aggregate across variables
        avg_monotonicity = np.mean(monotonicities) if monotonicities else 0.0
        avg_net_progression = np.mean(net_progressions) if net_progressions else 0.0
        avg_consistency = np.mean(scale_consistencies) if scale_consistencies else 0.0
        
        # Count variables showing meaningful scale movement (|net_progression| >= 1 bin)
        vars_with_scale = np.sum(np.abs(net_progressions) >= 1.0) if net_progressions else 0
        
        return ImageSchema('SCALE', {
            'monotonicity': float(avg_monotonicity),
            'scale_consistency': float(avg_consistency),
            'net_progression': float(avg_net_progression),
            'variables_with_scale': float(vars_with_scale),
            'n_deletion_net_progression': float(net_progressions[var_idx['N_deletion']] if 'N_deletion' in var_idx else 0.0),
            'atp_net_progression': float(net_progressions[var_idx['ATP']] if 'ATP' in var_idx else 0.0),
        })
    
    # ------------------------------------------------------------------
    # BALANCE — homeostatic regulation
    # ------------------------------------------------------------------
    def detect_balance(self, trajectory: CATrajectory) -> ImageSchema:
        """Detect BALANCE schema: homeostatic regulation around target values.
        
        Primary balance: N_healthy copy homeostasis (target ~ adequate bin).
        Also: NAD balance, membrane potential balance.
        
        Metrics: deviation from target, restoration rate, oscillation damping.
        """
        if trajectory.n_steps < 3:
            return ImageSchema('BALANCE', {
                'homeostatic_deviation': 0.0,
                'restoration_rate': 0.0,
                'oscillation_damping': 0.0,
                'balanced_variables': 0.0,
            })
        
        # Variables with homeostatic targets
        balance_vars = {
            'N_healthy': 2,  # target bin index = "adequate" (index 2)
            'NAD': 2,        # target = "robust" (index 2)
            'Membrane_potential': 2,  # target = "intact" (index 2)
        }
        
        var_idx = {var: i for i, var in enumerate(trajectory.var_names)}
        deviations = []
        restoration_rates = []
        damping_scores = []
        
        for var, target_idx in balance_vars.items():
            if var not in var_idx:
                continue
            idx = var_idx[var]
            bin_indices = trajectory.bin_indices[:, idx]
            
            # Mean absolute deviation from target
            mad = np.mean(np.abs(bin_indices - target_idx))
            max_dev = max(len(BIN_SCHEMA[var]["labels"]) - 1 - target_idx, target_idx)
            norm_dev = mad / (max_dev + 1e-10)
            deviations.append(norm_dev)
            
            # Restoration rate: correlation between deviation and subsequent correction
            if len(bin_indices) >= 3:
                devs = bin_indices - target_idx
                # Positive autocorrelation at lag 1 means persistence (poor restoration)
                # Negative autocorrelation means correction (good restoration)
                if np.std(devs) > 1e-6:
                    lag0 = devs[:-1]
                    lag1 = devs[1:]
                    # Degenerate lag slices can produce NaNs in corrcoef.
                    if np.std(lag0) <= 1e-6 or np.std(lag1) <= 1e-6:
                        restoration = 0.5  # neutral when correlation is not identifiable
                    else:
                        autocorr = np.corrcoef(lag0, lag1)[0, 1]
                        if np.isfinite(autocorr):
                            restoration = -autocorr  # negative autocorrelation = restoration
                            restoration = max(0.0, min(1.0, (restoration + 1) / 2))  # map [-1,1] → [0,1]
                        else:
                            restoration = 0.5
                else:
                    restoration = 1.0  # perfect restoration (no deviation)
                restoration_rates.append(restoration)
            
            # Oscillation damping: ratio of early vs late oscillation amplitude
            if len(bin_indices) >= 6:
                # Split into thirds
                third = len(bin_indices) // 3
                early = bin_indices[:third]
                late = bin_indices[-third:]
                amp_early = np.std(early) if len(early) > 1 else 0.0
                amp_late = np.std(late) if len(late) > 1 else 0.0
                if amp_early > 1e-6:
                    damping = 1.0 - (amp_late / amp_early)
                    damping = max(0.0, min(1.0, damping))
                else:
                    damping = 1.0  # no oscillation to damp
                damping_scores.append(damping)
        
        # Aggregate metrics
        avg_deviation = np.mean(deviations) if deviations else 0.0
        avg_restoration = np.mean(restoration_rates) if restoration_rates else 0.0
        avg_damping = np.mean(damping_scores) if damping_scores else 0.0
        
        # Count variables with good balance (deviation < 0.3, restoration > 0.7)
        balanced_count = 0
        for i, var in enumerate(balance_vars):
            if i < len(deviations) and i < len(restoration_rates):
                if deviations[i] < 0.3 and restoration_rates[i] > 0.7:
                    balanced_count += 1
        
        return ImageSchema('BALANCE', {
            'homeostatic_deviation': float(avg_deviation),
            'restoration_rate': float(avg_restoration),
            'oscillation_damping': float(avg_damping),
            'balanced_variables': float(balanced_count),
            'n_healthy_deviation': float(deviations[0] if deviations else 0.0),
        })
    
    # ------------------------------------------------------------------
    # FORCE — intervention-driven push
    # ------------------------------------------------------------------
    def detect_force(self, trajectory: CATrajectory) -> ImageSchema:
        """Detect FORCE schema: intervention-driven push against damage.
        
        Metrics: correlation between intervention intensity and health improvement,
        force magnitude (net bin transitions toward health), force efficiency.
        """
        if trajectory.n_steps < 2:
            return ImageSchema('FORCE', {
                'force_magnitude': 0.0,
                'force_efficiency': 0.0,
                'health_push_correlation': 0.0,
                'intervention_effectiveness': 0.0,
            })
        
        # Without direct intervention timeline, infer from trajectory shape
        # Force magnitude: net health-improving bin transitions
        # For each variable, count transitions toward health (considering direction)
        scale_vars = {
            'N_deletion': -1,    # decreasing bin index = improvement
            'ATP': +1,           # increasing bin index = improvement
            'ROS': -1,           # decreasing bin index = improvement
            'NAD': +1,           # increasing bin index = improvement
            'Senescent_fraction': -1,
            'Membrane_potential': +1,
            'N_point': -1,
            'N_healthy': +1,
        }
        
        var_idx = {var: i for i, var in enumerate(trajectory.var_names)}
        improvement_counts = []
        total_transitions = 0
        
        for var, direction in scale_vars.items():
            idx = var_idx[var]
            bin_indices = trajectory.bin_indices[:, idx]
            diffs = np.diff(bin_indices) * direction  # positive = improvement
            
            improvements = np.sum(diffs > 0)
            worsenings = np.sum(diffs < 0)
            total = len(diffs)
            
            if total > 0:
                improvement_counts.append(improvements / total)
                total_transitions += total
        
        # Force magnitude = average improvement rate across variables
        force_magnitude = np.mean(improvement_counts) if improvement_counts else 0.0
        
        # Force efficiency = improvement rate per transition (weighted by importance)
        # Weight N_deletion and ATP more heavily
        weights = {'N_deletion': 3.0, 'ATP': 3.0, 'ROS': 2.0, 'NAD': 2.0,
                  'Senescent_fraction': 1.5, 'Membrane_potential': 1.5,
                  'N_point': 1.0, 'N_healthy': 1.0}
        weighted_improvements = 0.0
        total_weight = 0.0
        
        for var, direction in scale_vars.items():
            idx = var_idx[var]
            bin_indices = trajectory.bin_indices[:, idx]
            diffs = np.diff(bin_indices) * direction
            improvements = np.sum(diffs > 0)
            total = len(diffs)
            
            if total > 0:
                rate = improvements / total
                weighted_improvements += rate * weights.get(var, 1.0)
                total_weight += weights.get(var, 1.0)
        
        force_efficiency = weighted_improvements / (total_weight + 1e-10)
        
        # Health push correlation: correlation between N_deletion and ATP movements
        # Negative correlation means when deletion increases (worsens), ATP decreases (worsens)
        del_idx = var_idx.get('N_deletion')
        atp_idx = var_idx.get('ATP')
        health_push_corr = 0.5  # default neutral
        if del_idx is not None and atp_idx is not None:
            del_diffs = np.diff(trajectory.bin_indices[:, del_idx])
            atp_diffs = np.diff(trajectory.bin_indices[:, atp_idx])
            try:
                if np.std(del_diffs) > 1e-6 and np.std(atp_diffs) > 1e-6:
                    corr = np.corrcoef(del_diffs, atp_diffs)[0, 1]
                    # Negative correlation is expected (deletion↑ → ATP↓)
                    # Convert to 0-1 score: -1 → 1, 0 → 0.5, 1 → 0
                    health_push_corr = (1.0 - corr) / 2.0  # maps [-1,1] → [1,0] but we want opposite
                    health_push_corr = 1.0 - health_push_corr  # Actually want: -1 → 1, 1 → 0
            except (ValueError, RuntimeWarning):
                health_push_corr = 0.5
        
        # Intervention effectiveness estimate (simplistic)
        # Based on net improvement in key variables
        key_vars = ['N_deletion', 'ATP', 'ROS']
        net_improvements = []
        for var in key_vars:
            if var in var_idx:
                idx = var_idx[var]
                direction = scale_vars[var]
                net = (trajectory.bin_indices[-1, idx] - trajectory.bin_indices[0, idx]) * direction
                net_improvements.append(net)
        
        intervention_effectiveness = np.mean(net_improvements) if net_improvements else 0.0
        
        return ImageSchema('FORCE', {
            'force_magnitude': float(force_magnitude),
            'force_efficiency': float(force_efficiency),
            'health_push_correlation': float(health_push_corr),
            'intervention_effectiveness': float(intervention_effectiveness),
            'weighted_improvement_rate': float(force_efficiency),
        })


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

def detect_schemas_in_trajectory(discrete_states: List[Dict[str, str]], 
                                 timestep_years: float = 0.25) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper: detect all schemas from discrete state trajectory."""
    trajectory = CATrajectory(discrete_states, timestep_years=timestep_years)
    detector = CAImageSchemaDetector()
    schemas = detector.detect_all(trajectory)
    return {name: schema.metrics for name, schema in schemas.items()}


def detect_schemas_from_simulation(patient=None, intervention=None, 
                                   sim_years=30.0, dt=0.25) -> Dict[str, Dict[str, float]]:
    """Run CA simulation and detect image schemas in the resulting trajectory."""
    trajectory = CATrajectory.from_simulation(
        patient=patient, intervention=intervention,
        sim_years=sim_years, dt=dt
    )
    detector = CAImageSchemaDetector()
    schemas = detector.detect_all(trajectory)
    return {name: schema.metrics for name, schema in schemas.items()}


if __name__ == '__main__':
    # Quick test
    print("Testing CA image schema detector...")
    schemas = detect_schemas_from_simulation(sim_years=5.0)
    for schema_name, metrics in schemas.items():
        print(f"\n{schema_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
