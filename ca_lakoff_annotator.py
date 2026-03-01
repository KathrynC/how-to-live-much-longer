"""CA-Lakoff annotator: dual-vocabulary annotation of CA trajectories with Lakoff image schemas.

Integrates:
1. CA simulation (single-cell or tissue grid)
2. CA analytics (rule stats, attractor classification, fidelity)
3. CA image schema detection (PATH, CYCLE, CONTAINER, SCALE, BALANCE, FORCE)
4. Lakoff archetype matching (conservative, aggressive, transplant_focused, metabolic_optimizer)
5. Metaphor violation detection

Outputs dual-vocabulary annotation dictionary mapping each time step to:
- Discrete CA state (bin labels)
- Image schema activations (schema names with strength metrics)
- Archetype similarity scores
- ICM violation flags

Follows Lakoff Maxim 7: ground first in observable bin transitions, then link to
cross-domain abstractions for cognitive accessibility.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ca_simulator import run_single_cell, run_tissue_grid
from ca_analytics import compute_ca_analytics
from ca_image_schemas import detect_schemas_in_trajectory, CATrajectory
from patterns.lakoff_integration import (
    ArchetypeLibrary, MetaphorAuditor, extract_features_from_analytics,
    get_feature_layer, FEATURE_ALIASES
)
from simulator import simulate
from analytics import compute_all


def extract_features_from_ca_analytics(ca_analytics_dict: Dict[str, Dict]) -> Dict[str, float]:
    """Flatten CA analytics dict into flat feature dictionary.
    
    Maps CA analytics keys to feature names defined in lakoff_integration.py
    FEATURE_LAYERS (e.g., ca_bin_agreement, ca_attractor).
    
    Args:
        ca_analytics_dict: Output from compute_ca_analytics() with keys:
            rule_stats, cascade_stats, attractor_stats, fidelity_stats,
            epoch_diagnostic.
    
    Returns:
        Flat dict with ca.* prefixed keys.
    """
    features = {}
    
    # Rule stats
    rule = ca_analytics_dict.get('rule_stats', {})
    features['ca_rule_firings_total'] = float(rule.get('total_firings', 0))
    features['ca_unique_rules'] = float(rule.get('unique_rules', 0))
    features['ca_mean_rules_per_step'] = float(rule.get('mean_rules_per_step', 0.0))
    
    # Cascade stats
    cascade = ca_analytics_dict.get('cascade_stats', {})
    features['ca_cascade_count'] = float(cascade.get('cascade_count', 0))
    features['ca_longest_cascade'] = float(cascade.get('longest_cascade', 0))
    
    # Attractor stats
    attractor = ca_analytics_dict.get('attractor_stats', {})
    # Map attractor name to index: healthy_aging=0, slow_decline=1, cliff_approaching=2, point_of_no_return=3
    attractor_map = {
        'healthy_aging': 0.0,
        'slow_decline': 1.0,
        'cliff_approaching': 2.0,
        'point_of_no_return': 3.0,
    }
    terminal = attractor.get('terminal_attractor', 'healthy_aging')
    features['ca_attractor'] = attractor_map.get(terminal, 0.0)
    features['ca_attractor_transitions'] = float(attractor.get('attractor_transitions', 0))
    
    # Fidelity stats (if ODE comparison available)
    fidelity = ca_analytics_dict.get('fidelity_stats', {})
    features['ca_bin_agreement'] = float(fidelity.get('bin_agreement', 0.0))
    features['ca_rmse'] = float(fidelity.get('rmse', 0.0))
    features['ca_cliff_crossing'] = float(fidelity.get('cliff_crossing', 0.0))
    features['ca_time_to_cliff'] = float(fidelity.get('time_to_cliff', float('nan')))
    
    # Epoch diagnostic
    epoch = ca_analytics_dict.get('epoch_diagnostic', {})
    # Summarize as number of variables that changed across age-65 transition
    changes = epoch.get('changes', {})
    features['ca_epoch_changes'] = float(len(changes))
    
    return features


def extract_features_from_image_schemas(schema_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten image schema metrics into flat feature dictionary.
    
    Maps schema metric keys to ca_* feature names.
    """
    features = {}
    for schema_name, metrics in schema_metrics.items():
        prefix = f'ca_{schema_name.lower()}'
        for metric_name, value in metrics.items():
            # Create canonical feature name (should match FEATURE_LAYERS)
            # e.g., ca_path_net_displacement, ca_cycle_dominant_frequency
            # Use mapping for consistency
            key = f'{prefix}_{metric_name}'
            features[key] = float(value)
    return features


def load_lakoff_library(archetypes_path: Optional[Path] = None) -> ArchetypeLibrary:
    """Load Lakoff archetype library from JSON file."""
    if archetypes_path is None:
        archetypes_path = Path(__file__).parent / 'patterns' / 'lakoff_archetypes.json'
    return ArchetypeLibrary.load(archetypes_path)


def annotate_ca_trajectory(
    ca_result: Dict[str, Any],
    ode_result: Optional[Dict[str, Any]] = None,
    archetypes_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Annotate a CA simulation result with Lakoff image schemas and archetypes.
    
    Args:
        ca_result: Output from ca_simulator.run_single_cell() or run_tissue_grid()
        ode_result: Optional ODE simulation result for fidelity comparison
        archetypes_path: Path to lakoff_archetypes.json (defaults to patterns/)
    
    Returns:
        Annotation dictionary with keys:
            - discrete_trajectory: list of bin label dicts
            - image_schemas: dict mapping schema names to metrics
            - ca_analytics: raw CA analytics dict
            - ca_features: flattened CA feature dict
            - archetype_similarities: dict mapping archetype names to scores
            - best_archetype: (name, score) tuple
            - metaphor_violations: list of violation descriptions
            - dual_vocabulary: list of per-step annotations
    """
    # 1. Extract discrete trajectory
    trajectory = ca_result['trajectory']
    
    # 2. Compute CA analytics
    ca_analytics = compute_ca_analytics(ca_result, ode_result)
    
    # 3. Extract CA features
    ca_features = extract_features_from_ca_analytics(ca_analytics)
    
    # 4. Detect image schemas
    schema_metrics = detect_schemas_in_trajectory(trajectory)
    schema_features = extract_features_from_image_schemas(schema_metrics)
    
    # 5. Extract ODE analytics features if ode_result provided
    ode_features = {}
    if ode_result is not None:
        from analytics import compute_all
        ode_analytics = compute_all(ode_result)
        ode_features = extract_features_from_analytics(ode_analytics)
    
    # Merge features (CA analytics + image schemas + ODE analytics)
    all_features = {**ode_features, **ca_features, **schema_features}
    
    # 5. Load Lakoff archetypes
    library = load_lakoff_library(archetypes_path)
    
    # 6. Compute archetype similarities
    archetype_similarities = library.similarity_vector(all_features)
    best_arch, best_score = library.best_match(all_features)
    
    # 7. Check metaphor violations
    auditor = MetaphorAuditor(library)
    audit_result = auditor.audit(all_features)
    
    # 8. Build dual-vocabulary per-step annotations
    dual_vocabulary = []
    # For each time step, we could compute image schema activations over a sliding window
    # For simplicity, we'll assign global schema metrics to all steps
    # Alternatively, we could compute per-step schema activations (future enhancement)
    for step, state in enumerate(trajectory):
        dual_vocabulary.append({
            'step': step,
            'age': ca_result.get('patient', {}).get('baseline_age', 70.0) + step * ca_result.get('dt', 0.25),
            'discrete_state': state,
            # Could add per-step schema activations here
        })
    
    return {
        'discrete_trajectory': trajectory,
        'image_schemas': schema_metrics,
        'ca_analytics': ca_analytics,
        'ca_features': ca_features,
        'schema_features': schema_features,
        'archetype_similarities': archetype_similarities,
        'best_archetype': (best_arch.name if best_arch else None, best_score),
        'metaphor_violations': audit_result.get('violations', []),
        'dual_vocabulary': dual_vocabulary,
        'all_features': all_features,
    }


def annotate_from_simulation(
    patient=None,
    intervention=None,
    sim_years=30.0,
    dt=0.25,
    archetypes_path=None,
) -> Dict[str, Any]:
    """Run both ODE and CA simulations and annotate the CA trajectory.
    
    Convenience wrapper that runs:
      1. ODE simulation (for fidelity comparison)
      2. CA simulation (single-cell)
      3. Annotation pipeline
    
    Returns the annotation dictionary.
    """
    from simulator import simulate
    from analytics import compute_all
    
    # Run ODE simulation
    ode_result = simulate(patient=patient, intervention=intervention,
                         sim_years=sim_years)
    
    # Run CA simulation
    ca_result = run_single_cell(patient=patient, intervention=intervention,
                               sim_years=sim_years, dt=dt)
    
    # Annotate
    return annotate_ca_trajectory(ca_result, ode_result, archetypes_path)


def save_annotation(annotation: Dict[str, Any], path: Path):
    """Save annotation dictionary to JSON file (with numpy serialization)."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)
    
    with open(path, 'w') as f:
        json.dump(annotation, f, cls=NumpyEncoder, indent=2)


if __name__ == '__main__':
    # Quick test with default parameters
    print("Running CA-Lakoff annotation test...")
    annotation = annotate_from_simulation(sim_years=5.0)
    
    print(f"\nBest archetype: {annotation['best_archetype'][0]} "
          f"(score: {annotation['best_archetype'][1]:.3f})")
    
    print("\nArchetype similarities:")
    for name, score in annotation['archetype_similarities'].items():
        print(f"  {name}: {score:.3f}")
    
    print("\nImage schemas detected:")
    for schema, metrics in annotation['image_schemas'].items():
        print(f"  {schema}: {len(metrics)} metrics")
    
    print("\nCA features (sample):")
    for key, val in list(annotation['ca_features'].items())[:5]:
        print(f"  {key}: {val:.4f}")
    
    # Save to file
    output_path = Path('output') / 'ca_lakoff_annotation_test.json'
    output_path.parent.mkdir(exist_ok=True)
    save_annotation(annotation, output_path)
    print(f"\nAnnotation saved to {output_path}")
