#!/usr/bin/env python3
"""
Adjust CA bin schema based on empirical ODE distribution from validation.

Loads validation_summary.json, computes new centers (and optionally thresholds)
to better match ODE continuous values, and outputs an updated BIN_SCHEMA.
"""

import json
import numpy as np
from pathlib import Path

def load_validation_stats():
    """Load global bin statistics from validation_summary.json."""
    path = Path("artifacts/ca_ode_validation/validation_summary.json")
    with open(path, 'r') as f:
        data = json.load(f)
    return data['global_bin_stats'], data['global_exemplar_comparison']

def compute_percentile_thresholds(bin_stats, n_bins):
    """Compute thresholds as percentiles of ODE values across all bins.
    
    Not used for now.
    """
    # Flatten all values per variable
    pass

def compute_bin_bounds(current_schema):
    """Return dict var->label->(lower, upper) based on thresholds."""
    bounds = {}
    for var_name, schema in current_schema.items():
        thresholds = schema['thresholds']
        labels = schema['labels']
        var_bounds = {}
        # First bin: lower bound = 0.0 (or maybe -inf)
        for i, label in enumerate(labels):
            lower = 0.0 if i == 0 else thresholds[i-1]
            upper = 1.0 if i == len(labels)-1 else thresholds[i]
            var_bounds[label] = (lower, upper)
        bounds[var_name] = var_bounds
    return bounds

def adjust_centers_free(bin_stats):
    """Return new centers dictionary {var: {label: new_center}}.
    
    Uses empirical mean of ODE values per bin (unconstrained).
    """
    new_centers = {}
    for var_name, bins in bin_stats.items():
        new_centers[var_name] = {}
        for bin_label, stat in bins.items():
            new_centers[var_name][bin_label] = stat['mean']
    return new_centers

def adjust_centers_clamped(bin_stats, current_schema):
    """Return new centers clamped to bin intervals."""
    bounds = compute_bin_bounds(current_schema)
    new_centers = {}
    for var_name, bins in bin_stats.items():
        new_centers[var_name] = {}
        for bin_label, stat in bins.items():
            mean = stat['mean']
            lower, upper = bounds[var_name][bin_label]
            clamped = np.clip(mean, lower, upper)
            new_centers[var_name][bin_label] = clamped
    return new_centers

def adjust_centers(bin_stats, current_schema=None, clamped=False):
    """Legacy: defaults to free."""
    if clamped and current_schema is not None:
        return adjust_centers_clamped(bin_stats, current_schema)
    return adjust_centers_free(bin_stats)

def adjust_thresholds(bin_stats, current_schema):
    """Suggest new thresholds based on bin distribution overlap.
    
    For each variable, compute thresholds that separate the empirical
    distributions (e.g., where cumulative distribution crosses).
    Simple approach: set threshold at midpoint between adjacent bin means.
    """
    # Not implemented yet
    return current_schema

def load_current_schema():
    """Load current BIN_SCHEMA from ca_schema.py by importing."""
    import sys
    sys.path.insert(0, '.')
    from ca_schema import BIN_SCHEMA
    return BIN_SCHEMA

def generate_updated_schema(current_schema, new_centers):
    """Generate updated BIN_SCHEMA dict with new centers."""
    updated = {}
    for var_name, schema in current_schema.items():
        updated_schema = schema.copy()
        labels = schema['labels']
        new_center_list = [new_centers[var_name][label] for label in labels]
        updated_schema['centers'] = new_center_list
        updated[var_name] = updated_schema
    return updated

def main():
    print("Adjusting CA bin schema based on empirical ODE distribution")
    print("=" * 60)
    
    # Load data
    bin_stats, exemplar_comp = load_validation_stats()
    current_schema = load_current_schema()
    
    # Compute new centers (free and clamped)
    new_centers_free = adjust_centers_free(bin_stats)
    new_centers_clamped = adjust_centers_clamped(bin_stats, current_schema)
    # Choose which to use for schema update
    use_clamped = True
    new_centers = new_centers_clamped if use_clamped else new_centers_free
    
    # Generate updated schema
    updated_schema = generate_updated_schema(current_schema, new_centers)
    
    # Print changes
    print("\nCenter adjustments (free vs clamped):")
    for var_name, schema in current_schema.items():
        print(f"\n{var_name}:")
        for idx, label in enumerate(schema['labels']):
            old = schema['centers'][idx]
            new_free = new_centers_free[var_name][label]
            new_clamp = new_centers_clamped[var_name][label]
            diff_free = new_free - old
            diff_clamp = new_clamp - old
            clamp_note = '' if new_free == new_clamp else ' (clamped)'
            print(f"  {label}: {old:.3f} → {new_free:.3f} (Δ{diff_free:+.3f}) / {new_clamp:.3f} (Δ{diff_clamp:+.3f}){clamp_note}")
    
    # Output new schema as Python code
    output_path = Path("artifacts/ca_ode_validation/updated_bin_schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable dict
    jsonable = {}
    for var_name, schema in updated_schema.items():
        jsonable[var_name] = {
            'index': schema['index'],
            'thresholds': schema['thresholds'],
            'labels': schema['labels'],
            'centers': schema['centers'],
            'unit': schema['unit'],
            'source': schema['source'] + ' + empirical adjustment',
        }
    
    with open(output_path, 'w') as f:
        json.dump(jsonable, f, indent=2)
    
    print(f"\nUpdated schema saved to {output_path}")
    
    # Also generate a patch for ca_schema.py
    patch_path = Path("artifacts/ca_ode_validation/schema_patch.py")
    with open(patch_path, 'w') as f:
        f.write("# Updated BIN_SCHEMA centers based on ODE empirical means\n")
        f.write("# Replace the 'centers' lists in ca_schema.py with these:\n")
        f.write("\n")
        for var_name, schema in updated_schema.items():
            centers = schema['centers']
            centers_str = ', '.join(f'{c:.3f}' for c in centers)
            f.write(f"# {var_name}: [{centers_str}]\n")
        f.write("\n")
        f.write("BIN_SCHEMA = {\n")
        for var_name, schema in updated_schema.items():
            f.write(f'    "{var_name}": {{\n')
            f.write(f'        "index": {schema["index"]},\n')
            thr_str = ', '.join(str(t) for t in schema["thresholds"])
            f.write(f'        "thresholds": [{thr_str}],\n')
            labels_str = ', '.join(f'"{l}"' for l in schema["labels"])
            f.write(f'        "labels": [{labels_str}],\n')
            centers_str = ', '.join(f'{c:.3f}' for c in schema["centers"])
            f.write(f'        "centers": [{centers_str}],\n')
            f.write(f'        "unit": "{schema["unit"]}",\n')
            f.write(f'        "source": "{schema["source"]}"\n')
            f.write('    },\n')
        f.write("}\n")
    
    print(f"Patch file saved to {patch_path}")
    
    # Compute expected RMSE improvement (rough)
    def compute_rmse(centers):
        total_sq_err = 0.0
        total_count = 0
        for var_name, bins in bin_stats.items():
            for bin_label, stat in bins.items():
                count = stat['count']
                mean = stat['mean']
                std = stat['std']
                center = centers[var_name][bin_label]
                sq_err = count * (std**2 + (mean - center)**2)
                total_sq_err += sq_err
                total_count += count
        return np.sqrt(total_sq_err / total_count) if total_count else 0.0
    
    rmse_old = compute_rmse({var: {label: schema['centers'][idx] for idx, label in enumerate(schema['labels'])} 
                             for var, schema in current_schema.items()})
    rmse_free = compute_rmse(new_centers_free)
    rmse_clamped = compute_rmse(new_centers_clamped)
    
    print("\nExpected continuous RMSE improvement:")
    print(f"  Old RMSE (across all bins): {rmse_old:.4f}")
    print(f"  Free centers RMSE: {rmse_free:.4f} (improvement {rmse_old - rmse_free:.4f}, {100*(rmse_old-rmse_free)/rmse_old:.1f}%)")
    print(f"  Clamped centers RMSE: {rmse_clamped:.4f} (improvement {rmse_old - rmse_clamped:.4f}, {100*(rmse_old-rmse_clamped)/rmse_old:.1f}%)")

if __name__ == "__main__":
    main()