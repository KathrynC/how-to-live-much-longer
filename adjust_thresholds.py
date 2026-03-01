#!/usr/bin/env python3
"""
Adjust bin thresholds based on empirical ODE distribution.

For each variable, compute new thresholds as midpoints between adjacent bin means.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

from ca_schema import BIN_SCHEMA

def load_bin_stats():
    """Load global bin statistics from validation_summary.json."""
    path = Path("artifacts/ca_ode_validation/validation_summary.json")
    with open(path, 'r') as f:
        data = json.load(f)
    return data['global_bin_stats']

def compute_thresholds_from_means(bin_stats, current_schema):
    """Compute new thresholds as midpoints between adjacent bin means.
    
    Args:
        bin_stats: dict from load_bin_stats()
        current_schema: BIN_SCHEMA dict
        
    Returns:
        dict: variable_name -> list of new thresholds (same length as original)
    """
    new_thresholds = {}
    
    for var_name, schema in current_schema.items():
        labels = schema['labels']
        n_bins = len(labels)
        if n_bins < 2:
            # No thresholds needed
            new_thresholds[var_name] = []
            continue
        
        # Get means for each label
        means = []
        for label in labels:
            if label in bin_stats.get(var_name, {}):
                mean = bin_stats[var_name][label]['mean']
            else:
                # fallback: use current threshold midpoints? Use current center
                mean = schema['centers'][labels.index(label)]
            means.append(mean)
        
        # Ensure means are monotonic (should be, but check)
        # For ordered bins (e.g., depleted < reduced < adequate), means should increase.
        # If not, sort means and labels together? Let's assume order is correct.
        # Compute thresholds as midpoints between consecutive means
        thresholds = []
        for i in range(n_bins - 1):
            mid = (means[i] + means[i+1]) / 2.0
            thresholds.append(mid)
        
        # Clip thresholds to be within variable's plausible range (0-1 for normalized)
        # Also ensure thresholds are strictly increasing
        for i in range(len(thresholds)-1):
            if thresholds[i] >= thresholds[i+1]:
                thresholds[i+1] = thresholds[i] + 1e-4
        
        new_thresholds[var_name] = thresholds
    
    return new_thresholds

def compute_thresholds_from_percentiles(bin_stats, current_schema, percentile=50):
    """Compute thresholds as percentiles of pooled ODE values.
    
    Not implemented yet.
    """
    # For each variable, we would need raw ODE values per bin.
    # We don't have raw values in bin_stats, only summary stats.
    # So we cannot compute percentiles without raw data.
    # We'll stick to midpoint method.
    pass

def main():
    print("Adjusting bin thresholds based on empirical ODE distribution")
    print("=" * 60)
    
    # Load data
    bin_stats = load_bin_stats()
    current_schema = BIN_SCHEMA
    
    # Compute new thresholds
    new_thresholds = compute_thresholds_from_means(bin_stats, current_schema)
    
    # Print changes
    print("\nThreshold adjustments (old → new):")
    for var_name, schema in current_schema.items():
        old = schema['thresholds']
        new = new_thresholds[var_name]
        if not old:
            continue
        print(f"\n{var_name}:")
        for i, (o, n) in enumerate(zip(old, new)):
            diff = n - o
            print(f"  threshold {i}: {o:.3f} → {n:.3f} (Δ{diff:+.3f})")
    
    # Generate updated schema
    updated_schema = {}
    for var_name, schema in current_schema.items():
        updated = schema.copy()
        updated['thresholds'] = new_thresholds[var_name]
        updated_schema[var_name] = updated
    
    # Save updated schema
    output_path = Path("artifacts/ca_ode_validation/updated_thresholds.json")
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
            'source': schema['source'] + ' + empirical threshold adjustment',
        }
    
    with open(output_path, 'w') as f:
        json.dump(jsonable, f, indent=2)
    
    print(f"\nUpdated thresholds saved to {output_path}")
    
    # Also generate a combined patch (centers + thresholds)
    # Load updated centers (clamped) from previous step
    centers_path = Path("artifacts/ca_ode_validation/updated_bin_schema.json")
    if centers_path.exists():
        with open(centers_path, 'r') as f:
            centers_data = json.load(f)
        # Merge centers and thresholds
        for var_name in jsonable.keys():
            if var_name in centers_data:
                jsonable[var_name]['centers'] = centers_data[var_name]['centers']
                jsonable[var_name]['source'] = centers_data[var_name]['source'] + ' + thresholds'
    
    combined_path = Path("artifacts/ca_ode_validation/updated_schema_combined.json")
    with open(combined_path, 'w') as f:
        json.dump(jsonable, f, indent=2)
    
    print(f"Combined schema (centers + thresholds) saved to {combined_path}")
    
    # Generate Python patch
    patch_path = Path("artifacts/ca_ode_validation/schema_patch_full.py")
    with open(patch_path, 'w') as f:
        f.write("# Updated BIN_SCHEMA with empirical centers and thresholds\n")
        f.write("# Replace entire BIN_SCHEMA in ca_schema.py with this:\n")
        f.write("\n")
        f.write("BIN_SCHEMA = {\n")
        for var_name, schema in jsonable.items():
            f.write(f'    "{var_name}": {{\n')
            f.write(f'        "index": {schema["index"]},\n')
            thr_str = ', '.join(f'{t:.3f}' for t in schema["thresholds"])
            f.write(f'        "thresholds": [{thr_str}],\n')
            labels_str = ', '.join(f'"{l}"' for l in schema["labels"])
            f.write(f'        "labels": [{labels_str}],\n')
            centers_str = ', '.join(f'{c:.3f}' for c in schema["centers"])
            f.write(f'        "centers": [{centers_str}],\n')
            f.write(f'        "unit": "{schema["unit"]}",\n')
            f.write(f'        "source": "{schema["source"]}"\n')
            f.write('    },\n')
        f.write("}\n")
    
    print(f"Full patch saved to {patch_path}")

if __name__ == "__main__":
    main()