#!/usr/bin/env python3
"""Quick test of CA-Lakoff annotator."""
import sys
sys.path.insert(0, '.')

from ca_lakoff_annotator import annotate_from_simulation

def main():
    print("Testing CA-Lakoff annotator with 2-year simulation...")
    annotation = annotate_from_simulation(sim_years=2.0)
    
    print(f"\nBest archetype: {annotation['best_archetype'][0]} "
          f"(score: {annotation['best_archetype'][1]:.3f})")
    
    print("\nArchetype similarities:")
    for name, score in annotation['archetype_similarities'].items():
        print(f"  {name}: {score:.3f}")
    
    print("\nImage schemas detected:")
    for schema, metrics in annotation['image_schemas'].items():
        print(f"  {schema}:")
        for k, v in list(metrics.items())[:3]:  # show first 3 metrics
            print(f"    {k}: {v:.4f}")
    
    print("\nCA features (sample):")
    for key, val in list(annotation['ca_features'].items())[:5]:
        print(f"  {key}: {val:.4f}")
    
    print("\nSchema features (sample):")
    for key, val in list(annotation['schema_features'].items())[:5]:
        print(f"  {key}: {val:.4f}")
    
    print("\nMetaphor violations:", len(annotation['metaphor_violations']))
    for v in annotation['metaphor_violations']:
        print(f"  - {v}")
    
    # Save
    import json
    from pathlib import Path
    output_path = Path('output') / 'ca_lakoff_test.json'
    output_path.parent.mkdir(exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)
    
    with open(output_path, 'w') as f:
        json.dump(annotation, f, cls=NumpyEncoder, indent=2)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()