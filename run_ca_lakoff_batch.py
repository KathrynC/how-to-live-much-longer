#!/usr/bin/env python3
"""Batch annotation of patient populations with CA-Lakoff dual vocabulary.

Loads patient profiles from artifacts/sample_patients_{100,edge}.json,
runs CA-Lakoff annotator for each patient (with optional intervention),
and saves annotations to output/ca_lakoff_batch/.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, '.')

from ca_lakoff_annotator import annotate_from_simulation, save_annotation


def load_patients(json_path: Path) -> List[Dict[str, Any]]:
    """Load patient list from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data['patients']


def annotate_patient_batch(
    patients: List[Dict[str, Any]],
    intervention: Optional[Dict[str, float]] = None,
    sim_years: float = 30.0,
    dt: float = 0.25,
    output_dir: Path = Path('output') / 'ca_lakoff_batch',
    max_patients: Optional[int] = None,
) -> Dict[str, Any]:
    """Run CA-Lakoff annotation for each patient and save individual results.
    
    Args:
        patients: List of patient dicts (each containing baseline_age, etc.)
        intervention: Optional intervention dict (if None, uses no treatment)
        sim_years: Simulation duration in years
        dt: Time step for CA simulation (years)
        output_dir: Directory to save annotation JSON files
        max_patients: Limit number of patients to process (for quick testing)
    
    Returns:
        Summary dict with patient IDs, best archetypes, and schema counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if max_patients is not None:
        patients = patients[:max_patients]
    
    summary = {
        'patients_processed': len(patients),
        'intervention': intervention,
        'sim_years': sim_years,
        'dt': dt,
        'results': [],
    }
    
    for i, patient in enumerate(patients):
        patient_id = patient.get('_id', i)
        patient_label = patient.get('_label', f'patient_{patient_id}')
        print(f"Processing patient {i+1}/{len(patients)}: {patient_label}")
        
        try:
            # Run annotation
            annotation = annotate_from_simulation(
                patient=patient,
                intervention=intervention,
                sim_years=sim_years,
                dt=dt,
            )
            
            # Extract summary info
            best_arch, best_score = annotation['best_archetype']
            schema_count = len(annotation['image_schemas'])
            violation_count = len(annotation['metaphor_violations'])
            
            # Save individual annotation
            patient_file = output_dir / f'patient_{patient_id:04d}.json'
            save_annotation(annotation, patient_file)
            
            summary['results'].append({
                'patient_id': patient_id,
                'patient_label': patient_label,
                'patient': patient,
                'best_archetype': best_arch,
                'best_score': best_score,
                'schema_count': schema_count,
                'violation_count': violation_count,
                'annotation_file': str(patient_file),
            })
            
            print(f"  → best archetype: {best_arch} ({best_score:.3f}), "
                  f"schemas: {schema_count}, violations: {violation_count}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            summary['results'].append({
                'patient_id': patient_id,
                'patient_label': patient_label,
                'error': str(e),
            })
    
    # Save summary
    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\nSaved batch summary to {summary_file}")
    return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch CA-Lakoff annotation for patient populations'
    )
    parser.add_argument('--population', choices=['normal', 'edge', 'both'], default='normal',
                        help='Patient population to process (default: normal)')
    parser.add_argument('--max-patients', type=int, default=None,
                        help='Limit number of patients for quick testing')
    parser.add_argument('--sim-years', type=float, default=30.0,
                        help='Simulation duration in years (default: 30)')
    parser.add_argument('--dt', type=float, default=0.25,
                        help='CA time step in years (default: 0.25)')
    parser.add_argument('--intervention', type=str, default=None,
                        help='Intervention JSON string, e.g., \'{"rapamycin_dose":0.5}\'')
    parser.add_argument('--no-treatment', action='store_true',
                        help='Use no treatment (default intervention)')
    parser.add_argument('--output-dir', type=str, default='output/ca_lakoff_batch',
                        help='Output directory (default: output/ca_lakoff_batch)')
    
    args = parser.parse_args()
    
    # Parse intervention
    intervention = None
    if args.intervention:
        intervention = json.loads(args.intervention)
    elif args.no_treatment:
        intervention = {}  # empty dict = no treatment
    
    # Load patient populations
    patients = []
    if args.population in ('normal', 'both'):
        normal_path = Path('artifacts') / 'sample_patients_100.json'
        if normal_path.exists():
            print(f"Loading normal population from {normal_path}")
            patients.extend(load_patients(normal_path))
        else:
            print(f"Warning: {normal_path} not found")
    
    if args.population in ('edge', 'both'):
        edge_path = Path('artifacts') / 'sample_patients_edge.json'
        if edge_path.exists():
            print(f"Loading edge population from {edge_path}")
            patients.extend(load_patients(edge_path))
        else:
            print(f"Warning: {edge_path} not found")
    
    if not patients:
        print("No patients loaded. Exiting.")
        sys.exit(1)
    
    print(f"Total patients to process: {len(patients)}")
    if args.max_patients:
        print(f"(Limited to first {args.max_patients})")
    
    # Run batch annotation
    summary = annotate_patient_batch(
        patients=patients,
        intervention=intervention,
        sim_years=args.sim_years,
        dt=args.dt,
        output_dir=Path(args.output_dir),
        max_patients=args.max_patients,
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("BATCH ANNOTATION SUMMARY")
    print("="*60)
    print(f"Patients processed: {summary['patients_processed']}")
    
    successful = [r for r in summary['results'] if 'error' not in r]
    if successful:
        archetype_counts = {}
        for r in successful:
            arch = r['best_archetype']
            archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        
        print(f"Successful annotations: {len(successful)}")
        print("Archetype distribution:")
        for arch, count in sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {arch}: {count} patients ({count/len(successful)*100:.1f}%)")
        
        avg_schemas = np.mean([r['schema_count'] for r in successful])
        avg_violations = np.mean([r['violation_count'] for r in successful])
        print(f"Average schemas detected: {avg_schemas:.2f}")
        print(f"Average metaphor violations: {avg_violations:.2f}")
    
    errors = [r for r in summary['results'] if 'error' in r]
    if errors:
        print(f"Errors: {len(errors)} patients")
        for r in errors[:3]:  # show first 3 errors
            print(f"  Patient {r['patient_id']}: {r['error']}")
        if len(errors) > 3:
            print(f"  ... and {len(errors)-3} more")
    
    print(f"\nIndividual annotations saved to {args.output_dir}/")
    print(f"Summary saved to {args.output_dir}/batch_summary.json")


if __name__ == '__main__':
    main()