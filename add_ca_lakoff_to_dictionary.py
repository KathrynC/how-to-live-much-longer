#!/usr/bin/env python3
"""Add CA-Lakoff annotations to an existing protocol dictionary.

Reads a protocol_dictionary.json, runs CA-Lakoff annotation for each record,
adds a 'lakoff_ca' field to each record, and saves a new dictionary.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add current directory to path for imports
sys.path.insert(0, '.')

from ca_lakoff_annotator import annotate_from_simulation
from protocol_record import ProtocolRecord


def load_dictionary(json_path: Path) -> List[Dict[str, Any]]:
    """Load protocol dictionary JSON and return list of record dicts."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get('records', [])


def add_ca_lakoff_to_record(
    record_dict: Dict[str, Any],
    sim_years: float = 30.0,
    dt: float = 0.25,
) -> Dict[str, Any]:
    """Run CA-Lakoff annotation for a single protocol record.
    
    Args:
        record_dict: Protocol record dict (must contain intervention, patient)
        sim_years: Simulation duration
        dt: CA time step
    
    Returns:
        Updated record dict with 'lakoff_ca' field added.
    """
    intervention = record_dict.get('intervention', {})
    patient = record_dict.get('patient', {})
    
    if not intervention or not patient:
        print("Warning: record missing intervention or patient, skipping")
        return record_dict
    
    try:
        annotation = annotate_from_simulation(
            patient=patient,
            intervention=intervention,
            sim_years=sim_years,
            dt=dt,
        )
        
        # Extract key information for storage (could store full annotation)
        # For space efficiency, keep only summary
        lakoff_ca = {
            'best_archetype': annotation['best_archetype'][0],
            'best_score': annotation['best_archetype'][1],
            'archetype_similarities': annotation['archetype_similarities'],
            'image_schemas': list(annotation['image_schemas'].keys()),
            'schema_metrics_summary': {
                schema: {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
                for schema, metrics in annotation['image_schemas'].items()
            },
            'metaphor_violations': annotation['metaphor_violations'],
            'dual_vocabulary_steps': len(annotation['dual_vocabulary']),
            'ca_features_count': len(annotation['ca_features']),
        }
        
        record_dict['lakoff_ca'] = lakoff_ca
        print(f"  → added CA-Lakoff: {lakoff_ca['best_archetype']} ({lakoff_ca['best_score']:.3f})")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        record_dict['lakoff_ca'] = {'error': str(e)}
    
    return record_dict


def process_dictionary(
    input_path: Path,
    output_path: Path,
    max_records: Optional[int] = None,
    sim_years: float = 30.0,
    dt: float = 0.25,
) -> Dict[str, Any]:
    """Process dictionary and save enriched version."""
    print(f"Loading dictionary from {input_path}")
    records = load_dictionary(input_path)
    
    if max_records is not None:
        records = records[:max_records]
        print(f"Limited to first {max_records} records")
    
    print(f"Processing {len(records)} records...")
    
    processed = []
    for i, rec in enumerate(records):
        print(f"[{i+1}/{len(records)}] Processing record {i}")
        processed.append(add_ca_lakoff_to_record(rec, sim_years, dt))
    
    # Save updated dictionary
    output_dict = {
        'records': processed,
        'meta': {
            'source': str(input_path),
            'processed_with': 'add_ca_lakoff_to_dictionary.py',
            'sim_years': sim_years,
            'dt': dt,
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2, default=lambda x: float(x) if isinstance(x, (float, int)) else x)
    
    print(f"\nSaved enriched dictionary to {output_path}")
    
    # Summary
    successful = [r for r in processed if 'lakoff_ca' in r and 'error' not in r.get('lakoff_ca', {})]
    archetype_counts = {}
    for r in successful:
        arch = r['lakoff_ca'].get('best_archetype')
        if arch:
            archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
    
    print(f"\nSuccessfully annotated: {len(successful)}/{len(processed)}")
    if archetype_counts:
        print("Archetype distribution:")
        for arch, count in sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {arch}: {count}")
    
    return output_dict


def main():
    parser = argparse.ArgumentParser(
        description='Add CA-Lakoff annotations to protocol dictionary'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input protocol_dictionary.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: input path with _ca_lakoff suffix)')
    parser.add_argument('--max-records', type=int, default=None,
                        help='Limit number of records to process (for testing)')
    parser.add_argument('--sim-years', type=float, default=30.0,
                        help='Simulation duration in years')
    parser.add_argument('--dt', type=float, default=0.25,
                        help='CA time step in years')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} not found")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_ca_lakoff.json"
    
    process_dictionary(
        input_path=input_path,
        output_path=output_path,
        max_records=args.max_records,
        sim_years=args.sim_years,
        dt=args.dt,
    )


if __name__ == '__main__':
    main()