#!/usr/bin/env python3
"""
Fix transplant_focused classification: records with transplant_rate < 0.1
should be reclassified as aggressive.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def fix_transplant_classification(input_path: Path, output_path: Path = None):
    """Fix transplant_focused classification."""
    if output_path is None:
        output_path = input_path
    
    print(f"Loading dictionary from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    records = data.get("records", [])
    print(f"Processing {len(records)} records")
    
    fixed_count = 0
    for i, record in enumerate(records):
        if "enrichment" not in record or "lakoff_archetype" not in record["enrichment"]:
            continue
        
        lakoff = record["enrichment"]["lakoff_archetype"]
        arch = lakoff.get("lakoff_archetype", "")
        
        if arch == "transplant_focused":
            transplant_rate = record["intervention"].get("transplant_rate", 0)
            if transplant_rate < 0.1:
                # Reclassify as aggressive
                # Find aggressive similarity score
                sim_vector = lakoff.get("similarity_vector", {})
                aggressive_score = sim_vector.get("aggressive", 0.5)
                
                # Update classification
                lakoff["lakoff_archetype"] = "aggressive"
                lakoff["lakoff_score"] = aggressive_score
                lakoff["fixed"] = True
                lakoff["original_archetype"] = "transplant_focused"
                lakoff["reason"] = f"transplant_rate={transplant_rate} < 0.1"
                
                fixed_count += 1
    
    print(f"Fixed {fixed_count} records")
    
    # Save fixed dictionary
    print(f"Saving fixed dictionary to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Count archetypes after fix
    arch_counts = {}
    for record in records:
        if "enrichment" in record and "lakoff_archetype" in record["enrichment"]:
            arch = record["enrichment"]["lakoff_archetype"].get("lakoff_archetype", "unknown")
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
    
    print("\nArchetype distribution after fix:")
    for arch, count in sorted(arch_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {arch}: {count} records ({count/len(records)*100:.1f}%)")
    
    return data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix transplant_focused classification')
    parser.add_argument('--input', default='artifacts/protocol_pipeline/protocol_dictionary_lakoff_v3.json',
                       help='Input dictionary path')
    parser.add_argument('--output', default='artifacts/protocol_pipeline/protocol_dictionary_lakoff_v4.json',
                       help='Output dictionary path')
    
    args = parser.parse_args()
    
    input_path = Path(ROOT / args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(ROOT / args.output)
    
    fix_transplant_classification(input_path, output_path)

if __name__ == "__main__":
    main()