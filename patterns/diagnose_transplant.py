#!/usr/bin/env python3
"""
Diagnose why young patients are classified as transplant_focused.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.lakoff_integration import (
    ArchetypeLibrary, MetaphorAuditor, extract_features_from_analytics,
    get_feature_layer
)

# Load adjusted archetypes
library_path = ROOT / "patterns" / "lakoff_archetypes_adjusted.json"
print(f"Loading archetypes from {library_path}")
library = ArchetypeLibrary.load(library_path)
print(f"Loaded {len(library.archetypes)} archetypes")

# Get transplant_focused archetype
tf_arch = None
for arch in library.archetypes:
    if arch.name == "transplant_focused":
        tf_arch = arch
        break

if tf_arch:
    print(f"\nTransplant_focused archetype:")
    print(f"  Description: {tf_arch.description}")
    print(f"  Grounding criteria ({len(tf_arch.grounding_criteria)}):")
    for crit in tf_arch.grounding_criteria:
        print(f"    - {crit.feature} {crit.predicate} {crit.value}")
    print(f"  ICM violation conditions ({len(tf_arch.icm.violation_conditions)}):")
    for vc in tf_arch.icm.violation_conditions:
        print(f"    - {vc.feature} {vc.predicate} {vc.value}")
else:
    print("Transplant_focused archetype not found!")

# Load a young patient record to examine
with open(ROOT / "output" / "archetype_transitions" / "archetype_transitions_raw.json") as f:
    data = json.load(f)

young_patient = None
for r in data:
    if r.get('patient_label') == 'baseline_age_min' and 'classification' in r:
        young_patient = r
        break

if young_patient:
    print(f"\nYoung patient analysis:")
    print(f"  Age: {young_patient['patient_params']['baseline_age']}")
    print(f"  Baseline het: {young_patient['patient_params']['baseline_heteroplasmy']}")
    print(f"  Final ATP: {young_patient['outcomes']['final_atp']:.3f}")
    print(f"  Final het: {young_patient['outcomes']['final_het']:.3f}")
    print(f"  Delta het: {young_patient['outcomes']['final_het'] - young_patient['patient_params']['baseline_heteroplasmy']:.3f}")
    
    # Check transplant_focused criteria manually
    if tf_arch:
        print(f"\nManual criteria check for transplant_focused:")
        analytics = young_patient.get('analytics', {})
        if not analytics:
            print("  No analytics in record")
        else:
            # We need to compute features from analytics
            features = extract_features_from_analytics(analytics)
            print(f"  Extracted {len(features)} features")
            
            # Create auditor and audit
            auditor = MetaphorAuditor(library)
            audit_results = auditor.audit(features)
            
            if 'transplant_focused' in audit_results:
                result = audit_results['transplant_focused']
                print(f"  Auditor result:")
                print(f"    Similarity: {result['similarity']}")
                print(f"    Criteria satisfied: {result.get('criteria_satisfied', [])}")
                print(f"    Criteria failed: {result.get('criteria_failed', [])}")
                if 'details' in result:
                    print(f"    Details: {result['details']}")