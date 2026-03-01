"""Archetype-scenario mappings for mitochondrial intervention scenarios.

Maps predefined scenarios (A-D) to Lakoff archetypes based on expected outcomes
and intervention profiles. Used to annotate scenarios with semantic meaning
and SystemViz vocabulary.
"""

from typing import Dict, List, Any
from .lakoff_integration import create_refined_archetypes, ArchetypeLibrary
from ..scenario_definitions import get_example_scenarios, Scenario, InterventionProfile


def map_scenarios_to_archetypes(
    scenarios: List[Scenario], 
    archetype_library: ArchetypeLibrary
) -> Dict[str, Dict[str, Any]]:
    """Map each scenario to its best-matching archetype(s).
    
    Args:
        scenarios: List of Scenario objects.
        archetype_library: ArchetypeLibrary with refined archetypes.
        
    Returns:
        Dict mapping scenario name to dict with archetype matches and scores.
    """
    # TODO: Actually simulate scenarios and extract features for proper mapping.
    # For now, use heuristic mapping based on intervention profiles.
    heuristic_mapping = {
        "A: Sleep + Alcohol Cessation": "conservative",
        "B: A + OTC Supplements + Keto": "metabolic_optimizer", 
        "C: B + Prescription": "aggressive",
        "D: C + Experimental": "transplant_focused",
    }
    
    results = {}
    for scenario in scenarios:
        scenario_name = scenario.name
        heuristic_arch = heuristic_mapping.get(scenario_name)
        
        # Get archetype object
        archetype = archetype_library.get(heuristic_arch) if heuristic_arch else None
        
        results[scenario_name] = {
            "scenario": scenario_name,
            "heuristic_archetype": heuristic_arch,
            "archetype_description": archetype.description if archetype else None,
            "notes": _get_scenario_notes(scenario),
        }
    
    return results


def _get_scenario_notes(scenario: Scenario) -> str:
    """Generate notes about scenario characteristics."""
    interventions = scenario.interventions
    
    notes = []
    if interventions.sleep_intervention > 0.5:
        notes.append("sleep intervention")
    if interventions.alcohol_intake == 0.0:
        notes.append("alcohol cessation")
    if interventions.diet_type == 'keto':
        notes.append("keto diet")
    if interventions.rapamycin_dose > 0.5:
        notes.append("rapamycin (mTOR inhibition)")
    if interventions.senolytic_dose > 0.5:
        notes.append("senolytics")
    if interventions.transplant_rate > 0.5:
        notes.append("mitochondrial transplant")
    if interventions.yamanaka_intensity > 0.3:
        notes.append("Yamanaka reprogramming")
    
    # Count supplements
    supplement_doses = [
        interventions.nr_dose, interventions.dha_dose, interventions.coq10_dose,
        interventions.resveratrol_dose, interventions.pqq_dose, interventions.ala_dose,
        interventions.vitamin_d_dose, interventions.b_complex_dose,
        interventions.magnesium_dose, interventions.zinc_dose, interventions.selenium_dose,
    ]
    active_supplements = sum(1 for d in supplement_doses if d > 0.5)
    if active_supplements > 5:
        notes.append(f"{active_supplements} OTC supplements")
    
    return ", ".join(notes) if notes else "minimal intervention"


def generate_scenario_annotations() -> Dict[str, Dict[str, Any]]:
    """Generate dual annotations (SystemViz + Lakoff) for all scenarios."""
    from .lakoff_integration import LakoffSystemVizBridge
    
    scenarios = get_example_scenarios()
    library = create_refined_archetypes()
    bridge = LakoffSystemVizBridge()
    
    mappings = map_scenarios_to_archetypes(scenarios, library)
    
    # Add SystemViz annotations based on mapped archetype
    for scenario_name, mapping in mappings.items():
        arch_name = mapping["heuristic_archetype"]
        if arch_name:
            # Get SystemViz terms relevant to this archetype
            arch_terms = bridge.get_systemviz_terms_for_archetype(arch_name)
            mapping["systemviz_terms"] = arch_terms
            # Get pattern stage alignment
            mapping["pattern_stage"] = _infer_pattern_stage(arch_name)
        else:
            mapping["systemviz_terms"] = []
            mapping["pattern_stage"] = "unknown"
    
    return mappings


def _infer_pattern_stage(archetype_name: str) -> str:
    """Infer which pattern language stage aligns with this archetype."""
    # Based on SystemViz term mapping and pattern language stages
    stage_mapping = {
        "conservative": "analytics",  # monitoring, stability
        "aggressive": "robustness",   # tipping points, criticality
        "transplant_focused": "global",  # control points, primary rejuvenation
        "metabolic_optimizer": "analytics",  # feedback loops, coupling
    }
    return stage_mapping.get(archetype_name, "global")


def main():
    """Print scenario-archetype mappings."""
    print("=" * 70)
    print("Scenario-Archetype Mappings (Heuristic)")
    print("=" * 70)
    
    mappings = generate_scenario_annotations()
    
    for scenario_name, data in mappings.items():
        print(f"\n{scenario_name}")
        print(f"  Archetype: {data['heuristic_archetype']}")
        if data['archetype_description']:
            print(f"  Description: {data['archetype_description']}")
        print(f"  Pattern stage: {data.get('pattern_stage', 'unknown')}")
        print(f"  SystemViz terms: {', '.join(data.get('systemviz_terms', []))[:80]}...")
        print(f"  Notes: {data['notes']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()