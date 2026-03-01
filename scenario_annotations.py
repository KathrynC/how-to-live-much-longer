"""Scenario annotations with SystemViz and Lakoff semantic vocabulary.

Enriches Scenario objects from scenario_definitions.py with dual annotations:
- SystemViz structural vocabulary (driver, signal, state, boundary, relation, domain)
- Lakoff cognitive semantics (grounded vs linking features, archetype mapping)

Provides functions to annotate scenarios and generate integrated reports.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

from scenario_definitions import Scenario, InterventionProfile, get_example_scenarios

# Try to import Lakoff integration; if not available, provide stubs
try:
    from patterns.lakoff_integration import (
        LakoffSystemVizBridge, create_refined_archetypes, 
        ArchetypeLibrary, get_feature_layer
    )
    LAKOFF_AVAILABLE = True
except ImportError:
    LAKOFF_AVAILABLE = False
    # Create minimal stubs for type checking
    class LakoffSystemVizBridge:
        def annotate_with_dual_vocabulary(self, pattern_name):
            return {"lakoff": {"pattern_name": pattern_name, "systemviz_terms": []}}
        def get_systemviz_terms_for_archetype(self, archetype_name):
            return []
    class ArchetypeLibrary:
        def __init__(self):
            self.archetypes = []
        def get(self, name):
            return None
    def create_refined_archetypes():
        return ArchetypeLibrary()
    def get_feature_layer(feature):
        return "linking"


@dataclass
class ScenarioAnnotation:
    """Dual annotation for a scenario combining SystemViz and Lakoff semantics."""
    scenario_name: str
    scenario_description: str
    
    # Lakoff cognitive semantics
    lakoff_archetype: str = ""
    lakoff_grounding_ratio: float = 0.0  # proportion of grounded features expected
    lakoff_icm_violations: List[str] = field(default_factory=list)
    
    # SystemViz structural vocabulary
    systemviz_terms: List[str] = field(default_factory=list)
    systemviz_categories: Dict[str, List[str]] = field(default_factory=dict)
    
    # Pattern language alignment
    pattern_stage: str = "global"  # global, ingest, analytics, robustness, classify, review, report
    pattern_id: str = ""
    
    # Intervention profile summary
    intervention_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Validation status (requires simulation)
    validation_note: str = ""  # e.g., "heuristic", "validated", "mismatch"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "lakoff_archetype": self.lakoff_archetype,
            "lakoff_grounding_ratio": self.lakoff_grounding_ratio,
            "lakoff_icm_violations": self.lakoff_icm_violations,
            "systemviz_terms": self.systemviz_terms,
            "systemviz_categories": self.systemviz_categories,
            "pattern_stage": self.pattern_stage,
            "pattern_id": self.pattern_id,
            "intervention_summary": self.intervention_summary,
            "validation_note": self.validation_note,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioAnnotation':
        """Create from dictionary."""
        return cls(**data)


class ScenarioAnnotator:
    """Annotate scenarios with SystemViz and Lakoff semantics."""
    
    def __init__(self):
        if LAKOFF_AVAILABLE:
            self.bridge = LakoffSystemVizBridge()
            self.archetype_library = create_refined_archetypes()
        else:
            self.bridge = LakoffSystemVizBridge()
            self.archetype_library = ArchetypeLibrary()
        
        # Heuristic mapping from scenario names to archetypes
        # Updated based on validation results (validate_archetype_mappings.py):
        # All scenarios produce conservative outcomes for the base 63yo APOE4 female patient.
        # Original intent mapping (commented):
        # A: conservative (matches) ✓
        # B: metabolic_optimizer → conservative ✗
        # C: aggressive → conservative ✗
        # D: transplant_focused → conservative ✗
        self.heuristic_mapping = {
            "A: Sleep + Alcohol Cessation": "conservative",
            "B: A + OTC Supplements + Keto": "conservative",
            "C: B + Prescription": "conservative",
            "D: C + Experimental": "conservative",
        }
        
        # Pattern stage mapping based on archetype
        self.pattern_stage_mapping = {
            "conservative": "analytics",        # monitoring, stability
            "aggressive": "robustness",         # tipping points, criticality
            "transplant_focused": "global",     # control points, primary rejuvenation
            "metabolic_optimizer": "analytics", # feedback loops, coupling
        }
    
    def annotate(self, scenario: Scenario) -> ScenarioAnnotation:
        """Create dual annotation for a scenario."""
        # Determine archetype heuristically (in future, simulate and match)
        archetype_name = self.heuristic_mapping.get(scenario.name, "")
        
        # Get SystemViz terms for this archetype
        systemviz_terms = []
        systemviz_categories = {}
        if LAKOFF_AVAILABLE and archetype_name:
            systemviz_terms = self.bridge.get_systemviz_terms_for_archetype(archetype_name)
            # Group by category
            for term in systemviz_terms:
                if '.' in term:
                    category = term.split('.')[0]
                    systemviz_categories.setdefault(category, []).append(term)
        
        # Estimate grounding ratio based on intervention intensity
        grounding_ratio = self._estimate_grounding_ratio(scenario.interventions)
        
        # Generate intervention summary
        intervention_summary = self._summarize_interventions(scenario.interventions)
        
        return ScenarioAnnotation(
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            lakoff_archetype=archetype_name,
            lakoff_grounding_ratio=grounding_ratio,
            systemviz_terms=systemviz_terms,
            systemviz_categories=systemviz_categories,
            pattern_stage=self.pattern_stage_mapping.get(archetype_name, "global"),
            pattern_id=self._infer_pattern_id(archetype_name),
            intervention_summary=intervention_summary,
            validation_note="heuristic mapping (not simulation-validated in this run)",
        )
    
    def _estimate_grounding_ratio(self, interventions: InterventionProfile) -> float:
        """Estimate proportion of grounded vs linking features expected.
        
        Higher intervention intensity tends to produce more observable (grounded)
        effects. This is a heuristic estimate.
        """
        # Count intervention components
        components = [
            interventions.rapamycin_dose,
            interventions.nad_supplement,
            interventions.senolytic_dose,
            interventions.yamanaka_intensity,
            interventions.transplant_rate,
            interventions.exercise_level,
            interventions.sleep_intervention,
            interventions.fasting_regimen,
            interventions.probiotic_intensity,
            interventions.therapy_intensity,
        ]
        # Supplement doses
        supplements = [
            interventions.nr_dose, interventions.dha_dose, interventions.coq10_dose,
            interventions.resveratrol_dose, interventions.pqq_dose, interventions.ala_dose,
            interventions.vitamin_d_dose, interventions.b_complex_dose,
            interventions.magnesium_dose, interventions.zinc_dose, interventions.selenium_dose,
        ]
        
        active_components = sum(1 for c in components if c > 0.3)
        active_supplements = sum(1 for s in supplements if s > 0.3)
        
        total_active = active_components + active_supplements
        
        # More active interventions → more observable effects → higher grounding ratio
        # Cap at 0.9 (some linking features always present)
        return min(0.9, 0.3 + 0.1 * total_active)
    
    def _summarize_interventions(self, interventions: InterventionProfile) -> Dict[str, Any]:
        """Generate structured summary of intervention profile."""
        summary = {
            "core_interventions": {},
            "supplements": {},
            "lifestyle": {},
            "experimental": {},
        }
        
        # Core interventions
        if interventions.rapamycin_dose > 0.1:
            summary["core_interventions"]["rapamycin"] = interventions.rapamycin_dose
        if interventions.nad_supplement > 0.1:
            summary["core_interventions"]["nad_supplement"] = interventions.nad_supplement
        if interventions.senolytic_dose > 0.1:
            summary["core_interventions"]["senolytic"] = interventions.senolytic_dose
        if interventions.yamanaka_intensity > 0.1:
            summary["experimental"]["yamanaka"] = interventions.yamanaka_intensity
        if interventions.transplant_rate > 0.1:
            summary["experimental"]["transplant"] = interventions.transplant_rate
        if interventions.exercise_level > 0.1:
            summary["lifestyle"]["exercise"] = interventions.exercise_level
        
        # Lifestyle
        if interventions.sleep_intervention > 0.1:
            summary["lifestyle"]["sleep"] = interventions.sleep_intervention
        if interventions.alcohol_intake == 0.0:
            summary["lifestyle"]["alcohol_cessation"] = True
        if interventions.diet_type != 'standard':
            summary["lifestyle"]["diet"] = interventions.diet_type
        if interventions.fasting_regimen > 0.1:
            summary["lifestyle"]["fasting"] = interventions.fasting_regimen
        if interventions.probiotic_intensity > 0.1:
            summary["lifestyle"]["probiotics"] = interventions.probiotic_intensity
        if interventions.therapy_intensity > 0.1:
            summary["lifestyle"]["therapy"] = interventions.therapy_intensity
        
        # Supplements
        supplement_keys = [
            ("nr_dose", "NR"),
            ("dha_dose", "DHA"),
            ("coq10_dose", "CoQ10"),
            ("resveratrol_dose", "resveratrol"),
            ("pqq_dose", "PQQ"),
            ("ala_dose", "ALA"),
            ("vitamin_d_dose", "vitamin D"),
            ("b_complex_dose", "B complex"),
            ("magnesium_dose", "magnesium"),
            ("zinc_dose", "zinc"),
            ("selenium_dose", "selenium"),
        ]
        for key, name in supplement_keys:
            dose = getattr(interventions, key)
            if dose > 0.1:
                summary["supplements"][name] = dose
        
        return summary
    
    def _infer_pattern_id(self, archetype_name: str) -> str:
        """Infer which pattern language pattern aligns with this archetype."""
        pattern_mapping = {
            "conservative": "analytics_profile",
            "aggressive": "robustness_assessment",
            "transplant_focused": "protocol_program",
            "metabolic_optimizer": "feedback_loop",
        }
        return pattern_mapping.get(archetype_name, "protocol_program")


def annotate_all_scenarios() -> Dict[str, ScenarioAnnotation]:
    """Annotate all example scenarios."""
    annotator = ScenarioAnnotator()
    scenarios = get_example_scenarios()
    
    annotations = {}
    for scenario in scenarios:
        annotation = annotator.annotate(scenario)
        annotations[scenario.name] = annotation
    
    return annotations


def generate_annotation_report() -> Dict[str, Any]:
    """Generate comprehensive annotation report."""
    annotations = annotate_all_scenarios()
    
    report = {
        "metadata": {
            "annotation_system": "SystemViz + Lakoff dual vocabulary",
            "scenario_count": len(annotations),
            "lakoff_available": LAKOFF_AVAILABLE,
        },
        "annotations": {name: ann.to_dict() for name, ann in annotations.items()},
        "summary": {
            "archetype_distribution": {},
            "pattern_stage_distribution": {},
            "systemviz_category_counts": {},
        }
    }
    
    # Calculate summary statistics
    for ann in annotations.values():
        archetype = ann.lakoff_archetype
        if archetype:
            report["summary"]["archetype_distribution"][archetype] = \
                report["summary"]["archetype_distribution"].get(archetype, 0) + 1
        
        stage = ann.pattern_stage
        report["summary"]["pattern_stage_distribution"][stage] = \
            report["summary"]["pattern_stage_distribution"].get(stage, 0) + 1
        
        for category, terms in ann.systemviz_categories.items():
            report["summary"]["systemviz_category_counts"][category] = \
                report["summary"]["systemviz_category_counts"].get(category, 0) + len(terms)
    
    return report


def save_annotations(output_path: str = "output/scenario_annotations.json"):
    """Save scenario annotations to JSON file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = generate_annotation_report()
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Scenario annotations saved to {output_path}")
    return output_path


def main():
    """Command-line interface for scenario annotation."""
    print("=" * 70)
    print("Scenario Annotations with SystemViz + Lakoff Semantics")
    print("=" * 70)
    
    if not LAKOFF_AVAILABLE:
        print("Warning: Lakoff integration not available. Using heuristic mappings only.")
    
    annotations = annotate_all_scenarios()
    
    for name, ann in annotations.items():
        print(f"\n{name}")
        print(f"  Archetype: {ann.lakoff_archetype or 'unknown'}")
        print(f"  Pattern stage: {ann.pattern_stage}")
        print(f"  Grounding ratio estimate: {ann.lakoff_grounding_ratio:.2f}")
        print(f"  SystemViz terms: {len(ann.systemviz_terms)}")
        if ann.systemviz_terms:
            print(f"    {', '.join(ann.systemviz_terms[:5])}...")
        print(f"  Interventions:")
        for category, items in ann.intervention_summary.items():
            if items:
                print(f"    {category}: {len(items)} items")
    
    # Generate and save report
    output_path = save_annotations()
    
    print(f"\nDetailed report saved to {output_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
