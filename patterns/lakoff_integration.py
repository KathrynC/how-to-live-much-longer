"""lakoff_integration.py — Lakoff cognitive semantics for mitochondrial intervention archetypes.

Implements Lakoff Maxim 7 (ground first, link second) for mitochondrial protocol classification.
Provides feature layer classification, grounding criteria, ICMs (Idealized Cognitive Models),
and metaphor violation detection.

Adapted from motion-analytics-toolkit's Lakoff implementation, tailored for mitochondrial
aging simulation metrics and intervention protocol semantics.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Feature layer classification (Lakoff Maxim 7)
# ------------------------------------------------------------------

# Maps canonical mitochondrial metric names to their Lakoff layer.
# Grounded = directly measurable biological quantities (ATP, heteroplasmy, ROS).
# Linking = cross-domain abstractions requiring interpretation (benefit-cost ratio, correlations).
FEATURE_LAYERS: Dict[str, str] = {
    # === Grounded (direct biological measurements) ===
    # Energy pillar
    "atp_initial": "grounded",
    "atp_final": "grounded",
    "atp_min": "grounded",
    "atp_max": "grounded",
    "atp_mean": "grounded",
    "atp_cv": "grounded",  # coefficient of variation (direct variability measure)
    
    # Damage pillar
    "het_initial": "grounded",
    "het_final": "grounded",
    "het_max": "grounded",
    "delta_het": "grounded",  # change is directly observable
    "deletion_het_initial": "grounded",
    "deletion_het_final": "grounded",
    "deletion_het_max": "grounded",
    "cliff_distance_initial": "grounded",  # distance to threshold
    "cliff_distance_final": "grounded",
    
    # Dynamics pillar
    "ros_amplitude": "grounded",  # ROS oscillation magnitude
    "ros_dominant_freq": "grounded",  # ROS oscillation frequency
    "membrane_potential_cv": "grounded",  # membrane potential variability
    "nad_slope": "grounded",  # NAD trend (slope is derived but captures direct measurement trend)
    "senescent_final": "grounded",  # final senescence fraction
    "senescent_slope": "grounded",  # senescence trend
    
    # Time metrics (direct temporal measurements)
    "time_to_crisis_years": "grounded",
    "time_to_cliff_years": "grounded",
    "frac_above_cliff": "grounded",  # proportion of time above cliff
    
    # === Linking (cross-domain abstractions) ===
    # Derived ratios and correlations
    "reserve_ratio": "linking",  # ATP headroom ratio (derived concept)
    "atp_slope": "linking",  # overall ATP trend (interpretive)
    "terminal_slope": "linking",  # terminal ATP trend (interpretive)
    "het_slope": "linking",  # heteroplasmy trend
    "het_acceleration": "linking",  # second derivative (highly interpretive)
    "membrane_potential_slope": "linking",  # membrane potential trend
    
    # Correlation metrics (statistical relationships)
    "ros_het_correlation": "linking",
    "ros_atp_correlation": "linking",
    
    # Intervention pillar (comparative/economic concepts)
    "atp_benefit_terminal": "linking",  # relative to baseline
    "atp_benefit_mean": "linking",
    "het_benefit_terminal": "linking",
    "benefit_cost_ratio": "linking",  # economic metaphor
    "energy_cost_per_year": "linking",
    "total_dose": "linking",  # aggregate intervention intensity
    "crisis_delay_years": "linking",  # temporal comparison
}

# Aliases mapping dotted pillar.metric names to canonical keys
FEATURE_ALIASES: Dict[str, str] = {
    # Pillar-prefixed names
    "energy.atp_initial": "atp_initial",
    "energy.atp_final": "atp_final",
    "energy.atp_min": "atp_min",
    "energy.atp_max": "atp_max",
    "energy.atp_mean": "atp_mean",
    "energy.atp_cv": "atp_cv",
    "energy.reserve_ratio": "reserve_ratio",
    "energy.atp_slope": "atp_slope",
    "energy.terminal_slope": "terminal_slope",
    "energy.time_to_crisis_years": "time_to_crisis_years",
    
    "damage.het_initial": "het_initial",
    "damage.het_final": "het_final",
    "damage.het_max": "het_max",
    "damage.delta_het": "delta_het",
    "damage.deletion_het_initial": "deletion_het_initial",
    "damage.deletion_het_final": "deletion_het_final",
    "damage.deletion_het_max": "deletion_het_max",
    "damage.cliff_distance_initial": "cliff_distance_initial",
    "damage.cliff_distance_final": "cliff_distance_final",
    "damage.het_slope": "het_slope",
    "damage.het_acceleration": "het_acceleration",
    "damage.time_to_cliff_years": "time_to_cliff_years",
    "damage.frac_above_cliff": "frac_above_cliff",
    
    "dynamics.ros_dominant_freq": "ros_dominant_freq",
    "dynamics.ros_amplitude": "ros_amplitude",
    "dynamics.membrane_potential_cv": "membrane_potential_cv",
    "dynamics.membrane_potential_slope": "membrane_potential_slope",
    "dynamics.nad_slope": "nad_slope",
    "dynamics.ros_het_correlation": "ros_het_correlation",
    "dynamics.ros_atp_correlation": "ros_atp_correlation",
    "dynamics.senescent_final": "senescent_final",
    "dynamics.senescent_slope": "senescent_slope",
    
    "intervention.atp_benefit_terminal": "atp_benefit_terminal",
    "intervention.atp_benefit_mean": "atp_benefit_mean",
    "intervention.het_benefit_terminal": "het_benefit_terminal",
    "intervention.benefit_cost_ratio": "benefit_cost_ratio",
    "intervention.energy_cost_per_year": "energy_cost_per_year",
    "intervention.total_dose": "total_dose",
    "intervention.crisis_delay_years": "crisis_delay_years",
}


def get_feature_layer(feature_name: str) -> str:
    """Return 'grounded' or 'linking' for a feature name.
    
    Aliases are resolved via FEATURE_ALIASES.
    Unknown features default to 'linking' (conservative: must be grounded to be grounded).
    """
    # Strip pillar prefix if present
    if '.' in feature_name:
        canonical = FEATURE_ALIASES.get(feature_name, feature_name)
    else:
        canonical = feature_name
    return FEATURE_LAYERS.get(canonical, "linking")


# ------------------------------------------------------------------
# Lakoff grounding structures
# ------------------------------------------------------------------

@dataclass
class GroundingCriterion:
    """A testable predicate that anchors a label to observable features.
    
    Following Lakoff Maxim 7 (ground first, link second): every metaphorical
    label must be grounded in sensorimotor observables before cross-domain
    linking is permitted.
    """
    feature: str           # metric key, e.g., 'atp_final' or 'damage.het_final'
    predicate: str         # 'gt', 'lt', 'between', 'near'
    value: float           # threshold or target
    tolerance: float = 0.0  # for 'near' predicate
    rationale: str = ""    # why this criterion grounds the label
    layer: str = ""        # 'grounded', 'linking', or '' (unconstrained)

    def check(self, features: Dict[str, float]) -> bool:
        """Return True if the criterion is satisfied by the given features."""
        actual = features.get(self.feature)
        if actual is None:
            return False
        if self.predicate == 'gt':
            return actual > self.value
        elif self.predicate == 'lt':
            return actual < self.value
        elif self.predicate == 'near':
            return abs(actual - self.value) <= self.tolerance
        elif self.predicate == 'between':
            # value encodes low bound, tolerance encodes high bound
            return self.value <= actual <= self.tolerance
        return False

    def layer_warning(self) -> str:
        """Return warning if this criterion claims grounded layer but references a linking feature."""
        if self.layer == 'grounded' and get_feature_layer(self.feature) == 'linking':
            return (f"Grounding criterion on '{self.feature}' claims grounded layer "
                    f"but feature is classified as linking")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'feature': self.feature,
            'predicate': self.predicate,
            'value': self.value,
            'tolerance': self.tolerance,
            'rationale': self.rationale,
        }
        if self.layer:
            d['layer'] = self.layer
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundingCriterion':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ICM:
    """Idealized Cognitive Model — background assumptions a label presupposes.
    
    When violation_conditions are met, the label's ICM breaks and the label
    should not be applied (or should be flagged as metaphor violation).
    """
    name: str
    background: List[str]                          # prose assumptions
    violation_conditions: List[GroundingCriterion]  # when the label breaks

    def check_violations(self, features: Dict[str, float]) -> List[str]:
        """Return list of violated condition descriptions (empty = ICM intact)."""
        violations = []
        for cond in self.violation_conditions:
            if cond.check(features):
                violations.append(
                    f"{cond.feature} {cond.predicate} {cond.value}: {cond.rationale}"
                )
        return violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'background': self.background,
            'violation_conditions': [c.to_dict() for c in self.violation_conditions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ICM':
        return cls(
            name=data['name'],
            background=data['background'],
            violation_conditions=[
                GroundingCriterion.from_dict(c) for c in data['violation_conditions']
            ],
        )


# ------------------------------------------------------------------
# Archetype base class
# ------------------------------------------------------------------

class Archetype:
    """
    Abstract base class representing a conceptual archetype for intervention protocols.
    
    Subclasses represent distinct intervention strategies (conservative, aggressive,
    transplant-focused, metabolic optimizer, etc.) with grounding criteria and ICMs.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        grounding_criteria: Optional[List[GroundingCriterion]] = None,
        icm: Optional[ICM] = None,
    ):
        self.name = name
        self.description = description
        self.grounding_criteria = grounding_criteria or []
        self.icm = icm

    def similarity_to(self, features: Dict[str, float]) -> float:
        """
        Return a similarity score (0 to 1) between the given features and this archetype.
        
        Default implementation: proportion of grounding criteria satisfied.
        Override for more sophisticated similarity measures.
        """
        if not self.grounding_criteria:
            return 0.5  # Neutral if no criteria defined
        
        satisfied = sum(1 for gc in self.grounding_criteria if gc.check(features))
        return satisfied / len(self.grounding_criteria)

    def check_grounding(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Test all grounding criteria against extracted features.
        
        Returns:
            (all_pass, list_of_failure_descriptions)
        """
        failures = []
        for gc in self.grounding_criteria:
            if not gc.check(features):
                failures.append(
                    f"{gc.feature} failed {gc.predicate} {gc.value}: {gc.rationale}"
                )
        return (len(failures) == 0, failures)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        d = {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__,
        }
        if self.grounding_criteria:
            d['grounding_criteria'] = [gc.to_dict() for gc in self.grounding_criteria]
        if self.icm:
            d['icm'] = self.icm.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Archetype':
        """Deserialize from dictionary (to be overridden by subclasses)."""
        raise NotImplementedError


class ArchetypeLibrary:
    """
    Collection of mitochondrial intervention archetypes with methods to load/save and query.
    """
    
    def __init__(self, archetypes: Optional[List[Archetype]] = None):
        self.archetypes = archetypes or []
        self._name_index = {a.name: a for a in self.archetypes}
    
    def add(self, archetype: Archetype):
        self.archetypes.append(archetype)
        self._name_index[archetype.name] = archetype
    
    def get(self, name: str) -> Optional[Archetype]:
        return self._name_index.get(name)
    
    def best_match(self, features: Dict[str, float]) -> Tuple[Archetype, float]:
        """
        Find the archetype with highest similarity to the given features.
        Returns (archetype, score).
        """
        best_score = -1.0
        best_arch = None
        for arch in self.archetypes:
            score = arch.similarity_to(features)
            if score > best_score:
                best_score = score
                best_arch = arch
        return best_arch, best_score
    
    def similarity_vector(self, features: Dict[str, float]) -> Dict[str, float]:
        """Return a dict mapping archetype names to similarity scores."""
        return {a.name: a.similarity_to(features) for a in self.archetypes}
    
    def save(self, path: Path):
        """Save library to JSON file."""
        data = [a.to_dict() for a in self.archetypes]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ArchetypeLibrary':
        """Load library from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        lib = cls()
        for item in data:
            arch = Archetype(
                name=item['name'],
                description=item.get('description', ''),
                grounding_criteria=[GroundingCriterion.from_dict(gc) 
                                   for gc in item.get('grounding_criteria', [])],
                icm=ICM.from_dict(item['icm']) if 'icm' in item else None
            )
            lib.add(arch)
        return lib


# ------------------------------------------------------------------
# Feature extraction helpers
# ------------------------------------------------------------------

def extract_features_from_analytics(analytics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Flatten 4-pillar analytics dict into a flat feature dictionary.
    
    Args:
        analytics_dict: Output from analytics.compute_all()
        
    Returns:
        Flat dict with pillar-prefixed keys (e.g., "energy.atp_final").
    """
    features = {}
    for pillar, metrics in analytics_dict.items():
        for key, value in metrics.items():
            features[f"{pillar}.{key}"] = float(value)
    return features


# ------------------------------------------------------------------
# Metaphor violation detection
# ------------------------------------------------------------------

class MetaphorAuditor:
    """Audit archetype labels against observed mitochondrial features."""
    
    def __init__(self, library: ArchetypeLibrary):
        self.library = library
    
    def audit(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """Audit all archetypes in the library against the given features.
        
        Returns:
            {archetype_name: {
                'similarity': float,
                'grounding_pass': bool,
                'failed_criteria': list[str],
                'icm_violated': bool,
                'icm_violations': list[str],
                'layer_warnings': list[str],
                'verdict': str,  # 'grounded', 'partial', 'violated'
            }}
        """
        results = {}
        for arch in self.library.archetypes:
            results[arch.name] = self.audit_single(arch, features)
        return results
    
    def audit_single(
        self,
        archetype: Archetype,
        features: Dict[str, float],
    ) -> Dict:
        """Audit one archetype against extracted features."""
        # Similarity score
        similarity = archetype.similarity_to(features)
        
        # Grounding check
        grounding_pass, failed_criteria = archetype.check_grounding(features)
        
        # ICM check
        icm_violated = False
        icm_violations: List[str] = []
        if archetype.icm is not None:
            icm_violations = archetype.icm.check_violations(features)
            icm_violated = len(icm_violations) > 0
        
        # Layer enforcement check
        layer_warnings: List[str] = []
        for gc in archetype.grounding_criteria:
            feature_layer = get_feature_layer(gc.feature)
            if feature_layer == 'linking':
                layer_warnings.append(
                    f"'{gc.feature}' is a linking feature used in grounding criterion: {gc.rationale}"
                )
        
        # Verdict
        if grounding_pass and not icm_violated:
            verdict = 'grounded'
        elif icm_violated:
            verdict = 'violated'
        else:
            verdict = 'partial'
        
        return {
            'similarity': float(similarity),
            'grounding_pass': grounding_pass,
            'failed_criteria': failed_criteria,
            'icm_violated': icm_violated,
            'icm_violations': icm_violations,
            'layer_warnings': layer_warnings,
            'verdict': verdict,
        }


# ------------------------------------------------------------------
# Integration with SystemViz mapping
# ------------------------------------------------------------------

class LakoffSystemVizBridge:
    """Bridge between Lakoff semantics and SystemViz structural vocabulary.
    
    Provides dual annotation: SystemViz structural terms + Lakoff cognitive semantics.
    Maps SystemViz categories to Lakoff feature layers and archetype relevance.
    """
    
    def __init__(self, systemviz_integration_path: Optional[Path] = None):
        if systemviz_integration_path is None:
            systemviz_integration_path = ROOT / "systemviz_integration.py"
        
        # Dynamically import SystemVizIntegration
        import sys
        sys.path.insert(0, str(ROOT.parent))
        from patterns.systemviz_integration import SystemVizIntegration
        self.systemviz = SystemVizIntegration()
        
        # Load Lakoff archetype library
        self.lakoff_library_path = ROOT / "lakoff_archetypes.json"
        self.library = self._load_library()
        
        # Build mapping between SystemViz categories and Lakoff concepts
        self._build_crosswalk()
    
    def _load_library(self) -> ArchetypeLibrary:
        """Load Lakoff archetype library, create default if missing."""
        if self.lakoff_library_path.exists():
            return ArchetypeLibrary.load(self.lakoff_library_path)
        else:
            return create_default_archetypes()
    
    def _build_crosswalk(self):
        """Build mapping between SystemViz terms and Lakoff concepts."""
        # Map SystemViz categories to likely Lakoff feature layers
        self.category_to_layer = {
            "driver": "linking",      # Drivers often abstract
            "signal": "grounded",     # Signals often observable
            "state": "grounded",      # States often measurable
            "boundary": "linking",    # Boundaries often conceptual
            "relation": "linking",    # Relations abstract
            "domain": "linking",      # Domains abstract
        }
        
        # Map SystemViz terms to relevant Lakoff archetypes
        # Based on term semantics and archetype focus
        self.term_to_archetypes = {
            # Control-related terms
            "driver.control_point": ["conservative", "transplant_focused"],
            "driver.conduit": ["metabolic_optimizer"],
            "driver.cycle": ["metabolic_optimizer"],
            "driver.cascade": ["aggressive", "transplant_focused"],
            "driver.inflection": ["aggressive"],
            "driver.equifinality": ["conservative", "metabolic_optimizer"],
            
            # Signal-related terms
            "signal.monitor": ["conservative"],  # Monitoring aligns with conservative
            "signal.indicator": ["conservative", "metabolic_optimizer"],
            "signal.framing": ["conservative"],
            "signal.feed_back": ["metabolic_optimizer"],  # Feedback loops
            
            # State-related terms
            "state.standard": ["conservative"],
            "state.fault_detection": ["conservative"],
            "state.tipping_point": ["aggressive", "transplant_focused"],
            "state.criticality": ["aggressive"],
            "state.transition": ["aggressive", "metabolic_optimizer"],
            
            # Boundary-related terms
            "boundary.gate": ["conservative"],
            "boundary.reactive_boundary": ["aggressive", "transplant_focused"],
            
            # Relation-related terms
            "relation.coupling": ["metabolic_optimizer"],
            "relation.network": ["metabolic_optimizer"],
            "relation.cluster": ["conservative", "metabolic_optimizer"],
            
            # Domain-related terms
            "domain.support_structure": ["conservative"],
            "domain.levels_of_scale": ["metabolic_optimizer"],
        }
        
        # Inverse mapping: archetype → relevant SystemViz terms
        self.archetype_to_terms = {}
        for term, archetypes in self.term_to_archetypes.items():
            for arch in archetypes:
                self.archetype_to_terms.setdefault(arch, []).append(term)
    
    def annotate_with_dual_vocabulary(self, pattern_id: str) -> Dict[str, Any]:
        """Annotate a pattern with both SystemViz and Lakoff semantics."""
        # Get SystemViz description
        sv_desc = self.systemviz.describe_pattern_with_lexicon(pattern_id)
        
        # Get pattern from pattern language
        pattern = self.systemviz.patterns_by_id.get(pattern_id, {})
        sv_terms = pattern.get("systemviz_tags", [])
        
        # Find relevant Lakoff archetypes based on SystemViz terms
        relevant_archetypes = set()
        for term in sv_terms:
            if term in self.term_to_archetypes:
                relevant_archetypes.update(self.term_to_archetypes[term])
        
        # For each relevant archetype, compute match strength
        lakoff_annotations = []
        for arch_name in sorted(relevant_archetypes):
            arch = self.library.get(arch_name)
            if arch:
                # Find shared terms between archetype and pattern
                shared_terms = [t for t in sv_terms 
                               if t in self.term_to_archetypes 
                               and arch_name in self.term_to_archetypes[t]]
                
                lakoff_annotations.append({
                    "archetype": arch_name,
                    "description": arch.description,
                    "shared_terms": shared_terms,
                    "match_strength": len(shared_terms) / max(len(sv_terms), 1),
                })
        
        # Classify pattern's SystemViz terms by Lakoff layer
        layer_analysis = {"grounded": [], "linking": []}
        for term in sv_terms:
            if "." in term:
                category = term.split(".", 1)[0]
                layer = self.category_to_layer.get(category, "linking")
                layer_analysis[layer].append(term)
        
        result = {
            "systemviz": sv_desc,
            "lakoff": {
                "pattern_id": pattern_id,
                "pattern_name": pattern.get("name", ""),
                "stage": pattern.get("stage", ""),
                "systemviz_terms": sv_terms,
                "layer_analysis": layer_analysis,
                "relevant_archetypes": lakoff_annotations,
                "grounded_ratio": len(layer_analysis["grounded"]) / max(len(sv_terms), 1),
            }
        }
        return result
    
    def analyze_pattern_language(self) -> Dict[str, Any]:
        """Analyze the entire pattern language through Lakoff+SystemViz lens."""
        patterns = self.systemviz.pattern_language["patterns"]
        
        analysis = {
            "patterns": [],
            "summary": {
                "total_patterns": len(patterns),
                "patterns_by_stage": {},
                "archetype_coverage": {arch.name: 0 for arch in self.library.archetypes},
                "layer_distribution": {"grounded": 0, "linking": 0},
            }
        }
        
        for pattern in patterns:
            pid = pattern["id"]
            dual = self.annotate_with_dual_vocabulary(pid)
            analysis["patterns"].append(dual)
            
            # Update summary statistics
            stage = pattern["stage"]
            analysis["summary"]["patterns_by_stage"][stage] = \
                analysis["summary"]["patterns_by_stage"].get(stage, 0) + 1
            
            # Update archetype coverage
            for arch_anno in dual["lakoff"]["relevant_archetypes"]:
                arch_name = arch_anno["archetype"]
                if arch_name in analysis["summary"]["archetype_coverage"]:
                    analysis["summary"]["archetype_coverage"][arch_name] += 1
            
            # Update layer distribution
            grounded_count = len(dual["lakoff"]["layer_analysis"]["grounded"])
            linking_count = len(dual["lakoff"]["layer_analysis"]["linking"])
            analysis["summary"]["layer_distribution"]["grounded"] += grounded_count
            analysis["summary"]["layer_distribution"]["linking"] += linking_count
        
        return analysis
    
    def generate_crosswalk_report(self) -> Dict[str, Any]:
        """Generate a crosswalk report between SystemViz and Lakoff vocabularies."""
        report = {
            "systemviz_categories": {},
            "lakoff_archetypes": {},
            "mapping_matrix": []
        }
        
        # SystemViz categories analysis
        for category in self.systemviz.lexicon["categories"]:
            terms = list(self.systemviz.lexicon["categories"][category]["terms"].keys())
            report["systemviz_categories"][category] = {
                "term_count": len(terms),
                "example_terms": terms[:3],
                "lakoff_layer": self.category_to_layer.get(category, "unknown"),
            }
        
        # Lakoff archetypes analysis
        for arch in self.library.archetypes:
            relevant_terms = self.archetype_to_terms.get(arch.name, [])
            report["lakoff_archetypes"][arch.name] = {
                "description": arch.description,
                "relevant_systemviz_terms": relevant_terms,
                "grounding_criteria_count": len(arch.grounding_criteria),
                "has_icm": arch.icm is not None,
            }
        
        # Mapping matrix
        for term in sorted(self.term_to_archetypes.keys()):
            if "." in term:
                category = term.split(".", 1)[0]
                report["mapping_matrix"].append({
                    "systemviz_term": term,
                    "category": category,
                    "lakoff_layer": self.category_to_layer.get(category, "unknown"),
                    "relevant_archetypes": self.term_to_archetypes.get(term, []),
                })
        
        return report


# ------------------------------------------------------------------
# Default archetype definitions
# ------------------------------------------------------------------

def create_default_archetypes() -> ArchetypeLibrary:
    """Create default mitochondrial intervention archetypes with grounding criteria."""
    library = ArchetypeLibrary()
    
    # 1. Conservative protocol archetype
    conservative_icm = ICM(
        name="conservative_assumptions",
        background=[
            "Patient is relatively healthy (heteroplasmy < 0.3)",
            "Intervention aims to maintain health, not reverse damage",
            "Energy cost of intervention should be minimal",
            "Safety and tolerability prioritized over maximal benefit",
        ],
        violation_conditions=[
            GroundingCriterion(
                feature="damage.het_initial",
                predicate="gt",
                value=0.5,
                rationale="Patient too damaged for conservative approach"
            ),
            GroundingCriterion(
                feature="intervention.energy_cost_per_year",
                predicate="gt",
                value=0.2,
                rationale="Energy cost too high for conservative protocol"
            ),
        ]
    )
    
    conservative = Archetype(
        name="conservative",
        description="Low-risk, maintenance-focused intervention with minimal energy cost",
        grounding_criteria=[
            GroundingCriterion(
                feature="intervention.total_dose",
                predicate="lt",
                value=1.5,
                rationale="Conservative protocols use low total intervention dose",
                layer="linking"
            ),
            GroundingCriterion(
                feature="intervention.benefit_cost_ratio",
                predicate="gt",
                value=2.0,
                rationale="Conservative protocols prioritize efficiency",
                layer="linking"
            ),
            GroundingCriterion(
                feature="energy.atp_final",
                predicate="gt",
                value=0.7,
                rationale="Conservative aim: maintain adequate ATP",
                layer="grounded"
            ),
        ],
        icm=conservative_icm
    )
    library.add(conservative)
    
    # 2. Aggressive protocol archetype
    aggressive_icm = ICM(
        name="aggressive_assumptions",
        background=[
            "Patient is significantly damaged (heteroplasmy > 0.3)",
            "Risk-benefit tradeoff favors aggressive intervention",
            "Energy reserves sufficient to support intensive treatment",
            "Goal is damage reversal, not just maintenance",
        ],
        violation_conditions=[
            GroundingCriterion(
                feature="energy.atp_initial",
                predicate="lt",
                value=0.5,
                rationale="Insufficient energy reserves for aggressive protocol"
            ),
            GroundingCriterion(
                feature="damage.cliff_distance_initial",
                predicate="lt",
                value=0.1,
                rationale="Too close to cliff for aggressive intervention safety"
            ),
        ]
    )
    
    aggressive = Archetype(
        name="aggressive",
        description="High-intensity intervention aiming for damage reversal",
        grounding_criteria=[
            GroundingCriterion(
                feature="intervention.total_dose",
                predicate="gt",
                value=3.0,
                rationale="Aggressive protocols use high total dose",
                layer="linking"
            ),
            GroundingCriterion(
                feature="damage.delta_het",
                predicate="lt",
                value=-0.1,
                rationale="Aggressive protocols should reduce heteroplasmy",
                layer="grounded"
            ),
            GroundingCriterion(
                feature="intervention.crisis_delay_years",
                predicate="gt",
                value=5.0,
                rationale="Aggressive protocols should significantly delay crisis",
                layer="linking"
            ),
        ],
        icm=aggressive_icm
    )
    library.add(aggressive)
    
    # 3. Transplant-focused archetype
    transplant_icm = ICM(
        name="transplant_assumptions",
        background=[
            "Deletion heteroplasmy is primary driver of pathology",
            "Transplant competes with damaged mtDNA via displacement",
            "Patient has sufficient NAD+ to support engraftment",
            "Transplant is the primary rejuvenation modality (Cramer C8)",
        ],
        violation_conditions=[
            GroundingCriterion(
                feature="dynamics.nad_slope",
                predicate="lt",
                value=-0.01,
                rationale="Declining NAD impairs transplant engraftment"
            ),
            GroundingCriterion(
                feature="damage.deletion_het_initial",
                predicate="lt",
                value=0.1,
                rationale="Transplant not indicated for low deletion heteroplasmy"
            ),
        ]
    )
    
    transplant = Archetype(
        name="transplant_focused",
        description="Protocol centered on mtDNA transplant as primary rejuvenation",
        grounding_criteria=[
            GroundingCriterion(
                feature="intervention.transplant_rate",
                predicate="gt",
                value=0.5,
                rationale="Transplant-focused protocols prioritize transplant dose",
                layer="grounded"
            ),
            GroundingCriterion(
                feature="damage.deletion_het_final",
                predicate="lt",
                value=0.4,
                rationale="Transplant should significantly reduce deletion heteroplasmy",
                layer="grounded"
            ),
            GroundingCriterion(
                feature="intervention.atp_benefit_terminal",
                predicate="gt",
                value=0.1,
                rationale="Transplant should provide substantial ATP benefit",
                layer="linking"
            ),
        ],
        icm=transplant_icm
    )
    library.add(transplant)
    
    # 4. Metabolic optimizer archetype
    metabolic_icm = ICM(
        name="metabolic_optimizer_assumptions",
        background=[
            "Metabolic flexibility enables hormetic adaptation",
            "Exercise-induced ROS triggers antioxidant upregulation",
            "NAD+ supports mitochondrial biogenesis and quality control",
            "Combined interventions have synergistic effects",
        ],
        violation_conditions=[
            GroundingCriterion(
                feature="dynamics.ros_amplitude",
                predicate="gt",
                value=0.3,
                rationale="Excessive ROS amplitude indicates poor metabolic control"
            ),
            GroundingCriterion(
                feature="energy.atp_cv",
                predicate="gt",
                value=0.2,
                rationale="High ATP variability indicates unstable metabolism"
            ),
        ]
    )
    
    metabolic = Archetype(
        name="metabolic_optimizer",
        description="Protocol optimizing metabolic flexibility via combined interventions",
        grounding_criteria=[
            GroundingCriterion(
                feature="intervention.exercise_level",
                predicate="gt",
                value=0.5,
                rationale="Metabolic optimizer includes substantial exercise",
                layer="grounded"
            ),
            GroundingCriterion(
                feature="intervention.nad_supplement",
                predicate="gt",
                value=0.5,
                rationale="Metabolic optimizer includes NAD+ support",
                layer="grounded"
            ),
            GroundingCriterion(
                feature="dynamics.ros_het_correlation",
                predicate="lt",
                value=-0.3,
                rationale="Metabolic optimizer should break ROS-damage vicious cycle",
                layer="linking"
            ),
        ],
        icm=metabolic_icm
    )
    library.add(metabolic)
    
    return library


# ------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Lakoff Integration for Mitochondrial Intervention Archetypes")
    print("=" * 70)
    
    # Create default archetypes
    library = create_default_archetypes()
    print(f"Created {len(library.archetypes)} default archetypes:")
    for arch in library.archetypes:
        print(f"  • {arch.name}: {arch.description}")
    
    # Save to file
    output_path = ROOT / "lakoff_archetypes.json"
    library.save(output_path)
    print(f"\nSaved to {output_path}")
    
    # Test feature extraction
    print("\nFeature layer classification examples:")
    test_features = ["energy.atp_final", "intervention.benefit_cost_ratio", 
                     "damage.het_slope", "dynamics.ros_amplitude"]
    for feat in test_features:
        layer = get_feature_layer(feat)
        print(f"  {feat:35s} → {layer}")
    
    # Test SystemViz-Lakoff bridge
    print("\n" + "=" * 70)
    print("SystemViz-Lakoff Bridge Test")
    print("=" * 70)
    
    try:
        bridge = LakoffSystemVizBridge()
        
        # Test dual annotation on a sample pattern
        sample_pattern = "analytics_profile"
        dual = bridge.annotate_with_dual_vocabulary(sample_pattern)
        print(f"\nDual annotation for pattern '{sample_pattern}':")
        print(f"  Pattern: {dual['lakoff']['pattern_name']} ({dual['lakoff']['stage']})")
        print(f"  SystemViz terms: {len(dual['lakoff']['systemviz_terms'])}")
        print(f"  Grounded ratio: {dual['lakoff']['grounded_ratio']:.2f}")
        print(f"  Relevant archetypes: {[a['archetype'] for a in dual['lakoff']['relevant_archetypes']]}")
        
        # Test crosswalk report
        report = bridge.generate_crosswalk_report()
        print(f"\nCrosswalk report summary:")
        print(f"  SystemViz categories: {len(report['systemviz_categories'])}")
        print(f"  Lakoff archetypes: {len(report['lakoff_archetypes'])}")
        print(f"  Mapping matrix entries: {len(report['mapping_matrix'])}")
        
        # Save crosswalk report
        crosswalk_path = ROOT / "lakoff_systemviz_crosswalk.json"
        with open(crosswalk_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nCrosswalk report saved to {crosswalk_path}")
        
    except Exception as e:
        print(f"\nBridge test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)