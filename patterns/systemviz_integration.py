"""systemviz_integration.py — Bridge between pattern language and Stoyko's SystemViz.

Provides semantic mapping between protocol pipeline patterns, mitochondrial simulator
components, and the SystemViz visual vocabulary.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent

# Paths to data files
PATTERN_LANGUAGE_PATH = ROOT / "protocol_pattern_language.v1.json"
SYSTEMVIZ_LEXICON_PATH = ROOT / "systemviz_lexicon.json"
SYSTEMVIZ_MAPPING_PATH = ROOT / "systemviz_mapping.json"


class SystemVizIntegration:
    """Integrates Stoyko's SystemViz lexicon with protocol pattern language."""
    
    def __init__(self):
        self.pattern_language = self._load_json(PATTERN_LANGUAGE_PATH)
        self.lexicon = self._load_json(SYSTEMVIZ_LEXICON_PATH)
        self.mapping = self._load_json(SYSTEMVIZ_MAPPING_PATH)
        
        # Index patterns by ID
        self.patterns_by_id = {p["id"]: p for p in self.pattern_language["patterns"]}
        
        # Build inverted index: term -> list of pattern IDs
        self.term_to_patterns: Dict[str, List[str]] = {}
        for p in self.pattern_language["patterns"]:
            for term in p.get("systemviz_tags", []):
                self.term_to_patterns.setdefault(term, []).append(p["id"])
        
        # Build inverted index: term -> list of component IDs
        self.term_to_components: Dict[str, List[Dict[str, Any]]] = {}
        for comp in self.mapping["mappings"]:
            for term in comp.get("systemviz_terms", []):
                self.term_to_components.setdefault(term, []).append(comp)
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON file with validation."""
        if not path.exists():
            raise FileNotFoundError(f"SystemViz data file not found: {path}")
        return json.loads(path.read_text())
    
    def get_patterns_by_term(self, term: str) -> List[Dict[str, Any]]:
        """Return all patterns tagged with the given SystemViz term."""
        pattern_ids = self.term_to_patterns.get(term, [])
        return [self.patterns_by_id[pid] for pid in pattern_ids]
    
    def get_components_by_term(self, term: str) -> List[Dict[str, Any]]:
        """Return all simulator components tagged with the given SystemViz term."""
        return self.term_to_components.get(term, [])
    
    def get_terms_by_pattern(self, pattern_id: str) -> List[str]:
        """Return SystemViz terms for a given pattern."""
        pattern = self.patterns_by_id.get(pattern_id)
        if not pattern:
            return []
        return pattern.get("systemviz_tags", [])
    
    def get_terms_by_component(self, component_id: str) -> List[str]:
        """Return SystemViz terms for a given simulator component."""
        for comp in self.mapping["mappings"]:
            if comp["id"] == component_id:
                return comp.get("systemviz_terms", [])
        return []
    
    def get_term_definition(self, category: str, term: str) -> Optional[str]:
        """Return definition of a SystemViz term."""
        cat = self.lexicon["categories"].get(category)
        if not cat:
            return None
        return cat["terms"].get(term)
    
    def describe_pattern_with_lexicon(self, pattern_id: str) -> Dict[str, Any]:
        """Generate rich description of a pattern using SystemViz lexicon."""
        pattern = self.patterns_by_id.get(pattern_id)
        if not pattern:
            return {}
        
        result = {
            "id": pattern["id"],
            "name": pattern["name"],
            "stage": pattern["stage"],
            "terms": []
        }
        
        for term in pattern.get("systemviz_tags", []):
            cat, term_name = term.split(".", 1)
            definition = self.get_term_definition(cat, term_name)
            result["terms"].append({
                "term": term,
                "category": cat,
                "definition": definition
            })
        
        return result
    
    def describe_component_with_lexicon(self, component_id: str) -> Dict[str, Any]:
        """Generate rich description of a simulator component using SystemViz lexicon."""
        for comp in self.mapping["mappings"]:
            if comp["id"] == component_id:
                result = {
                    "id": comp["id"],
                    "name": comp["name"],
                    "category": comp["category"],
                    "terms": []
                }
                for term in comp.get("systemviz_terms", []):
                    cat, term_name = term.split(".", 1)
                    definition = self.get_term_definition(cat, term_name)
                    result["terms"].append({
                        "term": term,
                        "category": cat,
                        "definition": definition
                    })
                return result
        return {}
    
    def find_bridging_terms(self) -> List[Dict[str, Any]]:
        """Find SystemViz terms that appear in both patterns and components."""
        pattern_terms = set(self.term_to_patterns.keys())
        component_terms = set(self.term_to_components.keys())
        bridge_terms = pattern_terms.intersection(component_terms)
        
        results = []
        for term in sorted(bridge_terms):
            cat, term_name = term.split(".", 1)
            definition = self.get_term_definition(cat, term_name)
            results.append({
                "term": term,
                "definition": definition,
                "pattern_count": len(self.term_to_patterns[term]),
                "component_count": len(self.term_to_components[term]),
                "patterns": self.get_patterns_by_term(term),
                "components": self.get_components_by_term(term)
            })
        return results
    
    def generate_crosswalk_report(self) -> Dict[str, Any]:
        """Generate comprehensive crosswalk between patterns and components."""
        bridge_terms = self.find_bridging_terms()
        
        # Group by category
        by_category: Dict[str, List] = {}
        for term_info in bridge_terms:
            cat, _ = term_info["term"].split(".", 1)
            by_category.setdefault(cat, []).append(term_info)
        
        return {
            "bridge_term_count": len(bridge_terms),
            "pattern_count": len(self.patterns_by_id),
            "component_count": len(self.mapping["mappings"]),
            "by_category": by_category,
            "bridge_terms": bridge_terms
        }


# Convenience singleton
_integration: Optional[SystemVizIntegration] = None

def get_integration() -> SystemVizIntegration:
    """Get or create the singleton SystemViz integration instance."""
    global _integration
    if _integration is None:
        _integration = SystemVizIntegration()
    return _integration


if __name__ == "__main__":
    # Quick demo when run directly
    integration = get_integration()
    print("SystemViz Integration Demo")
    print("=" * 40)
    
    # Show patterns tagged with driver.control_point
    term = "driver.control_point"
    patterns = integration.get_patterns_by_term(term)
    print(f"\nPatterns tagged with '{term}':")
    for p in patterns:
        print(f"  - {p['id']}: {p['name']}")
    
    # Show components tagged with state.tipping_point
    term2 = "state.tipping_point"
    components = integration.get_components_by_term(term2)
    print(f"\nComponents tagged with '{term2}':")
    for c in components:
        print(f"  - {c['id']}: {c['name']} ({c['category']})")
    
    # Find bridging terms
    bridges = integration.find_bridging_terms()
    print(f"\nFound {len(bridges)} bridging terms (appear in both patterns and components):")
    for b in bridges[:5]:  # Show first 5
        print(f"  - {b['term']}: {b['pattern_count']} patterns, {b['component_count']} components")
    
    # Generate crosswalk report
    report = integration.generate_crosswalk_report()
    print(f"\nCrosswalk summary:")
    print(f"  Total bridge terms: {report['bridge_term_count']}")
    for cat, terms in report['by_category'].items():
        print(f"  {cat}: {len(terms)} terms")