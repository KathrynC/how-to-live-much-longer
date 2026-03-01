#!/usr/bin/env python3
"""Comprehensive sentiment analysis of all SystemViz terms across all files."""

import json
import os
from collections import defaultdict

ROOT = os.path.dirname(__file__)

# Load all SystemViz data files
LEXICON_PATH = os.path.join(ROOT, "systemviz_lexicon.json")
MAPPING_PATH = os.path.join(ROOT, "systemviz_mapping.json")
CROSSWALK_PATH = os.path.join(ROOT, "lakoff_systemviz_crosswalk.json")

with open(LEXICON_PATH, "r") as f:
    lexicon = json.load(f)

with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

with open(CROSSWALK_PATH, "r") as f:
    crosswalk = json.load(f)

# Manual sentiment classification (extended with CA terms and side_effect)
MANUAL_SENTIMENT = {
    # Core lexicon terms
    "control_point": "neutral",
    "conduit": "neutral", 
    "amplifier": "neutral",
    "repeller": "negative",
    "enabler": "positive",
    "dampener": "neutral",
    "cycle": "neutral",
    "genator": "neutral",
    "cascade": "negative",
    "inflection": "neutral",
    "equifinality": "neutral",
    
    "framing": "neutral",
    "monitor": "neutral",
    "indicator": "neutral",
    "feed_back": "neutral",
    "status_display": "neutral",
    "alert": "negative",
    
    "stock": "neutral",
    "standard": "neutral",
    "fault_detection": "negative",
    "redundancy": "positive",
    "fault_recovery": "positive",
    "tipping_point": "negative",
    "criticality": "negative",
    "load": "negative",
    "mutation": "negative",
    "transition": "neutral",
    "accumulation": "neutral",
    "capacity": "positive",
    "reset": "neutral",
    "replication": "neutral",
    "assembly": "positive",
    "separation": "negative",
    "maintenance": "positive",
    
    "gate": "neutral",
    "buffer": "positive",
    "reactive_boundary": "neutral",
    "permissions": "neutral",
    "container": "neutral",
    "semi_permiable": "neutral",
    "divider": "neutral",
    "protector": "positive",
    "notional_boundary": "neutral",
    "edge": "neutral",
    
    "group": "positive",
    "rank": "neutral",
    "differentiation": "neutral",
    "mediation": "positive",
    "aggregate": "neutral",
    "coupling": "neutral",
    "network": "neutral",
    "cluster": "neutral",
    
    "support_structure": "positive",
    "reference_points": "neutral",
    "resources": "positive",
    "levels_of_scale": "neutral",
    
    # Additional terms found in mapping file (not in lexicon)
    "side_effect": "negative",
    
    # CA-specific terms (from crosswalk)
    "rule_firing": "neutral",
    "cascade": "negative",
    "attractor_transition": "neutral",
    "bin_transition": "neutral",
    "cliff_crossing_signal": "negative",
    "attractor": "neutral",
    "bin_state": "neutral",
    "discrete_state": "neutral",
    "cliff_boundary": "negative",
    "bin_threshold": "neutral",
    "bin_ordering": "neutral",
    "variable_coupling": "neutral",
    "state_space": "neutral",
    "bin_lattice": "neutral",
}

# Extract all terms from lexicon
lexicon_terms = []
for category_name, category_data in lexicon["categories"].items():
    for term_name in category_data["terms"]:
        lexicon_terms.append(f"{category_name}.{term_name}")

# Extract all terms from mapping file
mapping_terms = set()
for item in mapping["mappings"]:
    for term in item.get("systemviz_terms", []):
        mapping_terms.add(term)

# Extract all terms from crosswalk file
crosswalk_terms = set()
# From example_terms arrays
for category_name, category_data in crosswalk["systemviz_categories"].items():
    for term in category_data.get("example_terms", []):
        if "." in term:
            crosswalk_terms.add(term)
        else:
            crosswalk_terms.add(f"{category_name}.{term}")

# From relevant_systemviz_terms arrays
for archetype_name, archetype_data in crosswalk["lakoff_archetypes"].items():
    for term in archetype_data.get("relevant_systemviz_terms", []):
        crosswalk_terms.add(term)

# From mapping_matrix
for item in crosswalk.get("mapping_matrix", []):
    term = item.get("systemviz_term")
    if term:
        crosswalk_terms.add(term)

# Combine all terms
all_terms_set = set(lexicon_terms) | mapping_terms | crosswalk_terms
all_terms = sorted(all_terms_set)

# Classify each term
results = []
for full_term in all_terms:
    # Parse category and term name
    if "." in full_term:
        parts = full_term.split(".")
        if parts[0] in ["driver", "signal", "state", "boundary", "relation", "domain"]:
            category = parts[0]
            term_name = ".".join(parts[1:])  # Handle ca.xxx terms
        else:
            # Fallback: assume first part is category
            category = parts[0]
            term_name = ".".join(parts[1:])
    else:
        category = "unknown"
        term_name = full_term
    
    # Get definition if available
    definition = None
    if category in lexicon["categories"]:
        definition = lexicon["categories"][category]["terms"].get(term_name)
    
    # Determine sentiment
    sentiment = MANUAL_SENTIMENT.get(term_name, "neutral")
    
    results.append({
        "full_term": full_term,
        "category": category,
        "term_name": term_name,
        "definition": definition,
        "sentiment": sentiment,
        "sources": []
    })

# Tag sources for each term
for res in results:
    term = res["full_term"]
    sources = []
    if term in lexicon_terms:
        sources.append("lexicon")
    if term in mapping_terms:
        sources.append("mapping")
    if term in crosswalk_terms:
        sources.append("crosswalk")
    res["sources"] = sources

# Separate positive terms
positive_terms = [r for r in results if r["sentiment"] == "positive"]
negative_terms = [r for r in results if r["sentiment"] == "negative"]
neutral_terms = [r for r in results if r["sentiment"] == "neutral"]

# Print summary
print("SystemViz Comprehensive Sentiment Analysis")
print("=" * 70)
print(f"Total unique terms found: {len(all_terms)}")
print(f"Positive terms: {len(positive_terms)}")
print(f"Negative terms: {len(negative_terms)}")
print(f"Neutral terms: {len(neutral_terms)}")
print()

print("Most Positive Terms (sentiment='positive'):")
print("-" * 40)
for term in sorted([r["full_term"] for r in positive_terms]):
    print(f"  {term}")

print("\nSentiment by Category:")
print("-" * 40)
category_stats = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
for r in results:
    cat = r["category"]
    sentiment = r["sentiment"]
    category_stats[cat][sentiment] += 1

for cat in sorted(category_stats.keys()):
    stats = category_stats[cat]
    total = stats["positive"] + stats["negative"] + stats["neutral"]
    print(f"{cat:15s}: {stats['positive']:2d} pos, {stats['negative']:2d} neg, {stats['neutral']:2d} neu ({total} total)")

# Terms missing from lexicon
lexicon_term_set = set(lexicon_terms)
missing_terms = [t for t in all_terms if t not in lexicon_term_set]
print(f"\nTerms missing from lexicon ({len(missing_terms)}):")
for term in sorted(missing_terms):
    print(f"  {term}")

# Save detailed results
output_path = os.path.join(ROOT, "systemviz_comprehensive_sentiment.json")
with open(output_path, "w") as f:
    json.dump({
        "analysis_date": "2026-02-22",
        "method": "manual_classification_comprehensive",
        "total_unique_terms": len(all_terms),
        "sentiment_counts": {
            "positive": len(positive_terms),
            "negative": len(negative_terms),
            "neutral": len(neutral_terms)
        },
        "positive_terms": [r["full_term"] for r in positive_terms],
        "negative_terms": [r["full_term"] for r in negative_terms],
        "neutral_terms": [r["full_term"] for r in neutral_terms],
        "missing_from_lexicon": missing_terms,
        "detailed_results": results
    }, f, indent=2)

print(f"\nResults saved to: {output_path}")

# Also save just the positive terms as a simple list
positive_list_path = os.path.join(ROOT, "systemviz_positive_terms.txt")
with open(positive_list_path, "w") as f:
    for term in sorted([r["full_term"] for r in positive_terms]):
        f.write(f"{term}\n")

print(f"Positive terms list saved to: {positive_list_path}")