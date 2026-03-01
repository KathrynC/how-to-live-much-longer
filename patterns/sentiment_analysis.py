#!/usr/bin/env python3
"""Sentiment analysis of Stoyko's SystemViz terminology."""

import json
import os

# Load the lexicon
LEXICON_PATH = os.path.join(os.path.dirname(__file__), "systemviz_lexicon.json")

with open(LEXICON_PATH, "r") as f:
    lexicon = json.load(f)

# Pre-defined sentiment rules
POSITIVE_KEYWORDS = [
    "enabler", "protector", "redundancy", "recovery", "maintenance",
    "capacity", "reset", "assembly", "group", "mediation", "support",
    "resources", "standard", "permissions", "gate", "buffer", "container"
]

NEGATIVE_KEYWORDS = [
    "repeller", "fault", "mutation", "tipping_point", "criticality",
    "load", "separation", "dampener", "alert", "inflection", "cascade"
]

NEUTRAL_KEYWORDS = [
    "stock", "conduit", "cycle", "genator", "framing", "monitor",
    "indicator", "feed_back", "status_display", "transition",
    "accumulation", "replication", "fault_detection", "equifinality",
    "amplifier", "control_point", "differentiation", "aggregate",
    "coupling", "network", "cluster", "reference_points",
    "levels_of_scale", "semi_permiable", "divider", "notional_boundary",
    "edge", "rank", "reactive_boundary", "signal", "driver", "state",
    "boundary", "relation", "domain"
]

def classify_term(term_name, definition):
    """Classify term as positive, negative, or neutral."""
    term_lower = term_name.lower()
    
    # Check positive keywords
    for kw in POSITIVE_KEYWORDS:
        if kw in term_lower:
            return "positive"
    
    # Check negative keywords  
    for kw in NEGATIVE_KEYWORDS:
        if kw in term_lower:
            return "negative"
    
    # Check neutral keywords
    for kw in NEUTRAL_KEYWORDS:
        if kw in term_lower:
            return "neutral"
    
    # Default to neutral
    return "neutral"

# Analyze all terms
results = []
positive_terms = []

for category_name, category_data in lexicon["categories"].items():
    for term_name, definition in category_data["terms"].items():
        sentiment = classify_term(term_name, definition)
        
        results.append({
            "category": category_name,
            "term": term_name,
            "definition": definition,
            "sentiment": sentiment
        })
        
        if sentiment == "positive":
            positive_terms.append(f"{category_name}.{term_name}")

# Print results
print("SystemViz Terminology Sentiment Analysis")
print("=" * 60)
print(f"Total terms: {len(results)}")
print(f"Positive terms: {len([r for r in results if r['sentiment'] == 'positive'])}")
print(f"Negative terms: {len([r for r in results if r['sentiment'] == 'negative'])}")
print(f"Neutral terms: {len([r for r in results if r['sentiment'] == 'neutral'])}")
print()

print("Most Positive Terms (sentiment='positive'):")
print("-" * 40)
for term in sorted(positive_terms):
    print(f"  {term}")

# Also show by category
print("\nSentiment by Category:")
print("-" * 40)
for category_name in lexicon["categories"]:
    cat_terms = [r for r in results if r["category"] == category_name]
    pos = len([r for r in cat_terms if r["sentiment"] == "positive"])
    neg = len([r for r in cat_terms if r["sentiment"] == "negative"])
    neu = len([r for r in cat_terms if r["sentiment"] == "neutral"])
    print(f"{category_name:12s}: {pos:2d} pos, {neg:2d} neg, {neu:2d} neu")

# Save to JSON
output_path = os.path.join(os.path.dirname(__file__), "systemviz_sentiment.json")
with open(output_path, "w") as f:
    json.dump({
        "analysis_date": "2026-02-22",
        "total_terms": len(results),
        "sentiment_counts": {
            "positive": len([r for r in results if r["sentiment"] == "positive"]),
            "negative": len([r for r in results if r["sentiment"] == "negative"]),
            "neutral": len([r for r in results if r["sentiment"] == "neutral"])
        },
        "positive_terms": positive_terms,
        "detailed_results": results
    }, f, indent=2)

print(f"\nResults saved to: {output_path}")