#!/usr/bin/env python3
"""Refined sentiment analysis of Stoyko's SystemViz terminology with manual classification."""

import json
import os

# Load the lexicon
LEXICON_PATH = os.path.join(os.path.dirname(__file__), "systemviz_lexicon.json")

with open(LEXICON_PATH, "r") as f:
    lexicon = json.load(f)

# Manual sentiment classification based on systems thinking context
MANUAL_SENTIMENT = {
    # Driver category
    "control_point": "neutral",
    "conduit": "neutral", 
    "amplifier": "neutral",
    "repeller": "negative",
    "enabler": "positive",
    "dampener": "neutral",  # could be positive if dampening negative effects
    "cycle": "neutral",
    "genator": "neutral",
    "cascade": "negative",
    "inflection": "neutral",  # turning point could be positive or negative
    "equifinality": "neutral",
    
    # Signal category
    "framing": "neutral",
    "monitor": "neutral",
    "indicator": "neutral",
    "feed_back": "neutral",
    "status_display": "neutral",
    "alert": "negative",
    
    # State category
    "stock": "neutral",
    "standard": "neutral",  # standardization can be good or bad
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
    "reset": "neutral",  # could be positive if resetting to good state
    "replication": "neutral",
    "assembly": "positive",
    "separation": "negative",
    "maintenance": "positive",
    
    # Boundary category
    "gate": "neutral",
    "buffer": "positive",  # diminishes negative drivers
    "reactive_boundary": "neutral",
    "permissions": "neutral",
    "container": "neutral",
    "semi_permiable": "neutral",
    "divider": "neutral",
    "protector": "positive",
    "notional_boundary": "neutral",
    "edge": "neutral",
    
    # Relation category
    "group": "positive",
    "rank": "neutral",
    "differentiation": "neutral",
    "mediation": "positive",
    "aggregate": "neutral",
    "coupling": "neutral",
    "network": "neutral",
    "cluster": "neutral",
    
    # Domain category
    "support_structure": "positive",
    "reference_points": "neutral",
    "resources": "positive",
    "levels_of_scale": "neutral",
}

# Analyze all terms
results = []
positive_terms = []

for category_name, category_data in lexicon["categories"].items():
    for term_name, definition in category_data["terms"].items():
        # Use manual classification if available, otherwise default to neutral
        sentiment = MANUAL_SENTIMENT.get(term_name, "neutral")
        
        results.append({
            "category": category_name,
            "term": term_name,
            "definition": definition,
            "sentiment": sentiment
        })
        
        if sentiment == "positive":
            positive_terms.append(f"{category_name}.{term_name}")

# Print results
print("SystemViz Terminology Sentiment Analysis (Refined)")
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
output_path = os.path.join(os.path.dirname(__file__), "systemviz_sentiment_refined.json")
with open(output_path, "w") as f:
    json.dump({
        "analysis_date": "2026-02-22",
        "method": "manual_classification",
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