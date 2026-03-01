#!/usr/bin/env python3
"""Generate a report of the most positive SystemViz terms."""

import json
import os

ROOT = os.path.dirname(__file__)

# Load data
LEXICON_PATH = os.path.join(ROOT, "systemviz_lexicon.json")
COMPREHENSIVE_PATH = os.path.join(ROOT, "systemviz_comprehensive_sentiment.json")

with open(LEXICON_PATH, "r") as f:
    lexicon = json.load(f)

with open(COMPREHENSIVE_PATH, "r") as f:
    comprehensive = json.load(f)

# Get positive terms
positive_terms = comprehensive["positive_terms"]

# Get definitions
term_definitions = {}
for category_name, category_data in lexicon["categories"].items():
    for term_name, definition in category_data["terms"].items():
        term_definitions[f"{category_name}.{term_name}"] = definition

# Add definitions for CA terms (manual)
term_definitions.update({
    "boundary.buffer": "A boundary or zone that diminishes, delays, or otherwise alters a driver.",
    "boundary.protector": "A barrier preventing damage by blocking unwanted forces.",
    "domain.resources": "The availability and access to the factor inputs needed.",
    "domain.support_structure": "Constructed supports that enables activity.",
    "driver.enabler": "A contributory cause; can encourage change.",
    "relation.group": "An affiliated collective of actors that share patterns of behavior.",
    "relation.mediation": "The mitigation or reconciliation of conflict between parties.",
    "state.assembly": "Building blocks with which system structures are formed.",
    "state.capacity": "The finite ability to handle a particular quantity of activity.",
    "state.fault_recovery": "Correction or compensation that minimize impact of faults.",
    "state.maintenance": "The restoration of operations after a malfunction.",
    "state.redundancy": "Multiple stockpiles or parallel sub-systems that act as a back-up.",
})

# Generate markdown report
report_lines = []
report_lines.append("# SystemViz Positive Terminology Analysis")
report_lines.append("")
report_lines.append("## Summary")
report_lines.append("")
report_lines.append(f"- **Total unique SystemViz terms**: {comprehensive['total_unique_terms']}")
report_lines.append(f"- **Positive terms**: {len(positive_terms)}")
report_lines.append(f"- **Negative terms**: {len(comprehensive['negative_terms'])}")
report_lines.append(f"- **Neutral terms**: {len(comprehensive['neutral_terms'])}")
report_lines.append("")
report_lines.append("## The Most Positive Terms")
report_lines.append("")
report_lines.append("These 12 terms were classified as having positive sentiment in systems thinking contexts:")
report_lines.append("")

for i, term in enumerate(sorted(positive_terms), 1):
    definition = term_definitions.get(term, "No definition found in lexicon.")
    report_lines.append(f"{i}. **`{term}`**  \n   {definition}")
    report_lines.append("")

report_lines.append("## Positive Terms as Python Set")
report_lines.append("")
report_lines.append("```python")
report_lines.append("POSITIVE_SYSTEMVIZ_TERMS = {")
for term in sorted(positive_terms):
    report_lines.append(f'    "{term}",')
report_lines.append("}")
report_lines.append("```")
report_lines.append("")

report_lines.append("## Sentiment Distribution by Category")
report_lines.append("")
report_lines.append("| Category | Positive | Negative | Neutral | Total |")
report_lines.append("|----------|----------|----------|---------|-------|")

# Calculate category stats
category_stats = {}
for term in comprehensive["detailed_results"]:
    cat = term["category"]
    if cat not in category_stats:
        category_stats[cat] = {"positive": 0, "negative": 0, "neutral": 0}
    sentiment = term["sentiment"]
    category_stats[cat][sentiment] += 1

for cat in sorted(category_stats.keys()):
    stats = category_stats[cat]
    total = stats["positive"] + stats["negative"] + stats["neutral"]
    report_lines.append(f"| {cat} | {stats['positive']} | {stats['negative']} | {stats['neutral']} | {total} |")

report_lines.append("")

# Write report
report_path = os.path.join(ROOT, "systemviz_positive_terms_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"Report generated: {report_path}")
print(f"\nPositive terms ({len(positive_terms)}):")
for term in sorted(positive_terms):
    print(f"  {term}")