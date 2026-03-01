#!/usr/bin/env python3
"""
Update ca_schema.py with new BIN_SCHEMA.
"""
import re

# New BIN_SCHEMA block
new_block = '''BIN_SCHEMA: dict[str, dict] = {
    "N_healthy": {
        "index": 0,
        "thresholds": [0.3, 0.56],
        "labels": ["depleted", "reduced", "adequate"],
        "centers": [0.243, 0.27, 0.884],
        "unit": "normalized copies",
        "source": "C2 copy homeostasis",
    },
    "N_deletion": {
        "index": 1,
        "thresholds": [0.1, 0.3, 0.5],
        "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"],
        "centers": [0.123, 0.203, 0.416, 0.37],
        "unit": "deletion het fraction",
        "source": "HETEROPLASMY_CLIFF=0.50, Cramer Appendix 2",
    },
    "ATP": {
        "index": 2,
        "thresholds": [0.2, 0.5, 0.79],
        "labels": ["collapsed", "crisis", "compromised", "healthy"],
        "centers": [0.63, 0.632, 0.638, 0.886],
        "unit": "MU/day",
        "source": "ATP_CRISIS_FRACTION=0.5, Cramer Ch. VIII.A Table 3",
    },
    "ROS": {
        "index": 3,
        "thresholds": [0.1, 0.25],
        "labels": ["basal", "elevated", "pathological"],
        "centers": [0.05, 0.211, 0.309],
        "unit": "normalized",
        "source": "BASELINE_ROS=0.1, Cramer Ch. II.H",
    },
    "NAD": {
        "index": 4,
        "thresholds": [0.3, 0.7],
        "labels": ["depleted", "declining", "robust"],
        "centers": [0.293, 0.537, 0.924],
        "unit": "normalized",
        "source": "NAD_DECLINE_RATE=0.01/yr, Cramer Ch. VI.A.3",
    },
    "Senescent_fraction": {
        "index": 5,
        "thresholds": [0.1, 0.4],
        "labels": ["minimal", "emerging", "severe"],
        "centers": [0.085, 0.228, 0.319],
        "unit": "fraction",
        "source": "SENESCENCE_RATE=0.005/yr, Cramer Ch. VII.A",
    },
    "Membrane_potential": {
        "index": 6,
        "thresholds": [0.3, 0.5],
        "labels": ["collapsed", "impaired", "intact"],
        "centers": [0.15, 0.4, 0.75],
        "unit": "normalized ΔΨ",
        "source": "MITOPHAGY_ATP_MIDPOINT=0.6, Cramer Ch. VI.B",
    },
    "N_point": {
        "index": 7,
        "thresholds": [0.1, 0.3],
        "labels": ["low", "moderate", "high"],
        "centers": [0.05, 0.255, 0.441],
        "unit": "point het fraction",
        "source": "POINT_ERROR_RATE=0.001, Cramer Ch. II.H",
    },
}'''

# Read original file
with open('ca_schema.py', 'r') as f:
    content = f.read()

# Replace BIN_SCHEMA block using regex
pattern = r'BIN_SCHEMA: dict\[str, dict\] = \{.*?\n\}'
new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)

# Write back
with open('ca_schema.py', 'w') as f:
    f.write(new_content)

print("Updated ca_schema.py with new BIN_SCHEMA")