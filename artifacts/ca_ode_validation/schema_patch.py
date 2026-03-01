# Updated BIN_SCHEMA centers based on ODE empirical means
# Replace the 'centers' lists in ca_schema.py with these:

# N_healthy: [0.300, 0.560, 0.950]
# N_deletion: [0.054, 0.195, 0.300, 0.500]
# ATP: [0.200, 0.500, 0.790, 0.955]
# ROS: [0.090, 0.109, 0.250]
# NAD: [0.300, 0.700, 0.937]
# Senescent_fraction: [0.073, 0.100, 0.400]
# Membrane_potential: [0.300, 0.506, 0.866]
# N_point: [0.100, 0.100, 0.300]

BIN_SCHEMA = {
    "N_healthy": {
        "index": 0,
        "thresholds": [0.3, 0.56],
        "labels": ["depleted", "reduced", "adequate"],
        "centers": [0.300, 0.560, 0.950],
        "unit": "normalized copies",
        "source": "C2 copy homeostasis"
    },
    "N_deletion": {
        "index": 1,
        "thresholds": [0.1, 0.3, 0.5],
        "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"],
        "centers": [0.054, 0.195, 0.300, 0.500],
        "unit": "deletion het fraction",
        "source": "HETEROPLASMY_CLIFF=0.50, Cramer Appendix 2"
    },
    "ATP": {
        "index": 2,
        "thresholds": [0.2, 0.5, 0.79],
        "labels": ["collapsed", "crisis", "compromised", "healthy"],
        "centers": [0.200, 0.500, 0.790, 0.955],
        "unit": "MU/day",
        "source": "ATP_CRISIS_FRACTION=0.5, Cramer Ch. VIII.A Table 3"
    },
    "ROS": {
        "index": 3,
        "thresholds": [0.1, 0.25],
        "labels": ["basal", "elevated", "pathological"],
        "centers": [0.090, 0.109, 0.250],
        "unit": "normalized",
        "source": "BASELINE_ROS=0.1, Cramer Ch. II.H"
    },
    "NAD": {
        "index": 4,
        "thresholds": [0.3, 0.7],
        "labels": ["depleted", "declining", "robust"],
        "centers": [0.300, 0.700, 0.937],
        "unit": "normalized",
        "source": "NAD_DECLINE_RATE=0.01/yr, Cramer Ch. VI.A.3"
    },
    "Senescent_fraction": {
        "index": 5,
        "thresholds": [0.1, 0.4],
        "labels": ["minimal", "emerging", "severe"],
        "centers": [0.073, 0.100, 0.400],
        "unit": "fraction",
        "source": "SENESCENCE_RATE=0.005/yr, Cramer Ch. VII.A"
    },
    "Membrane_potential": {
        "index": 6,
        "thresholds": [0.3, 0.7],
        "labels": ["collapsed", "impaired", "intact"],
        "centers": [0.300, 0.506, 0.866],
        "unit": "normalized ΔΨ",
        "source": "MITOPHAGY_ATP_MIDPOINT=0.6, Cramer Ch. VI.B"
    },
    "N_point": {
        "index": 7,
        "thresholds": [0.1, 0.3],
        "labels": ["low", "moderate", "high"],
        "centers": [0.100, 0.100, 0.300],
        "unit": "point het fraction",
        "source": "POINT_ERROR_RATE=0.001, Cramer Ch. II.H"
    },
}
