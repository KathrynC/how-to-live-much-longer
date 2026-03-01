# Updated BIN_SCHEMA with empirical centers and thresholds
# Replace entire BIN_SCHEMA in ca_schema.py with this:

BIN_SCHEMA = {
    "N_healthy": {
        "index": 0,
        "thresholds": [0.736, 0.832],
        "labels": ["depleted", "reduced", "adequate"],
        "centers": [0.300, 0.560, 0.950],
        "unit": "normalized copies",
        "source": "C2 copy homeostasis + empirical adjustment + thresholds"
    },
    "N_deletion": {
        "index": 1,
        "thresholds": [0.124, 0.194, 0.194],
        "labels": ["minimal", "growing", "approaching_cliff", "past_cliff"],
        "centers": [0.054, 0.195, 0.300, 0.500],
        "unit": "deletion het fraction",
        "source": "HETEROPLASMY_CLIFF=0.50, Cramer Appendix 2 + empirical adjustment + thresholds"
    },
    "ATP": {
        "index": 2,
        "thresholds": [0.864, 0.864, 0.903],
        "labels": ["collapsed", "crisis", "compromised", "healthy"],
        "centers": [0.200, 0.500, 0.790, 0.955],
        "unit": "MU/day",
        "source": "ATP_CRISIS_FRACTION=0.5, Cramer Ch. VIII.A Table 3 + empirical adjustment + thresholds"
    },
    "ROS": {
        "index": 3,
        "thresholds": [0.100, 0.120],
        "labels": ["basal", "elevated", "pathological"],
        "centers": [0.090, 0.109, 0.250],
        "unit": "normalized",
        "source": "BASELINE_ROS=0.1, Cramer Ch. II.H + empirical adjustment + thresholds"
    },
    "NAD": {
        "index": 4,
        "thresholds": [0.627, 0.844],
        "labels": ["depleted", "declining", "robust"],
        "centers": [0.300, 0.700, 0.937],
        "unit": "normalized",
        "source": "NAD_DECLINE_RATE=0.01/yr, Cramer Ch. VI.A.3 + empirical adjustment + thresholds"
    },
    "Senescent_fraction": {
        "index": 5,
        "thresholds": [0.075, 0.169],
        "labels": ["minimal", "emerging", "severe"],
        "centers": [0.073, 0.100, 0.400],
        "unit": "fraction",
        "source": "SENESCENCE_RATE=0.005/yr, Cramer Ch. VII.A + empirical adjustment + thresholds"
    },
    "Membrane_potential": {
        "index": 6,
        "thresholds": [0.512, 0.686],
        "labels": ["collapsed", "impaired", "intact"],
        "centers": [0.300, 0.506, 0.866],
        "unit": "normalized ΔΨ",
        "source": "MITOPHAGY_ATP_MIDPOINT=0.6, Cramer Ch. VI.B + empirical adjustment + thresholds"
    },
    "N_point": {
        "index": 7,
        "thresholds": [0.100, 0.119],
        "labels": ["low", "moderate", "high"],
        "centers": [0.100, 0.100, 0.300],
        "unit": "point het fraction",
        "source": "POINT_ERROR_RATE=0.001, Cramer Ch. II.H + empirical adjustment + thresholds"
    },
}
