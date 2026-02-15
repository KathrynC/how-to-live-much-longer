"""Shared fixtures for the mitochondrial aging test suite."""
from __future__ import annotations

import pytest

from constants import DEFAULT_INTERVENTION, DEFAULT_PATIENT


@pytest.fixture
def default_patient() -> dict[str, float]:
    return dict(DEFAULT_PATIENT)


@pytest.fixture
def default_intervention() -> dict[str, float]:
    return dict(DEFAULT_INTERVENTION)


@pytest.fixture
def cocktail_intervention() -> dict[str, float]:
    return {
        "rapamycin_dose": 0.5,
        "nad_supplement": 0.75,
        "senolytic_dose": 0.5,
        "yamanaka_intensity": 0.0,
        "transplant_rate": 0.0,
        "exercise_level": 0.5,
    }


@pytest.fixture
def near_cliff_patient() -> dict[str, float]:
    return {
        "baseline_age": 80.0,
        "baseline_heteroplasmy": 0.65,
        "baseline_nad_level": 0.4,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.5,
    }


@pytest.fixture
def young_patient() -> dict[str, float]:
    return {
        "baseline_age": 30.0,
        "baseline_heteroplasmy": 0.01,
        "baseline_nad_level": 1.0,
        "genetic_vulnerability": 1.0,
        "metabolic_demand": 1.0,
        "inflammation_level": 0.0,
    }


# Sample LLM responses for parse testing
SAMPLE_LLM_CLEAN_JSON = (
    '{"rapamycin_dose": 0.5, "nad_supplement": 0.75, "senolytic_dose": 0.25, '
    '"yamanaka_intensity": 0.0, "transplant_rate": 0.1, "exercise_level": 0.5, '
    '"baseline_age": 70, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6, '
    '"genetic_vulnerability": 1.0, "metabolic_demand": 1.0, "inflammation_level": 0.25}'
)

SAMPLE_LLM_MARKDOWN_FENCED = (
    "Here is my recommendation:\n"
    "```json\n"
    '{"rapamycin_dose": 0.5, "nad_supplement": 0.75, "senolytic_dose": 0.25, '
    '"yamanaka_intensity": 0.0, "transplant_rate": 0.1, "exercise_level": 0.5, '
    '"baseline_age": 70, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6, '
    '"genetic_vulnerability": 1.0, "metabolic_demand": 1.0, "inflammation_level": 0.25}\n'
    "```"
)

SAMPLE_LLM_THINK_TAGS = (
    "<think>Let me analyze this patient carefully. They are 70 years old "
    "with moderate damage.</think>\n"
    '{"rapamycin_dose": 0.5, "nad_supplement": 0.75, "senolytic_dose": 0.25, '
    '"yamanaka_intensity": 0.0, "transplant_rate": 0.1, "exercise_level": 0.5, '
    '"baseline_age": 70, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6, '
    '"genetic_vulnerability": 1.0, "metabolic_demand": 1.0, "inflammation_level": 0.25}'
)

SAMPLE_LLM_PARTIAL = (
    '{"rapamycin_dose": 0.5, "nad_supplement": 0.75, "senolytic_dose": 0.25, '
    '"yamanaka_intensity": 0.0, "transplant_rate": 0.1, "exercise_level": 0.5}'
)

SAMPLE_LLM_FLATTENED = (
    '{"rapamycin_dose": 0.5, "nad_supplement": 0.75, "senolytic_dose": 0.25, '
    '"yamanaka_intensity": 0.0, "transplant_rate": 0.1, "exercise_level": 0.5, '
    '"baseline_age": 0.7, "baseline_heteroplasmy": 0.3, "baseline_nad_level": 0.6, '
    '"genetic_vulnerability": 0.3, "metabolic_demand": 0.4, "inflammation_level": 0.25}'
)
