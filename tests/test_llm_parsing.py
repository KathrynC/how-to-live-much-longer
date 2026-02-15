"""Tests for LLM response parsing utilities."""
from __future__ import annotations

import pytest

from llm_common import (
    strip_think_tags, strip_markdown_fences,
    parse_json_response, parse_intervention_vector,
    detect_flattening, validate_llm_output, split_vector,
)
from tests.conftest import (
    SAMPLE_LLM_CLEAN_JSON, SAMPLE_LLM_MARKDOWN_FENCED,
    SAMPLE_LLM_THINK_TAGS, SAMPLE_LLM_PARTIAL, SAMPLE_LLM_FLATTENED,
)


class TestStripThinkTags:
    def test_removes_think_tags(self):
        text = "<think>reasoning here</think>actual output"
        assert strip_think_tags(text) == "actual output"

    def test_no_think_tags_unchanged(self):
        text = "no tags here"
        assert strip_think_tags(text) == "no tags here"

    def test_nested_think_tags(self):
        text = "<think>outer<think>inner</think>middle</think>final"
        result = strip_think_tags(text)
        assert result == "final"


class TestStripMarkdownFences:
    def test_json_fence(self):
        text = "text\n```json\n{\"key\": \"value\"}\n```\nmore"
        result = strip_markdown_fences(text)
        assert result.startswith("{")

    def test_plain_fence(self):
        text = "text\n```\n{\"key\": \"value\"}\n```"
        result = strip_markdown_fences(text)
        assert result.startswith("{")

    def test_no_fence_unchanged(self):
        text = '{"key": "value"}'
        assert strip_markdown_fences(text) == text


class TestParseJsonResponse:
    def test_clean_json(self):
        result = parse_json_response(SAMPLE_LLM_CLEAN_JSON)
        assert result is not None
        assert result["rapamycin_dose"] == 0.5
        assert result["baseline_age"] == 70

    def test_markdown_fenced(self):
        result = parse_json_response(SAMPLE_LLM_MARKDOWN_FENCED)
        assert result is not None
        assert result["rapamycin_dose"] == 0.5

    def test_think_tags(self):
        result = parse_json_response(SAMPLE_LLM_THINK_TAGS)
        assert result is not None
        assert result["rapamycin_dose"] == 0.5

    def test_empty_input(self):
        assert parse_json_response("") is None
        assert parse_json_response(None) is None

    def test_no_json(self):
        assert parse_json_response("just plain text, no JSON") is None

    def test_malformed_json(self):
        assert parse_json_response("{ broken json: }") is None


class TestDetectFlattening:
    def test_flattened_age(self):
        params = {"baseline_age": 0.7}
        corrected, fixes = detect_flattening(params)
        assert corrected["baseline_age"] == pytest.approx(69.0, abs=1.0)
        assert len(fixes) > 0

    def test_flattened_vulnerability(self):
        params = {"genetic_vulnerability": 0.3}
        corrected, fixes = detect_flattening(params)
        assert corrected["genetic_vulnerability"] == pytest.approx(0.8, abs=0.01)

    def test_flattened_metabolic_demand(self):
        params = {"metabolic_demand": 0.4}
        corrected, fixes = detect_flattening(params)
        assert corrected["metabolic_demand"] == pytest.approx(0.9, abs=0.01)

    def test_normal_values_unchanged(self):
        params = {"baseline_age": 70, "genetic_vulnerability": 1.0,
                  "metabolic_demand": 1.5}
        corrected, fixes = detect_flattening(params)
        assert corrected["baseline_age"] == 70
        assert len(fixes) == 0


class TestParseInterventionVector:
    def test_complete_response(self):
        result = parse_intervention_vector(SAMPLE_LLM_CLEAN_JSON)
        assert result is not None
        assert "rapamycin_dose" in result
        assert "baseline_age" in result
        # Should be snapped to grid
        assert result["rapamycin_dose"] == 0.5
        assert result["baseline_age"] == 70.0

    def test_markdown_fenced_response(self):
        result = parse_intervention_vector(SAMPLE_LLM_MARKDOWN_FENCED)
        assert result is not None

    def test_think_tag_response(self):
        result = parse_intervention_vector(SAMPLE_LLM_THINK_TAGS)
        assert result is not None

    def test_partial_response_interventions_only(self):
        result = parse_intervention_vector(SAMPLE_LLM_PARTIAL)
        assert result is not None
        assert "rapamycin_dose" in result
        # Patient params not present — that's OK (only 6 needed)

    def test_flattened_response_corrected(self):
        result = parse_intervention_vector(SAMPLE_LLM_FLATTENED)
        assert result is not None
        # Age should be rescaled from 0.7 to ~69, then snapped to 70
        assert result["baseline_age"] == 70.0

    def test_too_few_params_returns_none(self):
        result = parse_intervention_vector('{"rapamycin_dose": 0.5}')
        assert result is None

    def test_empty_returns_none(self):
        assert parse_intervention_vector("") is None


class TestValidateLlmOutput:
    def test_valid_dict(self):
        raw = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
               "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
               "transplant_rate": 0.1, "exercise_level": 0.5,
               "baseline_age": 70, "baseline_heteroplasmy": 0.3}
        snapped, warnings = validate_llm_output(raw)
        assert snapped["rapamycin_dose"] == 0.5
        assert snapped["baseline_age"] == 70.0

    def test_out_of_range_warns(self):
        raw = {"rapamycin_dose": 1.5, "nad_supplement": 0.75,
               "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
               "transplant_rate": 0.1, "exercise_level": 0.5}
        snapped, warnings = validate_llm_output(raw)
        assert len(warnings) > 0  # pydantic should flag rapamycin_dose


class TestSplitVector:
    def test_split_complete(self):
        snapped = {"rapamycin_dose": 0.5, "nad_supplement": 0.75,
                   "senolytic_dose": 0.25, "yamanaka_intensity": 0.0,
                   "transplant_rate": 0.1, "exercise_level": 0.5,
                   "baseline_age": 70.0, "baseline_heteroplasmy": 0.3,
                   "baseline_nad_level": 0.6, "genetic_vulnerability": 1.0,
                   "metabolic_demand": 1.0, "inflammation_level": 0.25}
        intervention, patient = split_vector(snapped)
        assert len(intervention) == 6
        assert len(patient) == 6
        assert intervention["rapamycin_dose"] == 0.5
        assert patient["baseline_age"] == 70.0

    def test_split_with_defaults(self):
        snapped = {"rapamycin_dose": 0.5}
        intervention, patient = split_vector(snapped)
        assert intervention["rapamycin_dose"] == 0.5
        assert intervention["nad_supplement"] == 0.0  # default
        assert patient["baseline_age"] == 70.0  # default
