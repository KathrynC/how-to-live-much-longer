"""Tests for pydantic schema validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import InterventionVector, PatientProfile, FullProtocol


class TestInterventionVector:
    def test_valid(self):
        v = InterventionVector(
            rapamycin_dose=0.5, nad_supplement=0.75, senolytic_dose=0.25,
            yamanaka_intensity=0.0, transplant_rate=0.1, exercise_level=0.5,
        )
        assert v.rapamycin_dose == 0.5

    def test_boundary_values(self):
        v = InterventionVector(
            rapamycin_dose=0.0, nad_supplement=1.0, senolytic_dose=0.0,
            yamanaka_intensity=0.0, transplant_rate=0.0, exercise_level=1.0,
        )
        assert v.nad_supplement == 1.0

    def test_out_of_range(self):
        with pytest.raises(ValidationError):
            InterventionVector(
                rapamycin_dose=1.5, nad_supplement=0.75, senolytic_dose=0.25,
                yamanaka_intensity=0.0, transplant_rate=0.1, exercise_level=0.5,
            )

    def test_negative_value(self):
        with pytest.raises(ValidationError):
            InterventionVector(
                rapamycin_dose=-0.1, nad_supplement=0.75, senolytic_dose=0.25,
                yamanaka_intensity=0.0, transplant_rate=0.1, exercise_level=0.5,
            )


class TestPatientProfile:
    def test_valid(self):
        p = PatientProfile(
            baseline_age=70.0, baseline_heteroplasmy=0.3,
            baseline_nad_level=0.6, genetic_vulnerability=1.0,
            metabolic_demand=1.0, inflammation_level=0.25,
        )
        assert p.baseline_age == 70.0

    def test_age_out_of_range(self):
        with pytest.raises(ValidationError):
            PatientProfile(
                baseline_age=10.0, baseline_heteroplasmy=0.3,
                baseline_nad_level=0.6, genetic_vulnerability=1.0,
                metabolic_demand=1.0, inflammation_level=0.25,
            )

    def test_het_too_high(self):
        with pytest.raises(ValidationError):
            PatientProfile(
                baseline_age=70.0, baseline_heteroplasmy=0.99,
                baseline_nad_level=0.6, genetic_vulnerability=1.0,
                metabolic_demand=1.0, inflammation_level=0.25,
            )

    def test_vulnerability_range(self):
        with pytest.raises(ValidationError):
            PatientProfile(
                baseline_age=70.0, baseline_heteroplasmy=0.3,
                baseline_nad_level=0.6, genetic_vulnerability=0.1,
                metabolic_demand=1.0, inflammation_level=0.25,
            )


class TestFullProtocol:
    def test_all_optional(self):
        p = FullProtocol()
        assert p.rapamycin_dose is None
        assert p.baseline_age is None

    def test_partial_fields(self):
        p = FullProtocol(rapamycin_dose=0.5, baseline_age=70.0)
        assert p.rapamycin_dose == 0.5
        assert p.baseline_age == 70.0
        assert p.nad_supplement is None

    def test_model_dump_filters_none(self):
        p = FullProtocol(rapamycin_dose=0.5, baseline_age=70.0)
        d = {k: v for k, v in p.model_dump().items() if v is not None}
        assert len(d) == 2
        assert d["rapamycin_dose"] == 0.5

    def test_out_of_range_still_rejected(self):
        with pytest.raises(ValidationError):
            FullProtocol(rapamycin_dose=1.5)

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError):
            FullProtocol(rapamycin_dose="high")
