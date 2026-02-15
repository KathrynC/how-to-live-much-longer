"""Pydantic schema validation for LLM-generated intervention/patient vectors.

Provides strict validation with per-field range constraints derived from
the parameter space definitions in constants.py. Used as an additional
validation layer between JSON parsing and grid snapping.

Usage:
    from schemas import FullProtocol, InterventionVector, PatientProfile
    protocol = FullProtocol(rapamycin_dose=0.5, baseline_age=70)
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError


class InterventionVector(BaseModel):
    """6 intervention parameters, each constrained to 0.0-1.0."""

    rapamycin_dose: float = Field(ge=0.0, le=1.0)
    nad_supplement: float = Field(ge=0.0, le=1.0)
    senolytic_dose: float = Field(ge=0.0, le=1.0)
    yamanaka_intensity: float = Field(ge=0.0, le=1.0)
    transplant_rate: float = Field(ge=0.0, le=1.0)
    exercise_level: float = Field(ge=0.0, le=1.0)


class PatientProfile(BaseModel):
    """6 patient parameters with per-field ranges from constants.py."""

    baseline_age: float = Field(ge=20.0, le=90.0)
    baseline_heteroplasmy: float = Field(ge=0.0, le=0.95)
    baseline_nad_level: float = Field(ge=0.2, le=1.0)
    genetic_vulnerability: float = Field(ge=0.5, le=2.0)
    metabolic_demand: float = Field(ge=0.5, le=2.0)
    inflammation_level: float = Field(ge=0.0, le=1.0)


class FullProtocol(BaseModel):
    """Combined 12D vector. All fields optional (LLMs may omit some)."""

    rapamycin_dose: float | None = Field(default=None, ge=0.0, le=1.0)
    nad_supplement: float | None = Field(default=None, ge=0.0, le=1.0)
    senolytic_dose: float | None = Field(default=None, ge=0.0, le=1.0)
    yamanaka_intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    transplant_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    exercise_level: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_age: float | None = Field(default=None, ge=20.0, le=90.0)
    baseline_heteroplasmy: float | None = Field(default=None, ge=0.0, le=0.95)
    baseline_nad_level: float | None = Field(default=None, ge=0.2, le=1.0)
    genetic_vulnerability: float | None = Field(default=None, ge=0.5, le=2.0)
    metabolic_demand: float | None = Field(default=None, ge=0.5, le=2.0)
    inflammation_level: float | None = Field(default=None, ge=0.0, le=1.0)
