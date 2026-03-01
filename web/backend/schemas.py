"""Pydantic schemas for web backend API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunSimulationRequest(BaseModel):
    intervention: dict[str, Any] = Field(default_factory=dict)
    patient: dict[str, Any] = Field(default_factory=dict)
    sim_years: float = 30.0
    dt: float = 0.01
    tissue_type: str | None = None
    stochastic: bool = False
    noise_scale: float = 0.01
    n_trajectories: int = 1
    rng_seed: int | None = None
    async_job: bool = False


class RunSimulationResponse(BaseModel):
    run_id: str
    summary: dict[str, Any]
    analytics: dict[str, Any]
    series: dict[str, Any]
    status: Literal["completed"] = "completed"


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    kind: str
    run_id: str | None = None


class CompareRequest(BaseModel):
    baseline: RunSimulationRequest
    candidate: RunSimulationRequest
    async_job: bool = False


class CompareResponse(BaseModel):
    run_id: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    delta: dict[str, Any]
    status: Literal["completed"] = "completed"


class LlmSuggestRequest(BaseModel):
    scenario: str
    style: Literal["numeric", "diegetic", "contrastive"] = "numeric"
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 800
    min_intervention_keys: int = 4


class LlmSuggestResponse(BaseModel):
    vector: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    parse_status: str
    raw_excerpt: str | None = None
    raw_response: str | None = None
    provider: dict[str, Any] = Field(default_factory=dict)


class LlmExplainRequest(BaseModel):
    analytics: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    scenario: str | None = None
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 600


class LlmExplainResponse(BaseModel):
    explanation: str | None = None
    raw_response: str | None = None
    provider: dict[str, Any] = Field(default_factory=dict)


class RunIndexItem(BaseModel):
    run_id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    summary: dict[str, Any] = Field(default_factory=dict)


class RunListResponse(BaseModel):
    runs: list[RunIndexItem] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    kind: str
    run_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class MetaParametersResponse(BaseModel):
    intervention_params: dict[str, Any]
    patient_params: dict[str, Any]
    models: list[dict[str, Any]]
    prompt_styles: list[str]
