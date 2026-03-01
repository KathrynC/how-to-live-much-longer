"""FastAPI app for the local simulation + LLM workbench."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import traceback
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from constants import INTERVENTION_PARAMS, PATIENT_PARAMS
from web.backend.config import Settings, get_settings
from web.backend.schemas import (
    CompareRequest,
    CompareResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    LlmExplainRequest,
    LlmExplainResponse,
    LlmSuggestRequest,
    LlmSuggestResponse,
    MetaParametersResponse,
    RunListResponse,
    RunSimulationRequest,
    RunSimulationResponse,
)
from web.backend.services.history_store import HistoryStore
from web.backend.services.llm_service import LlmService
from web.backend.services.simulation_service import SimulationService


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobState:
    job_id: str
    kind: str
    status: str
    run_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


class JobManager:
    """Simple in-memory async job tracker."""

    def __init__(self, max_workers: int = 4):
        self._jobs: dict[str, JobState] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="web-job")

    def submit(self, kind: str, run_id: str, fn: Callable[[], dict[str, Any]]) -> JobState:
        job_id = uuid4().hex
        now = _utc_now_iso()
        job = JobState(
            job_id=job_id,
            kind=kind,
            status="pending",
            run_id=run_id,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job

        def _runner() -> None:
            self._set(job_id, status="running")
            try:
                result = fn()
                self._set(job_id, status="completed", result=result, error=None)
            except Exception as exc:
                tb = traceback.format_exc(limit=2)
                self._set(
                    job_id,
                    status="failed",
                    result=None,
                    error=f"{type(exc).__name__}: {exc}\n{tb}",
                )

        self._executor.submit(_runner)
        return job

    def _set(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = _utc_now_iso()

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)


def create_app(
    settings: Settings | None = None,
    history_store: HistoryStore | None = None,
    simulation_service: SimulationService | None = None,
    llm_service: LlmService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    history = history_store or HistoryStore(settings.web_runs_root)
    sim = simulation_service or SimulationService()
    llm = llm_service or LlmService()
    jobs = JobManager(max_workers=4)

    app = FastAPI(title="LLM Simulation Workbench", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.default_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _run_simulation_sync(run_id: str, request: RunSimulationRequest) -> dict[str, Any]:
        artifact = sim.run(request, run_id=run_id)
        payload = artifact.to_payload()
        history.save_run(
            run_id=run_id,
            kind="simulation",
            status="completed",
            payload=payload,
            summary=artifact.summary,
        )
        return {
            "run_id": run_id,
            "summary": artifact.summary,
            "analytics": artifact.analytics,
            "series": artifact.series,
            "status": "completed",
        }

    def _run_compare_sync(run_id: str, request: CompareRequest) -> dict[str, Any]:
        payload = sim.compare(request.baseline, request.candidate, run_id=run_id)
        history.save_run(
            run_id=run_id,
            kind="comparison",
            status="completed",
            payload=payload,
            summary=payload.get("delta", {}),
        )
        return {
            "run_id": run_id,
            "baseline": payload.get("baseline", {}),
            "candidate": payload.get("candidate", {}),
            "delta": payload.get("delta", {}),
            "status": "completed",
        }

    @app.get("/api/meta/parameters", response_model=MetaParametersResponse)
    def get_meta_parameters() -> MetaParametersResponse:
        return MetaParametersResponse(
            intervention_params=INTERVENTION_PARAMS,
            patient_params=PATIENT_PARAMS,
            models=llm.model_catalog(),
            prompt_styles=llm.prompt_styles(),
        )

    @app.get("/api/meta/health")
    def get_meta_health() -> dict[str, Any]:
        return {
            "ollama": llm.health(settings.ollama_url),
            "time_utc": _utc_now_iso(),
        }

    @app.post("/api/llm/suggest", response_model=LlmSuggestResponse)
    def post_llm_suggest(request: LlmSuggestRequest) -> LlmSuggestResponse:
        result = llm.suggest(request)
        return LlmSuggestResponse(
            vector=result.vector,
            warnings=result.warnings,
            parse_status=result.parse_status,
            raw_excerpt=result.raw_excerpt,
            raw_response=result.raw_response,
            provider={
                **result.provider,
                "model": result.model,
            },
        )

    @app.post("/api/llm/explain", response_model=LlmExplainResponse)
    def post_llm_explain(request: LlmExplainRequest) -> LlmExplainResponse:
        result = llm.explain(request)
        return LlmExplainResponse(
            explanation=result.explanation,
            raw_response=result.raw_response,
            provider={
                **result.provider,
                "model": result.model,
            },
        )

    @app.post(
        "/api/simulate/run",
        response_model=RunSimulationResponse | JobAcceptedResponse,
    )
    def post_simulate_run(
        request: RunSimulationRequest,
    ) -> RunSimulationResponse | JobAcceptedResponse:
        run_id = uuid4().hex
        if request.async_job:
            history.save_run(
                run_id=run_id,
                kind="simulation",
                status="pending",
                payload={
                    "run_id": run_id,
                    "kind": "simulation",
                    "status": "pending",
                    "request": request.model_dump(),
                    "created_at": _utc_now_iso(),
                },
                summary={},
            )

            def _job() -> dict[str, Any]:
                try:
                    return _run_simulation_sync(run_id, request)
                except Exception as exc:
                    history.save_run(
                        run_id=run_id,
                        kind="simulation",
                        status="failed",
                        payload={
                            "run_id": run_id,
                            "kind": "simulation",
                            "status": "failed",
                            "request": request.model_dump(),
                            "error": f"{type(exc).__name__}: {exc}",
                            "created_at": _utc_now_iso(),
                        },
                        summary={"error": str(exc)},
                    )
                    raise

            job = jobs.submit(kind="simulation", run_id=run_id, fn=_job)
            return JobAcceptedResponse(
                job_id=job.job_id,
                status=job.status,  # pending
                kind=job.kind,
                run_id=job.run_id,
            )

        try:
            response = _run_simulation_sync(run_id, request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc
        return RunSimulationResponse(**response)

    @app.post(
        "/api/simulate/compare",
        response_model=CompareResponse | JobAcceptedResponse,
    )
    def post_simulate_compare(
        request: CompareRequest,
    ) -> CompareResponse | JobAcceptedResponse:
        run_id = uuid4().hex
        if request.async_job:
            history.save_run(
                run_id=run_id,
                kind="comparison",
                status="pending",
                payload={
                    "run_id": run_id,
                    "kind": "comparison",
                    "status": "pending",
                    "request": request.model_dump(),
                    "created_at": _utc_now_iso(),
                },
                summary={},
            )

            def _job() -> dict[str, Any]:
                try:
                    return _run_compare_sync(run_id, request)
                except Exception as exc:
                    history.save_run(
                        run_id=run_id,
                        kind="comparison",
                        status="failed",
                        payload={
                            "run_id": run_id,
                            "kind": "comparison",
                            "status": "failed",
                            "request": request.model_dump(),
                            "error": f"{type(exc).__name__}: {exc}",
                            "created_at": _utc_now_iso(),
                        },
                        summary={"error": str(exc)},
                    )
                    raise

            job = jobs.submit(kind="comparison", run_id=run_id, fn=_job)
            return JobAcceptedResponse(
                job_id=job.job_id,
                status=job.status,
                kind=job.kind,
                run_id=job.run_id,
            )

        try:
            response = _run_compare_sync(run_id, request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc
        return CompareResponse(**response)

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job_status(job_id: str) -> JobStatusResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            kind=job.kind,
            run_id=job.run_id,
            result=job.result,
            error=job.error,
        )

    @app.get("/api/runs", response_model=RunListResponse)
    def get_runs(limit: int = 100) -> RunListResponse:
        rows = history.list_runs(limit=limit)
        runs = []
        for row in rows:
            runs.append(
                {
                    "run_id": row.get("run_id"),
                    "kind": row.get("kind", "simulation"),
                    "status": row.get("status", "unknown"),
                    "created_at": row.get("created_at", ""),
                    "updated_at": row.get("updated_at", ""),
                    "summary": row.get("summary", {}) or {},
                }
            )
        return RunListResponse(runs=runs)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        payload = history.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return payload

    return app


app = create_app()
