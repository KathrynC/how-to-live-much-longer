"""API contract and async lifecycle tests for the web workbench backend."""
from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from web.backend.app import create_app
from web.backend.config import Settings
from web.backend.services.llm_service import LlmExplainResult, LlmSuggestResult
from web.backend.services.simulation_service import SimulationArtifact


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FakeSimulationService:
    def __init__(self, delay_sec: float = 0.0, fail_run: bool = False):
        self.delay_sec = delay_sec
        self.fail_run = fail_run

    def run(self, request, run_id: str | None = None) -> SimulationArtifact:
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        if self.fail_run:
            raise RuntimeError("simulated run failure")

        rid = run_id or uuid4().hex
        summary = {
            "final_atp": 0.82,
            "final_heteroplasmy": 0.24,
            "time_to_cliff_years": 15.0,
            "time_to_crisis_years": 22.0,
        }
        analytics = {
            "energy": {"atp_final": 0.82},
            "damage": {"het_final": 0.24},
            "dynamics": {"ros_het_correlation": 0.3},
            "intervention": {"atp_benefit_terminal": 0.05},
            "symmathesy": {"adaptation_coherence": 0.1},
        }
        series = {
            "time": [0.0, 1.0, 2.0],
            "atp": [0.9, 0.85, 0.82],
            "heteroplasmy": [0.2, 0.22, 0.24],
        }
        return SimulationArtifact(
            run_id=rid,
            kind="simulation",
            created_at=_utc_now_iso(),
            request=request.model_dump(),
            summary=summary,
            analytics=analytics,
            series=series,
            raw_result={"ok": True},
            baseline_result={"ok": True},
        )

    def compare(self, baseline_request, candidate_request, run_id: str | None = None) -> dict[str, Any]:
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        rid = run_id or uuid4().hex
        if self.fail_run:
            raise RuntimeError("simulated compare failure")
        baseline = self.run(baseline_request, run_id=f"{rid}-baseline").to_payload()
        candidate = self.run(candidate_request, run_id=f"{rid}-candidate").to_payload()
        return {
            "run_id": rid,
            "kind": "comparison",
            "status": "completed",
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                "delta_final_atp": 0.07,
                "delta_final_heteroplasmy": -0.03,
                "delta_time_to_cliff_years": 3.0,
            },
        }


class FakeLlmService:
    def __init__(self):
        self.mode = "success"

    def suggest(self, request):
        model = request.model or "fake-ollama:model"
        if self.mode == "success":
            return LlmSuggestResult(
                vector={"rapamycin_dose": 0.5, "nad_supplement": 0.5},
                warnings=[],
                parse_status="ok",
                raw_excerpt='{"rapamycin_dose": 0.5}',
                raw_response='{"rapamycin_dose": 0.5}',
                provider={"ok": True, "latency_sec": 0.01},
                prompt="prompt",
                model=model,
            )
        if self.mode == "timeout":
            return LlmSuggestResult(
                vector=None,
                warnings=["LLMTimeoutError: timed out"],
                parse_status="provider_error",
                raw_excerpt=None,
                raw_response=None,
                provider={"ok": False, "error_type": "LLMTimeoutError", "error_message": "timed out"},
                prompt="prompt",
                model=model,
            )
        if self.mode == "malformed":
            return LlmSuggestResult(
                vector=None,
                warnings=[],
                parse_status="no_json_object",
                raw_excerpt="not-json",
                raw_response="not-json",
                provider={"ok": True, "latency_sec": 0.02},
                prompt="prompt",
                model=model,
            )
        if self.mode == "parse_warning":
            return LlmSuggestResult(
                vector=None,
                warnings=["insufficient intervention coverage: got 2 keys, need >= 4"],
                parse_status="insufficient_intervention_keys",
                raw_excerpt='{"baseline_age": 70}',
                raw_response='{"baseline_age": 70}',
                provider={"ok": True, "latency_sec": 0.02},
                prompt="prompt",
                model=model,
            )
        raise RuntimeError(f"unsupported fake mode: {self.mode}")

    def explain(self, request):
        model = request.model or "fake-ollama:model"
        if self.mode == "timeout":
            return LlmExplainResult(
                explanation=None,
                raw_response=None,
                provider={"ok": False, "error_type": "LLMTimeoutError", "error_message": "timed out"},
                prompt="prompt",
                model=model,
            )
        return LlmExplainResult(
            explanation="ATP improved while heteroplasmy rose slowly; consider increasing transplant_rate.",
            raw_response="ATP improved while heteroplasmy rose slowly; consider increasing transplant_rate.",
            provider={"ok": True, "latency_sec": 0.01},
            prompt="prompt",
            model=model,
        )

    def model_catalog(self):
        return [{"name": "fake-ollama:model", "type": "ollama"}]

    def prompt_styles(self):
        return ["numeric", "diegetic", "contrastive"]

    def health(self, ollama_generate_url: str):
        return {"ok": True, "url": ollama_generate_url, "models": ["fake-ollama:model"]}


def _make_client(tmp_path, sim_service: FakeSimulationService | None = None, llm_service: FakeLlmService | None = None):
    settings = Settings(web_runs_root=tmp_path / "web_runs")
    app = create_app(
        settings=settings,
        simulation_service=sim_service or FakeSimulationService(),
        llm_service=llm_service or FakeLlmService(),
    )
    return TestClient(app)


def _poll_job_until_terminal(client: TestClient, job_id: str, timeout_sec: float = 2.0):
    deadline = time.time() + timeout_sec
    seen_statuses: list[str] = []
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}")
        assert status.status_code == 200
        payload = status.json()
        seen_statuses.append(payload["status"])
        if payload["status"] in {"completed", "failed"}:
            return payload, seen_statuses
        time.sleep(0.03)
    raise AssertionError(f"job {job_id} did not reach terminal state; seen={seen_statuses}")


def test_meta_parameters_contract(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/api/meta/parameters")
    assert resp.status_code == 200
    payload = resp.json()
    assert "intervention_params" in payload
    assert "patient_params" in payload
    assert payload["prompt_styles"] == ["numeric", "diegetic", "contrastive"]
    assert payload["models"][0]["name"] == "fake-ollama:model"


def test_simulate_run_sync_contract_and_history(tmp_path):
    client = _make_client(tmp_path)
    resp = client.post(
        "/api/simulate/run",
        json={
            "patient": {"baseline_age": 65.0},
            "intervention": {"rapamycin_dose": 0.5},
            "sim_years": 2.0,
            "dt": 0.1,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "completed"
    assert "run_id" in payload
    assert "summary" in payload and "analytics" in payload and "series" in payload

    list_resp = client.get("/api/runs")
    assert list_resp.status_code == 200
    run_ids = [r["run_id"] for r in list_resp.json()["runs"]]
    assert payload["run_id"] in run_ids

    run_resp = client.get(f"/api/runs/{payload['run_id']}")
    assert run_resp.status_code == 200
    assert run_resp.json()["status"] == "completed"


def test_compare_sync_contract(tmp_path):
    client = _make_client(tmp_path)
    resp = client.post(
        "/api/simulate/compare",
        json={
            "baseline": {
                "patient": {"baseline_age": 70.0},
                "intervention": {"rapamycin_dose": 0.0},
                "sim_years": 2.0,
                "dt": 0.1,
            },
            "candidate": {
                "patient": {"baseline_age": 70.0},
                "intervention": {"rapamycin_dose": 0.5},
                "sim_years": 2.0,
                "dt": 0.1,
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "completed"
    assert "delta" in payload
    assert "delta_final_atp" in payload["delta"]


def test_llm_suggest_success(tmp_path):
    llm = FakeLlmService()
    client = _make_client(tmp_path, llm_service=llm)
    resp = client.post(
        "/api/llm/suggest",
        json={
            "scenario": "70-year-old near cliff",
            "style": "numeric",
            "model": "fake-ollama:model",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["parse_status"] == "ok"
    assert payload["vector"]["rapamycin_dose"] == 0.5


def test_llm_suggest_timeout_mock(tmp_path):
    llm = FakeLlmService()
    llm.mode = "timeout"
    client = _make_client(tmp_path, llm_service=llm)
    resp = client.post(
        "/api/llm/suggest",
        json={"scenario": "test timeout", "style": "numeric"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["parse_status"] == "provider_error"
    assert any("timed out" in w for w in payload["warnings"])


def test_llm_suggest_malformed_json_mock(tmp_path):
    llm = FakeLlmService()
    llm.mode = "malformed"
    client = _make_client(tmp_path, llm_service=llm)
    resp = client.post(
        "/api/llm/suggest",
        json={"scenario": "test malformed", "style": "numeric"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["parse_status"] == "no_json_object"
    assert payload["vector"] is None


def test_llm_suggest_parse_warning_mock(tmp_path):
    llm = FakeLlmService()
    llm.mode = "parse_warning"
    client = _make_client(tmp_path, llm_service=llm)
    resp = client.post(
        "/api/llm/suggest",
        json={"scenario": "patient-only payload", "style": "numeric"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["parse_status"] == "insufficient_intervention_keys"
    assert payload["vector"] is None
    assert payload["warnings"]


def test_async_job_lifecycle_submitted_running_completed(tmp_path):
    client = _make_client(tmp_path, sim_service=FakeSimulationService(delay_sec=0.15))
    submit = client.post(
        "/api/simulate/run",
        json={
            "patient": {"baseline_age": 72.0},
            "intervention": {"rapamycin_dose": 0.25},
            "sim_years": 2.0,
            "dt": 0.1,
            "async_job": True,
        },
    )
    assert submit.status_code == 200
    submit_payload = submit.json()
    assert submit_payload["status"] in {"pending", "running"}
    job_id = submit_payload["job_id"]

    terminal_payload, seen = _poll_job_until_terminal(client, job_id)
    assert terminal_payload["status"] == "completed"
    assert "running" in seen or "pending" in seen

    run_id = terminal_payload["run_id"]
    assert run_id is not None
    run_resp = client.get(f"/api/runs/{run_id}")
    assert run_resp.status_code == 200
    assert run_resp.json()["status"] == "completed"


def test_async_job_failure_path(tmp_path):
    client = _make_client(tmp_path, sim_service=FakeSimulationService(delay_sec=0.05, fail_run=True))
    submit = client.post(
        "/api/simulate/run",
        json={
            "patient": {"baseline_age": 72.0},
            "intervention": {"rapamycin_dose": 0.25},
            "sim_years": 2.0,
            "dt": 0.1,
            "async_job": True,
        },
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]
    run_id = submit.json()["run_id"]

    terminal_payload, _seen = _poll_job_until_terminal(client, job_id)
    assert terminal_payload["status"] == "failed"
    assert terminal_payload["error"] is not None

    run_resp = client.get(f"/api/runs/{run_id}")
    assert run_resp.status_code == 200
    assert run_resp.json()["status"] == "failed"
