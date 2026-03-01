# Web Workbench (V1)

## Scope
This workbench is a local, single-user interface for mitochondrial simulation and LLM-assisted protocol drafting.

- Backend: FastAPI (`web/backend`)
- Frontend: React + TypeScript + Vite (`web/frontend`)
- Persistence: file artifacts under `output/web_runs`
- LLM transport: Ollama-first via typed Python HTTP provider (`llm_provider.py`)

CA/Lakoff panels are intentionally deferred from V1 UI.

## Architecture

### Backend modules

- `web/backend/app.py`
  - API routes, async job manager, CORS, run persistence integration.
- `web/backend/schemas.py`
  - Pydantic request/response contracts.
- `web/backend/config.py`
  - Runtime settings (`WEB_RUNS_ROOT`, `OLLAMA_URL`).
- `web/backend/services/simulation_service.py`
  - Simulation orchestration (`simulate`, `compute_all`) and artifact shaping.
- `web/backend/services/llm_service.py`
  - Prompt wiring, model calls (`llm_common`), and health checks.
- `web/backend/services/history_store.py`
  - Run payload persistence + metadata index.

### Frontend modules

- `web/frontend/src/App.tsx`
  - Form-based workbench, AI assist actions, result charts, run history replay.
- `web/frontend/src/api.ts`
  - Typed API client.
- `web/frontend/src/components/LineChart.tsx`
  - Lightweight SVG chart component.

## API Contracts

### Metadata
- `GET /api/meta/parameters`
  - Returns intervention specs, patient specs, prompt styles, model catalog.
- `GET /api/meta/health`
  - Runtime Ollama reachability + model list from `/api/tags`.

### LLM
- `POST /api/llm/suggest`
  - Input: scenario/style/model.
  - Output: parsed vector, parse status, warnings, provider metadata.
- `POST /api/llm/explain`
  - Input: summary/analytics/scenario.
  - Output: short natural-language interpretation.

### Simulation
- `POST /api/simulate/run`
  - Input: patient/intervention/settings.
  - Returns completed payload or async job acceptance.
- `POST /api/simulate/compare`
  - Input: baseline and candidate simulation requests.
  - Returns completed delta payload or async job acceptance.

### Jobs and History
- `GET /api/jobs/{job_id}`
  - Job status (`pending`, `running`, `completed`, `failed`).
- `GET /api/runs`
  - Run index list.
- `GET /api/runs/{run_id}`
  - Full saved run payload.

## Run History Files

Stored under `output/web_runs/`:

- `index.json`
  - ordered metadata list (run id, kind, status, timestamps, summary).
- `runs/{run_id}.json`
  - full run payload (request, summary, analytics, series, provider metadata).

## Local Setup

1. Backend:
   - `make web-backend`
2. Frontend:
   - `npm --prefix web/frontend install`
   - `make web-frontend`
3. Open:
   - [http://127.0.0.1:5173](http://127.0.0.1:5173)

## Ollama Defaults and Overrides

- Default generate endpoint comes from `constants.OLLAMA_URL`.
- Override with environment variable:
  - `OLLAMA_URL=http://localhost:11434/api/generate`
- Optional output root override:
  - `WEB_RUNS_ROOT=/path/to/web_runs`

## Test Entry Points

- Backend API tests:
  - `make web-test-backend`
- Frontend tests:
  - `make web-test-frontend`
