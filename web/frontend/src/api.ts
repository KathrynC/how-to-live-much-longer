import type {
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
  StoredRunPayload,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`HTTP ${response.status}: ${text || response.statusText}`)
  }

  return response.json() as Promise<T>
}

export async function getMetaParameters(): Promise<MetaParametersResponse> {
  return requestJson<MetaParametersResponse>('/api/meta/parameters')
}

export async function suggestProtocol(payload: LlmSuggestRequest): Promise<LlmSuggestResponse> {
  return requestJson<LlmSuggestResponse>('/api/llm/suggest', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function explainResults(payload: LlmExplainRequest): Promise<LlmExplainResponse> {
  return requestJson<LlmExplainResponse>('/api/llm/explain', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function runSimulation(
  payload: RunSimulationRequest,
): Promise<RunSimulationResponse | JobAcceptedResponse> {
  return requestJson<RunSimulationResponse | JobAcceptedResponse>('/api/simulate/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function compareSimulation(
  payload: CompareRequest,
): Promise<CompareResponse | JobAcceptedResponse> {
  return requestJson<CompareResponse | JobAcceptedResponse>('/api/simulate/compare', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(`/api/jobs/${jobId}`)
}

export async function listRuns(): Promise<RunListResponse> {
  return requestJson<RunListResponse>('/api/runs')
}

export async function getRun(runId: string): Promise<StoredRunPayload> {
  return requestJson<StoredRunPayload>(`/api/runs/${runId}`)
}
