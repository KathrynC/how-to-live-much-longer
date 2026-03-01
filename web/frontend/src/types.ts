export type PromptStyle = 'numeric' | 'diegetic' | 'contrastive'

export type ParameterSpec = {
  description: string
  unit: string
  range: [number, number]
  grid: number[]
}

export type MetaParametersResponse = {
  intervention_params: Record<string, ParameterSpec>
  patient_params: Record<string, ParameterSpec>
  models: Array<{ name: string; type: string }>
  prompt_styles: PromptStyle[]
}

export type LlmSuggestRequest = {
  scenario: string
  style: PromptStyle
  model?: string
  temperature: number
  max_tokens: number
  min_intervention_keys: number
}

export type LlmSuggestResponse = {
  vector: Record<string, number> | null
  warnings: string[]
  parse_status: string
  raw_excerpt?: string | null
  raw_response?: string | null
  provider: Record<string, unknown>
}

export type LlmExplainRequest = {
  summary: Record<string, unknown>
  analytics: Record<string, unknown>
  scenario?: string
  model?: string
  temperature: number
  max_tokens: number
}

export type LlmExplainResponse = {
  explanation: string | null
  raw_response?: string | null
  provider: Record<string, unknown>
}

export type RunSimulationRequest = {
  intervention: Record<string, number>
  patient: Record<string, number>
  sim_years: number
  dt: number
  tissue_type?: string
  stochastic: boolean
  noise_scale: number
  n_trajectories: number
  rng_seed?: number
  async_job?: boolean
}

export type RunSimulationResponse = {
  run_id: string
  summary: Record<string, unknown>
  analytics: Record<string, unknown>
  series: Record<string, unknown>
  status: 'completed'
}

export type CompareRequest = {
  baseline: RunSimulationRequest
  candidate: RunSimulationRequest
  async_job?: boolean
}

export type CompareResponse = {
  run_id: string
  baseline: Record<string, unknown>
  candidate: Record<string, unknown>
  delta: Record<string, unknown>
  status: 'completed'
}

export type JobAcceptedResponse = {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  kind: string
  run_id?: string
}

export type JobStatusResponse = {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  kind: string
  run_id?: string
  result?: Record<string, unknown>
  error?: string
}

export type RunIndexItem = {
  run_id: string
  kind: string
  status: string
  created_at: string
  updated_at: string
  summary: Record<string, unknown>
}

export type RunListResponse = {
  runs: RunIndexItem[]
}

export type StoredRunPayload = {
  run_id: string
  kind: string
  status: string
  created_at?: string
  summary?: Record<string, unknown>
  analytics?: Record<string, unknown>
  series?: Record<string, unknown>
  delta?: Record<string, unknown>
  baseline?: Record<string, unknown>
  candidate?: Record<string, unknown>
  [k: string]: unknown
}

export const INTERVENTION_KEYS = [
  'rapamycin_dose',
  'nad_supplement',
  'senolytic_dose',
  'yamanaka_intensity',
  'transplant_rate',
  'exercise_level',
]

export const PATIENT_KEYS = [
  'baseline_age',
  'baseline_heteroplasmy',
  'baseline_nad_level',
  'genetic_vulnerability',
  'metabolic_demand',
  'inflammation_level',
]

export const DEFAULT_INTERVENTION: Record<string, number> = {
  rapamycin_dose: 0,
  nad_supplement: 0,
  senolytic_dose: 0,
  yamanaka_intensity: 0,
  transplant_rate: 0,
  exercise_level: 0,
}

export const DEFAULT_PATIENT: Record<string, number> = {
  baseline_age: 70,
  baseline_heteroplasmy: 0.3,
  baseline_nad_level: 0.6,
  genetic_vulnerability: 1,
  metabolic_demand: 1,
  inflammation_level: 0.25,
}
