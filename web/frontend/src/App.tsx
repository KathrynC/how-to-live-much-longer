import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'

import {
  compareSimulation,
  explainResults,
  getJob,
  getMetaParameters,
  getRun,
  listRuns,
  runSimulation,
  suggestProtocol,
} from './api'
import { LineChart } from './components/LineChart'
import {
  DEFAULT_INTERVENTION,
  DEFAULT_PATIENT,
  INTERVENTION_KEYS,
  PATIENT_KEYS,
  type JobAcceptedResponse,
  type JobStatusResponse,
  type ParameterSpec,
  type PromptStyle,
  type StoredRunPayload,
} from './types'

function keyLabel(key: string): string {
  return key
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function stepFromSpec(spec: ParameterSpec): number {
  if (spec.grid.length > 1) {
    let minDiff = Number.POSITIVE_INFINITY
    for (let i = 1; i < spec.grid.length; i += 1) {
      minDiff = Math.min(minDiff, Math.abs(spec.grid[i] - spec.grid[i - 1]))
    }
    if (Number.isFinite(minDiff) && minDiff > 0) {
      return minDiff
    }
  }
  const [lo, hi] = spec.range
  return Math.max((hi - lo) / 100, 0.01)
}

function asNumberArray(raw: unknown): number[] {
  if (!Array.isArray(raw)) {
    return []
  }
  return raw
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
}

function isJobResponse(value: unknown): value is JobAcceptedResponse {
  return !!value && typeof value === 'object' && 'job_id' in value
}

type ParamEditorProps = {
  title: string
  specs: Record<string, ParameterSpec>
  values: Record<string, number>
  onChange: (key: string, value: number) => void
}

function ParameterEditor({ title, specs, values, onChange }: ParamEditorProps) {
  const orderedEntries = useMemo(
    () =>
      Object.entries(specs).sort((a, b) => {
        const aIdx = [...INTERVENTION_KEYS, ...PATIENT_KEYS].indexOf(a[0])
        const bIdx = [...INTERVENTION_KEYS, ...PATIENT_KEYS].indexOf(b[0])
        return aIdx - bIdx
      }),
    [specs],
  )

  return (
    <section className="panel">
      <h3>{title}</h3>
      <div className="param-grid">
        {orderedEntries.map(([key, spec]) => {
          const current = values[key] ?? spec.grid[0] ?? spec.range[0]
          const [min, max] = spec.range
          const step = stepFromSpec(spec)
          return (
            <label key={key} className="param-card">
              <span className="param-title">{keyLabel(key)}</span>
              <span className="param-meta">{spec.description}</span>
              <div className="param-inputs">
                <input
                  data-testid={`range-${key}`}
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={current}
                  onChange={(event) => onChange(key, Number(event.target.value))}
                />
                <input
                  data-testid={`value-${key}`}
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={Number.isFinite(current) ? current : min}
                  onChange={(event) => onChange(key, Number(event.target.value))}
                />
              </div>
            </label>
          )
        })}
      </div>
    </section>
  )
}

function RunSummaryCards({ summary }: { summary?: Record<string, unknown> }) {
  const preferredKeys = [
    'final_atp',
    'final_heteroplasmy',
    'final_deletion_heteroplasmy',
    'time_to_cliff_years',
    'time_to_crisis_years',
  ]

  if (!summary) {
    return <div className="placeholder">Run a simulation to see summary metrics.</div>
  }

  return (
    <div className="metric-grid">
      {preferredKeys.map((key) => (
        <article key={key} className="metric-tile">
          <h4>{keyLabel(key)}</h4>
          <p>{summary[key] !== undefined && summary[key] !== null ? String(summary[key]) : 'N/A'}</p>
        </article>
      ))}
    </div>
  )
}

function DeltaTiles({ delta }: { delta?: Record<string, unknown> }) {
  if (!delta) {
    return null
  }

  const entries = Object.entries(delta)
  if (!entries.length) {
    return null
  }

  return (
    <div className="metric-grid">
      {entries.map(([key, value]) => (
        <article key={key} className="metric-tile">
          <h4>{keyLabel(key)}</h4>
          <p>{value === null || value === undefined ? 'N/A' : String(value)}</p>
        </article>
      ))}
    </div>
  )
}

export default function App() {
  const [intervention, setIntervention] = useState<Record<string, number>>({ ...DEFAULT_INTERVENTION })
  const [patient, setPatient] = useState<Record<string, number>>({ ...DEFAULT_PATIENT })
  const [scenario, setScenario] = useState('')
  const [promptStyle, setPromptStyle] = useState<PromptStyle>('numeric')
  const [model, setModel] = useState('')
  const [simYears, setSimYears] = useState(30)
  const [dt, setDt] = useState(0.01)
  const [stochastic, setStochastic] = useState(false)
  const [noiseScale, setNoiseScale] = useState(0.01)
  const [nTrajectories, setNTrajectories] = useState(1)

  const [activeRun, setActiveRun] = useState<StoredRunPayload | null>(null)
  const [comparisonRun, setComparisonRun] = useState<StoredRunPayload | null>(null)
  const [assistantWarnings, setAssistantWarnings] = useState<string[]>([])
  const [assistantStatus, setAssistantStatus] = useState<string>('idle')
  const [explanation, setExplanation] = useState<string>('')
  const [errorText, setErrorText] = useState('')

  const [historyOpen, setHistoryOpen] = useState(false)
  const [runJobId, setRunJobId] = useState<string | null>(null)
  const [compareJobId, setCompareJobId] = useState<string | null>(null)

  const metaQuery = useQuery({
    queryKey: ['meta-parameters'],
    queryFn: getMetaParameters,
  })

  const runsQuery = useQuery({
    queryKey: ['run-history'],
    queryFn: listRuns,
    refetchInterval: 4000,
  })

  const runJobQuery = useQuery({
    queryKey: ['job-status', runJobId],
    queryFn: () => getJob(runJobId as string),
    enabled: !!runJobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'completed' || status === 'failed' ? false : 1000
    },
  })

  const compareJobQuery = useQuery({
    queryKey: ['compare-job-status', compareJobId],
    queryFn: () => getJob(compareJobId as string),
    enabled: !!compareJobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'completed' || status === 'failed' ? false : 1000
    },
  })

  useEffect(() => {
    const job = runJobQuery.data
    if (!job) {
      return
    }
    if (job.status === 'completed' && job.run_id) {
      void getRun(job.run_id).then((payload) => {
        setActiveRun(payload)
        setComparisonRun(null)
        setRunJobId(null)
        setErrorText('')
        void runsQuery.refetch()
      })
    }
    if (job.status === 'failed') {
      setErrorText(job.error ?? 'Simulation job failed')
      setRunJobId(null)
    }
  }, [runJobQuery.data, runsQuery.refetch])

  useEffect(() => {
    const job = compareJobQuery.data
    if (!job) {
      return
    }
    if (job.status === 'completed' && job.run_id) {
      void getRun(job.run_id).then((payload) => {
        setComparisonRun(payload)
        setCompareJobId(null)
        setErrorText('')
        void runsQuery.refetch()
      })
    }
    if (job.status === 'failed') {
      setErrorText(job.error ?? 'Comparison job failed')
      setCompareJobId(null)
    }
  }, [compareJobQuery.data, runsQuery.refetch])

  const suggestMutation = useMutation({
    mutationFn: () =>
      suggestProtocol({
        scenario,
        style: promptStyle,
        model: model || undefined,
        temperature: 0.7,
        max_tokens: 800,
        min_intervention_keys: 4,
      }),
    onSuccess: (payload) => {
      setAssistantStatus(payload.parse_status)
      setAssistantWarnings(payload.warnings)
      if (!payload.vector) {
        return
      }
      const vector = payload.vector
      setIntervention((prev) => {
        const next = { ...prev }
        for (const key of INTERVENTION_KEYS) {
          if (typeof vector[key] === 'number') {
            next[key] = vector[key]
          }
        }
        return next
      })
      setPatient((prev) => {
        const next = { ...prev }
        for (const key of PATIENT_KEYS) {
          if (typeof vector[key] === 'number') {
            next[key] = vector[key]
          }
        }
        return next
      })
    },
    onError: (error: Error) => {
      setAssistantWarnings([error.message])
      setAssistantStatus('error')
    },
  })

  const explainMutation = useMutation({
    mutationFn: () =>
      explainResults({
        scenario,
        summary: (activeRun?.summary as Record<string, unknown>) ?? {},
        analytics: (activeRun?.analytics as Record<string, unknown>) ?? {},
        model: model || undefined,
        temperature: 0.3,
        max_tokens: 500,
      }),
    onSuccess: (payload) => {
      setExplanation(payload.explanation ?? '')
      if (!payload.explanation) {
        setAssistantWarnings(['LLM explain returned no text.'])
      }
    },
    onError: (error: Error) => {
      setAssistantWarnings([error.message])
    },
  })

  const runMutation = useMutation({
    mutationFn: () =>
      runSimulation({
        intervention,
        patient,
        sim_years: simYears,
        dt,
        stochastic,
        noise_scale: noiseScale,
        n_trajectories: nTrajectories,
        async_job: true,
      }),
    onSuccess: (payload) => {
      if (isJobResponse(payload)) {
        setRunJobId(payload.job_id)
        return
      }
      setActiveRun({
        run_id: payload.run_id,
        kind: 'simulation',
        status: payload.status,
        summary: payload.summary,
        analytics: payload.analytics,
        series: payload.series,
      })
      setComparisonRun(null)
      void runsQuery.refetch()
    },
    onError: (error: Error) => {
      setErrorText(error.message)
    },
  })

  const compareMutation = useMutation({
    mutationFn: () =>
      compareSimulation({
        baseline: {
          intervention: { ...DEFAULT_INTERVENTION },
          patient,
          sim_years: simYears,
          dt,
          stochastic,
          noise_scale: noiseScale,
          n_trajectories: nTrajectories,
        },
        candidate: {
          intervention,
          patient,
          sim_years: simYears,
          dt,
          stochastic,
          noise_scale: noiseScale,
          n_trajectories: nTrajectories,
        },
        async_job: true,
      }),
    onSuccess: (payload) => {
      if (isJobResponse(payload)) {
        setCompareJobId(payload.job_id)
        return
      }
      setComparisonRun({
        run_id: payload.run_id,
        kind: 'comparison',
        status: payload.status,
        baseline: payload.baseline,
        candidate: payload.candidate,
        delta: payload.delta,
      })
      void runsQuery.refetch()
    },
    onError: (error: Error) => {
      setErrorText(error.message)
    },
  })

  const loadRun = async (runId: string) => {
    const payload = await getRun(runId)
    if (payload.kind === 'comparison') {
      setComparisonRun(payload)
    } else {
      setActiveRun(payload)
      setComparisonRun(null)
    }
    setHistoryOpen(false)
  }

  const modelOptions = metaQuery.data?.models ?? []
  const styles = metaQuery.data?.prompt_styles ?? ['numeric', 'diegetic', 'contrastive']

  const time = asNumberArray(activeRun?.series?.time)
  const atp = asNumberArray(activeRun?.series?.atp)
  const het = asNumberArray(activeRun?.series?.heteroplasmy)
  const delHet = asNumberArray(activeRun?.series?.deletion_heteroplasmy)

  const overlay = (comparisonRun?.series_overlay ?? {}) as Record<string, unknown>
  const overlayTime = asNumberArray(overlay.time)

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>Simulation Workbench</h1>
          <p>Form-first mitochondrial simulation with AI-assisted drafting.</p>
        </div>
        <button className="history-toggle" onClick={() => setHistoryOpen((open) => !open)}>
          {historyOpen ? 'Close History' : 'Open History'}
        </button>
      </header>

      {errorText ? <div className="error-banner">{errorText}</div> : null}

      <main className="layout">
        <section className="left-column">
          <section className="panel">
            <h3>Run Controls</h3>
            <div className="inline-grid">
              <label>
                <span>Simulation Years</span>
                <input
                  type="number"
                  min={1}
                  max={120}
                  step={1}
                  value={simYears}
                  onChange={(event) => setSimYears(Number(event.target.value))}
                />
              </label>
              <label>
                <span>dt</span>
                <input
                  type="number"
                  min={0.001}
                  max={1}
                  step={0.001}
                  value={dt}
                  onChange={(event) => setDt(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Stochastic</span>
                <input
                  type="checkbox"
                  checked={stochastic}
                  onChange={(event) => setStochastic(event.target.checked)}
                />
              </label>
              <label>
                <span>Noise Scale</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.001}
                  value={noiseScale}
                  onChange={(event) => setNoiseScale(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Trajectories</span>
                <input
                  type="number"
                  min={1}
                  max={256}
                  step={1}
                  value={nTrajectories}
                  onChange={(event) => setNTrajectories(Number(event.target.value))}
                />
              </label>
            </div>
            <div className="button-row">
              <button
                data-testid="run-button"
                onClick={() => runMutation.mutate()}
                disabled={runMutation.isPending || !!runJobId}
              >
                {runMutation.isPending || runJobId ? 'Running...' : 'Run Simulation'}
              </button>
              <button
                onClick={() => compareMutation.mutate()}
                disabled={compareMutation.isPending || !!compareJobId}
              >
                {compareMutation.isPending || compareJobId ? 'Comparing...' : 'Compare vs Baseline'}
              </button>
            </div>
          </section>

          {metaQuery.data ? (
            <>
              <ParameterEditor
                title="Intervention Parameters"
                specs={metaQuery.data.intervention_params}
                values={intervention}
                onChange={(key, value) => setIntervention((prev) => ({ ...prev, [key]: value }))}
              />
              <ParameterEditor
                title="Patient Parameters"
                specs={metaQuery.data.patient_params}
                values={patient}
                onChange={(key, value) => setPatient((prev) => ({ ...prev, [key]: value }))}
              />
            </>
          ) : (
            <section className="panel">Loading parameter metadata...</section>
          )}

          <section className="panel ai-panel">
            <h3>AI Assist</h3>
            <label>
              <span>Scenario</span>
              <textarea
                value={scenario}
                onChange={(event) => setScenario(event.target.value)}
                placeholder="Describe the clinical scenario for protocol suggestion."
              />
            </label>
            <div className="inline-grid">
              <label>
                <span>Prompt Style</span>
                <select value={promptStyle} onChange={(event) => setPromptStyle(event.target.value as PromptStyle)}>
                  {styles.map((style) => (
                    <option key={style} value={style}>
                      {style}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Model</span>
                <select value={model} onChange={(event) => setModel(event.target.value)}>
                  <option value="">Default</option>
                  {modelOptions.map((entry) => (
                    <option key={entry.name} value={entry.name}>
                      {entry.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="button-row">
              <button
                data-testid="suggest-button"
                onClick={() => suggestMutation.mutate()}
                disabled={suggestMutation.isPending || !scenario.trim()}
              >
                {suggestMutation.isPending ? 'Suggesting...' : 'Suggest Protocol'}
              </button>
              <button
                onClick={() => explainMutation.mutate()}
                disabled={explainMutation.isPending || !activeRun}
              >
                {explainMutation.isPending ? 'Explaining...' : 'Explain Results'}
              </button>
            </div>

            <p className="status-line" data-testid="assistant-status">
              Parse status: <strong>{assistantStatus}</strong>
            </p>
            {assistantWarnings.length ? (
              <ul className="warning-list">
                {assistantWarnings.map((warning, idx) => (
                  <li key={`${warning}-${idx}`}>{warning}</li>
                ))}
              </ul>
            ) : null}
            {explanation ? <p className="explanation">{explanation}</p> : null}
          </section>
        </section>

        <section className="right-column">
          <section className="panel">
            <h3>Latest Simulation Summary</h3>
            <RunSummaryCards summary={activeRun?.summary as Record<string, unknown> | undefined} />
          </section>

          <section className="charts-grid">
            <LineChart
              title="ATP Trajectory"
              time={time}
              series={[{ name: 'ATP', color: '#05668d', values: atp }]}
            />
            <LineChart
              title="Heteroplasmy Trajectory"
              time={time}
              series={[
                { name: 'Total Het', color: '#f25c54', values: het },
                { name: 'Deletion Het', color: '#f7b267', values: delHet },
              ]}
            />
          </section>

          <section className="panel">
            <h3>Baseline vs Candidate Delta</h3>
            <DeltaTiles delta={comparisonRun?.delta as Record<string, unknown> | undefined} />
            <LineChart
              title="ATP Overlay"
              time={overlayTime}
              series={[
                {
                  name: 'Baseline ATP',
                  color: '#364156',
                  values: asNumberArray(overlay.baseline_atp),
                },
                {
                  name: 'Candidate ATP',
                  color: '#3da35d',
                  values: asNumberArray(overlay.candidate_atp),
                },
              ]}
            />
          </section>
        </section>
      </main>

      <aside className={`history-drawer ${historyOpen ? 'open' : ''}`}>
        <h3>Run History</h3>
        <p>Reopen saved run payloads.</p>
        <div className="history-list">
          {(runsQuery.data?.runs ?? []).map((run) => (
            <button key={run.run_id} className="history-item" onClick={() => void loadRun(run.run_id)}>
              <strong>{run.kind}</strong>
              <span>{run.status}</span>
              <small>{run.updated_at}</small>
            </button>
          ))}
          {!runsQuery.data?.runs.length ? <div className="placeholder">No saved runs yet.</div> : null}
        </div>
      </aside>
    </div>
  )
}
