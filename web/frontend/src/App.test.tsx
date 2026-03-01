import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, test, vi } from 'vitest'

import App from './App'

const INTERVENTION_KEYS = [
  'rapamycin_dose',
  'nad_supplement',
  'senolytic_dose',
  'yamanaka_intensity',
  'transplant_rate',
  'exercise_level',
]

const PATIENT_KEYS = [
  'baseline_age',
  'baseline_heteroplasmy',
  'baseline_nad_level',
  'genetic_vulnerability',
  'metabolic_demand',
  'inflammation_level',
]

function spec(range: [number, number], grid: number[]) {
  return {
    description: 'spec',
    unit: 'unit',
    range,
    grid,
  }
}

const META_FIXTURE = {
  intervention_params: Object.fromEntries(
    INTERVENTION_KEYS.map((key) => [key, spec([0, 1], [0, 0.1, 0.25, 0.5, 0.75, 1])]),
  ),
  patient_params: {
    baseline_age: spec([20, 90], [20, 30, 40, 50, 60, 70, 80, 90]),
    baseline_heteroplasmy: spec([0, 0.95], [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
    baseline_nad_level: spec([0.2, 1], [0.2, 0.4, 0.6, 0.8, 1]),
    genetic_vulnerability: spec([0.5, 2], [0.5, 0.75, 1, 1.5, 2]),
    metabolic_demand: spec([0.5, 2], [0.5, 0.75, 1, 1.5, 2]),
    inflammation_level: spec([0, 1], [0, 0.1, 0.25, 0.5, 0.75, 1]),
  },
  models: [{ name: 'fake:model', type: 'ollama' }],
  prompt_styles: ['numeric', 'diegetic', 'contrastive'],
}

function jsonResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  }
}

function renderApp() {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Simulation Workbench UI', () => {
  test('hydrates form state from AI suggestion', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/api/meta/parameters')) {
          return jsonResponse(META_FIXTURE)
        }
        if (url.endsWith('/api/runs')) {
          return jsonResponse({ runs: [] })
        }
        if (url.endsWith('/api/llm/suggest')) {
          return jsonResponse({
            vector: { rapamycin_dose: 0.75, baseline_age: 80 },
            warnings: [],
            parse_status: 'ok',
            raw_excerpt: '{}',
            raw_response: '{}',
            provider: { ok: true },
          })
        }
        throw new Error(`unhandled route ${url}`)
      }),
    )

    renderApp()

    await screen.findByTestId('value-rapamycin_dose')

    fireEvent.change(screen.getByPlaceholderText(/describe the clinical scenario/i), {
      target: { value: '80-year-old near cliff' },
    })
    await userEvent.click(screen.getByTestId('suggest-button'))

    await waitFor(() => {
      expect(screen.getByTestId('value-rapamycin_dose')).toHaveValue(0.75)
      expect(screen.getByTestId('value-baseline_age')).toHaveValue(80)
    })
  })

  test('does not auto-run simulation after suggestion', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/meta/parameters')) {
        return jsonResponse(META_FIXTURE)
      }
      if (url.endsWith('/api/runs')) {
        return jsonResponse({ runs: [] })
      }
      if (url.endsWith('/api/llm/suggest')) {
        return jsonResponse({
          vector: { rapamycin_dose: 0.5 },
          warnings: [],
          parse_status: 'ok',
          provider: { ok: true },
        })
      }
      if (url.endsWith('/api/simulate/run')) {
        return jsonResponse({ error: 'should not be called' }, 500)
      }
      throw new Error(`unhandled route ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()

    await screen.findByTestId('value-rapamycin_dose')
    fireEvent.change(screen.getByPlaceholderText(/describe the clinical scenario/i), {
      target: { value: 'protocol suggestion only' },
    })
    await userEvent.click(screen.getByTestId('suggest-button'))

    await waitFor(() => {
      const runCalls = fetchMock.mock.calls.filter((call) => String(call[0]).endsWith('/api/simulate/run'))
      expect(runCalls.length).toBe(0)
    })
  })

  test('renders ATP chart after run completion', async () => {
    let jobChecks = 0

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/api/meta/parameters')) {
          return jsonResponse(META_FIXTURE)
        }
        if (url.endsWith('/api/runs')) {
          return jsonResponse({
            runs: [
              {
                run_id: 'run-1',
                kind: 'simulation',
                status: 'completed',
                created_at: '2026-01-01T00:00:00Z',
                updated_at: '2026-01-01T00:00:00Z',
                summary: { final_atp: 0.88 },
              },
            ],
          })
        }
        if (url.endsWith('/api/simulate/run')) {
          return jsonResponse({ job_id: 'job-run', status: 'pending', kind: 'simulation', run_id: 'run-1' })
        }
        if (url.endsWith('/api/jobs/job-run')) {
          jobChecks += 1
          if (jobChecks < 2) {
            return jsonResponse({ job_id: 'job-run', status: 'running', kind: 'simulation', run_id: 'run-1' })
          }
          return jsonResponse({ job_id: 'job-run', status: 'completed', kind: 'simulation', run_id: 'run-1' })
        }
        if (url.endsWith('/api/runs/run-1')) {
          return jsonResponse({
            run_id: 'run-1',
            kind: 'simulation',
            status: 'completed',
            summary: {
              final_atp: 0.88,
              final_heteroplasmy: 0.22,
            },
            analytics: {},
            series: {
              time: [0, 1, 2],
              atp: [0.9, 0.89, 0.88],
              heteroplasmy: [0.2, 0.21, 0.22],
              deletion_heteroplasmy: [0.18, 0.19, 0.2],
            },
          })
        }
        throw new Error(`unhandled route ${url}`)
      }),
    )

    renderApp()

    await screen.findByTestId('run-button')
    await userEvent.click(screen.getByTestId('run-button'))

    await screen.findByText('0.88')

    const chart = await screen.findByLabelText('ATP Trajectory')
    expect(chart.querySelectorAll('path.chart-line').length).toBeGreaterThan(0)
  })

  test('replays run from history drawer', async () => {
    let runReadCount = 0
    let jobChecks = 0

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/api/meta/parameters')) {
          return jsonResponse(META_FIXTURE)
        }
        if (url.endsWith('/api/runs')) {
          return jsonResponse({
            runs: [
              {
                run_id: 'run-1',
                kind: 'simulation',
                status: 'completed',
                created_at: '2026-01-01T00:00:00Z',
                updated_at: '2026-01-01T00:00:00Z',
                summary: { final_atp: 0.88 },
              },
            ],
          })
        }
        if (url.endsWith('/api/simulate/run')) {
          return jsonResponse({ job_id: 'job-run', status: 'pending', kind: 'simulation', run_id: 'run-1' })
        }
        if (url.endsWith('/api/jobs/job-run')) {
          jobChecks += 1
          if (jobChecks < 2) {
            return jsonResponse({ job_id: 'job-run', status: 'running', kind: 'simulation', run_id: 'run-1' })
          }
          return jsonResponse({ job_id: 'job-run', status: 'completed', kind: 'simulation', run_id: 'run-1' })
        }
        if (url.endsWith('/api/runs/run-1')) {
          runReadCount += 1
          if (runReadCount === 1) {
            return jsonResponse({
              run_id: 'run-1',
              kind: 'simulation',
              status: 'completed',
              summary: { final_atp: 0.88, final_heteroplasmy: 0.22 },
              analytics: {},
              series: { time: [0, 1], atp: [0.9, 0.88], heteroplasmy: [0.2, 0.22], deletion_heteroplasmy: [0.18, 0.2] },
            })
          }
          return jsonResponse({
            run_id: 'run-1',
            kind: 'simulation',
            status: 'completed',
            summary: { final_atp: 0.93, final_heteroplasmy: 0.18 },
            analytics: {},
            series: { time: [0, 1], atp: [0.94, 0.93], heteroplasmy: [0.2, 0.18], deletion_heteroplasmy: [0.18, 0.16] },
          })
        }
        throw new Error(`unhandled route ${url}`)
      }),
    )

    renderApp()

    await screen.findByTestId('run-button')
    await userEvent.click(screen.getByTestId('run-button'))
    await screen.findByText('0.88')

    await userEvent.click(screen.getByText('Open History'))
    await screen.findByText('simulation')
    await userEvent.click(screen.getByText('simulation'))

    await screen.findByText('0.93')
  })
})
