import React from 'react'

type Series = {
  name: string
  color: string
  values: number[]
}

type LineChartProps = {
  title: string
  time: number[]
  series: Series[]
}

function buildPath(values: number[], time: number[], yMin: number, yMax: number): string {
  if (!values.length || values.length !== time.length) {
    return ''
  }

  const xMin = time[0]
  const xMax = time[time.length - 1]
  const xSpan = Math.max(xMax - xMin, 1e-6)
  const ySpan = Math.max(yMax - yMin, 1e-6)

  return values
    .map((value, idx) => {
      const x = ((time[idx] - xMin) / xSpan) * 560 + 24
      const y = 196 - ((value - yMin) / ySpan) * 160
      return `${idx === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}

export function LineChart({ title, time, series }: LineChartProps) {
  const combined = series.flatMap((s) => s.values)
  if (!time.length || !combined.length) {
    return (
      <section className="chart-card">
        <h3>{title}</h3>
        <div className="chart-empty">No data.</div>
      </section>
    )
  }

  const yMin = Math.min(...combined)
  const yMax = Math.max(...combined)

  return (
    <section className="chart-card">
      <h3>{title}</h3>
      <svg viewBox="0 0 600 220" role="img" aria-label={title}>
        <rect x="24" y="20" width="560" height="176" className="chart-bg" />
        <line x1="24" y1="196" x2="584" y2="196" className="chart-axis" />
        <line x1="24" y1="20" x2="24" y2="196" className="chart-axis" />
        {series.map((entry) => {
          const path = buildPath(entry.values, time, yMin, yMax)
          return (
            <path
              key={entry.name}
              d={path}
              className="chart-line"
              style={{ stroke: entry.color }}
              fill="none"
              strokeWidth="2.5"
            />
          )
        })}
      </svg>
      <div className="chart-legend">
        {series.map((entry) => (
          <span key={entry.name}>
            <i style={{ backgroundColor: entry.color }} />
            {entry.name}
          </span>
        ))}
      </div>
    </section>
  )
}
