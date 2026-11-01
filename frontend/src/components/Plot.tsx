/**
 * A thin Plotly wrapper: `Plotly.react` into a div whenever the data changes.
 *
 * Plotly is loaded with a dynamic `import()`, so its ~3 MB bundle is fetched
 * only by the pages that chart something (Model Lab, Embedding Map) and never
 * delays the Feed. Direct use rather than `react-plotly.js`: one effect is all
 * the integration this needs, and it saves a dependency.
 */

import { useEffect, useRef } from 'react'

type PlotlyModule = typeof import('plotly.js-dist-min')

let plotly: Promise<PlotlyModule> | null = null
const loadPlotly = () => (plotly ??= import('plotly.js-dist-min'))

export interface PlotProps {
  data: Plotly.Data[]
  layout?: Partial<Plotly.Layout>
  className?: string
  /** Called with the point index under the cursor (per trace), for hover thumbnails. */
  onHover?: (traceIndex: number, pointIndex: number) => void
  testId?: string
}

const AXIS: Partial<Plotly.LayoutAxis> = {
  gridcolor: 'rgba(148,163,184,0.10)',
  zerolinecolor: 'rgba(148,163,184,0.25)',
  linecolor: 'rgba(148,163,184,0.25)',
  tickfont: { color: '#8494ad' },
}

/** Dark HUD theme: transparent paper, faint grid, neon series colours. */
const DEFAULT_LAYOUT: Partial<Plotly.Layout> = {
  margin: { l: 44, r: 12, t: 36, b: 56 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { size: 11, color: '#aab6c8', family: '"Geist Mono Variable", ui-monospace, monospace' },
  colorway: ['#3fd8f0', '#a594f9', '#3ddc97', '#ffc145', '#ff6b81', '#60a5fa'],
  hoverlabel: { bgcolor: '#0a0e17', bordercolor: '#3fd8f0', font: { color: '#e6ecf5' } },
  showlegend: true,
  legend: { orientation: 'h', y: -0.22, yanchor: 'top', font: { color: '#8494ad' } },
}

/** The page's layout over the theme; axes are merged so the grid styling survives. */
function themed(layout: Partial<Plotly.Layout> | undefined): Partial<Plotly.Layout> {
  const merged: Record<string, unknown> = { ...DEFAULT_LAYOUT, ...layout }
  for (const axis of ['xaxis', 'yaxis', 'yaxis2'] as const) {
    const own = layout?.[axis] as Record<string, unknown> | undefined
    if (axis === 'yaxis2' && !own) continue
    merged[axis] = { ...AXIS, ...own }
  }
  // A right-hand axis needs room for its ticks and title.
  if (layout?.yaxis2 && !layout.margin) merged.margin = { ...DEFAULT_LAYOUT.margin, r: 56 }
  return merged as Partial<Plotly.Layout>
}

export function Plot({ data, layout, className, onHover, testId }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    const element = ref.current
    void loadPlotly().then((Plotly) => {
      if (cancelled || !element) return
      void Plotly.react(element, data, themed(layout), {
        displaylogo: false,
        responsive: true,
      })
      if (onHover) {
        const target = element as unknown as Plotly.PlotlyHTMLElement
        target.removeAllListeners?.('plotly_hover')
        target.on('plotly_hover', (event) => {
          const point = event.points[0]
          if (point) onHover(point.curveNumber, point.pointIndex)
        })
      }
    })
    return () => {
      cancelled = true
    }
  }, [data, layout, onHover])

  useEffect(() => {
    const element = ref.current
    return () => {
      if (element) void loadPlotly().then((Plotly) => Plotly.purge(element))
    }
  }, [])

  return <div ref={ref} className={className} data-testid={testId} />
}
