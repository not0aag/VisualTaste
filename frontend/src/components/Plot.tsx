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

const DEFAULT_LAYOUT: Partial<Plotly.Layout> = {
  margin: { l: 44, r: 12, t: 28, b: 36 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { size: 11 },
  showlegend: true,
  legend: { orientation: 'h', y: -0.2 },
}

export function Plot({ data, layout, className, onHover, testId }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    const element = ref.current
    void loadPlotly().then((Plotly) => {
      if (cancelled || !element) return
      void Plotly.react(element, data, { ...DEFAULT_LAYOUT, ...layout }, {
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
