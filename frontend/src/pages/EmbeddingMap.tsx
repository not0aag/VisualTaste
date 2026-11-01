/**
 * Embedding Map: all 50,000 images as points, SigLIP 2 and DINOv2 side by side.
 *
 * Each map is a UMAP projection (backend/visualtaste/viz/projection.py): close
 * on the map means close in that model's embedding, distances between distant
 * clumps mean little, and the axes mean nothing. The two maps show the same
 * images through two lenses: SigLIP groups by what an image depicts, DINOv2
 * by how it looks.
 *
 * WebGL scatter (`scattergl`) so 50,000 points stay interactive. Thumbnails
 * are fetched only for the point under the cursor. The slider walks the
 * preference vector through time, every 50 labels, placed with umap.transform.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { Projection, TastePath } from '../api/types'
import { Plot } from '../components/Plot'

const MODELS = [
  { key: 'siglip2_b16_224', label: 'SigLIP 2: what it depicts' },
  { key: 'dinov2_b14', label: 'DINOv2: how it looks' },
] as const

type ColourMode = 'cluster' | 'source' | 'score'

export function EmbeddingMap() {
  const [colour, setColour] = useState<ColourMode>('cluster')
  const [hovered, setHovered] = useState<string | null>(null)
  const [step, setStep] = useState(Number.MAX_SAFE_INTEGER)
  const [maxStep, setMaxStep] = useState(0)

  return (
    <main className="flex h-full flex-col" data-testid="embedding-map">
      <header className="flex flex-wrap items-center gap-4 px-4 py-3 text-sm sm:px-6">
        <div>
          <p className="eyebrow">Latent space · UMAP</p>
          <h1 className="text-lg font-semibold text-white">Embedding Map</h1>
        </div>
        <label className="flex items-center gap-2 font-mono text-xs uppercase tracking-wider text-canvas-400">
          Colour by{' '}
          <select value={colour} onChange={(e) => setColour(e.target.value as ColourMode)}>
            <option value="cluster">cluster</option>
            <option value="source">source</option>
            <option value="score">current model score</option>
          </select>
        </label>
        {maxStep > 0 && (
          <label className="panel flex items-center gap-3 px-3 py-1.5 font-mono text-xs text-canvas-300">
            Taste over time
            <input type="range" min={1} max={maxStep} value={Math.min(step, maxStep)} onChange={(e) => setStep(Number(e.target.value))} />
            <span className="text-accent">{Math.min(step, maxStep)}/{maxStep}</span>
          </label>
        )}
        {hovered && (
          <img
            // Keyed so each point gets a fresh element; the online demo bundles
            // only some thumbnails, and a missing one hides instead of breaking.
            key={hovered}
            src={api.thumbUrl(hovered)}
            alt=""
            onError={(event) => {
              event.currentTarget.style.visibility = 'hidden'
            }}
            className="ml-auto h-16 w-16 rounded-[3px] object-cover shadow-glow ring-1 ring-accent/50"
            data-testid="hover-thumb"
          />
        )}
      </header>
      <div className="grid min-h-0 flex-1 gap-3 px-3 pb-3 lg:grid-cols-2">
        {MODELS.map((m) => (
          <ModelMap key={m.key} model={m.key} title={m.label} colour={colour} step={step} onSteps={setMaxStep} onHover={setHovered} />
        ))}
      </div>
    </main>
  )
}

function ModelMap({
  model,
  title,
  colour,
  step,
  onSteps,
  onHover,
}: {
  model: string
  title: string
  colour: ColourMode
  step: number
  onSteps: (n: number) => void
  onHover: (imageId: string) => void
}) {
  const [projection, setProjection] = useState<Projection | null>(null)
  const [taste, setTaste] = useState<TastePath | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.projection(model).then(setProjection).catch((cause) => setError(cause instanceof ApiError ? cause.detail : String(cause)))
    api.tastePath(model).then((t) => {
      setTaste(t)
      onSteps(t.path.length)
    }).catch(() => setTaste(null))
  }, [model, onSteps])

  const data = useMemo(() => {
    if (!projection) return []
    const index = new Map(projection.image_ids.map((id, i) => [id, i]))
    const overlay = (ids: string[], name: string, color: string, symbol: string): Plotly.Data => {
      const rows = ids.map((id) => index.get(id)).filter((i): i is number => i !== undefined)
      return {
        x: rows.map((i) => projection.x[i] ?? 0), y: rows.map((i) => projection.y[i] ?? 0),
        type: 'scattergl', mode: 'markers', name, hoverinfo: 'skip',
        marker: { size: 6, color, symbol, line: { width: 1, color: '#05070c' } },
      }
    }
    const sources = [...new Set(projection.source)]
    const values =
      colour === 'score' && projection.score
        ? projection.score
        : colour === 'source'
          ? projection.source.map((s) => sources.indexOf(s))
          : projection.cluster
    const traces: Plotly.Data[] = [
      {
        x: projection.x, y: projection.y, type: 'scattergl', mode: 'markers', name: 'pool',
        hoverinfo: 'none',
        marker: { size: 2, color: values, colorscale: colour === 'score' ? 'Viridis' : 'Portland', opacity: 0.6, showscale: colour === 'score' },
      },
      overlay(projection.liked, 'liked', '#3ddc97', 'circle'),
      overlay(projection.disliked, 'disliked', '#ff6b81', 'x'),
      overlay(projection.saved, 'saved', '#ffc145', 'star'),
    ]
    if (taste && taste.path.length) {
      const shown = taste.path.slice(0, Math.min(step, taste.path.length))
      traces.push({
        x: shown.map((p) => p.x), y: shown.map((p) => p.y), text: shown.map((p) => `${p.labels} labels`),
        type: 'scatter', mode: 'lines+markers', name: 'taste over time', line: { color: '#e6ecf5', width: 2 },
      })
    }
    if (taste && taste.interests.length) {
      traces.push({
        x: taste.interests.map((p) => p.x), y: taste.interests.map((p) => p.y),
        text: taste.interests.map((p) => `interest #${p.labels}`),
        type: 'scatter', mode: 'markers', name: 'interests', marker: { size: 14, symbol: 'diamond', color: '#a594f9', line: { width: 1, color: '#fff' } },
      })
    }
    return traces
  }, [projection, taste, colour, step])

  const hover = useCallback(
    (trace: number, point: number) => {
      const imageId = projection?.image_ids[point]
      if (trace === 0 && imageId) onHover(imageId)
    },
    [projection, onHover],
  )

  const layout = useMemo<Partial<Plotly.Layout>>(
    () => ({ title: { text: title }, xaxis: { visible: false }, yaxis: { visible: false }, hovermode: 'closest', dragmode: 'pan' }),
    [title],
  )

  if (error) return <p className="panel p-4 font-mono text-sm text-canvas-400">{error}</p>
  if (!projection) return <p className="panel flex items-center justify-center p-4 font-mono text-xs uppercase tracking-wider text-canvas-400">Loading {title}…</p>
  return <Plot data={data} layout={layout} onHover={hover} className="panel h-full min-h-[360px] w-full p-1" testId={`map-${model}`} />
}
