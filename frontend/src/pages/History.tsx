/**
 * History: every verdict, newest first, with filters, a verdict changer, and
 * "more like this".
 *
 * Changing a verdict posts a new event; the server marks the old one
 * superseded (api/routes/events.py). Nothing is edited in place, so the log
 * stays a true record of what was decided when, which replay depends on.
 */

import { useCallback, useEffect, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { Action, HistoryResponse, SimilarResponse } from '../api/types'

const FILTERS: (Action | 'all')[] = ['all', 'like', 'save', 'dislike', 'skip']
const PAGE = 48

export function History() {
  const [filter, setFilter] = useState<Action | 'all'>('all')
  const [since, setSince] = useState('')
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<HistoryResponse | null>(null)
  const [similar, setSimilar] = useState<SimilarResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setPage(await api.history({ action: filter, since: since || undefined, limit: PAGE, offset }))
      setError(null)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }, [filter, since, offset])

  useEffect(() => {
    void load()
  }, [load])

  const change = async (imageId: string, action: Action) => {
    try {
      await api.recordEvent({ image_id: imageId, action })
      await load()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }

  return (
    <main className="h-full overflow-y-auto bg-canvas-200 dark:bg-canvas-900" data-testid="history">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 p-4 sm:p-6">
        <header className="flex flex-wrap items-center gap-3 text-sm">
          <h1 className="text-lg font-semibold">History</h1>
          {FILTERS.map((f) => (
            <button key={f} onClick={() => { setFilter(f); setOffset(0) }}
              className={`rounded px-2 py-0.5 ${filter === f ? 'bg-canvas-300 dark:bg-canvas-800' : ''}`}>
              {f}
            </button>
          ))}
          <label className="ml-auto">since <input type="date" value={since} onChange={(e) => { setSince(e.target.value); setOffset(0) }} className="rounded border border-canvas-400 bg-transparent px-1" /></label>
        </header>

        {error && <p className="text-sm text-red-600">{error}</p>}
        {page && page.items.length === 0 && <p className="text-sm text-canvas-500">Nothing here yet.</p>}

        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {page?.items.map((item) => (
            <li key={item.event_id} className="rounded-lg bg-canvas-100 p-2 text-xs dark:bg-canvas-950">
              <img src={item.image.thumb_url} alt={item.image.attribution.title ?? ''} loading="lazy" className="aspect-square w-full rounded object-cover" />
              <div className="mt-1 flex items-center justify-between">
                <select aria-label="verdict" value={item.action} onChange={(e) => change(item.image.image_id, e.target.value as Action)} className="bg-transparent">
                  {(['like', 'save', 'dislike', 'skip'] as Action[]).map((a) => <option key={a} value={a}>{a}</option>)}
                </select>
                <button className="underline" onClick={() => api.similar(item.image.image_id).then(setSimilar)}>more like this</button>
              </div>
              <p className="truncate text-canvas-500" title={item.created_at}>{item.created_at}</p>
            </li>
          ))}
        </ul>

        {page && page.total > PAGE && (
          <nav className="flex items-center gap-3 text-sm">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← newer</button>
            <span className="text-canvas-500">{offset + 1}-{Math.min(offset + PAGE, page.total)} of {page.total}</span>
            <button disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)}>older →</button>
          </nav>
        )}

        {similar && (
          <section aria-label="More like this">
            <h2 className="mb-1 text-sm font-medium">Nearest to that image ({similar.model})</h2>
            <ul className="flex gap-2 overflow-x-auto">
              {similar.neighbours.map((n) => (
                <li key={n.image.image_id} className="shrink-0 text-center text-xs">
                  <img src={n.image.thumb_url} alt="" className="h-24 w-24 rounded object-cover" />
                  <span className="text-canvas-500">{n.similarity.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </main>
  )
}
