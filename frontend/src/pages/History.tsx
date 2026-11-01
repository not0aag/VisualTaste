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

const VERDICT_TONE: Record<Action, string> = {
  like: 'text-like',
  save: 'text-save',
  dislike: 'text-dislike',
  skip: 'text-canvas-400',
}

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
    <main className="page" data-testid="history">
      <div className="page-inner">
        <div>
          <p className="eyebrow">Event log</p>
          <h1 className="page-title mt-1">History</h1>
        </div>
        <header className="panel flex flex-wrap items-center gap-2 p-2 text-sm">
          {FILTERS.map((f) => (
            <button key={f} onClick={() => { setFilter(f); setOffset(0) }}
              className={`rounded-md px-3 py-1 font-mono text-xs uppercase tracking-wider transition-colors ${filter === f ? 'bg-accent/15 text-white shadow-[inset_0_0_0_1px_rgba(34,211,238,0.4)]' : 'text-canvas-400 hover:bg-white/[0.04] hover:text-white'}`}>
              {f}
            </button>
          ))}
          <label className="ml-auto flex items-center gap-2 pr-1 font-mono text-xs text-canvas-400">since <input type="date" value={since} onChange={(e) => { setSince(e.target.value); setOffset(0) }} /></label>
        </header>

        {error && <p className="font-mono text-sm text-dislike">{error}</p>}
        {page && page.items.length === 0 && <p className="panel px-4 py-8 text-center font-mono text-sm text-canvas-500">Nothing here yet.</p>}

        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {page?.items.map((item) => (
            <li key={item.event_id} className="tile group animate-fade-up">
              <img src={item.image.thumb_url} alt={item.image.attribution.title ?? ''} loading="lazy" className="thumb group-hover:brightness-110" />
              <div className="mt-2 flex items-center justify-between gap-1">
                <select aria-label="verdict" value={item.action} onChange={(e) => change(item.image.image_id, e.target.value as Action)} className={`!px-1 !py-0.5 ${VERDICT_TONE[item.action]}`}>
                  {(['like', 'save', 'dislike', 'skip'] as Action[]).map((a) => <option key={a} value={a}>{a}</option>)}
                </select>
                <button className="link text-[11px]" onClick={() => api.similar(item.image.image_id).then(setSimilar)}>more like this</button>
              </div>
              <p className="mt-1 truncate font-mono text-[10px] text-canvas-500" title={item.created_at}>{item.created_at}</p>
            </li>
          ))}
        </ul>

        {page && page.total > PAGE && (
          <nav className="flex items-center justify-center gap-3 text-sm">
            <button className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>← newer</button>
            <span className="font-mono text-xs text-canvas-400">{offset + 1}-{Math.min(offset + PAGE, page.total)} of {page.total}</span>
            <button className="btn btn-sm" disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)}>older →</button>
          </nav>
        )}

        {similar && (
          <section aria-label="More like this" className="panel p-4">
            <p className="eyebrow mb-1">Similarity search</p>
            <h2 className="mb-3 text-sm font-medium text-white">Nearest to that image <span className="chip ml-1">{similar.model}</span></h2>
            <ul className="flex gap-3 overflow-x-auto pb-1">
              {similar.neighbours.map((n) => (
                <li key={n.image.image_id} className="shrink-0 text-center text-xs">
                  <img src={n.image.thumb_url} alt="" className="h-24 w-24 rounded-lg object-cover ring-1 ring-white/10" />
                  <span className="mt-1 block font-mono text-[10px] text-accent">{n.similarity.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </main>
  )
}
