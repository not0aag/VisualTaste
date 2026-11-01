/**
 * Search: describe something in words and see the closest photographs.
 *
 * SigLIP 2 puts text and images in one space (`search/text.py`), so a query
 * like "foggy harbour at dawn" ranks the 50,000 images directly. The text
 * model loads on the first search, which therefore takes a while; the page
 * says so instead of looking stuck.
 */

import { useEffect, useRef, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { SearchResponse } from '../api/types'

const EXAMPLES = ['a lighthouse on a rocky coast', 'old hand-drawn maps', 'red vintage cars', 'moths']

export function Search() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [slow, setSlow] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const slowTimer = useRef<number>()

  useEffect(() => () => window.clearTimeout(slowTimer.current), [])

  const run = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    setQuery(trimmed)
    setBusy(true)
    setError(null)
    setSlow(false)
    // The first query loads the text model; say so once it is clearly slow.
    slowTimer.current = window.setTimeout(() => setSlow(true), 2500)
    try {
      setResult(await api.search(trimmed, 24))
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    } finally {
      window.clearTimeout(slowTimer.current)
      setBusy(false)
      setSlow(false)
    }
  }

  return (
    <main className="page" data-testid="search">
      <div className="page-inner">
        <div>
          <p className="eyebrow">Text to image · SigLIP 2</p>
          <h1 className="page-title mt-1">Search</h1>
        </div>

        <form
          className="plate flex flex-wrap items-center gap-2 p-2"
          onSubmit={(event) => {
            event.preventDefault()
            void run(query)
          }}
        >
          <label htmlFor="search-query" className="sr-only">
            Describe what you want to see
          </label>
          <input
            id="search-query"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Describe what you want to see"
            maxLength={200}
            className="min-w-0 flex-1 rounded-plate border border-white/10 bg-canvas-950/60 px-3 py-2 text-sm text-white outline-none placeholder:text-canvas-500 focus:border-accent/60"
          />
          <button type="submit" className="btn" disabled={busy || !query.trim()}>
            {busy ? 'Searching…' : 'Search'}
          </button>
        </form>

        <p className="-mt-2 flex flex-wrap items-center gap-2 text-xs text-canvas-500">
          Try
          {EXAMPLES.map((example) => (
            <button key={example} className="chip hover:border-accent/40" onClick={() => void run(example)}>
              {example}
            </button>
          ))}
        </p>

        {slow && (
          <p className="panel px-4 py-3 font-mono text-xs text-accent" role="status">
            Loading the text model for the first search; this can take a minute. Later searches are instant.
          </p>
        )}
        {error && <p className="font-mono text-sm text-dislike">{error}</p>}

        {result && result.results.length > 0 && (
          <section aria-label="Results">
            <p className="eyebrow mb-3">
              {result.results.length} closest to “{result.query}”
            </p>
            <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
              {result.results.map(({ image, similarity }) => (
                <li key={image.image_id} className="tile">
                  <img src={image.thumb_url} alt={image.attribution.title ?? ''} loading="lazy" className="thumb" />
                  <p className="readout mt-1.5 flex justify-between">
                    <span>{image.attribution.source.replace('_', ' ')}</span>
                    <span className="text-accent">{similarity.toFixed(3)}</span>
                  </p>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </main>
  )
}
