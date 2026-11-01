/**
 * Saves: the images you kept, as a gallery at their own proportions.
 *
 * History shows every verdict as a square tile; saves deserve more room, so
 * here they keep their aspect ratio, in columns, newest first, each with its
 * credit. Data is History's (`/api/history?action=save`), so nothing new is
 * stored.
 */

import { useEffect, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { HistoryItem } from '../api/types'
import { AttributionLine } from '../components/AttributionLine'

const LIMIT = 200

export function Saves() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [total, setTotal] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .history({ action: 'save', limit: LIMIT })
      .then((page) => {
        setItems(page.items)
        setTotal(page.total)
      })
      .catch((cause) => setError(cause instanceof ApiError ? cause.detail : String(cause)))
  }, [])

  return (
    <main className="page" data-testid="saves">
      <div className="page-inner">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Collection</p>
            <h1 className="page-title mt-1">Saves</h1>
          </div>
          {items && items.length > 0 && (
            <span className="chip">
              {total.toLocaleString()} saved
              {total > LIMIT ? ` · newest ${LIMIT} shown` : ''}
            </span>
          )}
        </header>

        {error && <p className="font-mono text-sm text-dislike">{error}</p>}

        {items && items.length === 0 && (
          <div className="plate px-6 py-10 text-center">
            <p className="font-display text-lg font-semibold uppercase tracking-wide text-white">
              Nothing saved yet
            </p>
            <p className="mt-2 text-sm text-canvas-400">
              Press <span className="kbd">S</span> or drag a photo upwards in the Feed to keep it here.
            </p>
          </div>
        )}

        {items && items.length > 0 && (
          <ul className="columns-1 gap-4 sm:columns-2 lg:columns-3">
            {items.map((item) => (
              <li key={item.event_id} className="mb-4 break-inside-avoid">
                <figure className="panel overflow-hidden p-2 transition-all hover:border-save/40 hover:shadow-glow-save">
                  <img
                    src={item.image.display_url}
                    alt={item.image.attribution.title ?? 'A saved image'}
                    loading="lazy"
                    width={item.image.width ?? undefined}
                    height={item.image.height ?? undefined}
                    className="h-auto w-full rounded-[3px]"
                  />
                  <figcaption className="pt-2">
                    <AttributionLine attribution={item.image.attribution} />
                  </figcaption>
                </figure>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  )
}
