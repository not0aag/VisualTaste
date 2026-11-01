/**
 * Review: the images the safety screen quarantined, to keep or remove.
 *
 * The automated screen is deliberately over-cautious (it flags the top 1%
 * whatever they show), so most of what lands here is harmless. "Remove" is
 * recorded in `removals`; nothing is ever deleted from disk.
 */

import { useCallback, useEffect, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { ReviewItem } from '../api/types'

export function Review() {
  const [items, setItems] = useState<ReviewItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setItems(await api.review())
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const decide = async (imageId: string, decision: 'keep' | 'remove') => {
    try {
      await api.decide(imageId, decision)
      setItems((current) => current?.filter((item) => item.image.image_id !== imageId) ?? null)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.detail : String(cause))
    }
  }

  return (
    <main className="page" data-testid="review">
      <div className="page-inner">
        <div>
          <p className="eyebrow">Safety screen</p>
          <h1 className="page-title mt-1">Review</h1>
        </div>
        <p className="max-w-3xl text-sm text-canvas-400">
          Quarantined by the automated safety screen. Keep returns an image to the corpus; remove records it in
          the removals table. Nothing is deleted.
        </p>
        {error && <p className="font-mono text-sm text-dislike">{error}</p>}
        {items && items.length === 0 && <p className="panel px-4 py-8 text-center font-mono text-sm text-canvas-500">Nothing waiting for review.</p>}
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {items?.map((item) => (
            <li key={item.image.image_id} className="tile group">
              <img src={item.image.thumb_url} alt="" loading="lazy" className="thumb blur-md hover:blur-none" />
              <div className="mt-2 flex justify-between gap-1">
                <button className="btn btn-sm flex-1" onClick={() => decide(item.image.image_id, 'keep')}>keep</button>
                <button className="btn btn-sm btn-danger flex-1" onClick={() => decide(item.image.image_id, 'remove')}>remove</button>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </main>
  )
}
