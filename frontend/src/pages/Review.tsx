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
    <main className="h-full overflow-y-auto bg-canvas-200 dark:bg-canvas-900" data-testid="review">
      <div className="mx-auto max-w-6xl p-4 sm:p-6">
        <h1 className="mb-1 text-lg font-semibold">Review</h1>
        <p className="mb-4 text-sm text-canvas-500">
          Quarantined by the automated safety screen. Keep returns an image to the corpus; remove records it in
          the removals table. Nothing is deleted.
        </p>
        {error && <p className="text-sm text-red-600">{error}</p>}
        {items && items.length === 0 && <p className="text-sm text-canvas-500">Nothing waiting for review.</p>}
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {items?.map((item) => (
            <li key={item.image.image_id} className="rounded-lg bg-canvas-100 p-2 text-xs dark:bg-canvas-950">
              <img src={item.image.thumb_url} alt="" loading="lazy" className="aspect-square w-full rounded object-cover blur-sm hover:blur-none" />
              <div className="mt-1 flex justify-between">
                <button className="underline" onClick={() => decide(item.image.image_id, 'keep')}>keep</button>
                <button className="text-red-600 underline" onClick={() => decide(item.image.image_id, 'remove')}>remove</button>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </main>
  )
}
