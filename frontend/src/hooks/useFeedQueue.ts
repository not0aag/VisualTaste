/**
 * The card queue: fetching ahead, swiping optimistically, and undoing.
 *
 * Three behaviours the brief asks for, and the reasoning behind each:
 *
 * **Preload the next three.** A feed that waits for a network round trip after
 * every swipe feels broken even when it is working. Cards are fetched in
 * batches and the next few images are decoded by the browser before they are
 * needed, so a flick shows the next photograph instantly.
 *
 * **Optimistic, with rollback.** The card advances the moment the gesture
 * commits, and the event is posted afterwards. If the post fails the card
 * comes back and the error is shown -- rather than the swipe being silently
 * lost, which would mean a label Alen believes he gave that the model never
 * saw.
 *
 * **Undo restores the exact card.** Not a refetch: the same image, with its
 * original impression id, so re-rating it attaches to the offer that was
 * actually made.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api } from '../api/client'
import type { Action, FeedCard } from '../api/types'

/** Refill when fewer than this many cards remain, so the queue never empties. */
const REFILL_THRESHOLD = 3

/** Images to decode ahead of the current one. */
const PRELOAD_AHEAD = 3

const BATCH_SIZE = 5

interface UndoableSwipe {
  card: FeedCard
  eventId: number
  action: Action
}

export interface FeedQueueState {
  current: FeedCard | null
  upcoming: FeedCard[]
  labels: number
  rung: string
  modelVersion: string
  loading: boolean
  exhausted: boolean
  error: string | null
  stillBuilding: boolean
  lastSwipe: UndoableSwipe | null
  retrainQueued: boolean
}

export function useFeedQueue() {
  const [queue, setQueue] = useState<FeedCard[]>([])
  const [labels, setLabels] = useState(0)
  const [rung, setRung] = useState('')
  const [modelVersion, setModelVersion] = useState('')
  const [loading, setLoading] = useState(true)
  const [exhausted, setExhausted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stillBuilding, setStillBuilding] = useState(false)
  const [lastSwipe, setLastSwipe] = useState<UndoableSwipe | null>(null)
  const [retrainQueued, setRetrainQueued] = useState(false)

  /** Guards against two refills racing when swipes come fast. */
  const fetching = useRef(false)
  /** Ids already queued, so a refill cannot introduce a duplicate card. */
  const queuedIds = useRef(new Set<string>())

  const refill = useCallback(async () => {
    if (fetching.current) return
    fetching.current = true
    try {
      const batch = await api.nextBatch(BATCH_SIZE)
      setRung(batch.rung)
      setModelVersion(batch.model_version)
      setLabels(batch.labels_so_far)
      setError(null)
      setStillBuilding(false)

      const fresh = batch.cards.filter((card) => !queuedIds.current.has(card.image.image_id))
      fresh.forEach((card) => queuedIds.current.add(card.image.image_id))

      if (batch.cards.length === 0) setExhausted(true)
      if (fresh.length > 0) setQueue((previous) => [...previous, ...fresh])
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(cause.detail)
        setStillBuilding(cause.isStillBuilding)
      } else {
        setError('Something went wrong loading the feed.')
      }
    } finally {
      fetching.current = false
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refill()
  }, [refill])

  useEffect(() => {
    if (!loading && !exhausted && queue.length < REFILL_THRESHOLD) void refill()
  }, [queue.length, loading, exhausted, refill])

  // Ask the browser to decode the next few images now, so the swipe after this
  // one is instant. `new Image()` plus `src` is enough: the response carries an
  // immutable cache header, so the real request happens exactly once.
  useEffect(() => {
    queue.slice(1, 1 + PRELOAD_AHEAD).forEach((card) => {
      const image = new Image()
      image.src = card.image.display_url
    })
  }, [queue])

  const swipe = useCallback(
    async (action: Action, dwellMs: number | null) => {
      const card = queue[0]
      if (!card) return

      // Optimistic: the card leaves immediately.
      setQueue((previous) => previous.slice(1))
      setError(null)

      try {
        const result = await api.recordEvent({
          image_id: card.image.image_id,
          action,
          impression_id: card.impression_id,
          dwell_ms: dwellMs,
          client_ts: new Date().toISOString(),
        })
        setLabels(result.labels_so_far)
        setRetrainQueued(result.retrain_queued)
        setLastSwipe({ card, eventId: result.event_id, action })
      } catch (cause) {
        // Rollback. A silently dropped swipe would mean a label Alen believes
        // he gave that the model never received.
        setQueue((previous) => [card, ...previous])
        setError(
          cause instanceof ApiError
            ? `Could not save that: ${cause.detail}`
            : 'Could not save that swipe.',
        )
      }
    },
    [queue],
  )

  const undo = useCallback(async () => {
    if (!lastSwipe) return
    try {
      const result = await api.undoEvent(lastSwipe.eventId)
      setLabels(result.labels_so_far)
      // The same card object, so its impression id survives and a re-rating
      // still attaches to the offer that was actually made.
      setQueue((previous) => [lastSwipe.card, ...previous])
      setLastSwipe(null)
      setExhausted(false)
      setError(null)
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? `Could not undo: ${cause.detail}`
          : 'Could not undo that.',
      )
    }
  }, [lastSwipe])

  const state: FeedQueueState = {
    current: queue[0] ?? null,
    upcoming: queue.slice(1, 1 + PRELOAD_AHEAD),
    labels,
    rung,
    modelVersion,
    loading,
    exhausted: exhausted && queue.length === 0,
    error,
    stillBuilding,
    lastSwipe,
    retrainQueued,
  }

  return { ...state, swipe, undo, refill }
}
