/**
 * Tests for the feed queue's three interesting behaviours: preloading ahead,
 * optimistic swipes that roll back on failure, and undo restoring the exact
 * card rather than refetching.
 *
 * `fetch` is stubbed rather than the client module mocked, so the real client
 * -- including its error translation -- is exercised.
 */

import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useFeedQueue } from '../src/hooks/useFeedQueue'
import type { FeedCard, FeedResponse } from '../src/api/types'

function card(n: number): FeedCard {
  return {
    image: {
      image_id: `image${n}`,
      width: 800,
      height: 600,
      dominant_colours: ['#123456'],
      attribution: {
        license_url: 'https://creativecommons.org/publicdomain/zero/1.0/',
        author: `Author ${n}`,
        author_url: null,
        source: 'pd12m',
        source_url: `https://example.invalid/${n}`,
        title: `Image ${n}`,
      },
      display_url: `/api/images/image${n}/display`,
      thumb_url: `/api/images/image${n}/thumb`,
    },
    impression_id: 1000 + n,
    rank: n,
    reason: 'cold_start',
    sentence: 'Still learning what you like.',
    explanation: {},
    score: 0.5,
    sampled_score: null,
    propensity: null,
  }
}

function batch(start: number, count = 5): FeedResponse {
  return {
    batch_id: `batch${start}`,
    model_version: 'r0_bootstrap',
    rung: 'r0',
    cards: Array.from({ length: count }, (_, i) => card(start + i)),
    candidates_considered: 50_000,
    labels_so_far: start,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: 'OK',
    json: async () => body,
  } as Response
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  let nextStart = 0
  fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
    if (path.startsWith('/api/feed/next')) {
      const response = jsonResponse(batch(nextStart))
      nextStart += 5
      return response
    }
    if (path === '/api/events' && init?.method === 'POST') {
      const payload = JSON.parse(String(init.body)) as { image_id: string }
      return jsonResponse({
        event_id: 7,
        image_id: payload.image_id,
        action: 'like',
        created_at: '2026-10-31T00:00:00Z',
        labels_so_far: 1,
        superseded_event_id: null,
        retrain_queued: false,
      })
    }
    if (path.endsWith('/undo')) {
      return jsonResponse({
        event_id: 7,
        image_id: 'image0',
        undone: true,
        labels_so_far: 0,
      })
    }
    throw new Error(`unexpected request: ${path}`)
  })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('useFeedQueue', () => {
  it('loads a first batch and exposes the first card', async () => {
    const { result } = renderHook(() => useFeedQueue())

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(result.current.current?.image.image_id).toBe('image0')
    expect(result.current.rung).toBe('r0')
  })

  it('keeps the next few cards ready, so a swipe is instant', async () => {
    const { result } = renderHook(() => useFeedQueue())

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(result.current.upcoming.length).toBeGreaterThanOrEqual(3)
  })

  it('advances to the next card immediately on a swipe', async () => {
    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.swipe('like', 1200)
    })

    expect(result.current.current?.image.image_id).toBe('image1')
  })

  it('sends the impression id with the event, so outcomes can be attributed', async () => {
    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.swipe('like', 900)
    })

    const call = fetchMock.mock.calls.find(([path]) => path === '/api/events')
    expect(call).toBeDefined()
    const payload = JSON.parse(String((call![1] as RequestInit).body))
    expect(payload.impression_id).toBe(1000)
    expect(payload.dwell_ms).toBe(900)
  })

  it('rolls the card back when saving the event fails', async () => {
    // A silently dropped swipe would mean a label Alen believes he gave that
    // the model never received.
    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    fetchMock.mockImplementationOnce(async () =>
      jsonResponse({ detail: 'database is busy' }, 503),
    )

    await act(async () => {
      await result.current.swipe('like', 500)
    })

    expect(result.current.current?.image.image_id).toBe('image0')
    expect(result.current.error).toContain('database is busy')
  })

  it('refills before the queue can empty', async () => {
    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    for (let i = 0; i < 3; i += 1) {
      await act(async () => {
        await result.current.swipe('skip', 100)
      })
    }

    await waitFor(() => {
      const feedCalls = fetchMock.mock.calls.filter(([path]) =>
        String(path).startsWith('/api/feed/next'),
      )
      expect(feedCalls.length).toBeGreaterThan(1)
    })
    expect(result.current.current).not.toBeNull()
  })

  it('undo restores the same card with its original impression id', async () => {
    // Not a refetch: re-rating must attach to the offer that was actually made.
    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.swipe('like', 400)
    })
    await act(async () => {
      await result.current.undo()
    })

    expect(result.current.current?.image.image_id).toBe('image0')
    expect(result.current.current?.impression_id).toBe(1000)
  })

  it('clears the undo offer once it has been used', async () => {
    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.swipe('like', 400)
    })
    expect(result.current.lastSwipe).not.toBeNull()

    await act(async () => {
      await result.current.undo()
    })

    expect(result.current.lastSwipe).toBeNull()
  })

  it('reports an unbuilt corpus as still building, not as an error', async () => {
    fetchMock.mockImplementation(async () =>
      jsonResponse({ detail: 'no embeddings are loaded. Run jobs.embed' }, 503),
    )
    const { result } = renderHook(() => useFeedQueue())

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(result.current.stillBuilding).toBe(true)
    expect(result.current.error).toContain('jobs.embed')
  })

  it('reports an exhausted corpus rather than spinning', async () => {
    fetchMock.mockImplementation(async () => jsonResponse({ ...batch(0, 0) }))
    const { result } = renderHook(() => useFeedQueue())

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(result.current.exhausted).toBe(true)
    expect(result.current.current).toBeNull()
  })

  it('says the backend is unreachable when fetch itself fails', async () => {
    fetchMock.mockImplementation(async () => {
      throw new TypeError('Failed to fetch')
    })
    const { result } = renderHook(() => useFeedQueue())

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(result.current.error).toContain('dev.ps1')
  })

  it('does not queue the same image twice across refills', async () => {
    // Unrated cards are offered again by the backend, so a refill must not
    // introduce a duplicate.
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) => {
      if (String(path).startsWith('/api/feed/next')) return jsonResponse(batch(0))
      if (path === '/api/events') {
        const payload = JSON.parse(String(init!.body)) as { image_id: string }
        return jsonResponse({
          event_id: 1,
          image_id: payload.image_id,
          action: 'like',
          created_at: 'now',
          labels_so_far: 1,
          superseded_event_id: null,
          retrain_queued: false,
        })
      }
      throw new Error('unexpected')
    })

    const { result } = renderHook(() => useFeedQueue())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.swipe('like', 100)
    })
    await act(async () => {
      await result.current.refill()
    })

    const ids = [result.current.current, ...result.current.upcoming]
      .filter(Boolean)
      .map((c) => c!.image.image_id)
    expect(new Set(ids).size).toBe(ids.length)
  })
})
