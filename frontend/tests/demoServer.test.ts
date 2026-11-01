import { describe, expect, it } from 'vitest'

import type { FeedCard, HistoryItem, ImageSummary } from '../src/api/types'
import { DemoServer, rewriteImageUrls, type DemoFixtures } from '../src/demo/server'

const id = (n: number) => n.toString(16).padStart(16, '0')

function image(n: number): ImageSummary {
  return {
    image_id: id(n),
    width: 800,
    height: 600,
    dominant_colours: ['#112233'],
    attribution: {
      license_url: 'https://creativecommons.org/publicdomain/zero/1.0/',
      author: null,
      author_url: null,
      source: 'pd12m',
      source_url: null,
      title: null,
    },
    display_url: `/api/images/${id(n)}/display`,
    thumb_url: `/api/images/${id(n)}/thumb`,
  }
}

const card = (n: number): FeedCard => ({
  image: image(n),
  impression_id: n,
  rank: n,
  reason: 'model',
  sentence: 'Close to images you liked.',
  explanation: {},
  score: 0.7,
  sampled_score: null,
  propensity: null,
})

const past = (n: number, action: HistoryItem['action']): HistoryItem => ({
  event_id: n,
  image: image(100 + n),
  action,
  dwell_ms: 900,
  created_at: `2026-11-01 09:00:0${n}`,
  undone: false,
})

function fixtures(): DemoFixtures {
  return {
    routes: {
      '/api/profile': { profile_id: 1, labels: 40, counts: {}, is_simulated: false },
      '/api/stats/learning-curve?window=50': { window: 50 },
      '/api/models': { versions: [] },
    },
    feed: Array.from({ length: 12 }, (_, i) => card(i + 1)),
    model_version: 'r4_demo',
    rung: 'r4',
    history: [past(2, 'like'), past(1, 'dislike')],
    similar: { [id(102)]: { image_id: id(102), model: 'siglip2_b16_224', neighbours: [] } },
    retrain_every: 20,
  }
}

const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

describe('the demo server', () => {
  it('serves the captured feed in order, five at a time, without repeats', () => {
    const server = new DemoServer(fixtures())
    const first = server.handle('/api/feed/next?n=5') as { cards: FeedCard[]; rung: string }
    const second = server.handle('/api/feed/next?n=5') as { cards: FeedCard[] }

    expect(first.rung).toBe('r4')
    expect(first.cards.map((c) => c.impression_id)).toEqual([1, 2, 3, 4, 5])
    expect(second.cards.map((c) => c.impression_id)).toEqual([6, 7, 8, 9, 10])
  })

  it('returns an empty batch when the recording runs out', () => {
    const server = new DemoServer(fixtures())
    for (let i = 0; i < 3; i++) server.handle('/api/feed/next?n=5')
    const done = server.handle('/api/feed/next?n=5') as { cards: FeedCard[] }
    expect(done.cards).toEqual([])
  })

  it('counts swipes from the captured profile, and flags a retrain every 20', () => {
    const server = new DemoServer(fixtures())
    let last = { labels_so_far: 0, retrain_queued: false }
    for (let i = 0; i < 20; i++) {
      last = server.handle('/api/events', post({ image_id: id(1), action: 'like' })) as typeof last
    }
    expect(last.labels_so_far).toBe(60)
    expect(last.retrain_queued).toBe(true)
    expect((server.handle('/api/profile') as { labels: number }).labels).toBe(60)
  })

  it('puts swipes at the top of History, and undo takes them back out', () => {
    const server = new DemoServer(fixtures())
    const event = server.handle('/api/events', post({ image_id: id(3), action: 'save' })) as {
      event_id: number
    }
    const page = server.handle('/api/history?action=all&limit=48&offset=0') as {
      items: HistoryItem[]
      total: number
    }
    expect(page.total).toBe(3)
    expect(page.items[0]!.image.image_id).toBe(id(3))

    const undone = server.handle(`/api/events/${event.event_id}/undo`, { method: 'POST' }) as {
      labels_so_far: number
    }
    expect(undone.labels_so_far).toBe(40)
    const after = server.handle('/api/history?action=all&limit=48&offset=0') as { total: number }
    expect(after.total).toBe(2)
  })

  it('filters History by verdict', () => {
    const server = new DemoServer(fixtures())
    const likes = server.handle('/api/history?action=like&limit=48&offset=0') as {
      items: HistoryItem[]
    }
    expect(likes.items.map((i) => i.action)).toEqual(['like'])
  })

  it('answers captured routes by exact path first, then by bare path', () => {
    const server = new DemoServer(fixtures())
    expect(server.handle('/api/stats/learning-curve?window=50')).toEqual({ window: 50 })
    expect(server.handle('/api/models')).toEqual({ versions: [] })
  })

  it('refuses anything that would change state, and says why', () => {
    const server = new DemoServer(fixtures())
    expect(() => server.handle('/api/models/train', post({ rung: 'r3' }))).toThrow()
    try {
      server.handle('/api/profile/reset', { method: 'POST' })
    } catch (error) {
      expect(error).toMatchObject({ status: 403 })
      expect((error as { detail: string }).detail).toMatch(/run it locally/)
    }
  })

  it('serves similar images only where they were captured', () => {
    const server = new DemoServer(fixtures())
    expect(server.handle(`/api/images/${id(102)}/similar?k=12`)).toMatchObject({ neighbours: [] })
    expect(() => server.handle(`/api/images/${id(999)}/similar?k=12`)).toThrow()
  })

  it('a route that was never captured is a 404, not a crash', () => {
    const server = new DemoServer(fixtures())
    expect(() => server.handle('/api/stats/clusters')).toThrow()
  })
})

describe('rewriteImageUrls', () => {
  it('points the backend image routes at the files bundled with the demo', () => {
    const json = JSON.stringify({ a: `/api/images/${id(7)}/display`, b: `/api/images/${id(7)}/thumb` })
    const rewritten = JSON.parse(rewriteImageUrls(json, '/VisualTaste/')) as { a: string; b: string }
    expect(rewritten.a).toBe(`/VisualTaste/demo/display/${id(7)}.webp`)
    expect(rewritten.b).toBe(`/VisualTaste/demo/thumb/${id(7)}.webp`)
  })
})
