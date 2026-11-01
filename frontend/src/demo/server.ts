/**
 * The in-browser stand-in for the API, used only by the GitHub Pages demo
 * (`VITE_DEMO=1`). It answers the same requests as the FastAPI backend from
 * fixtures captured from a real sandbox session (`scripts/capture_demo.py`).
 *
 * What is live and what is a replay:
 *   - the Feed serves the model's real picks, captured batch by batch, in
 *     order; your swipes are kept in memory, so the counter, the learning
 *     ring and History all respond;
 *   - Profile, Model Lab and the Embedding Map show the captured state of the
 *     simulated user the session continues;
 *   - anything that needs the real backend (training, reset, review
 *     decisions) answers with a 403 explaining how to run it locally.
 *
 * Nothing is sent anywhere: there is no backend behind the demo.
 */

import type {
  Action,
  EventPayload,
  EventResponse,
  FeedCard,
  FeedResponse,
  HistoryItem,
  HistoryResponse,
  ProfileResponse,
  UndoResponse,
} from '../api/types'

export interface DemoFixtures {
  /** Exact request path (with its query) -> captured JSON response. */
  routes: Record<string, unknown>
  /** The model's picks in the order it served them, five per batch. */
  feed: FeedCard[]
  model_version: string
  rung: string
  /** The simulated user's history before the demo begins, newest first. */
  history: HistoryItem[]
  /** Image id -> `/api/images/{id}/similar` response, for History's "more like this". */
  similar: Record<string, unknown>
  /** Labels between online retrains, as the backend's `VT_RETRAIN_EVERY`. */
  retrain_every: number
}

/** The same shape as `ApiError` in client.ts, without importing it (no cycle). */
export interface DemoError {
  status: number
  detail: string
}

const READ_ONLY =
  'The demo is a replay running in your browser, so it cannot train, reset or ' +
  'change anything. Clone the repository and run it locally to do that.'

/** Rewrites the backend's image URLs to the files bundled with the demo. */
export function rewriteImageUrls(json: string, base: string): string {
  return json.replace(
    /\/api\/images\/([0-9a-f]{16})\/(display|thumb)/g,
    (_, id: string, kind: string) => `${base}demo/${kind}/${id}.webp`,
  )
}

export class DemoServer {
  private cursor = 0
  private nextEventId = 1_000_000
  private labels: number
  private swipes: HistoryItem[] = []

  constructor(private readonly fixtures: DemoFixtures) {
    const profile = fixtures.routes['/api/profile'] as ProfileResponse | undefined
    this.labels = profile?.labels ?? 0
  }

  handle(path: string, init?: RequestInit): unknown {
    const method = (init?.method ?? 'GET').toUpperCase()
    const url = new URL(path, 'http://demo.local')
    const route = url.pathname

    if (method === 'GET' && route === '/api/feed/next') return this.nextBatch(url)
    if (method === 'POST' && route === '/api/events') return this.record(init)
    const undo = route.match(/^\/api\/events\/(\d+)\/undo$/)
    if (method === 'POST' && undo) return this.undo(Number(undo[1]))
    if (method === 'GET' && route === '/api/history') return this.historyPage(url)
    if (method === 'GET' && route === '/api/profile') return this.profile()
    const similar = route.match(/^\/api\/images\/([0-9a-f]{16})\/similar$/)
    if (method === 'GET' && similar) {
      const found = this.fixtures.similar[similar[1]!]
      if (found) return found
      throw this.error(404, 'Similar images were not captured for this one in the demo.')
    }
    if (method !== 'GET') throw this.error(403, READ_ONLY)

    const exact = this.fixtures.routes[`${route}${url.search}`]
    if (exact !== undefined) return exact
    const bare = this.fixtures.routes[route]
    if (bare !== undefined) return bare
    throw this.error(404, `Not part of the demo: ${route}`)
  }

  private nextBatch(url: URL): FeedResponse {
    const n = Number(url.searchParams.get('n') ?? 5)
    const cards = this.fixtures.feed.slice(this.cursor, this.cursor + n)
    this.cursor += cards.length
    return {
      batch_id: `demo-${this.cursor}`,
      model_version: this.fixtures.model_version,
      rung: this.fixtures.rung,
      cards,
      candidates_considered: 50_000,
      labels_so_far: this.labels,
    }
  }

  private record(init?: RequestInit): EventResponse {
    const payload = JSON.parse(String(init?.body ?? '{}')) as EventPayload
    const card = this.fixtures.feed.find((c) => c.image.image_id === payload.image_id)
    const eventId = this.nextEventId++
    const createdAt = new Date().toISOString().slice(0, 19).replace('T', ' ')
    this.labels += 1
    if (card) {
      this.swipes.unshift({
        event_id: eventId,
        image: card.image,
        action: payload.action,
        dwell_ms: payload.dwell_ms ?? null,
        created_at: createdAt,
        undone: false,
      })
    }
    return {
      event_id: eventId,
      image_id: payload.image_id,
      action: payload.action,
      created_at: createdAt,
      labels_so_far: this.labels,
      superseded_event_id: null,
      retrain_queued: this.labels % this.fixtures.retrain_every === 0,
    }
  }

  private undo(eventId: number): UndoResponse {
    const index = this.swipes.findIndex((s) => s.event_id === eventId)
    if (index === -1) throw this.error(404, 'Nothing to undo.')
    const [removed] = this.swipes.splice(index, 1)
    this.labels -= 1
    return { event_id: eventId, image_id: removed!.image.image_id, undone: true, labels_so_far: this.labels }
  }

  private historyPage(url: URL): HistoryResponse {
    const action = (url.searchParams.get('action') ?? 'all') as Action | 'all'
    const limit = Number(url.searchParams.get('limit') ?? 50)
    const offset = Number(url.searchParams.get('offset') ?? 0)
    const since = url.searchParams.get('since')
    const all = [...this.swipes, ...this.fixtures.history].filter(
      (item) =>
        (action === 'all' || item.action === action) && (!since || item.created_at >= since),
    )
    return { items: all.slice(offset, offset + limit), total: all.length, offset, limit }
  }

  private profile(): ProfileResponse {
    const captured = this.fixtures.routes['/api/profile'] as ProfileResponse
    return { ...captured, labels: this.labels }
  }

  private error(status: number, detail: string): DemoError {
    return { status, detail }
  }
}

let server: Promise<DemoServer> | null = null

/** The demo server, created once from the bundled fixtures. */
export function demoServer(): Promise<DemoServer> {
  const base = import.meta.env.BASE_URL
  server ??= fetch(`${base}demo/fixtures.json`)
    .then((response) => response.text())
    .then((text) => new DemoServer(JSON.parse(rewriteImageUrls(text, base)) as DemoFixtures))
  return server
}
