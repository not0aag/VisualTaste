/**
 * The only place that talks to the backend.
 *
 * One function per endpoint, each returning a typed promise. No fetch call
 * anywhere else in the app, for two reasons: a mismatch with the Python
 * schema shows up in one file rather than five, and error handling is uniform
 * so the UI can treat "the API said no" the same way everywhere.
 *
 * Relative URLs throughout. In development Vite proxies `/api` to port 8000,
 * and in a built app the frontend is served from the same origin -- so there
 * is no base URL to configure and no way to accidentally point this at
 * something that is not loopback.
 */

import type {
  Action,
  EventPayload,
  EventResponse,
  FeedResponse,
  HealthResponse,
  HistoryResponse,
  ImageDetail,
  ProfileResponse,
  ResetResponse,
  SimilarResponse,
  UndoResponse,
} from './types'

/** An API call that failed, carrying enough to show the user something true. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`${status}: ${detail}`)
    this.name = 'ApiError'
  }

  /**
   * 503 means the corpus is still being built -- embeddings missing, or a
   * background job holding the database lock. Worth distinguishing, because
   * the UI should say "still building" rather than "something broke".
   */
  get isStillBuilding(): boolean {
    return this.status === 503
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch (cause) {
    // A network-level failure here means the backend is not running, which on
    // a loopback-only app is the single most likely cause.
    throw new ApiError(
      0,
      'Could not reach the backend. Is it running? Try `.\\scripts\\dev.ps1`.',
    )
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // A non-JSON error body (a proxy page, say). The status line will do.
    }
    throw new ApiError(response.status, detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),

  nextBatch: (n = 5) => request<FeedResponse>(`/api/feed/next?n=${n}`),

  recordEvent: (payload: EventPayload) =>
    request<EventResponse>('/api/events', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  undoEvent: (eventId: number) =>
    request<UndoResponse>(`/api/events/${eventId}/undo`, { method: 'POST' }),

  image: (imageId: string) => request<ImageDetail>(`/api/images/${imageId}`),

  similar: (imageId: string, model?: string, k = 12) => {
    const params = new URLSearchParams({ k: String(k) })
    if (model) params.set('model', model)
    return request<SimilarResponse>(`/api/images/${imageId}/similar?${params}`)
  },

  history: (options: {
    action?: Action | 'all'
    includeUndone?: boolean
    since?: string
    until?: string
    limit?: number
    offset?: number
  } = {}) => {
    const params = new URLSearchParams()
    if (options.action) params.set('action', options.action)
    if (options.includeUndone) params.set('include_undone', 'true')
    if (options.since) params.set('since', options.since)
    if (options.until) params.set('until', options.until)
    params.set('limit', String(options.limit ?? 50))
    params.set('offset', String(options.offset ?? 0))
    return request<HistoryResponse>(`/api/history?${params}`)
  },

  profile: () => request<ProfileResponse>('/api/profile'),

  resetProfile: () =>
    request<ResetResponse>('/api/profile/reset', { method: 'POST' }),
}
