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
  CalibrationStats,
  ClusterStat,
  DiversityStats,
  EvalRunResponse,
  Experiment,
  ExplorationStats,
  LearningCurve,
  ModelVersion,
  ModelsResponse,
  ProfileListItem,
  Projection,
  ReviewItem,
  ScoreDistribution,
  SimilarResponse,
  StatsSummary,
  TastePath,
  TasteResponse,
  TrainResponse,
  UndoResponse,
} from './types'

/** `?profile_id=` when a specific (e.g. simulated) profile is being viewed. */
const forProfile = (profileId?: number | null, extra: Record<string, string> = {}) => {
  const params = new URLSearchParams(extra)
  if (profileId != null) params.set('profile_id', String(profileId))
  const query = params.toString()
  return query ? `?${query}` : ''
}

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

  taste: () => request<TasteResponse>('/api/profile/taste'),

  thumbUrl: (imageId: string) => `/api/images/${imageId}/thumb`,

  resetProfile: () =>
    request<ResetResponse>('/api/profile/reset', { method: 'POST' }),

  // --- Model Lab ------------------------------------------------------------

  profiles: (includeSimulated = false) =>
    request<ProfileListItem[]>(`/api/profiles?include_simulated=${includeSimulated}`),

  summary: (profileId?: number | null) =>
    request<StatsSummary>(`/api/stats/summary${forProfile(profileId)}`),

  learningCurve: (profileId?: number | null, window = 50) =>
    request<LearningCurve>(
      `/api/stats/learning-curve${forProfile(profileId, { window: String(window) })}`,
    ),

  clusters: (profileId?: number | null) =>
    request<ClusterStat[]>(`/api/stats/clusters${forProfile(profileId)}`),

  diversity: (profileId?: number | null) =>
    request<DiversityStats>(`/api/stats/diversity${forProfile(profileId)}`),

  calibration: (profileId?: number | null) =>
    request<CalibrationStats>(`/api/stats/calibration${forProfile(profileId)}`),

  exploration: (profileId?: number | null) =>
    request<ExplorationStats>(`/api/stats/exploration${forProfile(profileId)}`),

  scores: () => request<ScoreDistribution>('/api/stats/scores'),

  models: () => request<ModelsResponse>('/api/models'),

  train: (rung: string | null, activate = true) =>
    request<TrainResponse>('/api/models/train', {
      method: 'POST',
      body: JSON.stringify({ rung, activate }),
    }),

  activate: (version: string) =>
    request<ModelVersion>(`/api/models/${encodeURIComponent(version)}/activate`, {
      method: 'POST',
    }),

  evaluate: (version: string) =>
    request<ModelVersion>(`/api/models/${encodeURIComponent(version)}/evaluate`, {
      method: 'POST',
    }),

  experiments: () => request<Experiment[]>('/api/experiments'),

  startExperiment: (versionA: string, versionB: string) =>
    request<Experiment>('/api/experiments', {
      method: 'POST',
      body: JSON.stringify({ version_a: versionA, version_b: versionB }),
    }),

  stopExperiment: (experimentId: number) =>
    request<Experiment>(`/api/experiments/${experimentId}/stop`, { method: 'POST' }),

  studies: () => request<string[]>('/api/eval/studies'),

  runStudy: (study: string, seeds: number) =>
    request<EvalRunResponse>('/api/eval/run', {
      method: 'POST',
      body: JSON.stringify({ study, seeds }),
    }),

  // --- Embedding Map, Review ------------------------------------------------

  projection: (model: string, profileId?: number | null) =>
    request<Projection>(`/api/viz/projection${forProfile(profileId, { model })}`),

  tastePath: (model: string, profileId?: number | null, every = 50) =>
    request<TastePath>(
      `/api/viz/taste-path${forProfile(profileId, { model, every: String(every) })}`,
    ),

  review: () => request<ReviewItem[]>('/api/review'),

  decide: (imageId: string, decision: 'keep' | 'remove') =>
    request<ReviewItem>(`/api/review/${imageId}`, {
      method: 'POST',
      body: JSON.stringify({ decision }),
    }),
}
