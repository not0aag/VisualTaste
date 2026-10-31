/**
 * The wire format, mirroring `backend/visualtaste/api/schemas.py`.
 *
 * Hand-written rather than generated. The API has about twenty endpoints and
 * generating types would add a build step and a codegen dependency to save a
 * few hundred lines -- but every field here must match the Python model
 * exactly, including nullability, because `strict` mode will not catch a
 * mismatch against a lie.
 */

export type Action = 'like' | 'dislike' | 'skip' | 'save'

export type ImageStatus = 'ok' | 'near_dup' | 'removed' | 'quarantined' | 'rejected'

/**
 * Credit for an image. Not decoration: Open Images is CC BY 2.0 and requires
 * attribution, and the Unsplash terms ask for the photographer and photo page.
 * The nullable fields are nullable because a public-domain archive genuinely
 * may not name an author -- never because it is optional to show.
 */
export interface Attribution {
  license_url: string
  author: string | null
  author_url: string | null
  source: string
  source_url: string | null
  title: string | null
}

export interface ImageSummary {
  image_id: string
  width: number | null
  height: number | null
  dominant_colours: string[]
  attribution: Attribution
  display_url: string
  thumb_url: string
}

export interface ImageDetail extends ImageSummary {
  status: ImageStatus
  selected: boolean
  cluster64: number | null
  cluster400: number | null
  phash: string | null
  dup_of: string | null
  created_at: string | null
}

export interface FeedCard {
  image: ImageSummary
  /** Echoed back with the event, so an outcome can be traced to its offer. */
  impression_id: number
  rank: number
  reason: string
  sentence: string
  explanation: Record<string, unknown>
  score: number
  sampled_score: number | null
  propensity: number | null
}

export interface FeedResponse {
  batch_id: string
  model_version: string
  rung: string
  cards: FeedCard[]
  candidates_considered: number
  labels_so_far: number
}

export interface EventPayload {
  image_id: string
  action: Action
  impression_id?: number | null
  dwell_ms?: number | null
  client_ts?: string | null
}

export interface EventResponse {
  event_id: number
  image_id: string
  action: Action
  created_at: string
  labels_so_far: number
  superseded_event_id: number | null
  retrain_queued: boolean
}

export interface UndoResponse {
  event_id: number
  image_id: string
  undone: boolean
  labels_so_far: number
}

export interface HistoryItem {
  event_id: number
  image: ImageSummary
  action: Action
  dwell_ms: number | null
  created_at: string
  undone: boolean
}

export interface HistoryResponse {
  items: HistoryItem[]
  total: number
  offset: number
  limit: number
}

export interface SimilarImage {
  image: ImageSummary
  similarity: number
}

export interface SimilarResponse {
  image_id: string
  model: string
  neighbours: SimilarImage[]
}

export interface ProfileResponse {
  profile_id: number
  started_at: string
  is_simulated: boolean
  labels: number
  counts: Record<string, number>
  active_model_version: string | null
  active_rung: string | null
  coldstart_active: boolean
}

export interface ResetResponse {
  previous_profile_id: number | null
  new_profile_id: number
  archived_labels: number
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  schema_version: number
  images_ok: number
  images_selected: number
  embeddings: Record<string, number>
  active_model_version: string | null
  notes: string[]
}
