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

/** Beta(1 + likes, 1 + dislikes): its mean and central 90% interval. */
export interface LikeRateInterval {
  likes: number
  dislikes: number
  mean: number
  low: number
  high: number
}

export interface InterestCard {
  interest_id: number
  label: string
  n_likes: number
  /** Most central liked images first, up to six. */
  central_image_ids: string[]
  like_rate: LikeRateInterval
}

export interface ConceptWord {
  concept: string
  z_score: number
}

/** One recluster of the interests: ids that appeared and ids retired. */
export interface InterestEvent {
  likes: number
  born: number[]
  retired: number[]
}

export interface TasteResponse {
  profile_id: number
  likes: number
  dislikes: number
  /** The serving model's version, or "computed" when fitted on the fly. */
  source: string
  interests: InterestCard[]
  likes_in_words: ConceptWord[]
  dislikes_in_words: ConceptWord[]
  history: InterestEvent[]
  note: string
}

// --- Analytics (Phase 13) --------------------------------------------------

export interface Interval {
  successes: number
  trials: number
  mean: number
  low: number
  high: number
}

export interface ProfileListItem {
  profile_id: number
  started_at: string
  ended_at: string | null
  is_simulated: boolean
  note: string | null
  labels: number
}

export interface StatsSummary {
  profile_id: number
  is_simulated: boolean
  labels: number
  counts: Record<string, number>
  impressions: number
  like_rate: number | null
  like_rate_last_100: number | null
  serving_version: string
  serving_rung: string
  versions: number
}

export interface RungSwitch {
  index: number
  model_version: string
}

export interface LearningCurve {
  window: number
  like_rate: number[]
  coverage: number[]
  interests: number[]
  switches: RungSwitch[]
}

export interface ClusterStat {
  cluster64: number
  shown: number
  rate: Interval
}

export interface DiversityStats {
  batch_diversity: number[]
  batch_novelty: number[]
  coverage_last_500: number
}

export interface CalibrationBin {
  low: number
  high: number
  count: number
  mean_predicted: number
  observed_rate: number
}

export interface CalibrationStats {
  bins: CalibrationBin[]
  ece: number | null
  samples: number
}

export interface ExplorationStats {
  explore: Interval
  exploit: Interval
}

export interface ScoreDistribution {
  model_version: string
  edges: number[]
  counts: number[]
  batch_scores: number[]
}

export interface ModelVersion {
  version: string
  rung: string
  embedding_key: string | null
  params: Record<string, unknown>
  trained_on_events: number
  metrics: Record<string, unknown>
  is_active: boolean
  created_at: string
}

export interface ModelsResponse {
  versions: ModelVersion[]
  serving_version: string
  serving_rung: string
  default_rung: string
}

export interface TrainResponse {
  status: string
  rung: string | null
  version: string | null
  detail: string
  metrics: Record<string, unknown> | null
}

export interface Experiment {
  experiment_id: number
  version_a: string
  version_b: string
  method: string
  started_at: string
  ended_at: string | null
  active: boolean
  wins_a: number
  wins_b: number
  ties: number
  p_value: number | null
  like_rate_a: number | null
  like_rate_b: number | null
  summary: string
}

export interface Projection {
  model: string
  image_ids: string[]
  x: number[]
  y: number[]
  cluster: number[]
  source: string[]
  score: number[] | null
  liked: string[]
  disliked: string[]
  saved: string[]
}

export interface TastePoint {
  labels: number
  x: number
  y: number
}

export interface TastePath {
  model: string
  every: number
  path: TastePoint[]
  interests: TastePoint[]
}

export interface ReviewItem {
  image: ImageSummary
  status: string
}

export interface EvalRunResponse {
  pid: number
  command: string
  log_path: string
}

/** `GET /api/search`: images closest to a description, best first. */
export interface SearchResponse {
  query: string
  results: SimilarImage[]
}

/** `/api/profile/seed-text`: the described taste, or null when there is none. */
export interface SeedTextResponse {
  text: string | null
}
