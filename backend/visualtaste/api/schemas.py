"""Request and response shapes for the API.

Pydantic models everywhere, per brief §7.8, for a reason beyond tidiness: the
frontend is TypeScript in `strict` mode, and these schemas are the contract it
is written against. A field that is optional here must be handled as possibly
absent there, so being precise about nullability in this file prevents a class
of bug in the other language.

One convention worth noting: the API speaks in `image_id` and never in row
positions or array indices. Those are internal to the scoring path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# The four things a person can do to an image (schema CHECK constraint, §7.4).
Action = Literal["like", "dislike", "skip", "save"]

# The status values an image can hold.
ImageStatus = Literal["ok", "near_dup", "removed", "quarantined", "rejected"]


class Attribution(BaseModel):
    """Everything needed to credit an image, shown on every card.

    Not optional in spirit: Open Images is CC BY 2.0 and requires attribution,
    and the Unsplash terms ask for the photographer and the photo page. The
    fields are nullable only because a public-domain source genuinely may not
    name an author.
    """

    license_url: str
    author: str | None = None
    author_url: str | None = None
    source: str
    source_url: str | None = None
    title: str | None = None


class ImageSummary(BaseModel):
    """An image as the feed and the history grid see it."""

    image_id: str
    width: int | None = None
    height: int | None = None
    dominant_colours: list[str] = Field(default_factory=list)
    attribution: Attribution
    display_url: str
    thumb_url: str


class ImageDetail(ImageSummary):
    """An image with the internals the Model Lab and Review page want."""

    status: ImageStatus
    selected: bool
    cluster64: int | None = None
    cluster400: int | None = None
    phash: str | None = None
    dup_of: str | None = None
    created_at: str | None = None


class FeedCardOut(BaseModel):
    """One card: the image, why it was chosen, and the id to report back."""

    image: ImageSummary

    #: Echoed back with the event so the outcome can be tied to the offer.
    impression_id: int

    rank: int
    reason: str
    sentence: str
    explanation: dict[str, Any] = Field(default_factory=dict)

    #: The model's own score. Not comparable across rungs; see `rec/base.py`.
    score: float
    sampled_score: float | None = None
    propensity: float | None = None


class FeedResponse(BaseModel):
    batch_id: str
    model_version: str
    rung: str
    cards: list[FeedCardOut]
    candidates_considered: int
    labels_so_far: int


class EventIn(BaseModel):
    """A swipe, as the client reports it."""

    image_id: str
    action: Action

    #: Echoed from the card. Nullable because a verdict changed from the
    #: History page has no impression behind it -- nothing was shown.
    impression_id: int | None = None

    #: Milliseconds the card was actually visible. Capped client-side at 30 s,
    #: paused on blur. Logged, never used as a label (§7.5).
    dwell_ms: int | None = Field(default=None, ge=0)

    #: The client's own clock, kept for ordering diagnostics only; the server's
    #: `created_at` is authoritative.
    client_ts: str | None = None


class EventOut(BaseModel):
    event_id: int
    image_id: str
    action: Action
    created_at: str
    labels_so_far: int

    #: Set when this event replaced an earlier verdict for the same image.
    superseded_event_id: int | None = None

    #: True when this event crossed a retraining boundary and a background
    #: retrain was queued, so the UI can say the model is updating.
    retrain_queued: bool = False


class UndoResponse(BaseModel):
    event_id: int
    image_id: str
    undone: bool
    labels_so_far: int


class HistoryItem(BaseModel):
    event_id: int
    image: ImageSummary
    action: Action
    dwell_ms: int | None = None
    created_at: str
    undone: bool


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    offset: int
    limit: int


class SimilarImage(BaseModel):
    image: ImageSummary
    similarity: float


class SimilarResponse(BaseModel):
    image_id: str
    model: str
    neighbours: list[SimilarImage]


class ProfileResponse(BaseModel):
    profile_id: int
    started_at: str
    is_simulated: bool
    labels: int
    counts: dict[str, int]
    active_model_version: str | None = None
    active_rung: str | None = None
    coldstart_active: bool


class LikeRateInterval(BaseModel):
    """Beta(1 + likes, 1 + dislikes): mean and central 90% interval."""

    likes: int
    dislikes: int
    mean: float
    low: float
    high: float


class InterestCard(BaseModel):
    interest_id: int
    label: str
    n_likes: int
    #: The liked images nearest the centroid, most central first (up to 6).
    central_image_ids: list[str]
    like_rate: LikeRateInterval


class ConceptWord(BaseModel):
    concept: str
    z_score: float


class InterestEvent(BaseModel):
    """One recluster: which interest ids appeared and which were retired."""

    likes: int
    born: list[int]
    retired: list[int]


class TasteResponse(BaseModel):
    """The Profile page: interests and taste in words (Part 2 §7.2)."""

    profile_id: int
    likes: int
    dislikes: int
    #: Where the interests came from: the serving model's version, or
    #: "computed" when the serving model has none and they were fitted here.
    source: str
    interests: list[InterestCard]
    likes_in_words: list[ConceptWord]
    dislikes_in_words: list[ConceptWord]
    history: list[InterestEvent]
    #: Why something is empty, in words, when it is.
    note: str = ""


class ResetResponse(BaseModel):
    """`POST /profile/reset` archives and starts fresh; it never deletes."""

    previous_profile_id: int | None
    new_profile_id: int
    archived_labels: int


class ModelVersionOut(BaseModel):
    """One row of `rec_models`, for the Model Lab's version table."""

    version: str
    rung: str
    embedding_key: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    trained_on_events: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    created_at: str


class ModelsResponse(BaseModel):
    versions: list[ModelVersionOut]
    #: What is actually serving right now, which may be `r0_bootstrap` even
    #: when a version is active (it may not have enough labels yet).
    serving_version: str
    serving_rung: str
    default_rung: str


class TrainRequest(BaseModel):
    #: None trains the rung the online retrain would pick (api/training.py).
    rung: str | None = None
    activate: bool = True


class TrainResponse(BaseModel):
    status: str
    rung: str | None = None
    version: str | None = None
    detail: str = ""
    metrics: dict[str, Any] | None = None


class ExperimentStart(BaseModel):
    version_a: str
    version_b: str


class ExperimentOut(BaseModel):
    """An interleaving experiment and its result, recomputed from the log."""

    experiment_id: int
    version_a: str
    version_b: str
    method: str
    started_at: str
    ended_at: str | None = None
    active: bool
    wins_a: int
    wins_b: int
    ties: int
    p_value: float | None = None
    like_rate_a: float | None = None
    like_rate_b: float | None = None
    summary: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    schema_version: int
    images_ok: int
    images_selected: int
    embeddings: dict[str, int]
    active_model_version: str | None = None
    notes: list[str] = Field(default_factory=list)
    #: Free space where the data lives, against the guard the jobs respect.
    #: Below the guard, downloads and ingestion refuse to run; the feed is
    #: unaffected, so this does not change `status`.
    disk_free_gb: float | None = None
    disk_min_free_gb: int | None = None
    disk_ok: bool | None = None


# --- Analytics (Phase 13): every number on a dashboard comes from one of these.


class Interval(BaseModel):
    """A Beta(1 + k, 1 + n - k) posterior over a rate: mean and 90% interval."""

    successes: int
    trials: int
    mean: float
    low: float
    high: float


class ProfileListItem(BaseModel):
    profile_id: int
    started_at: str
    ended_at: str | None = None
    is_simulated: bool
    note: str | None = None
    labels: int


class StatsSummary(BaseModel):
    profile_id: int
    is_simulated: bool
    labels: int
    counts: dict[str, int]
    impressions: int
    like_rate: float | None = None
    like_rate_last_100: float | None = None
    serving_version: str
    serving_rung: str
    versions: int


class RungSwitch(BaseModel):
    """The swipe index at which the serving model version changed."""

    index: int
    model_version: str


class LearningCurve(BaseModel):
    """Like rate, coverage and interests on one timeline (indexed by swipe)."""

    window: int
    like_rate: list[float]
    coverage: list[float]
    interests: list[int]
    switches: list[RungSwitch]


class ClusterStat(BaseModel):
    cluster64: int
    shown: int
    rate: Interval


class DiversityStats(BaseModel):
    batch_diversity: list[float]
    batch_novelty: list[float]
    coverage_last_500: float


class CalibrationBinOut(BaseModel):
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


class CalibrationStats(BaseModel):
    bins: list[CalibrationBinOut]
    ece: float | None = None
    samples: int


class ExplorationStats(BaseModel):
    explore: Interval
    exploit: Interval


class ScoreDistribution(BaseModel):
    """Histogram of the serving model's scores over unseen images, current batch marked."""

    model_version: str
    edges: list[float]
    counts: list[int]
    batch_scores: list[float]


class ProjectionOut(BaseModel):
    """The Embedding Map, columnar: one entry per pool image in each list."""

    model: str
    image_ids: list[str]
    x: list[float]
    y: list[float]
    cluster: list[int]
    source: list[str]
    score: list[float] | None = None
    liked: list[str]
    disliked: list[str]
    saved: list[str]


class TastePoint(BaseModel):
    labels: int
    x: float
    y: float


class TastePath(BaseModel):
    """The preference vector after every N labels, and the interests, on the map."""

    model: str
    every: int
    path: list[TastePoint]
    interests: list[TastePoint]


class ReviewItem(BaseModel):
    image: ImageSummary
    status: str


class ReviewDecision(BaseModel):
    decision: Literal["keep", "remove"]


class EvalRunRequest(BaseModel):
    study: str
    seeds: int = Field(default=20, ge=1, le=50)
    swipes: int = Field(default=1_000, ge=100, le=10_000)


class EvalRunResponse(BaseModel):
    pid: int
    command: str
    log_path: str
