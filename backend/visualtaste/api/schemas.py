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


class ResetResponse(BaseModel):
    """`POST /profile/reset` archives and starts fresh; it never deletes."""

    previous_profile_id: int | None
    new_profile_id: int
    archived_labels: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    schema_version: int
    images_ok: int
    images_selected: int
    embeddings: dict[str, int]
    active_model_version: str | None = None
    notes: list[str] = Field(default_factory=list)
