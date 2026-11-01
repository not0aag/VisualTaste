"""The feed: score every unseen image, pick five, and record what was shown.

This is the request path the whole system exists to serve, and it has a latency
budget of 50 ms at the 95th percentile (brief §7.5). That budget shapes the
design more than anything else here.

The pipeline
------------
    1. candidates   every selected image this profile has not seen
    2. score        the active model scores all of them, in one array operation
    3. sample       R4+ redraws scores from the posterior (exploration)
    4. re-rank      MMR for diversity, plus the exploration quota
    5. explain      one reason code and sentence per card
    6. record       one `impressions` row per card, before the response

Why score everything rather than a top-k search
-----------------------------------------------
A FAISS top-k would be faster in isolation, but steps 3 and 4 need the whole
distribution: MMR compares candidates against each other, and the exploration
quota needs to know what is available in each cluster. Scoring 50,000 rows is
one matrix multiply -- a few milliseconds -- so the simple thing is also the
fast thing.

The bootstrap, and why it lives here
------------------------------------
Every rung above R0 has a minimum label count: Rocchio needs likes to average,
kNN needs neighbours to be near. Below it the model is unfitted and scoring
raises. But the feed still has to return five cards -- a fresh profile with R1
active is a completely ordinary state, and it is the state the very first swipe
happens in.

So `build_batch` takes a `bootstrap` model, used whenever the active one is not
fitted yet. That is the ladder in `§7.5` made operational: R0 covers the
collection until there is enough signal for the active rung to take over, and
the handover happens by itself. The impression records the rung that actually
served, not the one that was configured, because attributing a cold-start card
to R1 would corrupt every later comparison between them.

Why impressions are written before the response
-----------------------------------------------
An impression is the record that an image was *offered*. Without it, a like
cannot be attributed to the model version that produced it, propensity cannot
be computed, and the exploration-yield comparison has nothing to compare. So the
write happens on the serving path, not afterwards: a card that was shown but not
recorded would be a hole in the evaluation data.

The cost of being wrong here is silent. Everything would still appear to work;
only the measurements would be quietly meaningless.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from visualtaste.rec.labels import seen_image_ids

if TYPE_CHECKING:
    from visualtaste.rec.base import FeatureBundle, Recommender

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 5


@dataclass
class FeedCard:
    """One image offered to the user, with everything the UI needs to show it."""

    image_id: str
    rank: int
    score: float
    sampled_score: float | None
    propensity: float | None
    reason: str
    sentence: str
    explanation: dict[str, Any] = field(default_factory=dict)
    impression_id: int | None = None
    #: Index into the batch's candidate array, for the propensity replay.
    candidate_offset: int = -1
    #: Interleaving only: which arm picked this card, and that arm's version.
    arm: str | None = None
    model_version: str | None = None


@dataclass
class FeedBatch:
    """A batch of cards, plus the context needed to interpret them later."""

    batch_id: str
    model_version: str
    rung: str
    cards: list[FeedCard]
    candidates_considered: int
    #: The candidate rows this batch was chosen from (kept for R4's propensity
    #: estimate, which must replay the batch over exactly the same set).
    candidates: np.ndarray | None = None
    #: (batch size, exploration cards) as served, captured at serving time so
    #: a later request cannot change what the propensity replay reproduces.
    shape: tuple[int, int] | None = None
    #: Set when this batch is part of an interleaving experiment.
    experiment_id: int | None = None

    @property
    def image_ids(self) -> list[str]:
        return [card.image_id for card in self.cards]


def candidate_positions(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    profile_id: int,
) -> np.ndarray:
    """Row indices of pool images this profile has not acted on.

    The bundle *is* the selected pool (see `api/deps.AppState._load_features`),
    so every row is a candidate until it has been seen. That makes this the set
    difference between all rows and the seen ones.

    "Seen" uses `seen_image_ids`, which includes undone events: an image rated
    and then un-rated has still been seen, and offering it again as if it were
    new would look like a bug.

    Why it is written this way. The obvious version queried the 50,000 selected
    ids on every request and then looped over every row of the bundle testing
    membership -- around 130,000 Python-level operations on the hot path, which
    measured at a 245 ms p95 against the brief's 50 ms budget. Now the only
    per-request work is proportional to the number of images already *seen*, a
    few thousand at most even after months of use, plus one array write.
    """
    seen = seen_image_ids(connection, profile_id)
    if not seen:
        return np.arange(len(features), dtype=np.int64)

    seen_positions = [
        position
        for position in (features.position(image_id) for image_id in seen)
        if position is not None
    ]
    if not seen_positions:
        return np.arange(len(features), dtype=np.int64)

    available = np.ones(len(features), dtype=bool)
    available[np.asarray(seen_positions, dtype=np.int64)] = False
    return np.flatnonzero(available).astype(np.int64)


def build_batch(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    model: Recommender,
    *,
    profile_id: int,
    model_version: str,
    n: int = DEFAULT_BATCH_SIZE,
    rerank=None,
    bootstrap: Recommender | None = None,
) -> FeedBatch:
    """Produce the next `n` cards and write their impressions.

    `rerank` is an optional callable applied to the scored candidates, which is
    where MMR and the exploration quota plug in from Phase 10. Keeping it a
    parameter rather than an import means the feed can be tested, and the
    simulator can be run, with and without exploration -- which is exactly the
    comparison Phase 10 has to make.

    `bootstrap` serves in place of `model` while `model` cannot serve: too few
    labels, labels of one class only, or a failure while scoring (Part 2 §4.4).
    The reason goes into every card's `explanation_json` under `fallback`, and
    `model_version` records the rung that actually served. Passing no bootstrap
    keeps the strict behaviour: an unfitted model raises rather than silently
    serving something else, which is what the tests want.
    """
    batch_id = uuid.uuid4().hex[:12]

    # Why the configured rung is not the one serving, if it is not. Written
    # into every card's explanation so the Model Lab can show the handover.
    fallback: dict[str, Any] | None = None

    if not model.is_fitted and bootstrap is not None:
        fallback = {"from": model.rung, "reason": model.unfit_reason or "not fitted"}
        logger.info(
            "%s cannot serve (%s); %s is serving this batch",
            model.rung,
            fallback["reason"],
            bootstrap.rung,
        )
        model = bootstrap
        model_version = f"{bootstrap.rung}_bootstrap"

    candidates = candidate_positions(connection, features, profile_id)

    if len(candidates) == 0:
        logger.warning("profile %d has seen every available image", profile_id)
        return FeedBatch(batch_id, model_version, model.rung, [], 0)

    try:
        scores = np.asarray(model.score(features, candidates), dtype=np.float32)
        if len(scores) != len(candidates):
            raise ValueError(
                f"{model.rung} returned {len(scores)} scores for {len(candidates)} candidates"
            )
    except Exception as exc:
        # A rung that fails at scoring time -- a missing feature block, a
        # corrupt artefact -- must not take the feed down with it (Part 2
        # §4.4). Fall one step down the chain; with no step left, re-raise,
        # because then there is genuinely nothing to serve.
        if bootstrap is None or model is bootstrap:
            raise
        logger.exception("%s failed to score; falling back to %s", model.rung, bootstrap.rung)
        fallback = {"from": model.rung, "reason": f"scoring failed: {exc}"}
        model = bootstrap
        model_version = f"{bootstrap.rung}_bootstrap"
        scores = np.asarray(model.score(features, candidates), dtype=np.float32)

    # A rung with its own batch logic (R4: posterior draw, MMR, quota) supplies
    # it as `model.rerank`; an explicit `rerank` argument overrides it.
    rerank = rerank or getattr(model, "rerank", None)
    if rerank is not None:
        chosen, sampled, propensities = rerank(features, candidates, scores, n)
    else:
        # Plain top-n. `argpartition` avoids sorting 50,000 rows to find five.
        take = min(n, len(scores))
        top = np.argpartition(-scores, take - 1)[:take]
        chosen = top[np.argsort(-scores[top])]
        sampled = None
        propensities = None

    if propensities is None and not getattr(model, "stochastic", False):
        propensities = deterministic_propensities(model, features, candidates, chosen)

    cards: list[FeedCard] = []
    for rank, offset in enumerate(chosen):
        position = int(candidates[offset])
        explanation = model.explain(features, position)
        if fallback is not None:
            explanation.detail["fallback"] = fallback
        cards.append(
            FeedCard(
                image_id=features.image_ids[position],
                rank=rank,
                score=float(scores[offset]),
                sampled_score=None if sampled is None else float(sampled[offset]),
                propensity=None if propensities is None else float(propensities[offset]),
                reason=explanation.reason,
                sentence=explanation.sentence,
                explanation=explanation.detail,
                candidate_offset=int(offset),
            )
        )

    batch = FeedBatch(
        batch_id=batch_id,
        model_version=model_version,
        rung=model.rung,
        cards=cards,
        candidates_considered=len(candidates),
        candidates=candidates,
        shape=getattr(model, "last_batch", None) if getattr(model, "stochastic", False) else None,
    )
    record_impressions(connection, batch, profile_id=profile_id)

    # R0 keeps a round-robin position across requests, so it has to be told
    # what was actually served or it would restart from cluster 0 every time.
    note_shown = getattr(model, "note_shown", None)
    if note_shown is not None and features.cluster64 is not None:
        clusters = np.asarray(features.cluster64)
        for card in batch.cards:
            index = features.position(card.image_id)
            if index is not None:
                note_shown(int(clusters[index]))

    return batch


def deterministic_propensities(
    model: Recommender, features: FeatureBundle, candidates: np.ndarray, chosen: np.ndarray
) -> np.ndarray:
    """P(shown) for rungs whose batch is not sampled (Part 2 §6.3).

    A deterministic ranker shows its top cards with certainty: propensity 1.
    R0 picks a random unseen image from the cluster whose turn it is, so each
    card had probability 1 / (unseen images in its cluster).
    Indexed like `scores`, i.e. by candidate offset.
    """
    out = np.ones(len(candidates), dtype=np.float32)
    if model.rung == "r0" and features.cluster64 is not None:
        clusters = np.asarray(features.cluster64)[candidates]
        sizes = np.bincount(clusters - clusters.min())
        for offset in chosen:
            out[offset] = 1.0 / sizes[clusters[offset] - clusters.min()]
    return out


def model_ranking(
    model: Recommender, features: FeatureBundle, candidates: np.ndarray, n: int
) -> list[int]:
    """A rung's preference order over candidate offsets, for team-draft.

    The rung's own batch first (for R4 that includes its posterior draw, MMR
    and probe, so interleaving compares the policies as they really serve),
    then the rest of its top scores, so the draft can continue past cards the
    other arm already took.
    """
    scores = np.asarray(model.score(features, candidates), dtype=np.float32)
    head: list[int] = []
    own = getattr(model, "rerank", None)
    if own is not None:
        head = [int(c) for c in own(features, candidates, scores, n)[0]]
    take = min(len(scores), 4 * n)
    top = np.argpartition(-scores, take - 1)[:take]
    tail = [int(c) for c in top[np.argsort(-scores[top])] if int(c) not in head]
    return head + tail


def build_interleaved_batch(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    arms: dict[str, tuple[Recommender, str]],
    *,
    profile_id: int,
    experiment_id: int,
    n: int = DEFAULT_BATCH_SIZE,
) -> FeedBatch:
    """A team-draft batch from two versions (brief §7.7, Part 2 §6.4).

    `arms` is {"a": (model, version), "b": (model, version)}. Each card records
    the arm that picked it and that arm's version, so `eval.interleave.
    result_from_impressions` can credit likes from the log alone. Both models
    must be fitted; the caller falls back to an ordinary batch otherwise.
    """
    from visualtaste.eval.interleave import team_draft

    batch_id = uuid.uuid4().hex[:12]
    candidates = candidate_positions(connection, features, profile_id)
    if len(candidates) == 0:
        return FeedBatch(batch_id, "interleaved", "interleaved", [], 0)

    rankings = {arm: model_ranking(m, features, candidates, n) for arm, (m, _) in arms.items()}
    merged = team_draft(
        rankings["a"], rankings["b"], slots=n, rng=np.random.default_rng(int(batch_id, 16))
    )

    cards: list[FeedCard] = []
    for rank, (offset, arm) in enumerate(zip(merged.chosen, merged.arms, strict=True)):
        model, version = arms[arm]
        position = int(candidates[offset])
        explanation = model.explain(features, position)
        explanation.detail["arm"] = arm
        cards.append(
            FeedCard(
                image_id=features.image_ids[position],
                rank=rank,
                score=float(model.score(features, np.array([position]))[0]),
                sampled_score=None,
                propensity=None,
                reason=explanation.reason,
                sentence=explanation.sentence,
                explanation=explanation.detail,
                candidate_offset=int(offset),
                arm=arm,
                model_version=version,
            )
        )

    batch = FeedBatch(
        batch_id=batch_id,
        model_version="interleaved",
        rung="interleaved",
        cards=cards,
        candidates_considered=len(candidates),
        candidates=candidates,
        experiment_id=experiment_id,
    )
    record_impressions(connection, batch, profile_id=profile_id)
    return batch


#: Part 2 §6.3: inclusion probabilities below this are clipped, and flagged.
PROPENSITY_FLOOR = 0.005


def write_estimated_propensities(
    connection: sqlite3.Connection,
    model: Recommender,
    features: FeatureBundle,
    batch: FeedBatch,
    *,
    draws: int = 200,
) -> np.ndarray | None:
    """Estimate and store each card's inclusion probability (stochastic rungs).

    Runs after the response is sent (a FastAPI background task, or inline in
    the simulator when a study needs propensities). Probabilities below
    `PROPENSITY_FLOOR` are clipped to it and marked `propensity_clipped` in the
    impression's explanation, because an IPS weight of 1/0.0001 would let one
    card dominate any estimate built on the log.
    """
    estimate = getattr(model, "inclusion_probabilities", None)
    if estimate is None or batch.candidates is None or batch.shape is None or not batch.cards:
        return None
    n, n_explore = batch.shape
    raw = estimate(
        features,
        batch.candidates,
        [card.candidate_offset for card in batch.cards],
        n=n,
        n_explore=n_explore,
        draws=draws,
        # The batch id is hex, so the replay's seed is reproducible from it
        # (Python's own string hash is randomised per process).
        seed=int(batch.batch_id, 16),
    )
    clipped = np.maximum(raw, PROPENSITY_FLOOR)

    from visualtaste.db.repo import transaction

    with transaction(connection):
        for card, value, original in zip(batch.cards, clipped, raw, strict=True):
            card.propensity = float(value)
            connection.execute(
                """
                UPDATE impressions
                SET propensity = ?,
                    explanation_json = json_set(explanation_json,
                        '$.propensity_clipped', json(?), '$.propensity_draws', ?)
                WHERE impression_id = ?
                """,
                (
                    float(value),
                    "true" if original < PROPENSITY_FLOOR else "false",
                    draws,
                    card.impression_id,
                ),
            )
    return clipped


def record_impressions(
    connection: sqlite3.Connection, batch: FeedBatch, *, profile_id: int
) -> None:
    """Write one row per card, and stamp each card with its impression id.

    The id goes back onto the card so the client can attach it to the event it
    posts. That link is what lets an outcome be traced to the offer that caused
    it -- the basis of propensity, exploration yield and interleaving.
    """
    import json

    from visualtaste.db.repo import transaction

    with transaction(connection):
        for card in batch.cards:
            cursor = connection.execute(
                """
                INSERT INTO impressions (profile_id, image_id, model_version,
                                         experiment_id, arm,
                                         batch_id, rank, score, sampled_score,
                                         propensity, reason, explanation_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    card.image_id,
                    card.model_version or batch.model_version,
                    batch.experiment_id,
                    card.arm,
                    batch.batch_id,
                    card.rank,
                    card.score,
                    card.sampled_score,
                    card.propensity,
                    card.reason,
                    json.dumps({"sentence": card.sentence, **card.explanation}),
                ),
            )
            card.impression_id = int(cursor.lastrowid or 0)
