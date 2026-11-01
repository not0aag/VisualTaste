"""`GET /feed/next` -- the endpoint the whole system exists to serve."""

from __future__ import annotations

import logging
import sqlite3
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from visualtaste.api.deps import (
    AppState,
    DatabaseBusy,
    FeaturesUnavailable,
    ensure_profile,
    get_db,
    get_state,
)
from visualtaste.api.routes.images import summary_from_row
from visualtaste.api.schemas import FeedCardOut, FeedResponse
from visualtaste.db import repo
from visualtaste.rec.feed import (
    DEFAULT_BATCH_SIZE,
    build_batch,
    build_interleaved_batch,
    write_estimated_propensities,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feed"])


@router.get("/feed/next", response_model=FeedResponse)
def next_batch(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    background: BackgroundTasks,
    n: Annotated[int, Query(ge=1, le=20)] = DEFAULT_BATCH_SIZE,
) -> FeedResponse:
    """The next `n` cards, with their impressions already recorded.

    Returns 503 rather than an empty list when the embeddings are missing: an
    empty feed and an unbuilt corpus look identical to the UI, and the
    difference matters enormously to whoever is debugging it.
    """
    try:
        features = state.require_features()
    except FeaturesUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        profile_id = ensure_profile(connection, state)
    except DatabaseBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Read as a pair: an online retrain may swap the model mid-request.
    model, model_version = state.serving()

    experiment = state.experiment
    if experiment is not None:
        experiment_id, arms = experiment
        batch = build_interleaved_batch(
            connection, features, arms, profile_id=profile_id, experiment_id=experiment_id, n=n
        )
        return _response(connection, state, batch)

    batch = build_batch(
        connection,
        features,
        model,
        profile_id=profile_id,
        model_version=model_version,
        n=n,
        # R0 covers the collection until the active rung has enough labels to
        # be fitted. Without this, activating R1 on a fresh profile would make
        # the first request fail instead of starting the ladder.
        bootstrap=state.require_bootstrap(),
    )

    if batch.shape is not None:
        # R4's propensities: 200 replayed posterior draws, after the response
        # is sent, never on the request path (Part 2 §6.3).
        background.add_task(_estimate_propensities, state, model, features, batch)

    return _response(connection, state, batch)


def _response(connection: sqlite3.Connection, state: AppState, batch) -> FeedResponse:
    cards: list[FeedCardOut] = []
    for card in batch.cards:
        row = repo.get_image(connection, card.image_id)
        if row is None:  # pragma: no cover - the id came from the database
            continue
        cards.append(
            FeedCardOut(
                image=summary_from_row(row),
                impression_id=card.impression_id or 0,
                rank=card.rank,
                reason=card.reason,
                sentence=card.sentence,
                explanation=card.explanation,
                score=card.score,
                sampled_score=card.sampled_score,
                propensity=card.propensity,
            )
        )

    return FeedResponse(
        batch_id=batch.batch_id,
        model_version=batch.model_version,
        rung=batch.rung,
        cards=cards,
        candidates_considered=batch.candidates_considered,
        labels_so_far=state.label_count(connection),
    )


def _estimate_propensities(state: AppState, model, features, batch) -> None:
    """Background task: its own connection, and a failure is logged, not raised."""
    connection = state.connect()
    try:
        write_estimated_propensities(connection, model, features, batch)
    except Exception:
        logger.exception("propensity estimate failed for batch %s", batch.batch_id)
    finally:
        connection.close()
