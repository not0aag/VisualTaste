"""Search by words, and a described taste for the cold start.

    GET    /search?q=foggy+forest&k=24   images closest to a description
    GET    /profile/seed-text            the current profile's description, if any
    POST   /profile/seed-text {text}     describe what you like (aims the cold start)
    DELETE /profile/seed-text            forget it

Both embed text with SigLIP 2's text tower (`search/text.py`), loaded on the
first request that needs it. The first search therefore waits for the model to
load (about 15 seconds on an idle machine, over a minute on a busy one); later
ones take milliseconds.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from visualtaste.api.deps import (
    AppState,
    DatabaseBusy,
    FeaturesUnavailable,
    ensure_profile,
    get_db,
    get_state,
)
from visualtaste.api.routes.images import summary_from_row
from visualtaste.api.schemas import SimilarImage
from visualtaste.search import text as text_search

router = APIRouter(tags=["search"])


class SearchResponse(BaseModel):
    query: str
    results: list[SimilarImage]


class SeedText(BaseModel):
    text: str = Field(min_length=1, max_length=text_search.MAX_TEXT_CHARACTERS)


class SeedTextResponse(BaseModel):
    """The described taste; `text` is None when there is none."""

    text: str | None


def _features(state: AppState):
    try:
        return state.require_features()
    except FeaturesUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _profile(connection: sqlite3.Connection, state: AppState) -> int:
    try:
        return ensure_profile(connection, state)
    except DatabaseBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _embed(state: AppState, text: str):
    try:
        return state.text_encoder.embed(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/search", response_model=SearchResponse)
def search_images(
    q: Annotated[str, Query(min_length=1, max_length=text_search.MAX_TEXT_CHARACTERS)],
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    k: Annotated[int, Query(ge=1, le=100)] = 24,
) -> SearchResponse:
    """The `k` selected images closest to `q`, best first."""
    features = _features(state)
    query = _embed(state, q)
    hits = text_search.search(features.siglip, query, k)
    ids = [features.image_ids[row] for row, _ in hits]
    rows = {
        row["image_id"]: row
        for row in connection.execute(
            f"SELECT * FROM images WHERE image_id IN ({','.join('?' for _ in ids)})", ids
        )
    }
    results = [
        SimilarImage(image=summary_from_row(rows[image_id]), similarity=round(score, 4))
        for image_id, (_, score) in zip(ids, hits, strict=True)
        if image_id in rows
    ]
    return SearchResponse(query=text_search.clean(q), results=results)


@router.get("/profile/seed-text", response_model=SeedTextResponse)
def get_seed_text(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> SeedTextResponse:
    seed = text_search.load_seed(state.settings.data, _profile(connection, state))
    return SeedTextResponse(text=seed[0] if seed else None)


@router.post("/profile/seed-text", response_model=SeedTextResponse)
def set_seed_text(
    payload: SeedText,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> SeedTextResponse:
    """Save a description; until a learned rung can serve, the cold start leans on it."""
    _features(state)
    text = text_search.clean(payload.text)
    vector = _embed(state, text)
    text_search.save_seed(state.settings.data, _profile(connection, state), text, vector)
    return SeedTextResponse(text=text)


@router.delete("/profile/seed-text", response_model=SeedTextResponse)
def clear_seed_text(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> SeedTextResponse:
    text_search.clear_seed(state.settings.data, _profile(connection, state))
    return SeedTextResponse(text=None)
