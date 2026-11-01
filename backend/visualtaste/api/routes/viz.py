"""The Embedding Map's data: the 2-D projection, overlays, and the taste path.

    GET /viz/projection?model=siglip2_b16_224    all 50,000 points, columnar
    GET /viz/taste-path?model=...&every=50       where the taste sat on the map, over time

Columnar (one list per field) rather than one object per point: 50,000 objects
of six fields each is several megabytes of repeated key names, and Plotly
wants columns anyway. Thumbnails are not included; the page fetches one on
hover, from `/api/images/{id}/thumb`.

The projections are computed offline by `python -m visualtaste.jobs.project`.
Until they exist these endpoints return 404 with that instruction.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from visualtaste.api.deps import AppState, FeaturesUnavailable, get_db, get_state
from visualtaste.api.routes.stats import _profile
from visualtaste.api.schemas import ProjectionOut, TastePath, TastePoint
from visualtaste.rec.labels import build_label_set
from visualtaste.viz import projection as projections

router = APIRouter(tags=["viz"])

MODELS = ("siglip2_b16_224", "dinov2_b14")


def _require_features(state: AppState):
    try:
        return state.require_features()
    except FeaturesUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _missing(model: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"no projection for {model}; run `python -m visualtaste.jobs.project`",
    )


@router.get("/viz/projection", response_model=ProjectionOut)
def projection(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    model: Annotated[str, Query(pattern="^(siglip2_b16_224|dinov2_b14)$")] = MODELS[0],
    profile_id: int | None = None,
    with_scores: bool = True,
) -> ProjectionOut:
    features = _require_features(state)
    loaded = projections.load_coords(state.settings.features_dir, model)
    if loaded is None:
        raise _missing(model)
    ids, coords = loaded

    # Re-index the map to the feature bundle's order, so every column lines up.
    order = [features.position(i) for i in ids]
    keep = [i for i, p in enumerate(order) if p is not None]
    rows = np.array([order[i] for i in keep], dtype=np.int64)
    coords = coords[keep]

    sources = dict(connection.execute("SELECT image_id, source FROM images").fetchall())
    clusters = (
        np.asarray(features.cluster64)[rows].tolist()
        if features.cluster64 is not None
        else [-1] * len(rows)
    )

    score = None
    serving, _ = state.serving()
    if with_scores and serving.is_fitted:
        score = np.round(serving.score(features, rows).astype(float), 4).tolist()

    pid = _profile(connection, state, profile_id)
    labels = build_label_set(connection, pid)
    by_action: dict[str, list[str]] = {"like": [], "dislike": [], "save": []}
    for image_id, action in zip(labels.image_ids, labels.actions, strict=True):
        if action in by_action:
            by_action[action].append(image_id)

    image_ids = [features.image_ids[r] for r in rows]
    return ProjectionOut(
        model=model,
        image_ids=image_ids,
        x=np.round(coords[:, 0], 3).tolist(),
        y=np.round(coords[:, 1], 3).tolist(),
        cluster=[int(c) for c in clusters],
        source=[sources.get(i, "") for i in image_ids],
        score=score,
        liked=by_action["like"],
        disliked=by_action["dislike"],
        saved=by_action["save"],
    )


_PROJECTION_CACHE: dict[tuple[str, str], projections.Projection] = {}


@router.get("/viz/taste-path", response_model=TastePath)
def taste_path(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    model: Annotated[str, Query(pattern="^(siglip2_b16_224|dinov2_b14)$")] = MODELS[0],
    profile_id: int | None = None,
    every: Annotated[int, Query(ge=10, le=500)] = 50,
) -> TastePath:
    """The Rocchio preference vector after every `every` labels, placed with
    `umap.transform`, plus the serving model's interest centroids if it has any.

    The path is computed in the model's own space (SigLIP 2 or DINOv2), so the
    two maps side by side show the same taste through two different lenses.
    """
    from visualtaste.rec.r1_rocchio import rocchio_vector
    from visualtaste.rec.r5_interests import tracker_of

    features = _require_features(state)
    key = (str(state.settings.features_dir), model)
    if key not in _PROJECTION_CACHE:
        loaded = projections.load(state.settings.features_dir, model)
        if loaded is None:
            raise _missing(model)
        _PROJECTION_CACHE[key] = loaded
    fitted = _PROJECTION_CACHE[key]
    vectors = features.siglip if model.startswith("siglip") else features.dinov2
    if vectors is None:
        raise _missing(model)

    pid = _profile(connection, state, profile_id)
    labels = build_label_set(connection, pid)
    directions, counts = [], []
    for n in range(every, len(labels) + 1, every):
        liked = [
            i
            for i, a in zip(labels.image_ids[:n], labels.actions[:n], strict=True)
            if a in ("like", "save")
        ]
        disliked = [
            i
            for i, a in zip(labels.image_ids[:n], labels.actions[:n], strict=True)
            if a == "dislike"
        ]
        liked_rows, disliked_rows = features.positions(liked), features.positions(disliked)
        if len(liked_rows):
            directions.append(rocchio_vector(vectors[liked_rows], vectors[disliked_rows]))
            counts.append(n)

    path = []
    if directions:
        placed = fitted.transform(np.stack(directions))
        path = [
            TastePoint(labels=n, x=float(p[0]), y=float(p[1]))
            for n, p in zip(counts, placed, strict=True)
        ]

    interests = []
    tracker = tracker_of(state.serving()[0])
    if tracker is not None and tracker.state.k and model.startswith("siglip"):
        placed = fitted.transform(tracker.state.centroids)
        interests = [
            TastePoint(labels=int(i), x=float(p[0]), y=float(p[1]))
            for i, p in zip(tracker.state.ids, placed, strict=True)
        ]
    return TastePath(model=model, every=every, path=path, interests=interests)
