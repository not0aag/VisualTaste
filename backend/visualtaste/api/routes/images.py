"""Image metadata and the two derivative files, served to the browser.

Caching
-------
Derivatives are immutable: `image_id` is the SHA-256 of the original bytes, and
the WebP is a deterministic function of that. So a URL's content can never
change, and the response carries

    Cache-Control: public, max-age=31536000, immutable

which means the browser re-requests a thumbnail it has already seen exactly
never. That matters for the Feed, which preloads the next three images, and for
the Embedding Map, which shows thumbnails on hover across 50,000 points.

`public` is safe here because nothing is served beyond `127.0.0.1`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from visualtaste.api.deps import AppState, get_db, get_state
from visualtaste.api.schemas import (
    Attribution,
    ImageDetail,
    ImageSummary,
    SimilarImage,
    SimilarResponse,
)
from visualtaste.db import repo
from visualtaste.vision.encoders import DINOV2_KEY, SIGLIP2_KEY

router = APIRouter(tags=["images"])

#: Every image id is the first 16 hex digits of the original's SHA-256
#: (`data/http.image_id_for`). Rejecting anything else before a file path is
#: built means no request can reach outside the derivatives folders.
IMAGE_ID_PATTERN = r"^[0-9a-f]{16}$"
ImageId = Annotated[
    str, PathParam(pattern=IMAGE_ID_PATTERN, description="16 lowercase hex characters")
]

IMMUTABLE_CACHE = "public, max-age=31536000, immutable"


def summary_from_row(row: sqlite3.Row) -> ImageSummary:
    """Build the wire format from a database row.

    Shared by the feed, history and similar endpoints so that an image looks
    identical everywhere in the UI -- including its attribution, which must
    appear on every card.
    """
    colours: list[str] = []
    if row["dominant_colours"]:
        try:
            colours = json.loads(row["dominant_colours"])
        except (TypeError, ValueError):
            colours = []

    return ImageSummary(
        image_id=row["image_id"],
        width=row["width"],
        height=row["height"],
        dominant_colours=colours,
        attribution=Attribution(
            license_url=row["license_url"],
            author=row["author"],
            author_url=row["author_url"],
            source=row["source"],
            source_url=row["source_url"],
            title=row["title"],
        ),
        display_url=f"/api/images/{row['image_id']}/display",
        thumb_url=f"/api/images/{row['image_id']}/thumb",
    )


def detail_from_row(row: sqlite3.Row) -> ImageDetail:
    base = summary_from_row(row)
    return ImageDetail(
        **base.model_dump(),
        status=row["status"],
        selected=bool(row["selected"]),
        cluster64=row["cluster64"],
        cluster400=row["cluster400"],
        phash=row["phash"],
        dup_of=row["dup_of"],
        created_at=row["created_at"],
    )


def fetch_row(connection: sqlite3.Connection, image_id: str) -> sqlite3.Row:
    row = repo.get_image(connection, image_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no such image: {image_id}")
    return row


@router.get("/images/{image_id}", response_model=ImageDetail)
def get_image(
    image_id: ImageId, connection: Annotated[sqlite3.Connection, Depends(get_db)]
) -> ImageDetail:
    return detail_from_row(fetch_row(connection, image_id))


def _serve(path: Path, image_id: str, kind: str) -> FileResponse:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"no {kind} derivative for {image_id}. It may not have been "
                f"ingested yet; run `python -m visualtaste.jobs.ingest`."
            ),
        )
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": IMMUTABLE_CACHE},
    )


@router.get("/images/{image_id}/display")
def get_display(image_id: ImageId, state: Annotated[AppState, Depends(get_state)]) -> FileResponse:
    """The 1024 px rendition the Feed shows."""
    return _serve(state.settings.display_path(image_id), image_id, "display")


@router.get("/images/{image_id}/thumb")
def get_thumb(image_id: ImageId, state: Annotated[AppState, Depends(get_state)]) -> FileResponse:
    """The 256 px rendition for grids and hover previews."""
    return _serve(state.settings.thumb_path(image_id), image_id, "thumb")


@router.get("/images/{image_id}/similar", response_model=SimilarResponse)
def get_similar(
    image_id: ImageId,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    model: Annotated[str, Query(pattern=f"^({SIGLIP2_KEY}|{DINOV2_KEY})$")] = SIGLIP2_KEY,
    k: Annotated[int, Query(ge=1, le=100)] = 12,
) -> SimilarResponse:
    """Nearest neighbours of one image, by either encoder.

    Both encoders are offered because they disagree usefully: SigLIP 2 finds
    images *about* the same thing, DINOv2 finds images that *look* the same.
    Seeing the two side by side is one of the clearer ways to understand what
    an embedding actually encodes, which is why the Embedding Map shows both.
    """
    fetch_row(connection, image_id)

    index = state.indexes.get(model)
    if index is None:
        raise HTTPException(
            status_code=503,
            detail=f"no {model} index is loaded; run `python -m visualtaste.jobs.index`",
        )

    query = index.vector_for(image_id)
    if query is None:
        raise HTTPException(status_code=404, detail=f"{image_id} has no {model} embedding")

    neighbours: list[SimilarImage] = []
    for neighbour_id, similarity in index.search(query, k, exclude={image_id}):
        row = repo.get_image(connection, neighbour_id)
        if row is None or row["status"] != "ok":
            # Quarantined or folded-away images stay out of the UI even when
            # they are still in the index.
            continue
        neighbours.append(SimilarImage(image=summary_from_row(row), similarity=similarity))

    return SimilarResponse(image_id=image_id, model=model, neighbours=neighbours)
