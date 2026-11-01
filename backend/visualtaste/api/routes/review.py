"""Review: the images the safety screen quarantined, for a human to keep or remove.

    GET  /review              quarantined images
    POST /review/{image_id}   {"decision": "keep" | "remove"}

The automated screen (`data/safety.py`) is deliberately over-cautious: it
flags the top 1% by unsafe-prompt score whatever they show, so most flagged
images are false alarms -- a red sunset that resembled "graphic violence" to a
text prompt. A person decides.

"remove" records the image in `removals` and sets its status to `removed`;
nothing is deleted from disk (brief §0.3). "keep" returns it to status `ok`.
It is not re-added to the feed's pool, because selection was fixed in Phase 5
and changing it would move every model's feature statistics.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from visualtaste.api.deps import get_db
from visualtaste.api.routes.images import ImageId, summary_from_row
from visualtaste.api.schemas import ReviewDecision, ReviewItem
from visualtaste.db import repo

router = APIRouter(tags=["review"])


@router.get("/review", response_model=list[ReviewItem])
def list_review(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ReviewItem]:
    rows = connection.execute(
        "SELECT * FROM images WHERE status = 'quarantined' ORDER BY image_id LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [ReviewItem(image=summary_from_row(row), status=row["status"]) for row in rows]


@router.post("/review/{image_id}", response_model=ReviewItem)
def decide(
    image_id: ImageId,
    payload: ReviewDecision,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
) -> ReviewItem:
    row = repo.get_image(connection, image_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no such image: {image_id}")
    if row["status"] not in ("quarantined", "removed", "ok"):
        raise HTTPException(
            status_code=409, detail=f"{image_id} is {row['status']}, not reviewable"
        )

    with repo.transaction(connection):
        if payload.decision == "remove":
            connection.execute(
                """
                INSERT OR REPLACE INTO removals (image_id, reason, at)
                VALUES (?, 'removed after safety review', datetime('now'))
                """,
                (image_id,),
            )
            connection.execute(
                "UPDATE images SET status = 'removed' WHERE image_id = ?", (image_id,)
            )
        else:
            # Not a DELETE: a reversed removal stays on record, marked as such.
            connection.execute(
                """
                UPDATE removals SET reason = 'restored after safety review', at = datetime('now')
                WHERE image_id = ?
                """,
                (image_id,),
            )
            connection.execute("UPDATE images SET status = 'ok' WHERE image_id = ?", (image_id,))

    row = repo.get_image(connection, image_id)
    return ReviewItem(image=summary_from_row(row), status=row["status"])
