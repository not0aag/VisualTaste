"""Recording swipes, changing verdicts, and undoing them.

The invariant this file protects
--------------------------------
Nothing is ever deleted, and at most one verdict per image is live.

Those two together are what make the log trustworthy. `eval/replay.py` has to
be able to reconstruct what the model knew at any past moment, which needs every
event ever recorded; and training needs exactly one example per image, which
needs a single live verdict. So a changed verdict is an insert plus a
supersede, in one transaction -- never an update in place.

Three operations, three shapes
------------------------------
    POST /events              a new verdict (supersedes any earlier one)
    POST /events/{id}/undo    take back a verdict, reviving the one it replaced
    (no DELETE)               by design

Undo revives rather than just marks: if a like replaced a dislike and the like
is then undone, the dislike becomes live again. Without that, undoing would
leave the image unlabelled even though the user had expressed an opinion about
it twice.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from visualtaste.api.deps import AppState, DatabaseBusy, ensure_profile, get_db, get_state
from visualtaste.api.schemas import EventIn, EventOut, UndoResponse
from visualtaste.api.training import retrain
from visualtaste.db import repo
from visualtaste.rec.labels import label_count

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])

# Dwell is capped client-side at 30 s, but a client is not to be trusted with
# an invariant: a laptop asleep for an hour would otherwise log 3,600,000 ms.
MAX_DWELL_MS = 30_000


@router.post("/events", response_model=EventOut, status_code=201)
def record_event(
    payload: EventIn,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    background: BackgroundTasks,
) -> EventOut:
    """Record a verdict, superseding any earlier live one for the same image."""
    if repo.get_image(connection, payload.image_id) is None:
        raise HTTPException(status_code=404, detail=f"no such image: {payload.image_id}")

    try:
        profile_id = ensure_profile(connection, state)
    except DatabaseBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    dwell = payload.dwell_ms
    if dwell is not None:
        dwell = min(dwell, MAX_DWELL_MS)

    previous = _live_event_for(connection, profile_id, payload.image_id)

    with repo.transaction(connection):
        cursor = connection.execute(
            """
            INSERT INTO events (impression_id, profile_id, image_id, action,
                                dwell_ms, client_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.impression_id,
                profile_id,
                payload.image_id,
                payload.action,
                dwell,
                payload.client_ts,
            ),
        )
        event_id = int(cursor.lastrowid or 0)

        if previous is not None:
            # Both statements inside the transaction: a crash between them
            # would leave two live verdicts for one image.
            connection.execute(
                "UPDATE events SET undone = 1, superseded_by = ? WHERE event_id = ?",
                (event_id, previous),
            )

    row = connection.execute(
        "SELECT created_at FROM events WHERE event_id = ?", (event_id,)
    ).fetchone()
    labels = label_count(connection, profile_id)

    retrain_queued = _should_retrain(labels, state)
    if retrain_queued:
        logger.info("label %d crossed a retraining boundary; retraining after the response", labels)
        # Runs after the response is sent, off the request path (§5.5).
        background.add_task(retrain, state)

    return EventOut(
        event_id=event_id,
        image_id=payload.image_id,
        action=payload.action,
        created_at=row["created_at"],
        labels_so_far=labels,
        superseded_event_id=previous,
        retrain_queued=retrain_queued,
    )


@router.post("/events/{event_id}/undo", response_model=UndoResponse)
def undo_event(
    event_id: int,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> UndoResponse:
    """Take back a verdict, reviving whichever one it replaced."""
    row = connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no such event: {event_id}")
    if row["undone"]:
        raise HTTPException(status_code=409, detail=f"event {event_id} has already been undone")

    with repo.transaction(connection):
        connection.execute("UPDATE events SET undone = 1 WHERE event_id = ?", (event_id,))
        # Revive the verdict this one replaced, if any. Otherwise undoing a
        # changed verdict would leave the image unlabelled even though the user
        # had expressed an opinion about it twice.
        connection.execute(
            """
            UPDATE events SET undone = 0, superseded_by = NULL
            WHERE superseded_by = ?
            """,
            (event_id,),
        )

    labels = state.label_count(connection)
    return UndoResponse(
        event_id=event_id,
        image_id=row["image_id"],
        undone=True,
        labels_so_far=labels,
    )


def _live_event_for(connection: sqlite3.Connection, profile_id: int, image_id: str) -> int | None:
    """The current live verdict for this image, if there is one."""
    row = connection.execute(
        """
        SELECT event_id FROM events
        WHERE profile_id = ? AND image_id = ? AND undone = 0 AND superseded_by IS NULL
        ORDER BY event_id DESC LIMIT 1
        """,
        (profile_id, image_id),
    ).fetchone()
    return int(row["event_id"]) if row else None


def _should_retrain(labels: int, state: AppState) -> bool:
    """Whether this label crossed a retraining boundary.

    Every `VT_RETRAIN_EVERY` labels (default 20). The retrain runs as a
    background task (`api/training.py`); the flag also lets the UI show "the
    model is updating" at the right moments.
    """
    every = max(1, state.settings.retrain_every)
    return labels > 0 and labels % every == 0
