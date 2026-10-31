"""`GET /history` -- everything this profile has rated, newest first.

Shows the log as it is, including superseded and undone events, because the
History page's purpose is partly to let Alen see and change his own verdicts.
A page that hid superseded rows would make a changed verdict look like it had
never happened.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from visualtaste.api.deps import AppState, get_db, get_state
from visualtaste.api.routes.images import summary_from_row
from visualtaste.api.schemas import HistoryItem, HistoryResponse

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryResponse)
def get_history(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    action: Annotated[Literal["like", "dislike", "skip", "save", "all"], Query()] = "all",
    include_undone: Annotated[bool, Query()] = False,
    since: Annotated[str | None, Query(description="ISO date, inclusive")] = None,
    until: Annotated[str | None, Query(description="ISO date, inclusive")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryResponse:
    """Filtered, paginated history.

    The filters are built as a parameterised WHERE clause rather than by string
    interpolation -- these values come from a URL, and even on loopback a query
    assembled by concatenation is a habit worth not having.
    """
    if state.profile_id is None:
        return HistoryResponse(items=[], total=0, offset=offset, limit=limit)

    clauses = ["e.profile_id = ?"]
    params: list[object] = [state.profile_id]

    if action != "all":
        clauses.append("e.action = ?")
        params.append(action)
    if not include_undone:
        clauses.append("e.undone = 0")
    if since:
        clauses.append("date(e.created_at) >= date(?)")
        params.append(since)
    if until:
        clauses.append("date(e.created_at) <= date(?)")
        params.append(until)

    where = " AND ".join(clauses)

    total = int(
        connection.execute(f"SELECT COUNT(*) AS n FROM events e WHERE {where}", params).fetchone()[
            "n"
        ]
    )

    rows = connection.execute(
        f"""
        SELECT e.event_id, e.action, e.dwell_ms, e.created_at, e.undone, i.*
        FROM events e
        JOIN images i ON i.image_id = e.image_id
        WHERE {where}
        ORDER BY e.event_id DESC
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()

    items = [
        HistoryItem(
            event_id=row["event_id"],
            image=summary_from_row(row),
            action=row["action"],
            dwell_ms=row["dwell_ms"],
            created_at=row["created_at"],
            undone=bool(row["undone"]),
        )
        for row in rows
    ]

    return HistoryResponse(items=items, total=total, offset=offset, limit=limit)
