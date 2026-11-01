"""The profile, and resetting it -- which archives rather than deletes.

Why reset does not delete
-------------------------
`POST /profile/reset` closes the current profile and starts a new one. The old
events stay exactly where they are, attached to the old `profile_id`.

Three reasons. The history is the only irreplaceable data in the project --
every image and embedding can be regenerated, those swipes cannot. A reset is
often a mistake, and an undoable mistake is a much better one. And the old
profile remains available to the evaluation harness, so a replay comparison can
still be run over it afterwards.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from visualtaste.api.deps import AppState, DatabaseBusy, ensure_profile, get_db, get_state
from visualtaste.api.schemas import ProfileResponse, ResetResponse
from visualtaste.db import repo
from visualtaste.rec.labels import build_label_set, label_count
from visualtaste.rec.r0_coldstart import coldstart_is_active

router = APIRouter(tags=["profile"])


@router.get("/profile", response_model=ProfileResponse)
def get_profile(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ProfileResponse:
    try:
        profile_id = ensure_profile(connection, state)
    except DatabaseBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = connection.execute(
        "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    labels = label_count(connection, profile_id)
    counts = build_label_set(connection, profile_id).counts()
    model, version = state.serving()

    return ProfileResponse(
        profile_id=profile_id,
        started_at=row["started_at"] if row else "",
        is_simulated=bool(row["is_simulated"]) if row else False,
        labels=labels,
        counts=counts,
        active_model_version=version,
        # The rung actually serving: R0 while the configured one is unfitted.
        active_rung=model.rung if model.is_fitted else "r0",
        coldstart_active=coldstart_is_active(labels),
    )


@router.post("/profile/reset", response_model=ResetResponse)
def reset_profile(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ResetResponse:
    """Close the current profile and begin a new, empty one.

    The old events are left untouched. Nothing is deleted here or anywhere
    else in this application.
    """
    previous = state.profile_id
    archived = label_count(connection, previous) if previous is not None else 0

    with repo.transaction(connection):
        if previous is not None:
            repo.end_profile(connection, previous)
        new_profile = repo.create_profile(
            connection, note=f"reset from profile {previous}" if previous else "first"
        )

    state.profile_id = new_profile

    # Back to cold start: a fresh profile has no labels, so any model trained
    # on the old one would be describing someone else's taste. The old version
    # is deactivated too, or a restart would load it straight back. The default
    # rung starts unfitted and takes over once the new profile can train it.
    from visualtaste.api.deps import load_rung
    from visualtaste.rec import registry

    registry.deactivate_all(connection)
    state.swap(load_rung(state.settings.default_rung, state.settings), "r0_bootstrap")

    return ResetResponse(
        previous_profile_id=previous,
        new_profile_id=new_profile,
        archived_labels=archived,
    )
