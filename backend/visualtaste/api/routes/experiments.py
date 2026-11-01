"""Interleaving experiments: compare two model versions on the live feed.

    GET  /experiments              every experiment, its result recomputed from the log
    POST /experiments              start one: {"version_a": ..., "version_b": ...}
    POST /experiments/{id}/stop    end it and store the result

While an experiment runs, every `/feed/next` batch is a team-draft merge of the
two versions (`rec/feed.build_interleaved_batch`): a coin decides who picks
first, then they alternate, and each card is credited to the arm that picked
it. A batch is won by the arm whose cards got more likes; the result is the
sign test over won batches (brief §7.7).

Why interleaving rather than an A/B split: both versions are judged on the same
person, the same moment, the same mood, in the same batch. Differences that
would take thousands of separate impressions to show up in an A/B test show up
in a few hundred interleaved batches.

Only one experiment runs at a time; starting a new one stops the old one.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from visualtaste.api.deps import AppState, get_db, get_state, load_version
from visualtaste.api.schemas import ExperimentOut, ExperimentStart
from visualtaste.eval.interleave import finish_experiment, result_from_impressions, start_experiment

router = APIRouter(tags=["experiments"])


def _finite(value: float) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def _out(connection: sqlite3.Connection, row: sqlite3.Row) -> ExperimentOut:
    result = result_from_impressions(connection, row["experiment_id"])
    return ExperimentOut(
        experiment_id=row["experiment_id"],
        version_a=row["version_a"],
        version_b=row["version_b"],
        method=row["method"] or "team_draft",
        started_at=row["started_at"] or "",
        ended_at=row["ended_at"],
        active=row["ended_at"] is None,
        wins_a=result.wins_a,
        wins_b=result.wins_b,
        ties=result.ties,
        p_value=_finite(result.p_value),
        like_rate_a=_finite(result.like_rate_a),
        like_rate_b=_finite(result.like_rate_b),
        summary=result.summarise(row["version_a"], row["version_b"]),
    )


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[ExperimentOut]:
    rows = connection.execute("SELECT * FROM experiments ORDER BY experiment_id DESC").fetchall()
    return [_out(connection, row) for row in rows]


@router.post("/experiments", response_model=ExperimentOut, status_code=201)
def start(
    payload: ExperimentStart,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ExperimentOut:
    if payload.version_a == payload.version_b:
        raise HTTPException(status_code=400, detail="an experiment needs two different versions")

    arms = {}
    for arm, version in (("a", payload.version_a), ("b", payload.version_b)):
        try:
            model = load_version(connection, state.settings, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=f"{version}: {exc}") from exc
        if not model.is_fitted:
            raise HTTPException(status_code=409, detail=f"{version} is not fitted")
        arms[arm] = (model, version)

    _stop_active(connection, state)
    experiment_id = start_experiment(
        connection, version_a=payload.version_a, version_b=payload.version_b
    )
    state.experiment = (experiment_id, arms)
    row = connection.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()
    return _out(connection, row)


@router.post("/experiments/{experiment_id}/stop", response_model=ExperimentOut)
def stop(
    experiment_id: int,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ExperimentOut:
    row = connection.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no such experiment: {experiment_id}")
    if row["ended_at"] is None:
        finish_experiment(
            connection, experiment_id, result_from_impressions(connection, experiment_id)
        )
    if state.experiment is not None and state.experiment[0] == experiment_id:
        state.experiment = None
    row = connection.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
    ).fetchone()
    return _out(connection, row)


def _stop_active(connection: sqlite3.Connection, state: AppState) -> None:
    active = connection.execute(
        "SELECT experiment_id FROM experiments WHERE ended_at IS NULL"
    ).fetchall()
    for row in active:
        finish_experiment(
            connection,
            row["experiment_id"],
            result_from_impressions(connection, row["experiment_id"]),
        )
    state.experiment = None
