"""The Model Lab's numbers. Each endpoint is one documented query (ARCHITECTURE.md).

    GET /profiles                   every profile, for the SIMULATED / REAL switch
    GET /stats/summary              counters
    GET /stats/learning-curve       like rate, coverage and interests on one timeline
    GET /stats/clusters             per-cluster64 like rate with Beta intervals
    GET /stats/diversity            per-batch diversity and novelty, recent coverage
    GET /stats/calibration          predicted vs observed, ECE
    GET /stats/exploration          like rate in probe vs exploit slots
    GET /stats/scores               the serving model's score distribution, batch marked

Every endpoint takes `?profile_id=`; without it, the current real profile.
Simulated profiles are never chosen by default, so a dashboard only shows a
simulation when someone asked for it (and the UI then says SIMULATED).

The timeline is the impression log joined to its events, in impression order:
what was shown, in the order it was shown, and what happened to it.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from visualtaste.api.deps import (
    AppState,
    DatabaseBusy,
    FeaturesUnavailable,
    ensure_profile,
    get_db,
    get_state,
)
from visualtaste.api.schemas import (
    CalibrationBinOut,
    CalibrationStats,
    ClusterStat,
    DiversityStats,
    ExplorationStats,
    Interval,
    LearningCurve,
    ProfileListItem,
    RungSwitch,
    ScoreDistribution,
    StatsSummary,
)
from visualtaste.eval.metrics import (
    beta_interval,
    calibration_bins,
    expected_calibration_error,
    intra_batch_diversity,
    like_rate,
    rolling_like_rate,
)
from visualtaste.rec.labels import build_label_set, label_count

router = APIRouter(tags=["stats"])

COVERAGE_WINDOW = 500
CLUSTERS = 64
POSITIVE = ("like", "save")
#: Rungs whose impression `score` is a probability, so calibration means something.
PROBABILITY_PREFIXES = ("r3", "r4", "r6", "latency_r3", "latency_r4")


def _profile(connection, state: AppState, profile_id: int | None) -> int:
    if profile_id is not None:
        return profile_id
    try:
        return ensure_profile(connection, state)
    except DatabaseBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def timeline(connection: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    """Impressions in order, each with the action it received (NULL if unswiped).

    One row per impression. An event later undone still counts for what was
    shown and seen; for its *verdict* the live (non-undone) event is used.
    """
    return connection.execute(
        """
        SELECT i.impression_id, i.batch_id, i.image_id, i.model_version, i.reason,
               i.score, img.cluster64,
               (SELECT e.action FROM events e
                 WHERE e.impression_id = i.impression_id AND e.undone = 0
                 ORDER BY e.event_id DESC LIMIT 1) AS action
        FROM impressions i JOIN images img ON img.image_id = i.image_id
        WHERE i.profile_id = ?
        ORDER BY i.impression_id
        """,
        (profile_id,),
    ).fetchall()


def _interval(successes: int, trials: int) -> Interval:
    b = beta_interval(successes, trials)
    return Interval(successes=b.successes, trials=b.trials, mean=b.mean, low=b.low, high=b.high)


@router.get("/profiles", response_model=list[ProfileListItem])
def list_profiles(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    include_simulated: bool = False,
) -> list[ProfileListItem]:
    rows = connection.execute(
        "SELECT * FROM profiles WHERE is_simulated = 0 OR ? ORDER BY profile_id DESC LIMIT 200",
        (int(include_simulated),),
    ).fetchall()
    return [
        ProfileListItem(
            profile_id=row["profile_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            is_simulated=bool(row["is_simulated"]),
            note=row["note"],
            labels=label_count(connection, row["profile_id"]),
        )
        for row in rows
    ]


@router.get("/stats/summary", response_model=StatsSummary)
def summary(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    profile_id: int | None = None,
) -> StatsSummary:
    pid = _profile(connection, state, profile_id)
    labels = build_label_set(connection, pid)
    actions = [row["action"] for row in timeline(connection, pid) if row["action"]]
    profile = connection.execute(
        "SELECT is_simulated FROM profiles WHERE profile_id = ?", (pid,)
    ).fetchone()
    model, version = state.serving()
    versions = connection.execute("SELECT COUNT(*) AS n FROM rec_models").fetchone()["n"]
    impressions = connection.execute(
        "SELECT COUNT(*) AS n FROM impressions WHERE profile_id = ?", (pid,)
    ).fetchone()["n"]
    return StatsSummary(
        profile_id=pid,
        is_simulated=bool(profile["is_simulated"]) if profile else False,
        labels=len(labels),
        counts=labels.counts(),
        impressions=impressions,
        like_rate=like_rate(actions) if actions else None,
        like_rate_last_100=like_rate(actions[-100:]) if actions else None,
        serving_version=version,
        serving_rung=model.rung if model.is_fitted else "r0",
        versions=versions,
    )


@router.get("/stats/learning-curve", response_model=LearningCurve)
def learning_curve(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    profile_id: int | None = None,
    window: Annotated[int, Query(ge=5, le=500)] = 50,
) -> LearningCurve:
    """The F5 lesson on one timeline: like rate *and* coverage *and* interests.

    A model can raise the like rate by narrowing what it shows. Plotting
    coverage beside it makes that visible the moment it happens.
    """
    pid = _profile(connection, state, profile_id)
    rows = [row for row in timeline(connection, pid) if row["action"]]
    actions = [row["action"] for row in rows]
    clusters = [row["cluster64"] for row in rows]

    coverage = []
    for end in range(1, len(rows) + 1):
        recent = clusters[max(0, end - COVERAGE_WINDOW) : end]
        coverage.append(len({c for c in recent if c is not None}) / CLUSTERS)

    switches, previous = [], None
    for index, row in enumerate(rows):
        if row["model_version"] != previous:
            switches.append(RungSwitch(index=index, model_version=row["model_version"]))
            previous = row["model_version"]

    # Interests over time: the count recorded with each version that had any,
    # stepped at the label count the version was trained on.
    steps = connection.execute(
        """
        SELECT m.trained_on_events AS at, COUNT(i.interest_id) AS k
        FROM interests i JOIN rec_models m ON m.version = i.model_version
        WHERE i.profile_id = ? GROUP BY m.version ORDER BY at
        """,
        (pid,),
    ).fetchall()
    interests, current, cursor = [], 0, 0
    for index in range(len(rows)):
        while cursor < len(steps) and steps[cursor]["at"] <= index:
            current = steps[cursor]["k"]
            cursor += 1
        interests.append(current)

    return LearningCurve(
        window=window,
        like_rate=rolling_like_rate(actions, window=window) if actions else [],
        coverage=coverage,
        interests=interests,
        switches=switches,
    )


@router.get("/stats/clusters", response_model=list[ClusterStat])
def clusters(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    profile_id: int | None = None,
) -> list[ClusterStat]:
    pid = _profile(connection, state, profile_id)
    shown: dict[int, int] = {}
    liked: dict[int, int] = {}
    rated: dict[int, int] = {}
    for row in timeline(connection, pid):
        cluster = row["cluster64"]
        if cluster is None:
            continue
        shown[cluster] = shown.get(cluster, 0) + 1
        if row["action"] in POSITIVE or row["action"] == "dislike":
            rated[cluster] = rated.get(cluster, 0) + 1
            liked[cluster] = liked.get(cluster, 0) + (row["action"] in POSITIVE)
    return [
        ClusterStat(
            cluster64=c,
            shown=shown.get(c, 0),
            rate=_interval(liked.get(c, 0), rated.get(c, 0)),
        )
        for c in range(CLUSTERS)
    ]


@router.get("/stats/diversity", response_model=DiversityStats)
def diversity(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    profile_id: int | None = None,
    batches: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> DiversityStats:
    pid = _profile(connection, state, profile_id)
    try:
        features = state.require_features()
    except FeaturesUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    rows = timeline(connection, pid)

    by_batch: dict[str, list[int]] = {}
    for row in rows:
        position = features.position(row["image_id"])
        if position is not None:
            by_batch.setdefault(row["batch_id"], []).append(position)

    diversity_values, novelty_values, seen = [], [], []
    for positions in by_batch.values():
        vectors = features.siglip[positions]
        diversity_values.append(intra_batch_diversity(vectors) if len(positions) > 1 else 0.0)
        if seen:
            novelty_values.append(
                float((1 - (vectors @ features.siglip[seen].T).max(axis=1)).mean())
            )
        else:
            novelty_values.append(1.0)
        seen.extend(positions)

    recent = [row["cluster64"] for row in rows[-COVERAGE_WINDOW:] if row["cluster64"] is not None]
    return DiversityStats(
        batch_diversity=diversity_values[-batches:],
        batch_novelty=novelty_values[-batches:],
        coverage_last_500=len(set(recent)) / CLUSTERS,
    )


@router.get("/stats/calibration", response_model=CalibrationStats)
def calibration(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    profile_id: int | None = None,
) -> CalibrationStats:
    """Over impressions served by probability rungs (R3, R4, R6), skips excluded."""
    pid = _profile(connection, state, profile_id)
    pairs = [
        (1.0 if row["action"] in POSITIVE else 0.0, float(row["score"]))
        for row in timeline(connection, pid)
        if row["action"] in ("like", "save", "dislike")
        and row["score"] is not None
        and (row["model_version"] or "").startswith(PROBABILITY_PREFIXES)
    ]
    if not pairs:
        return CalibrationStats(bins=[], ece=None, samples=0)
    targets, scores = (np.array(v) for v in zip(*pairs, strict=True))
    bins = [
        CalibrationBinOut(
            low=b.low,
            high=b.high,
            count=b.count,
            mean_predicted=b.mean_predicted,
            observed_rate=b.observed_rate,
        )
        for b in calibration_bins(targets, scores)
        if b.count
    ]
    return CalibrationStats(
        bins=bins, ece=expected_calibration_error(targets, scores), samples=len(pairs)
    )


@router.get("/stats/exploration", response_model=ExplorationStats)
def exploration(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    profile_id: int | None = None,
) -> ExplorationStats:
    """Like rate of `explore` cards vs every other card (both rated, skips excluded)."""
    pid = _profile(connection, state, profile_id)
    counts = {"explore": [0, 0], "exploit": [0, 0]}
    for row in timeline(connection, pid):
        if row["action"] not in ("like", "save", "dislike"):
            continue
        slot = "explore" if row["reason"] == "explore" else "exploit"
        counts[slot][0] += row["action"] in POSITIVE
        counts[slot][1] += 1
    return ExplorationStats(
        explore=_interval(*counts["explore"]), exploit=_interval(*counts["exploit"])
    )


@router.get("/stats/scores", response_model=ScoreDistribution)
def scores(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
    bins: Annotated[int, Query(ge=5, le=100)] = 40,
) -> ScoreDistribution:
    """How the serving model scores every unseen image, with the last batch marked."""
    from visualtaste.rec.feed import candidate_positions

    pid = _profile(connection, state, None)
    try:
        features = state.require_features()
    except FeaturesUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    model, version = state.serving()
    if not model.is_fitted:
        model, version = state.require_bootstrap(), "r0_bootstrap"
    candidates = candidate_positions(connection, features, pid)
    values = np.asarray(model.score(features, candidates), dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins)
    last = connection.execute(
        """
        SELECT score FROM impressions
        WHERE profile_id = ? AND batch_id = (
            SELECT batch_id FROM impressions WHERE profile_id = ?
            ORDER BY impression_id DESC LIMIT 1)
        """,
        (pid, pid),
    ).fetchall()
    return ScoreDistribution(
        model_version=version,
        edges=edges.tolist(),
        counts=counts.tolist(),
        batch_scores=[row["score"] for row in last if row["score"] is not None],
    )
