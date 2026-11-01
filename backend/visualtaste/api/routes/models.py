"""Model versions: list them, train one, activate one. The Model Lab's controls.

    GET  /models                     every version, newest first, plus what is serving
    POST /models/train               fit a rung now (body: {"rung": "r3"}), optionally activate
    POST /models/{version}/activate  serve an existing version

Training here runs synchronously, unlike the online retrain: the person who
pressed "train" is waiting for the result and wants to see the validation curve.
R3 fits in well under a second at the label counts this app will see. R6 is not
trainable from here at all -- it is a CLI job (`jobs.train_neural`) that must
never run inside the API process (brief §7.5).
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from visualtaste.api.deps import AppState, get_db, get_state, load_version
from visualtaste.api.schemas import ModelsResponse, ModelVersionOut, TrainRequest, TrainResponse
from visualtaste.api.training import ONLINE_RUNGS, retrain
from visualtaste.rec import registry

router = APIRouter(tags=["models"])


def _out(version: registry.ModelVersion) -> ModelVersionOut:
    return ModelVersionOut(
        version=version.version,
        rung=version.rung,
        embedding_key=version.embedding_key,
        params=version.params,
        trained_on_events=version.trained_on_events,
        metrics=version.metrics,
        is_active=version.is_active,
        created_at=version.created_at,
    )


@router.get("/models", response_model=ModelsResponse)
def list_models(
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ModelsResponse:
    model, version = state.serving()
    return ModelsResponse(
        versions=[_out(v) for v in registry.list_versions(connection)],
        serving_version=version,
        serving_rung=model.rung if model.is_fitted else "r0",
        default_rung=state.settings.default_rung,
    )


@router.post("/models/train", response_model=TrainResponse)
def train_model(
    payload: TrainRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> TrainResponse:
    if payload.rung is not None and payload.rung not in ONLINE_RUNGS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{payload.rung!r} cannot be trained from the API. Online rungs are "
                f"{sorted(ONLINE_RUNGS)}; R6 trains with `python -m visualtaste.jobs.train_neural`."
            ),
        )
    outcome = retrain(state, rung=payload.rung, activate=payload.activate)
    if outcome.status == "busy":
        raise HTTPException(status_code=409, detail=outcome.detail)
    return TrainResponse(
        status=outcome.status,
        rung=outcome.rung,
        version=outcome.version,
        detail=outcome.detail,
        metrics=outcome.metrics,
    )


@router.post("/models/{version}/activate", response_model=ModelVersionOut)
def activate_model(
    version: str,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ModelVersionOut:
    """Load a stored version and make it the one serving.

    Loaded *before* it is marked active, so a broken artefact is a 409 here
    rather than a model that the next restart cannot load.
    """
    if registry.get_version(connection, version) is None:
        raise HTTPException(status_code=404, detail=f"no such model version: {version}")

    try:
        model = load_version(connection, state.settings, version)
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail=f"version {version} cannot be loaded: {exc}"
        ) from exc

    registry.set_active(connection, version)
    state.swap(model, version)
    refreshed = registry.get_version(connection, version)
    assert refreshed is not None
    return _out(refreshed)


@router.post("/models/{version}/evaluate", response_model=ModelVersionOut)
def evaluate_model(
    version: str,
    connection: Annotated[sqlite3.Connection, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> ModelVersionOut:
    """Fill a version's row of the Model Lab table: AUC, log loss, NDCG@10, ECE, p95.

    - AUC, log loss, NDCG@10: time-ordered replay (eval/replay.py) of this
      version's rung over the current profile's history, refitted at each cut.
    - ECE: over the impressions this version actually served, skips excluded.
    - p95: of `score` over the whole pool, 50 timed runs of the loaded model.

    Merged into `metrics_json` (registry.update_metrics), so a version keeps
    every evaluation it has had.
    """
    import time

    import numpy as np

    from visualtaste.eval.metrics import expected_calibration_error
    from visualtaste.eval.replay import replay
    from visualtaste.eval.study import RungSpec, replay_fit_and_score
    from visualtaste.rec.labels import build_label_set

    stored = registry.get_version(connection, version)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"no such model version: {version}")
    features = state.require_features()
    profile_id = state.profile_id
    if profile_id is None:
        raise HTTPException(status_code=409, detail="no profile to evaluate on")

    labels = build_label_set(connection, profile_id)
    metrics: dict[str, float | int | None] = {"evaluated_on_labels": len(labels)}
    if stored.rung not in ("r0", "r6"):
        replayed = replay(
            labels.actions,
            replay_fit_and_score(
                RungSpec.parse(stored.rung),
                state.settings,
                features,
                labels.image_ids,
                labels.actions,
            ),
            label=version,
        )
        metrics.update(
            replay_auc=replayed.mean("auc"),
            replay_log_loss=replayed.mean("log_loss"),
            replay_ndcg_at_10=replayed.mean("ndcg_at_10"),
            replay_cuts=replayed.usable_cuts,
        )

    served = connection.execute(
        """
        SELECT i.score, e.action FROM impressions i
        JOIN events e ON e.impression_id = i.impression_id AND e.undone = 0
        WHERE i.model_version = ? AND e.action != 'skip' AND i.score IS NOT NULL
        """,
        (version,),
    ).fetchall()
    if served and stored.rung not in ("r0", "r1", "r2", "r5"):
        targets = np.array([row["action"] in ("like", "save") for row in served], dtype=float)
        metrics["ece"] = expected_calibration_error(
            targets, np.array([row["score"] for row in served])
        )

    try:
        model = load_version(connection, state.settings, version)
        candidates = np.arange(len(features))
        model.score(features, candidates)  # warm-up (R6 builds its pool cache here)
        timings = []
        for _ in range(50):
            started = time.perf_counter()
            model.score(features, candidates)
            timings.append((time.perf_counter() - started) * 1_000)
        metrics["score_p95_ms"] = float(np.percentile(timings, 95))
    except (OSError, KeyError, ValueError) as exc:
        metrics["score_p95_ms"] = None
        metrics["latency_error"] = str(exc)  # type: ignore[assignment]

    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}
    registry.update_metrics(connection, version, clean)
    refreshed = registry.get_version(connection, version)
    assert refreshed is not None
    return _out(refreshed)
