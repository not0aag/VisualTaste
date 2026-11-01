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

from visualtaste.api.deps import AppState, get_db, get_state, load_rung
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
    stored = registry.get_version(connection, version)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"no such model version: {version}")

    try:
        model = load_rung(stored.rung, state.settings)
        model.load(
            registry.artifact_dir(connection, version, models_root=state.settings.models_dir)
        )
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail=f"version {version} cannot be loaded: {exc}"
        ) from exc

    registry.set_active(connection, version)
    state.swap(model, version)
    refreshed = registry.get_version(connection, version)
    assert refreshed is not None
    return _out(refreshed)
