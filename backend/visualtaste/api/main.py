"""The FastAPI application.

Run:  .\\scripts\\dev.ps1
      or: python -m uvicorn visualtaste.api.main:app --host 127.0.0.1 --port 8000

Loopback only
-------------
Nothing here ever binds to anything but `127.0.0.1` (brief §0.7). The CORS
policy allows only the Vite dev server's loopback origins, which exist because
the frontend runs on port 5173 in development; in a packaged build the frontend
is served as static files from the same origin and CORS is irrelevant.

There is no authentication, and that is correct rather than an omission: the
server is reachable only from this machine, and adding a login would be
security theatre plus one more thing to go wrong. The brief forbids it
explicitly (§0.10).

Startup and shutdown
--------------------
Startup loads the vectors, the FAISS indexes and the active model once, because
the feed's 50 ms budget does not allow reading 300 MB per request. Shutdown
takes a database snapshot to the archive -- the interaction history is the only
irreplaceable data in the project, and the cheapest moment to protect it is
when the process is stopping anyway.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from visualtaste.api import deps
from visualtaste.api.deps import AppState, get_db, get_state, set_offline_env
from visualtaste.api.routes import (
    eval,
    events,
    experiments,
    feed,
    history,
    images,
    models,
    profile,
    review,
    stats,
    viz,
)
from visualtaste.api.schemas import HealthResponse
from visualtaste.config import get_settings
from visualtaste.db import repo
from visualtaste.vision.encoders import DINOV2_KEY, SIGLIP2_KEY
from visualtaste.vision.store import EmbeddingStore

logger = logging.getLogger(__name__)

# The Vite dev server. Both spellings of loopback, because browsers do not
# treat them as the same origin.
DEV_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything once, serve, then snapshot the database on the way out."""
    set_offline_env()

    # Reuse a state that has already been injected, and only build one from the
    # process-wide settings if nothing has. Without this, a test that points
    # the app at a temporary corpus would have its state silently replaced here
    # -- and would then read, and write backups of, the real database.
    state = deps.peek_state()
    if state is None:
        state = AppState(settings=get_settings())
        deps.set_state(state)
    settings = state.settings
    settings.ensure_dirs()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    state.load()
    for note in state.notes:
        logger.warning("startup note: %s", note)

    # A profile with enough labels but no trained model (a fresh install over
    # an existing history, or a deleted artefact) trains one now rather than
    # waiting up to 20 swipes for the next boundary. R3 fits in well under a
    # second at these sizes, so this is cheap enough to do synchronously.
    if state.model is not None and not state.model.is_fitted and state.profile_id is not None:
        connection = state.connect()
        try:
            labels = state.label_count(connection)
        finally:
            connection.close()
        if labels >= settings.serve_min_labels:
            from visualtaste.api.training import retrain

            outcome = retrain(state)
            logger.info("startup retrain: %s %s", outcome.status, outcome.detail)
    logger.info(
        "serving on http://%s:%d (profile %s, model %s)",
        settings.api_host,
        settings.api_port,
        state.profile_id,
        state.model_version,
    )

    yield

    try:
        from visualtaste.db.backup import snapshot_database

        snapshot_database(settings)
    except Exception:  # a failed backup must not mask a clean shutdown
        logger.exception("shutdown backup failed")


app = FastAPI(
    title="VisualTaste",
    version="0.1.0",
    summary="A local image recommender that learns one person's visual taste.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

for module in (
    feed,
    events,
    images,
    history,
    profile,
    models,
    experiments,
    stats,
    viz,
    review,
    eval,
):
    app.include_router(module.router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health(
    connection: Annotated[object, Depends(get_db)],
    state: Annotated[AppState, Depends(get_state)],
) -> HealthResponse:
    """What is loaded, what is missing, and whether the feed can serve.

    Reports `degraded` rather than failing when the corpus is partially built,
    and lists exactly what is absent. During a long build that is the
    difference between "the API is broken" and "embeddings are not done yet".
    """
    import sqlite3

    assert isinstance(connection, sqlite3.Connection)

    counts = repo.count_images_by_status(connection)
    selected = int(
        connection.execute("SELECT COUNT(*) AS n FROM images WHERE selected = 1").fetchone()["n"]
    )

    embeddings: dict[str, int] = {}
    for model_key in (SIGLIP2_KEY, DINOV2_KEY):
        store = EmbeddingStore(state.settings.embedding_dir(model_key), model_key)
        meta = store.read_meta()
        embeddings[model_key] = meta.count if meta else 0

    notes = list(state.notes)
    status = "ok" if state.features is not None and not notes else "degraded"

    return HealthResponse(
        status=status,
        schema_version=repo.schema_version(connection),
        images_ok=counts.get("ok", 0),
        images_selected=selected,
        embeddings=embeddings,
        active_model_version=state.model_version,
        notes=notes,
    )


def main() -> int:  # pragma: no cover - a convenience entry point
    """`python -m visualtaste.api.main` for running without uvicorn's CLI."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "visualtaste.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
