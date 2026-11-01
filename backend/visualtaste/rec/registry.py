"""Versioned models: what was trained, when, on how much, and how well.

Why a registry rather than one live model
-----------------------------------------
The brief's whole premise is that the ladder's rungs can be compared. That is
only possible if a trained model is a *thing* with an identity -- something you
can name, score, activate, and go back to -- rather than whatever happens to be
in memory.

So every fit produces a version:

    r3_logreg_20261031_0450

which is `<rung>_<date>_<label count>`. From that name alone you can tell which
rung it is, when it was trained, and how much it had learned from, which is
exactly the context needed to read a metric next to it.

Each version owns a directory under `<DATA>/models/<version>/` holding its
artefacts, and a row in `rec_models` holding its parameters and metrics. One
version is `is_active = 1`; that is the one the feed uses.

Activation is a database write, not a restart
---------------------------------------------
`POST /models/{version}/activate` flips the flag and the API reloads the model.
That is what makes the Model Lab's comparison honest: the same code path serves
both versions, so a difference in outcomes is a difference between the models
rather than between two deployments.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visualtaste.rec.base import Recommender

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """One row of `rec_models`, as a Python object."""

    version: str
    rung: str
    embedding_key: str | None
    params: dict[str, Any]
    artifact_path: str | None
    trained_on_events: int
    metrics: dict[str, Any]
    is_active: bool
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ModelVersion:
        return cls(
            version=row["version"],
            rung=row["rung"],
            embedding_key=row["embedding_key"],
            params=json.loads(row["params_json"] or "{}"),
            artifact_path=row["artifact_path"],
            trained_on_events=int(row["trained_on_events"] or 0),
            metrics=json.loads(row["metrics_json"] or "{}"),
            is_active=bool(row["is_active"]),
            created_at=row["created_at"] or "",
        )


def make_version_name(rung: str, label_count: int, *, now: datetime | None = None) -> str:
    """`<rung>_<yyyymmdd_hhmm>_<labels>`, e.g. `r3_20261031_0450_152`.

    The label count is in the name on purpose. "R6 scored 0.71 AUC" means
    nothing without it; "R6 trained on 152 labels scored 0.71" is a finding.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M")
    return f"{rung}_{stamp}_{label_count}"


def register(
    connection: sqlite3.Connection,
    model: Recommender,
    *,
    models_root: Path,
    embedding_key: str | None = None,
    metrics: dict[str, Any] | None = None,
    activate: bool = False,
    version: str | None = None,
) -> ModelVersion:
    """Save a fitted model's artefacts and record it in `rec_models`.

    The artefacts are written before the row is inserted: a row pointing at a
    directory that does not exist would make the API fail to start, while a
    directory with no row is merely unused disk.
    """
    version = version or make_version_name(model.rung, model.trained_on_events)
    artifact_dir = models_root / version

    model.save(artifact_dir)

    connection.execute(
        """
        INSERT INTO rec_models (version, rung, embedding_key, params_json,
                                artifact_path, trained_on_events, metrics_json,
                                is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, datetime('now'))
        ON CONFLICT (version) DO UPDATE SET
            params_json = excluded.params_json,
            artifact_path = excluded.artifact_path,
            trained_on_events = excluded.trained_on_events,
            metrics_json = excluded.metrics_json
        """,
        (
            version,
            model.rung,
            embedding_key,
            json.dumps(model.params),
            str(artifact_dir),
            model.trained_on_events,
            json.dumps(metrics or {}),
        ),
    )
    connection.commit()

    if activate:
        set_active(connection, version)

    logger.info(
        "registered %s (%s, %d labels)%s",
        version,
        model.rung,
        model.trained_on_events,
        " and activated it" if activate else "",
    )
    return get_version(connection, version)  # type: ignore[return-value]


def set_active(connection: sqlite3.Connection, version: str) -> None:
    """Make one version active, in a single transaction.

    Both statements together, because a moment with no active model would make
    the feed fall back to cold start -- and a moment with two would make
    "which model served this impression?" unanswerable.
    """
    row = connection.execute(
        "SELECT version FROM rec_models WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no such model version: {version!r}")

    from visualtaste.db.repo import transaction

    with transaction(connection):
        connection.execute("UPDATE rec_models SET is_active = 0 WHERE is_active = 1")
        connection.execute("UPDATE rec_models SET is_active = 1 WHERE version = ?", (version,))
    logger.info("activated model version %s", version)


def deactivate_all(connection: sqlite3.Connection) -> None:
    """No version active: the feed serves cold start (used by profile reset)."""
    connection.execute("UPDATE rec_models SET is_active = 0 WHERE is_active = 1")
    connection.commit()
    logger.info("deactivated every model version")


def active_version(connection: sqlite3.Connection) -> ModelVersion | None:
    row = connection.execute("SELECT * FROM rec_models WHERE is_active = 1 LIMIT 1").fetchone()
    return ModelVersion.from_row(row) if row else None


def get_version(connection: sqlite3.Connection, version: str) -> ModelVersion | None:
    row = connection.execute("SELECT * FROM rec_models WHERE version = ?", (version,)).fetchone()
    return ModelVersion.from_row(row) if row else None


def list_versions(
    connection: sqlite3.Connection, *, rung: str | None = None, limit: int = 100
) -> list[ModelVersion]:
    """Versions, newest first. The Model Lab's table."""
    if rung:
        rows = connection.execute(
            """
            SELECT * FROM rec_models WHERE rung = ?
            ORDER BY created_at DESC, version DESC LIMIT ?
            """,
            (rung, limit),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM rec_models ORDER BY created_at DESC, version DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [ModelVersion.from_row(row) for row in rows]


def update_metrics(connection: sqlite3.Connection, version: str, metrics: dict[str, Any]) -> None:
    """Attach evaluation results to a version, merging with what is there.

    Merging rather than replacing, because a version is typically scored more
    than once -- a replay run, then a simulator study, then an interleaving
    experiment -- and each should add to the record rather than erase it.
    """
    existing = get_version(connection, version)
    if existing is None:
        raise KeyError(f"no such model version: {version!r}")

    merged = {**existing.metrics, **metrics}
    connection.execute(
        "UPDATE rec_models SET metrics_json = ? WHERE version = ?",
        (json.dumps(merged), version),
    )
    connection.commit()


def artifact_dir(connection: sqlite3.Connection, version: str, *, models_root: Path) -> Path:
    """Where a version's artefacts live, preferring the recorded path.

    Falls back to `<models_root>/<version>` so that a database copied between
    machines still resolves, rather than pointing at an absolute path that only
    existed on the machine where training happened.
    """
    stored = get_version(connection, version)
    if stored and stored.artifact_path:
        recorded = Path(stored.artifact_path)
        if recorded.is_dir():
            return recorded
    return models_root / version
