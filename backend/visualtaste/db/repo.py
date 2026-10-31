"""The database: connections, migrations, and every query in one place.

Why SQLite, and why no ORM: this is a single-user application on one laptop.
The entire interaction history will be tens of thousands of rows. SQLite in WAL
mode handles that with room to spare, needs no server, and the file can be
copied to back it up. An ORM would add a layer of indirection between Alen and
the SQL he is trying to learn from, so the queries are written out.

Three settings are applied to every connection, and all three matter:

`journal_mode=WAL`
    Readers do not block the writer and the writer does not block readers. The
    API reads while a background retrain writes, so without WAL the feed would
    intermittently stall on a locked database.

`foreign_keys=ON`
    SQLite ignores foreign keys unless you ask, per connection. The schema is
    full of references (`events.image_id`, `impressions.profile_id`), and they
    are worth nothing unenforced.

`busy_timeout`
    Rather than failing instantly on a locked database, wait. Five seconds is
    far longer than any write here takes, and it turns a rare crash into a rare
    pause.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
BUSY_TIMEOUT_MS = 5_000

# Bumped whenever a migration is added. The API refuses to start against a
# database it does not recognise, rather than failing later in a confusing way.
EXPECTED_SCHEMA_VERSION = 1


class SchemaVersionMismatch(RuntimeError):
    """The database on disk is not the schema this code expects."""


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with the pragmas this project depends on.

    `read_only` opens via a URI so that a reader genuinely cannot write; it is
    used by the analysis jobs, which should not be able to mutate state by
    accident.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # `check_same_thread=False` deserves its justification, because the default
    # exists for a good reason.
    #
    # Python's guard raises if a connection is touched from a thread other than
    # the one that created it. FastAPI runs synchronous endpoints in a thread
    # pool and -- this is the part that bites -- a generator dependency's setup
    # and its teardown can land on *different* pool threads. So a connection
    # opened for a request was being closed from another thread, and swiping
    # quickly produced intermittent 500s:
    #
    #     sqlite3.ProgrammingError: SQLite objects created in a thread can only
    #     be used in that same thread
    #
    # Disabling the check is safe here because the invariant it protects is
    # still held by construction: one connection per request, never shared
    # between concurrent requests. The threads touch it one after another, not
    # at the same time. Concurrent *writes* are handled by WAL plus
    # `busy_timeout`, which is a different mechanism entirely.
    if read_only:
        connection = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
        )
    else:
        connection = sqlite3.connect(
            db_path, timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False
        )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if not read_only:
        connection.execute("PRAGMA journal_mode = WAL")
        # NORMAL rather than FULL: on a crash we could lose the last
        # transaction, which here is one swipe. In exchange, writes do not wait
        # for a disk flush, which keeps the feed responsive. The archive backup
        # (db/backup.py) is the real durability story.
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block in one transaction, committing or rolling back as a unit.

    Used wherever two writes must not be separable -- inserting an event and
    marking the previous one superseded, for instance, where a crash in between
    would leave two live verdicts for one image.
    """
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


def schema_version(connection: sqlite3.Connection) -> int:
    """The highest applied migration number, or 0 for an empty database.

    Stored in SQLite's own `user_version` pragma rather than a table of our own:
    it is one integer, it is already there, and it cannot be accidentally
    dropped by a migration.
    """
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create or upgrade the database, then return an open connection.

    Migrations are the numbered `.sql` files in `migrations/`, applied in order,
    each in a single transaction. A half-applied migration is therefore
    impossible: either the file ran or it did not.
    """
    connection = connect(db_path)
    current = schema_version(connection)

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        number = int(path.name.split("_", 1)[0])
        if number <= current:
            continue

        logger.info("applying migration %s", path.name)
        with transaction(connection):
            connection.executescript(path.read_text(encoding="utf-8"))
            # executescript commits implicitly, so the pragma is set after it
            # to leave the version and the tables consistent.
            connection.execute(f"PRAGMA user_version = {number}")
        current = number

    logger.info("database ready at %s (schema version %d)", db_path, current)
    return connection


def verify_schema_version(connection: sqlite3.Connection) -> None:
    """Raise unless the database matches `EXPECTED_SCHEMA_VERSION`.

    Called at API startup (brief §7.8), so a stale database produces one clear
    error instead of a scattering of "no such column" failures later.
    """
    found = schema_version(connection)
    if found != EXPECTED_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"database is at schema version {found}, this code expects "
            f"{EXPECTED_SCHEMA_VERSION}. Run `python -m visualtaste.jobs.ingest "
            f"--init-only` to apply pending migrations."
        )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def upsert_image(
    connection: sqlite3.Connection,
    *,
    image_id: str,
    source: str,
    source_id: str,
    license_url: str,
    orig_path: str,
    source_url: str | None = None,
    author: str | None = None,
    author_url: str | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    bytes_: int | None = None,
    phash: str | None = None,
    dominant_colours: list[str] | None = None,
    status: str = "ok",
) -> bool:
    """Insert an image, or record an extra source for one we already have.

    Returns True if a new image row was created.

    The same photograph genuinely appears in more than one of our four sources,
    and because `image_id` is the hash of the bytes, those arrive as the same
    id. Rather than duplicating the image or discarding the provenance, the
    first arrival owns the `images` row and every arrival is recorded in
    `image_sources`.
    """
    colours_json = json.dumps(dominant_colours) if dominant_colours else None

    cursor = connection.execute(
        """
        INSERT INTO images (image_id, source, source_id, source_url, author, author_url,
                            title, license_url, orig_path, width, height, bytes, phash,
                            dominant_colours, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (image_id) DO NOTHING
        """,
        (
            image_id,
            source,
            source_id,
            source_url,
            author,
            author_url,
            title,
            license_url,
            orig_path,
            width,
            height,
            bytes_,
            phash,
            colours_json,
            status,
        ),
    )
    created = cursor.rowcount > 0

    connection.execute(
        """
        INSERT INTO image_sources (image_id, source, source_id)
        VALUES (?, ?, ?)
        ON CONFLICT (source, source_id) DO NOTHING
        """,
        (image_id, source, source_id),
    )
    return created


def set_image_status(
    connection: sqlite3.Connection,
    image_id: str,
    status: str,
    *,
    dup_of: str | None = None,
) -> None:
    """Change an image's status. The only mutation ingestion makes after insert."""
    connection.execute(
        "UPDATE images SET status = ?, dup_of = ? WHERE image_id = ?",
        (status, dup_of, image_id),
    )


def record_removal(connection: sqlite3.Connection, image_id: str, reason: str) -> None:
    """Audit an image leaving the pool, from the safety screen or the Review page."""
    connection.execute(
        """
        INSERT INTO removals (image_id, reason, at) VALUES (?, ?, datetime('now'))
        ON CONFLICT (image_id) DO UPDATE SET reason = excluded.reason, at = excluded.at
        """,
        (image_id, reason),
    )


def get_image(connection: sqlite3.Connection, image_id: str) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()


def count_images_by_status(connection: sqlite3.Connection) -> dict[str, int]:
    """Status histogram, for the reconciliation check in the Phase 3 DoD."""
    rows = connection.execute("SELECT status, COUNT(*) AS n FROM images GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


def count_images_by_source(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT source, COUNT(*) AS n FROM images GROUP BY source ORDER BY source"
    ).fetchall()
    return {row["source"]: row["n"] for row in rows}


def image_ids_with_status(connection: sqlite3.Connection, status: str = "ok") -> list[str]:
    rows = connection.execute(
        "SELECT image_id FROM images WHERE status = ? ORDER BY image_id", (status,)
    ).fetchall()
    return [row["image_id"] for row in rows]


def existing_image_ids(connection: sqlite3.Connection) -> set[str]:
    """Every image already in the database, for the ingestion job's skip check."""
    return {row["image_id"] for row in connection.execute("SELECT image_id FROM images").fetchall()}


def existing_source_keys(connection: sqlite3.Connection) -> set[tuple[str, str]]:
    """Every (source, source_id) already recorded, for the provenance pass.

    Distinct from `existing_image_ids`: an image can be known by its bytes while
    a *second* source that also supplied it is not yet recorded against it.
    """
    return {
        (row["source"], row["source_id"])
        for row in connection.execute("SELECT source, source_id FROM image_sources")
    }


def images_with_phash(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Candidates for near-duplicate detection, largest first.

    Ordered by pixel count so that the first member of a duplicate group
    encountered is the highest-resolution copy, which is the one §7.2 keeps.
    """
    return connection.execute(
        """
        SELECT image_id, phash, width, height
        FROM images
        WHERE phash IS NOT NULL AND status = 'ok'
        ORDER BY (COALESCE(width, 0) * COALESCE(height, 0)) DESC, image_id
        """
    ).fetchall()


# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------


def register_embedding_model(
    connection: sqlite3.Connection,
    *,
    model_key: str,
    family: str,
    checkpoint: str,
    dim: int,
    preprocess: str,
) -> None:
    connection.execute(
        """
        INSERT INTO embedding_models (model_key, family, checkpoint, dim, preprocess, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT (model_key) DO UPDATE SET
            family = excluded.family, checkpoint = excluded.checkpoint,
            dim = excluded.dim, preprocess = excluded.preprocess
        """,
        (model_key, family, checkpoint, dim, preprocess),
    )


def record_embedding_error(
    connection: sqlite3.Connection, image_id: str, model_key: str, error: str
) -> None:
    """One image failed to embed. Recorded so a batch is never lost to it."""
    connection.execute(
        """
        INSERT INTO embedding_errors (image_id, model_key, error, at)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (image_id, model_key, error[:500]),
    )


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def create_profile(
    connection: sqlite3.Connection, *, note: str | None = None, is_simulated: bool = False
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO profiles (started_at, note, is_simulated)
        VALUES (datetime('now'), ?, ?)
        """,
        (note, 1 if is_simulated else 0),
    )
    return int(cursor.lastrowid or 0)


def active_profile_id(connection: sqlite3.Connection) -> int | None:
    """The newest real (non-simulated), un-ended profile.

    Simulated profiles are excluded here rather than filtered by callers,
    because §7.4 requires that simulator data never leak into the real
    profile's training data or dashboards by default.
    """
    row = connection.execute(
        """
        SELECT profile_id FROM profiles
        WHERE is_simulated = 0 AND ended_at IS NULL
        ORDER BY profile_id DESC LIMIT 1
        """
    ).fetchone()
    return int(row["profile_id"]) if row else None


def end_profile(connection: sqlite3.Connection, profile_id: int) -> None:
    """Close a profile. Used by `POST /profile/reset`, which never deletes."""
    connection.execute(
        "UPDATE profiles SET ended_at = datetime('now') WHERE profile_id = ?",
        (profile_id,),
    )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def query(
    connection: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> list[sqlite3.Row]:
    """Run a read-only query. For one-off analytics that do not deserve a function."""
    return connection.execute(sql, params).fetchall()


def table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}
