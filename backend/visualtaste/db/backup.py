"""Protecting the only irreplaceable data in the project.

Everything else can be rebuilt. The images can be re-downloaded, the
derivatives re-encoded, the embeddings recomputed, the models retrained. Alen's
swipes cannot: they are a record of a few thousand decisions made once, and
losing them means starting over as a different person.

So they are backed up two ways, both cheap because the data is tiny -- a few
thousand rows, well under a megabyte.

**A database snapshot** (`conn.backup`), on shutdown and once a day. SQLite's
online backup API copies a consistent snapshot even while the database is in
use, which a file copy cannot promise: copying a WAL-mode database with `cp`
can capture a main file and a write-ahead log that disagree. Fourteen are kept
(brief §6.2), oldest pruned.

**A monthly JSONL export**, appended rather than rewritten. This is the one that
survives the failure a snapshot does not: a schema change, a corrupted database
file, or this project being abandoned and the data wanted in another form. One
JSON object per line, no schema needed to read it, greppable by hand.

The honest caveat, from DECISIONS.md #2: on this machine the archive is on the
same physical disk as everything else, so neither copy survives a drive
failure. Setting `VT_ARCHIVE_ROOT` to an external drive fixes that, and the
detection ladder will find one automatically if it is named
`VisualTasteArchive`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from visualtaste.config import Settings
from visualtaste.db import repo

logger = logging.getLogger(__name__)

# Brief §6.2: keep the newest fourteen snapshots.
SNAPSHOTS_KEPT = 14


def snapshot_database(settings: Settings) -> Path | None:
    """Copy the database to the archive and prune old snapshots.

    Returns the snapshot path, or None if there is no database to copy yet.
    """
    if not settings.db_path.exists():
        logger.info("no database at %s yet; nothing to snapshot", settings.db_path)
        return None

    destination_dir = settings.backups_dir / "db"
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    destination = destination_dir / f"visualtaste_{stamp}.db"

    source = repo.connect(settings.db_path, read_only=True)
    try:
        # Written to a temporary name and renamed, so a snapshot interrupted
        # half way never looks like a complete one.
        tmp = destination.with_name(f".{destination.name}.part")
        target = sqlite3.connect(tmp)
        try:
            # The online backup API: a consistent snapshot of a live database.
            # A plain file copy of a WAL-mode database can capture a main file
            # and a write-ahead log that disagree with each other.
            source.backup(target)
        finally:
            target.close()
        tmp.replace(destination)
    finally:
        source.close()

    size_kb = destination.stat().st_size / 1024
    logger.info("database snapshot: %s (%.0f KB)", destination.name, size_kb)

    pruned = prune_snapshots(settings)
    if pruned:
        logger.info("pruned %d old snapshot(s)", pruned)
    return destination


def prune_snapshots(settings: Settings, *, keep: int = SNAPSHOTS_KEPT) -> int:
    """Delete all but the newest `keep` snapshots. Returns how many went.

    Sorted by name rather than by modification time: the names carry a
    sortable timestamp, and copying the archive between machines does not
    preserve mtimes.
    """
    directory = settings.backups_dir / "db"
    if not directory.is_dir():
        return 0

    snapshots = sorted(directory.glob("visualtaste_*.db"))
    doomed = snapshots[:-keep] if len(snapshots) > keep else []
    for path in doomed:
        path.unlink(missing_ok=True)
    return len(doomed)


def export_events_jsonl(settings: Settings, *, profile_id: int | None = None) -> Path | None:
    """Append any events not yet exported to this month's JSONL file.

    Append-only and idempotent: the file records the highest `event_id` it
    already holds, so running this repeatedly adds only what is new and never
    duplicates a line. That is what makes it safe to call on every shutdown.
    """
    if not settings.db_path.exists():
        return None

    directory = settings.backups_dir / "events"
    directory.mkdir(parents=True, exist_ok=True)
    month = datetime.now(UTC).strftime("%Y%m")
    destination = directory / f"events_{month}.jsonl"

    already = _highest_exported_event_id(destination)

    connection = repo.connect(settings.db_path, read_only=True)
    try:
        if profile_id is None:
            rows = connection.execute(
                """
                SELECT e.*, i.source, i.license_url
                FROM events e JOIN images i ON i.image_id = e.image_id
                WHERE e.event_id > ? ORDER BY e.event_id
                """,
                (already,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT e.*, i.source, i.license_url
                FROM events e JOIN images i ON i.image_id = e.image_id
                WHERE e.event_id > ? AND e.profile_id = ? ORDER BY e.event_id
                """,
                (already, profile_id),
            ).fetchall()
    finally:
        connection.close()

    if not rows:
        logger.info("events export: nothing new since event %d", already)
        return destination

    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    logger.info("events export: appended %d event(s) to %s", len(rows), destination.name)
    return destination


def _highest_exported_event_id(path: Path) -> int:
    """The largest `event_id` already in the file, or 0.

    Reads the whole file, which is fine: one line per swipe means a year of
    heavy use is a few megabytes. Reading backwards for the last line would be
    faster and would break if the file were ever sorted differently.
    """
    if not path.exists():
        return 0

    highest = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                highest = max(highest, int(json.loads(line)["event_id"]))
            except (ValueError, KeyError, TypeError):
                # A truncated final line from an interrupted append. Skipped:
                # the worst case is re-exporting one event, and a duplicate
                # line is better than refusing to back anything up.
                continue
    return highest


def run_backup(settings: Settings) -> tuple[Path | None, Path | None]:
    """Both backups together. The daily job and the shutdown hook call this."""
    return snapshot_database(settings), export_events_jsonl(settings)
