"""Restoring the database from a snapshot (Phase 14 robustness check)."""

from __future__ import annotations

from visualtaste.db import repo
from visualtaste.db.backup import (
    list_snapshots,
    restore_database,
    snapshot_database,
    table_counts,
)
from visualtaste.jobs import backup as backup_job

IMAGE = "a" * 16


def _add_events(settings, n_events: int) -> None:
    connection = repo.init_db(settings.db_path)
    if repo.get_image(connection, IMAGE) is None:
        repo.upsert_image(
            connection,
            image_id=IMAGE,
            source="pd12m",
            source_id="s1",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            orig_path="p",
        )
    profile = repo.create_profile(connection, note="restore test")
    for _ in range(n_events):
        connection.execute(
            "INSERT INTO events (profile_id, image_id, action) VALUES (?, ?, 'like')",
            (profile, IMAGE),
        )
    connection.commit()
    connection.close()


def test_a_restore_brings_back_exactly_the_snapshot(settings):
    settings.ensure_dirs()
    _add_events(settings, 5)
    snapshot = snapshot_database(settings)
    expected = table_counts(snapshot)

    _add_events(settings, 7)  # changes made after the snapshot
    assert table_counts(settings.db_path)["events"] == 12

    restored = restore_database(settings, snapshot)

    assert restored == expected
    assert restored["events"] == 5
    # The overwritten state was snapshotted first, under its own name: a
    # same-minute snapshot gets a suffix instead of replacing the original.
    assert table_counts(snapshot)["events"] == 5
    assert table_counts(list_snapshots(settings)[0])["events"] == 12


def test_the_job_refuses_an_unknown_snapshot(settings, capsys):
    settings.ensure_dirs()
    assert backup_job.main(["--restore", "nope.db"]) == 2
    assert "no snapshot" in capsys.readouterr().err
