"""Tests for the schema, migrations and query layer."""

from __future__ import annotations

import sqlite3

import pytest

from visualtaste.db import repo
from visualtaste.db.repo import (
    EXPECTED_SCHEMA_VERSION,
    SchemaVersionMismatch,
    connect,
    init_db,
    schema_version,
    table_names,
    transaction,
    verify_schema_version,
)

# Every table brief §7.4 requires. Asserted by name so a dropped table in a
# future migration is caught rather than discovered at runtime.
REQUIRED_TABLES = {
    "images",
    "image_sources",
    "embedding_models",
    "profiles",
    "rec_models",
    "impressions",
    "events",
    "interests",
    "experiments",
    "embedding_errors",
    "removals",
}


@pytest.fixture
def db(tmp_path):
    connection = init_db(tmp_path / "test.db")
    yield connection
    connection.close()


class TestMigrations:
    def test_creates_every_table_the_brief_specifies(self, db):
        assert table_names(db) >= REQUIRED_TABLES

    def test_records_the_schema_version(self, db):
        assert schema_version(db) == EXPECTED_SCHEMA_VERSION

    def test_running_init_twice_changes_nothing(self, tmp_path):
        path = tmp_path / "t.db"
        first = init_db(path)
        tables = table_names(first)
        first.close()

        second = init_db(path)

        assert table_names(second) == tables
        assert schema_version(second) == EXPECTED_SCHEMA_VERSION
        second.close()

    def test_an_existing_database_is_not_wiped(self, tmp_path):
        path = tmp_path / "t.db"
        first = init_db(path)
        repo.upsert_image(
            first,
            image_id="a" * 16,
            source="pd12m",
            source_id="x",
            license_url="https://cc0",
            orig_path="p",
        )
        first.commit()
        first.close()

        second = init_db(path)

        assert repo.get_image(second, "a" * 16) is not None
        second.close()

    def test_verify_schema_version_accepts_a_current_database(self, db):
        verify_schema_version(db)  # must not raise

    def test_verify_schema_version_rejects_a_stale_database(self, tmp_path):
        # The API calls this at startup so a stale database gives one clear
        # error instead of scattered "no such column" failures later.
        path = tmp_path / "stale.db"
        connection = connect(path)
        connection.execute("PRAGMA user_version = 0")
        connection.commit()

        with pytest.raises(SchemaVersionMismatch, match="schema version"):
            verify_schema_version(connection)
        connection.close()


class TestPragmas:
    def test_wal_mode_is_on(self, db):
        # Without WAL, a background retrain writing would block the feed reading.
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    def test_foreign_keys_are_enforced(self, db):
        # SQLite ignores foreign keys unless asked, per connection.
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_a_dangling_reference_is_actually_rejected(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO events (profile_id, image_id, action) VALUES (?, ?, ?)",
                (1, "no_such_image", "like"),
            )
            db.commit()

    def test_a_read_only_connection_cannot_write(self, tmp_path):
        path = tmp_path / "t.db"
        init_db(path).close()
        reader = connect(path, read_only=True)

        with pytest.raises(sqlite3.OperationalError):
            reader.execute("INSERT INTO removals (image_id) VALUES ('x')")
        reader.close()


class TestImages:
    def test_inserting_an_image_reports_that_it_was_created(self, db):
        created = repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="x1",
            license_url="https://cc0",
            orig_path="p",
            width=800,
            height=600,
        )

        assert created is True
        assert repo.get_image(db, "a" * 16)["width"] == 800

    def test_the_same_bytes_from_another_source_do_not_duplicate_the_image(self, db):
        # image_id is the hash of the bytes, so the same photograph arriving
        # from two of our four sources is genuinely the same row.
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="x1",
            license_url="https://cc0",
            orig_path="p",
        )

        created_again = repo.upsert_image(
            db,
            image_id="a" * 16,
            source="megalith",
            source_id="y1",
            license_url="https://pdm",
            orig_path="p2",
        )

        assert created_again is False
        assert len(repo.query(db, "SELECT * FROM images")) == 1

    def test_but_both_sources_are_recorded_as_provenance(self, db):
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="x1",
            license_url="https://cc0",
            orig_path="p",
        )
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="megalith",
            source_id="y1",
            license_url="https://pdm",
            orig_path="p2",
        )

        rows = repo.query(db, "SELECT source, source_id FROM image_sources ORDER BY source")

        assert [tuple(r) for r in rows] == [("megalith", "y1"), ("pd12m", "x1")]

    def test_re_ingesting_the_very_same_row_is_a_no_op(self, db):
        kwargs = {
            "image_id": "a" * 16,
            "source": "pd12m",
            "source_id": "x1",
            "license_url": "https://cc0",
            "orig_path": "p",
        }
        repo.upsert_image(db, **kwargs)

        assert repo.upsert_image(db, **kwargs) is False
        assert len(repo.query(db, "SELECT * FROM image_sources")) == 1

    def test_dominant_colours_round_trip_as_json(self, db):
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="x1",
            license_url="https://cc0",
            orig_path="p",
            dominant_colours=["#112233", "#445566"],
        )

        stored = repo.get_image(db, "a" * 16)["dominant_colours"]

        assert stored == '["#112233", "#445566"]'

    def test_status_defaults_to_ok(self, db):
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="x1",
            license_url="https://cc0",
            orig_path="p",
        )

        assert repo.get_image(db, "a" * 16)["status"] == "ok"

    def test_marking_a_near_duplicate_records_what_it_duplicates(self, db):
        for suffix in ("a", "b"):
            repo.upsert_image(
                db,
                image_id=suffix * 16,
                source="pd12m",
                source_id=suffix,
                license_url="https://cc0",
                orig_path="p",
            )

        repo.set_image_status(db, "b" * 16, "near_dup", dup_of="a" * 16)

        row = repo.get_image(db, "b" * 16)
        assert row["status"] == "near_dup"
        assert row["dup_of"] == "a" * 16

    def test_status_and_source_histograms(self, db):
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="1",
            license_url="https://cc0",
            orig_path="p",
        )
        repo.upsert_image(
            db,
            image_id="b" * 16,
            source="megalith",
            source_id="2",
            license_url="https://pdm",
            orig_path="p",
            status="near_dup",
        )

        assert repo.count_images_by_status(db) == {"ok": 1, "near_dup": 1}
        assert repo.count_images_by_source(db) == {"megalith": 1, "pd12m": 1}

    def test_images_with_phash_are_returned_largest_first(self, db):
        # The duplicate grouper relies on this order to keep the biggest copy.
        sizes = {"small": (400, 300), "huge": (4000, 3000), "mid": (1200, 900)}
        ids = {name: name.ljust(16, "0") for name in sizes}
        for name, (width, height) in sizes.items():
            repo.upsert_image(
                db,
                image_id=ids[name],
                source="pd12m",
                source_id=name,
                license_url="https://cc0",
                orig_path="p",
                width=width,
                height=height,
                phash="0000000000000000",
            )

        rows = repo.images_with_phash(db)

        assert [r["image_id"] for r in rows] == [ids["huge"], ids["mid"], ids["small"]]

    def test_only_ok_images_are_offered_for_deduplication(self, db):
        repo.upsert_image(
            db,
            image_id="a" * 16,
            source="pd12m",
            source_id="1",
            license_url="https://cc0",
            orig_path="p",
            phash="0" * 16,
        )
        repo.upsert_image(
            db,
            image_id="b" * 16,
            source="pd12m",
            source_id="2",
            license_url="https://cc0",
            orig_path="p",
            phash="0" * 16,
            status="quarantined",
        )

        rows = repo.images_with_phash(db)

        assert [r["image_id"] for r in rows] == ["a" * 16]

    def test_recording_a_removal_is_idempotent(self, db):
        repo.record_removal(db, "a" * 16, "unsafe")
        repo.record_removal(db, "a" * 16, "manual review")

        rows = repo.query(db, "SELECT * FROM removals")
        assert len(rows) == 1
        assert rows[0]["reason"] == "manual review"


class TestProfiles:
    def test_a_new_profile_becomes_the_active_one(self, db):
        profile_id = repo.create_profile(db, note="first")

        assert repo.active_profile_id(db) == profile_id

    def test_a_simulated_profile_is_never_the_active_one(self, db):
        # Brief §7.4: simulator data must not leak into the real profile's
        # training data or dashboards by default.
        real = repo.create_profile(db, note="real")
        repo.create_profile(db, note="sim", is_simulated=True)

        assert repo.active_profile_id(db) == real

    def test_ending_a_profile_removes_it_from_consideration(self, db):
        first = repo.create_profile(db)
        repo.end_profile(db, first)

        assert repo.active_profile_id(db) is None

    def test_after_a_reset_the_newest_profile_wins(self, db):
        old = repo.create_profile(db)
        repo.end_profile(db, old)
        new = repo.create_profile(db)

        assert repo.active_profile_id(db) == new

    def test_no_profile_yet_returns_none(self, db):
        assert repo.active_profile_id(db) is None


class TestEmbeddingBookkeeping:
    def test_registering_a_model_is_idempotent_and_updates(self, db):
        repo.register_embedding_model(
            db,
            model_key="siglip2_b16_224",
            family="open_clip",
            checkpoint="ViT-B-16-SigLIP2/webli",
            dim=768,
            preprocess="resize224",
        )
        repo.register_embedding_model(
            db,
            model_key="siglip2_b16_224",
            family="open_clip",
            checkpoint="ViT-B-16-SigLIP2/webli",
            dim=768,
            preprocess="updated",
        )

        rows = repo.query(db, "SELECT * FROM embedding_models")
        assert len(rows) == 1
        assert rows[0]["preprocess"] == "updated"

    def test_embedding_errors_are_recorded_per_image_and_model(self, db):
        repo.record_embedding_error(db, "a" * 16, "dinov2_b14", "CUDA out of memory")

        row = repo.query(db, "SELECT * FROM embedding_errors")[0]
        assert row["image_id"] == "a" * 16
        assert "CUDA" in row["error"]

    def test_a_very_long_error_is_truncated(self, db):
        repo.record_embedding_error(db, "a" * 16, "m", "x" * 5000)

        assert len(repo.query(db, "SELECT error FROM embedding_errors")[0]["error"]) <= 500


class TestTransaction:
    def test_a_failure_rolls_the_whole_block_back(self, db):
        # Used where two writes must not be separable -- inserting an event and
        # superseding the previous one, for instance.
        with pytest.raises(RuntimeError), transaction(db):
            repo.upsert_image(
                db,
                image_id="a" * 16,
                source="pd12m",
                source_id="1",
                license_url="https://cc0",
                orig_path="p",
            )
            raise RuntimeError("boom")

        assert repo.get_image(db, "a" * 16) is None

    def test_a_successful_block_commits(self, db):
        with transaction(db):
            repo.upsert_image(
                db,
                image_id="a" * 16,
                source="pd12m",
                source_id="1",
                license_url="https://cc0",
                orig_path="p",
            )

        assert repo.get_image(db, "a" * 16) is not None
