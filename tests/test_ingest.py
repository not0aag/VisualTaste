"""End-to-end ingestion tests, including the kill-and-resume contract.

These build a small archive on disk -- originals plus a manifest -- and run the
real ingestion job against it, so the manifest reading, worker pool, database
writes, colour feature file and near-duplicate marking are all exercised
together rather than mocked apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import fixtures
from visualtaste.config import Settings
from visualtaste.data.http import image_id_for
from visualtaste.data.manifest import Candidate, Manifest
from visualtaste.db import repo
from visualtaste.jobs import ingest


@pytest.fixture
def settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        min_free_gb=0,
        # One worker: a process pool of 14 for eight test images would spend all
        # its time spawning interpreters, and spawned workers on Windows cannot
        # see a tmp_path-scoped module anyway.
        worker_processes=1,
    )


@pytest.fixture
def db(settings):
    settings.ensure_dirs()
    connection = repo.init_db(settings.db_path)
    yield connection
    connection.close()


def stock_archive(settings, source="pd12m", *, names=None):
    """Write fixture images into the archive and build a matching manifest.

    Mirrors what Phase 2 leaves behind: content-addressed originals plus a
    parquet manifest whose rows carry the licence and attribution.
    """
    staging = settings.project_root / "staging"
    built = fixtures.build_all(staging)
    chosen = names or [
        "valid_jpeg",
        "valid_png",
        "second_valid",
        "cmyk",
        "palette",
        "exif_rotated",
        "tiny",
        "truncated",
        "corrupt",
    ]

    manifest = Manifest.load(settings.manifest_path(source), source)
    stored: dict[str, str] = {}

    for index, name in enumerate(chosen):
        data = built[name].read_bytes()
        image_id = image_id_for(data)
        suffix = built[name].suffix
        destination = settings.original_path(source, image_id, suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

        candidate = Candidate(
            source=source,
            source_id=f"id{index:03d}",
            url=f"https://example.invalid/{name}",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            source_url=f"https://example.invalid/page/{name}",
            author=f"Author {index}",
            author_url="https://example.invalid/author",
            title=name,
        )
        candidate.succeed(image_id, len(data))
        manifest.add_all([candidate])
        stored[name] = image_id

    manifest.flush(force=True)
    return stored


class TestIngestSource:
    def test_valid_images_are_inserted_and_bad_ones_rejected(self, settings, db):
        stock_archive(settings)

        stats, colours = ingest.ingest_source(settings, db, "pd12m")

        # tiny, truncated and corrupt must all be rejected; the other six kept.
        assert stats.inserted == 6
        assert stats.total_rejected == 3
        assert set(stats.rejected) >= {"too_small", "truncated_or_corrupt"}
        assert len(colours) == 6

    def test_derivatives_are_written_for_every_inserted_image(self, settings, db):
        stock_archive(settings)

        ingest.ingest_source(settings, db, "pd12m")

        for image_id in repo.image_ids_with_status(db, "ok"):
            assert settings.display_path(image_id).exists()
            assert settings.thumb_path(image_id).exists()

    def test_attribution_from_the_manifest_reaches_the_database(self, settings, db):
        # Every card in the Feed shows author, licence and source link, so this
        # metadata has to survive the journey from manifest to row.
        stock_archive(settings)

        ingest.ingest_source(settings, db, "pd12m")

        row = repo.query(db, "SELECT * FROM images WHERE title = 'valid_jpeg'")[0]
        assert row["license_url"] == "https://creativecommons.org/publicdomain/zero/1.0/"
        assert row["author"].startswith("Author ")
        assert row["source_url"].startswith("https://example.invalid/page/")

    def test_every_stored_image_has_a_licence_url(self, settings, db):
        # The §7.1 gate, checked at the far end of the pipeline: an image with
        # no licence must not exist in the database at all.
        stock_archive(settings)

        ingest.ingest_source(settings, db, "pd12m")

        rows = repo.query(db, "SELECT license_url FROM images")
        assert rows
        assert all(row["license_url"] for row in rows)

    def test_phash_and_dominant_colours_are_stored(self, settings, db):
        stock_archive(settings)

        ingest.ingest_source(settings, db, "pd12m")

        row = repo.query(db, "SELECT * FROM images WHERE title = 'valid_jpeg'")[0]
        assert len(row["phash"]) == 16
        assert row["dominant_colours"].startswith("[")

    def test_dimensions_are_the_oriented_ones(self, settings, db):
        # The EXIF-rotated fixture is stored 600x400 but displays as 400x600.
        stock_archive(settings)

        ingest.ingest_source(settings, db, "pd12m")

        row = repo.query(db, "SELECT * FROM images WHERE title = 'exif_rotated'")[0]
        assert (row["width"], row["height"]) == (400, 600)

    def test_a_limit_stops_early(self, settings, db):
        stock_archive(settings)

        stats, _ = ingest.ingest_source(settings, db, "pd12m", limit=3)

        assert stats.considered == 3

    def test_nothing_to_do_is_not_an_error(self, settings, db):
        stats, colours = ingest.ingest_source(settings, db, "pd12m")

        assert stats.considered == 0
        assert colours == {}


class TestResumability:
    def test_a_second_run_ingests_nothing_new(self, settings, db):
        # The Phase 3 Definition of Done, and the property that makes a
        # multi-hour job safe to interrupt.
        stock_archive(settings)
        first, _ = ingest.ingest_source(settings, db, "pd12m")

        second, colours = ingest.ingest_source(settings, db, "pd12m")

        assert first.inserted == 6
        assert second.considered == 0
        assert colours == {}

    def test_a_run_killed_half_way_resumes_from_where_it_stopped(self, settings, db):
        stock_archive(settings)

        # Simulate a kill after three images by limiting the first pass.
        ingest.ingest_source(settings, db, "pd12m", limit=3)
        after_first = len(repo.query(db, "SELECT * FROM images"))

        ingest.ingest_source(settings, db, "pd12m")

        usable = len(repo.image_ids_with_status(db, "ok"))
        assert after_first <= 3
        assert usable == 6, "resume did not finish the remaining images"
        # The three bad files are remembered as rejected rather than retried.
        assert len(repo.query(db, "SELECT * FROM images WHERE status = 'rejected'")) == 3

    def test_resume_does_not_duplicate_rows_or_files(self, settings, db):
        stock_archive(settings)
        ingest.ingest_source(settings, db, "pd12m", limit=4)
        ingest.ingest_source(settings, db, "pd12m")
        ingest.ingest_source(settings, db, "pd12m")

        rows = repo.query(db, "SELECT image_id, COUNT(*) AS n FROM images GROUP BY image_id")

        assert all(row["n"] == 1 for row in rows)
        displays = list(settings.display_dir.rglob("*.webp"))
        assert len(displays) == len({p.name for p in displays})

    def test_a_missing_derivative_is_repaired_without_a_duplicate_row(self, settings, db):
        stock_archive(settings)
        ingest.ingest_source(settings, db, "pd12m")
        image_ids = repo.image_ids_with_status(db, "ok")
        settings.display_path(image_ids[0]).unlink()

        # The row already exists, so ingestion skips it. The repair path is
        # `--overwrite`-free by design: derivatives are rebuilt by deleting the
        # row, or by the dedicated rebuild in build_derivatives(overwrite=True).
        from visualtaste.data.derivatives import build_derivatives

        original = ingest._locate_original(settings, "pd12m", image_ids[0])
        build_derivatives(
            original, settings.display_path(image_ids[0]), settings.thumb_path(image_ids[0])
        )

        assert settings.display_path(image_ids[0]).exists()
        assert len(repo.image_ids_with_status(db, "ok")) == 6

    def test_manifest_rows_with_no_file_on_disk_are_skipped_not_fatal(self, settings, db):
        stored = stock_archive(settings)
        # Delete one original, as a partially-synced archive would have.
        victim = stored["valid_jpeg"]
        ingest._locate_original(settings, "pd12m", victim).unlink()

        stats, _ = ingest.ingest_source(settings, db, "pd12m")

        assert stats.inserted == 5


class TestExactDuplicates:
    def test_the_same_bytes_in_two_sources_collapse_to_one_row(self, settings, db):
        # Megalith and Open Images both draw on Flickr, so this happens for real.
        stock_archive(settings, "pd12m", names=["valid_jpeg"])
        stock_archive(settings, "megalith", names=["valid_jpeg"])

        ingest.ingest_source(settings, db, "pd12m")
        ingest.ingest_source(settings, db, "megalith")

        assert len(repo.image_ids_with_status(db, "ok")) == 1
        assert len(repo.query(db, "SELECT * FROM image_sources")) == 2

    def test_both_provenances_are_kept(self, settings, db):
        stock_archive(settings, "pd12m", names=["valid_jpeg"])
        stock_archive(settings, "megalith", names=["valid_jpeg"])
        ingest.ingest_source(settings, db, "pd12m")
        ingest.ingest_source(settings, db, "megalith")

        sources = {row["source"] for row in repo.query(db, "SELECT source FROM image_sources")}

        assert sources == {"pd12m", "megalith"}


class TestNearDuplicateMarking:
    def test_a_re_encoded_copy_is_folded_into_the_original(self, settings, db):
        stock_archive(settings, names=["valid_jpeg", "near_duplicate", "second_valid"])
        ingest.ingest_source(settings, db, "pd12m")

        folded = ingest.mark_near_duplicates(db)

        assert folded == 1
        near_dups = repo.query(db, "SELECT * FROM images WHERE status = 'near_dup'")
        assert len(near_dups) == 1
        assert near_dups[0]["dup_of"] is not None

    def test_the_larger_copy_survives(self, settings, db):
        stock_archive(settings, names=["valid_jpeg", "near_duplicate"])
        ingest.ingest_source(settings, db, "pd12m")
        ingest.mark_near_duplicates(db)

        kept = repo.query(db, "SELECT * FROM images WHERE status = 'ok'")
        folded = repo.query(db, "SELECT * FROM images WHERE status = 'near_dup'")

        assert len(kept) == 1
        assert folded[0]["dup_of"] == kept[0]["image_id"]

    def test_distinct_images_are_not_folded(self, settings, db):
        stock_archive(settings, names=["valid_jpeg", "second_valid", "valid_png"])
        ingest.ingest_source(settings, db, "pd12m")

        folded = ingest.mark_near_duplicates(db)

        assert folded == 0

    def test_running_it_twice_folds_nothing_further(self, settings, db):
        stock_archive(settings, names=["valid_jpeg", "near_duplicate"])
        ingest.ingest_source(settings, db, "pd12m")
        ingest.mark_near_duplicates(db)

        assert ingest.mark_near_duplicates(db) == 0


class TestColourFeatures:
    def test_writes_vectors_and_a_matching_id_index(self, settings, db):
        stock_archive(settings)
        _, colours = ingest.ingest_source(settings, db, "pd12m")

        ingest.save_colour_features(settings, db, colours)

        vectors = np.load(settings.features_dir / "colour64.npy")
        ids = np.load(settings.features_dir / "colour64_ids.npy")
        assert vectors.shape == (6, 64)
        assert len(ids) == 6
        assert vectors.dtype == np.float32

    def test_every_descriptor_is_a_unit_vector(self, settings, db):
        stock_archive(settings)
        _, colours = ingest.ingest_source(settings, db, "pd12m")
        ingest.save_colour_features(settings, db, colours)

        vectors = np.load(settings.features_dir / "colour64.npy")

        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)

    def test_the_ids_are_sorted_so_the_file_is_deterministic(self, settings, db):
        stock_archive(settings)
        _, colours = ingest.ingest_source(settings, db, "pd12m")
        ingest.save_colour_features(settings, db, colours)

        ids = np.load(settings.features_dir / "colour64_ids.npy").tolist()

        assert ids == sorted(ids)

    def test_a_second_batch_is_merged_rather_than_replacing_the_first(self, settings, db):
        stock_archive(settings, names=["valid_jpeg"])
        _, first = ingest.ingest_source(settings, db, "pd12m")
        ingest.save_colour_features(settings, db, first)

        stock_archive(settings, "megalith", names=["second_valid"])
        _, second = ingest.ingest_source(settings, db, "megalith")
        ingest.save_colour_features(settings, db, second)

        ids = np.load(settings.features_dir / "colour64_ids.npy")
        assert len(ids) == 2


class TestReconciliation:
    def test_the_summary_balances_considered_against_outcomes(self, settings, db):
        stock_archive(settings)
        stats, _ = ingest.ingest_source(settings, db, "pd12m")

        summary = ingest.render_summary({"pd12m": stats}, db)

        assert "Reconciliation" in summary
        assert "considered this run: 9" in summary
        assert "rejected: 3" in summary

    def test_the_summary_reports_the_rejection_reasons(self, settings, db):
        stock_archive(settings)
        stats, _ = ingest.ingest_source(settings, db, "pd12m")

        summary = ingest.render_summary({"pd12m": stats}, db)

        assert "too_small" in summary
        assert "truncated_or_corrupt" in summary
