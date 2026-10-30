"""Tests for the resumability contract (brief §6.2).

The claim being tested: a job killed at any point and re-run produces the same
final result, with no duplicates and no repeated downloads.
"""

from __future__ import annotations

import pytest

from visualtaste.data.manifest import Candidate, Manifest


def make_candidate(source_id: str, **overrides) -> Candidate:
    kwargs = {
        "source": "pd12m",
        "source_id": source_id,
        "url": f"https://example.invalid/{source_id}.jpg",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    }
    kwargs.update(overrides)
    return Candidate(**kwargs)


class TestCandidate:
    def test_a_candidate_without_a_licence_cannot_be_constructed(self):
        # The brief's §7.1 gate: no licence URL means the image is excluded.
        # Enforcing it in __post_init__ means no code path can forget.
        with pytest.raises(ValueError, match="no licence URL"):
            make_candidate("x1", license_url="")

    def test_a_fresh_candidate_is_unresolved(self):
        assert make_candidate("x1").resolved is False

    def test_succeed_records_identity_and_size(self):
        candidate = make_candidate("x1")

        candidate.succeed("abc123def4567890", 51_234)

        assert candidate.resolved is True
        assert candidate.status == "ok"
        assert candidate.image_id == "abc123def4567890"
        assert candidate.bytes == 51_234
        assert candidate.resolved_at is not None

    def test_rejected_and_failed_are_different_outcomes(self):
        # This distinction is what keeps the <5% failure gate meaningful: an
        # intentionally skipped 20 MB scan is not a failure to download.
        rejected = make_candidate("x1")
        rejected.reject("too_large")
        failed = make_candidate("x2")
        failed.fail("http_404")

        assert rejected.status == "rejected"
        assert failed.status == "failed"
        assert rejected.resolved and failed.resolved


class TestManifestRoundTrip:
    def test_an_absent_manifest_loads_as_empty(self, tmp_path):
        manifest = Manifest.load(tmp_path / "nothing.parquet", "pd12m")

        assert manifest.candidates == {}
        assert manifest.pending() == []

    def test_every_field_survives_a_save_and_load(self, tmp_path):
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all(
            [
                make_candidate(
                    "a1",
                    source_url="https://example.invalid/page",
                    author="A Photographer",
                    author_url="https://example.invalid/author",
                    title="A Title",
                    expected_md5="d41d8cd98f00b204e9800998ecf8427e",
                    declared_width=1024,
                    declared_height=768,
                )
            ]
        )
        manifest.candidates["a1"].succeed("0123456789abcdef", 12_345)
        manifest.flush(force=True)

        reloaded = Manifest.load(path, "pd12m")
        got = reloaded.candidates["a1"]

        assert got.source_id == "a1"
        assert got.license_url == "https://creativecommons.org/publicdomain/zero/1.0/"
        assert got.author == "A Photographer"
        assert got.title == "A Title"
        assert got.expected_md5 == "d41d8cd98f00b204e9800998ecf8427e"
        assert got.declared_width == 1024
        assert got.declared_height == 768
        assert got.status == "ok"
        assert got.image_id == "0123456789abcdef"
        assert got.bytes == 12_345

    def test_missing_optional_fields_come_back_as_none_not_nan(self, tmp_path):
        # pandas stores a missing string as the float NaN. Left unrepaired, that
        # reaches SQLite as the literal text "nan" and poisons the attribution
        # line in the UI, which is the kind of bug that is found months later.
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all([make_candidate("a1")])
        manifest.flush(force=True)

        got = Manifest.load(path, "pd12m").candidates["a1"]

        assert got.author is None
        assert got.title is None
        assert got.expected_md5 is None
        assert got.declared_width is None
        assert got.reason is None


class TestResumability:
    def test_replanning_never_discards_finished_work(self, tmp_path):
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all([make_candidate(f"a{i}") for i in range(5)])
        manifest.candidates["a0"].succeed("0000000000000000", 100)
        manifest.candidates["a1"].reject("too_large")
        manifest.flush(force=True)

        # Simulate a restart: the source module re-plans the same candidate list.
        resumed = Manifest.load(path, "pd12m")
        newly_added = resumed.add_all([make_candidate(f"a{i}") for i in range(5)])

        assert newly_added == 0, "re-planning must not duplicate rows"
        assert resumed.candidates["a0"].status == "ok"
        assert resumed.candidates["a0"].image_id == "0000000000000000"
        assert resumed.candidates["a1"].status == "rejected"

    def test_only_unresolved_candidates_are_handed_out_after_a_restart(self, tmp_path):
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all([make_candidate(f"a{i}") for i in range(5)])
        manifest.candidates["a0"].succeed("0000000000000000", 100)
        manifest.candidates["a1"].fail("http_404")
        manifest.candidates["a2"].reject("too_large")
        manifest.flush(force=True)

        pending = Manifest.load(path, "pd12m").pending()

        assert {c.source_id for c in pending} == {"a3", "a4"}

    def test_a_completed_manifest_has_nothing_left_to_do(self, tmp_path):
        # "Re-run downloads nothing new" -- the Phase 2 Definition of Done.
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all([make_candidate(f"a{i}") for i in range(3)])
        for candidate in manifest.candidates.values():
            candidate.succeed("0" * 16, 10)
        manifest.flush(force=True)

        resumed = Manifest.load(path, "pd12m")
        resumed.add_all([make_candidate(f"a{i}") for i in range(3)])

        assert resumed.pending() == []
        assert resumed.succeeded == 3

    def test_pending_respects_a_limit_so_a_job_can_stop_at_its_target(self, tmp_path):
        manifest = Manifest.load(tmp_path / "m.parquet", "pd12m")
        manifest.add_all([make_candidate(f"a{i}") for i in range(10)])

        assert len(manifest.pending(limit=4)) == 4

    def test_candidate_order_is_stable_so_a_seeded_plan_is_reproducible(self, tmp_path):
        path = tmp_path / "m.parquet"
        manifest = Manifest.load(path, "pd12m")
        ids = [f"a{i}" for i in range(20)]
        manifest.add_all([make_candidate(i) for i in ids])
        manifest.flush(force=True)

        reloaded = Manifest.load(path, "pd12m")

        assert [c.source_id for c in reloaded.pending()] == ids


class TestReporting:
    def test_reason_counts_group_by_cause_for_the_run_summary(self, tmp_path):
        manifest = Manifest.load(tmp_path / "m.parquet", "pd12m")
        manifest.add_all([make_candidate(f"a{i}") for i in range(6)])
        manifest.candidates["a0"].reject("too_large")
        manifest.candidates["a1"].reject("too_large")
        manifest.candidates["a2"].reject("too_small")
        manifest.candidates["a3"].fail("http_404")

        assert manifest.reason_counts("rejected") == {"too_large": 2, "too_small": 1}
        assert manifest.reason_counts("failed") == {"http_404": 1}

    def test_image_ids_lists_only_what_was_actually_stored(self, tmp_path):
        manifest = Manifest.load(tmp_path / "m.parquet", "pd12m")
        manifest.add_all([make_candidate(f"a{i}") for i in range(3)])
        manifest.candidates["a0"].succeed("1111111111111111", 10)
        manifest.candidates["a1"].reject("too_large")

        assert manifest.image_ids() == ["1111111111111111"]


class TestAtomicFlush:
    def test_flush_leaves_no_temporary_file_behind(self, tmp_path):
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all([make_candidate("a1")])

        manifest.flush(force=True)

        assert path.exists()
        assert list(tmp_path.glob(".*.part")) == []

    def test_flush_is_skipped_until_enough_has_changed(self, tmp_path):
        # Rewriting a 30,000-row parquet file once per image would dominate the
        # runtime; the manifest batches writes and only loses seconds on a crash.
        path = tmp_path / "manifest_pd12m.parquet"
        manifest = Manifest.load(path, "pd12m")
        manifest.add_all([make_candidate("a1")])
        manifest._unflushed = 0

        manifest.flush()

        assert not path.exists()
