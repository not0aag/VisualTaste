"""Tests for the layout, the archive ladder and the disk guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualtaste import config as config_module
from visualtaste.config import (
    ARCHIVE_DRIVE_MIN_FREE_GB,
    DiskGuardTripped,
    Settings,
    check_disk_headroom,
    free_gb,
    resolve_archive_root,
)


class TestLayout:
    def test_roots_derive_from_project_root(self, tmp_path):
        s = Settings(project_root=tmp_path, archive_root=tmp_path / "archive")
        assert s.data == tmp_path / "data"
        assert s.db_path == tmp_path / "data" / "db" / "visualtaste.db"
        assert s.hf_home == tmp_path / "data" / "cache" / "hf"
        assert s.originals_dir == tmp_path / "archive" / "originals"

    def test_paths_are_sharded_by_the_first_two_hex_characters(self, settings):
        image_id = "a1b2c3d4e5f60718"

        assert settings.original_path(image_id, ".jpg").parent.name == "a1"
        assert settings.display_path(image_id).parent.name == "a1"
        assert settings.thumb_path(image_id).parent.name == "a1"

    def test_original_path_accepts_extension_with_or_without_dot(self, settings):
        image_id = "0123456789abcdef"
        assert settings.original_path(image_id, "jpg") == settings.original_path(image_id, ".jpg")
        assert settings.original_path(image_id, "jpg").name == f"{image_id}.jpg"

    def test_derivatives_are_always_webp(self, settings):
        image_id = "0123456789abcdef"
        assert settings.display_path(image_id).suffix == ".webp"
        assert settings.thumb_path(image_id).suffix == ".webp"

    def test_ensure_dirs_creates_everything_and_is_idempotent(self, settings):
        settings.ensure_dirs()
        settings.ensure_dirs()  # must not raise on the second call

        for path in (
            settings.originals_dir,
            settings.manifests_dir,
            settings.backups_dir / "db",
            settings.backups_dir / "events",
            settings.display_dir,
            settings.thumb_dir,
            settings.embeddings_dir,
            settings.features_dir,
            settings.indexes_dir,
            settings.models_dir,
            settings.db_path.parent,
            settings.hf_home,
        ):
            assert path.is_dir(), f"{path} was not created"

    def test_apply_hf_env_points_the_cache_inside_the_project(self, settings, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        settings.apply_hf_env()

        import os

        assert os.environ["HF_HOME"] == str(settings.hf_home)
        assert settings.hf_home.is_dir()

    def test_worker_processes_leaves_headroom_for_the_os(self):
        s = Settings(archive_root=Path("/tmp/archive"))
        assert s.worker_processes >= 1
        assert s.worker_processes <= max(1, (__import__("os").cpu_count() or 4) - 2)


class TestDiskGuard:
    def test_free_gb_reports_a_positive_number_for_an_existing_path(self, tmp_path):
        assert free_gb(tmp_path) > 0

    def test_free_gb_walks_up_to_an_existing_parent(self, tmp_path):
        # The guard has to answer before the directory it protects exists --
        # on a fresh install, nothing under data/ has been created yet.
        missing = tmp_path / "not" / "created" / "yet"
        assert free_gb(missing) == pytest.approx(free_gb(tmp_path), rel=0.01)

    def test_headroom_check_passes_when_there_is_room(self, tmp_path):
        check_disk_headroom(tmp_path, min_free=0.001)

    def test_headroom_check_raises_when_the_disk_is_too_full(self, tmp_path):
        # Ask for more space than any real drive has, so the guard must trip.
        with pytest.raises(DiskGuardTripped, match="below the"):
            check_disk_headroom(tmp_path, min_free=1_000_000)

    def test_the_error_tells_the_reader_how_to_recover(self, tmp_path):
        with pytest.raises(DiskGuardTripped) as excinfo:
            check_disk_headroom(tmp_path, min_free=1_000_000)

        message = str(excinfo.value)
        assert "VT_MIN_FREE_GB" in message
        assert "resumes" in message


class TestArchiveLadder:
    """The ladder is the brief's section 6.1. Each rule is tested in isolation.

    `_candidate_drives` and `free_gb` are patched because the real answers
    depend on what is plugged into the machine running the suite.
    """

    def test_rule_2_prefers_an_existing_archive_folder(self, tmp_path, monkeypatch):
        existing = tmp_path / "D_drive" / "VisualTasteArchive"
        existing.mkdir(parents=True)

        monkeypatch.setattr(config_module, "_candidate_drives", lambda: [("D", 500.0)])
        # Rule 2 probes `D:/VisualTasteArchive`; redirect that probe to tmp_path.
        real_is_dir = Path.is_dir
        monkeypatch.setattr(
            Path,
            "is_dir",
            lambda self: True if str(self) == "D:\\VisualTasteArchive" else real_is_dir(self),
        )

        path, rule = resolve_archive_root(tmp_path)
        assert path == Path("D:/VisualTasteArchive")
        assert "rule 2" in rule

    def test_rule_3_claims_a_roomy_drive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            config_module,
            "_candidate_drives",
            lambda: [("E", ARCHIVE_DRIVE_MIN_FREE_GB + 50)],
        )
        monkeypatch.setattr(Path, "is_dir", lambda self: False)

        path, rule = resolve_archive_root(tmp_path)
        assert path == Path("E:/VisualTasteArchive")
        assert "rule 3" in rule

    def test_rule_3_skips_a_drive_that_is_too_small(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            config_module,
            "_candidate_drives",
            lambda: [("E", ARCHIVE_DRIVE_MIN_FREE_GB - 1)],
        )
        monkeypatch.setattr(Path, "is_dir", lambda self: False)
        monkeypatch.setattr(config_module, "free_gb", lambda _: ARCHIVE_DRIVE_MIN_FREE_GB + 10)

        path, rule = resolve_archive_root(tmp_path)
        assert path == tmp_path / "archive"
        assert "rule 4" in rule

    def test_rule_4_falls_back_to_the_repo_when_the_internal_drive_is_roomy(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config_module, "_candidate_drives", lambda: [])
        monkeypatch.setattr(config_module, "free_gb", lambda _: ARCHIVE_DRIVE_MIN_FREE_GB + 1)

        path, rule = resolve_archive_root(tmp_path)
        assert path == tmp_path / "archive"
        assert "rule 4" in rule

    def test_rule_5_still_returns_the_fallback_but_warns(self, tmp_path, monkeypatch, caplog):
        """The owner approved the cramped internal fallback (DECISIONS.md #2).

        So rule 5 must not refuse to start -- but it must leave a warning in the
        log, because the choice carries a real backup risk.
        """
        monkeypatch.setattr(config_module, "_candidate_drives", lambda: [])
        monkeypatch.setattr(config_module, "free_gb", lambda _: 100.0)

        with caplog.at_level("WARNING"):
            path, rule = resolve_archive_root(tmp_path)

        assert path == tmp_path / "archive"
        assert "rule 5" in rule
        assert "DECISIONS.md #2" in caplog.text

    def test_an_explicit_archive_root_bypasses_the_ladder(self, tmp_path, monkeypatch):
        def explode() -> list[tuple[str, float]]:
            raise AssertionError("the ladder must not run when VT_ARCHIVE_ROOT is set")

        monkeypatch.setattr(config_module, "_candidate_drives", explode)

        s = Settings(project_root=tmp_path, archive_root=tmp_path / "chosen")
        assert s.archive == tmp_path / "chosen"
