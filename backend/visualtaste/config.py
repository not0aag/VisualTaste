r"""Settings and filesystem layout, resolved once per process.

Everything that touches the disk asks this module where to write, so there is
exactly one place that knows the layout. Two parts are worth reading closely:
the archive-root detection ladder (`resolve_archive_root`) and the disk guard
(`check_disk_headroom`), because both exist to stop a long download job from
filling the only drive this machine has.

Paths are `pathlib.Path` everywhere. Never build a path by string
concatenation: this project runs on Windows, where the separator is a backslash
but forward slashes also work, and mixing the two by hand is how you get bugs
that only appear on one machine.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import string
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# The repo root is two levels up from this file:
#   <repo>/backend/visualtaste/config.py -> parents[2] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Rules 3 and 4 of the archive ladder: a drive needs this much free space
# before we are willing to put ~80 GB of originals on it.
ARCHIVE_DRIVE_MIN_FREE_GB = 150

# Windows GetDriveTypeW return values we are willing to use as an archive.
_DRIVE_REMOVABLE = 2
_DRIVE_FIXED = 3

_BYTES_PER_GB = 1024**3


class Settings(BaseSettings):
    """Configuration, overridable by `.env` or the environment with a `VT_` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="VT_",
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = _REPO_ROOT
    data_root: Path | None = None
    archive_root: Path | None = None

    seed: int = 1729

    # Disk guard. `min_free_gb` is headroom the jobs refuse to eat into;
    # `max_original_mb` drops the handful of very large museum scans.
    min_free_gb: int = 25
    max_original_mb: int = 8

    http_concurrency: int = 4
    user_agent: str = "VisualTaste/0.1 (personal research; local use)"

    retrain_every: int = 20
    neural_min_labels: int = 1500

    # --- The handover chain (Part 2 §4.4) -------------------------------------
    #
    # R0 serves until the profile has `serve_min_labels` labels with at least
    # `serve_min_per_class` of each class; then `default_rung` takes over. A
    # rung that cannot serve falls back down the chain rather than erroring.
    # `default_rung` is chosen from the final simulator study (Phase 14) and
    # recorded in docs/DECISIONS.md; R2 and R6 are never eligible.
    default_rung: str = "r3"
    serve_min_labels: int = 30
    serve_min_per_class: int = 5

    # Weight of a skip as a negative label. Phase 9 compares 0 / 0.25 / exclude
    # on the simulator; the outcome is in docs/EVALUATION.md.
    skip_weight: float = 0.25

    # R4's posterior temperature: draws are w* + sqrt(tau) * (Laplace deviation).
    # Part 2 specifies 1.0; the value used is chosen from phase10_sensitivity
    # on the H3 criterion and recorded in docs/DECISIONS.md #25.
    r4_tau_post: float = 1.0

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Worker count for CPU-bound pools. Leave two cores for the OS and for
    # whatever else is running, so the laptop stays usable during a long job.
    worker_processes: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 4) - 2))

    # --- Derived roots -------------------------------------------------------

    @property
    def data(self) -> Path:
        """Internal-SSD root: derivatives, embeddings, DB, models, caches."""
        return self.data_root or (self.project_root / "data")

    @property
    def archive(self) -> Path:
        """Originals + backups root. Resolved by the ladder if not configured."""
        if self.archive_root is not None:
            return self.archive_root
        return resolve_archive_root(self.project_root)[0]

    # --- Layout (section 6.2 of the brief) -----------------------------------

    @property
    def originals_dir(self) -> Path:
        return self.archive / "originals"

    @property
    def manifests_dir(self) -> Path:
        return self.archive / "manifests"

    @property
    def backups_dir(self) -> Path:
        return self.archive / "backups"

    @property
    def display_dir(self) -> Path:
        return self.data / "derived" / "display"

    @property
    def thumb_dir(self) -> Path:
        return self.data / "derived" / "thumb"

    @property
    def embeddings_dir(self) -> Path:
        return self.data / "embeddings"

    @property
    def features_dir(self) -> Path:
        return self.data / "derived_features"

    @property
    def indexes_dir(self) -> Path:
        return self.data / "indexes"

    @property
    def models_dir(self) -> Path:
        return self.data / "models"

    #: Overrides the database file. Used to run a second simulator study in
    #: parallel against its own copy, so two study processes never contend for
    #: SQLite's single write lock (docs/DECISIONS.md #22).
    db_file: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.db_file or (self.data / "db" / "visualtaste.db")

    @property
    def hf_home(self) -> Path:
        return self.data / "cache" / "hf"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    # --- Sharded paths -------------------------------------------------------
    #
    # Files are grouped into folders named after the first two hex characters of
    # the image_id, giving 256 folders of a few hundred files each. One flat
    # folder with 80,000 entries makes NTFS lookups and Explorer slow, and on
    # Windows it also pushes paths towards the 260-character limit.

    def original_path(self, source: str, image_id: str, ext: str) -> Path:
        """`<ARCHIVE>/originals/<source>/<id[:2]>/<image_id>.<ext>` (brief §6.2).

        The source name is part of the path so that the archive stays readable
        by a human and so that one source can be re-acquired or audited without
        touching the others. Derivatives are *not* sharded by source: by then an
        image is just an image, and the DB records where it came from.
        """
        ext = ext if ext.startswith(".") else f".{ext}"
        return self.originals_dir / source / image_id[:2] / f"{image_id}{ext}"

    def display_path(self, image_id: str) -> Path:
        return self.display_dir / image_id[:2] / f"{image_id}.webp"

    def thumb_path(self, image_id: str) -> Path:
        return self.thumb_dir / image_id[:2] / f"{image_id}.webp"

    def manifest_path(self, source: str) -> Path:
        return self.manifests_dir / f"manifest_{source}.parquet"

    def embedding_dir(self, model_key: str) -> Path:
        return self.embeddings_dir / model_key

    def index_path(self, model_key: str) -> Path:
        return self.indexes_dir / f"{model_key}.faiss"

    def model_dir(self, version: str) -> Path:
        return self.models_dir / version

    # --- Startup -------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create every directory the project writes to. Safe to call repeatedly."""
        for path in (
            self.originals_dir,
            self.manifests_dir,
            self.backups_dir / "db",
            self.backups_dir / "events",
            self.display_dir,
            self.thumb_dir,
            self.embeddings_dir,
            self.features_dir,
            self.indexes_dir,
            self.models_dir,
            self.db_path.parent,
            self.hf_home,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def apply_hf_env(self) -> None:
        """Point Hugging Face caches inside the project, before any model import.

        `transformers` and `huggingface_hub` read this variable when they are
        first imported, so this has to run early -- hence the call in every job
        entry point rather than at module import time.
        """
        self.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.hf_home)


class DiskGuardTripped(RuntimeError):
    """Raised when a job would write into the free-space headroom we reserved."""


def free_gb(path: Path) -> float:
    """Free space in GB on the volume containing `path`.

    The path need not exist yet; we walk up to the nearest existing parent,
    which for a fresh install is usually the drive root.
    """
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / _BYTES_PER_GB


def check_disk_headroom(path: Path, min_free: float) -> None:
    """Raise `DiskGuardTripped` if the volume holding `path` is nearly full.

    Called by long jobs between batches rather than per file: the point is to
    stop cleanly at a resumable boundary, not to shave the last megabyte.
    """
    available = free_gb(path)
    if available < min_free:
        raise DiskGuardTripped(
            f"{available:.1f} GB free on the volume holding {path}, "
            f"below the {min_free:.0f} GB reserve. Free space, or lower "
            f"VT_MIN_FREE_GB, then re-run -- every job resumes where it stopped."
        )


def _drive_type(letter: str) -> int:
    """Windows drive type for `letter`, via GetDriveTypeW. Returns 0 elsewhere."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return 0
    return int(windll.kernel32.GetDriveTypeW(f"{letter}:\\"))


def _candidate_drives() -> list[tuple[str, float]]:
    """Attached non-system fixed/removable drives, as (letter, free GB) pairs."""
    system_drive = os.environ.get("SYSTEMDRIVE", "C:")
    system_letter = system_drive.rstrip(":\\/").upper()[:1] or "C"
    found: list[tuple[str, float]] = []
    for letter in string.ascii_uppercase:
        if letter == system_letter:
            continue
        root = Path(f"{letter}:/")
        if not root.exists():
            continue
        if _drive_type(letter) not in (_DRIVE_FIXED, _DRIVE_REMOVABLE):
            continue
        try:
            found.append((letter, shutil.disk_usage(root).free / _BYTES_PER_GB))
        except OSError:
            # A card reader with no card, or a drive that vanished mid-scan.
            continue
    return found


def resolve_archive_root(project_root: Path) -> tuple[Path, str]:
    r"""Decide where originals and backups live, and say which rule decided it.

    The ladder is the brief's section 6.1, in order:

      1. `VT_ARCHIVE_ROOT` in the environment or `.env`  (handled by the caller)
      2. a non-system drive that already has a `VisualTasteArchive` folder
      3. a non-system drive with >= 150 GB free -> create `X:\VisualTasteArchive`
      4. `<repo>/archive` on the internal drive, if it has >= 150 GB free
      5. otherwise: a human decides (brief section 3, Q2)

    Returns the path plus a one-line explanation, which callers log so that the
    choice is visible afterwards rather than being silently re-derived.
    """
    drives = _candidate_drives()

    for letter, _ in drives:
        existing = Path(f"{letter}:/VisualTasteArchive")
        if existing.is_dir():
            return existing, f"rule 2: existing archive folder on drive {letter}:"

    for letter, available in drives:
        if available >= ARCHIVE_DRIVE_MIN_FREE_GB:
            return (
                Path(f"{letter}:/VisualTasteArchive"),
                f"rule 3: drive {letter}: has {available:.0f} GB free "
                f"(>= {ARCHIVE_DRIVE_MIN_FREE_GB} GB)",
            )

    internal = project_root / "archive"
    available = free_gb(internal)
    if available >= ARCHIVE_DRIVE_MIN_FREE_GB:
        return internal, f"rule 4: internal fallback, {available:.0f} GB free"

    # Rule 5 would normally stop the build and ask. On this machine it fired on
    # 2026-10-30 (135.9 GB free, no second drive) and the owner chose the
    # internal fallback with the full candidate pool -- see docs/DECISIONS.md #2.
    # So we take the fallback and say so loudly instead of refusing to start.
    logger.warning(
        "Archive ladder reached rule 5: no external drive and only %.1f GB free. "
        "Using the internal fallback by prior decision (docs/DECISIONS.md #2).",
        available,
    )
    return internal, f"rule 5 (owner-approved fallback): internal, {available:.0f} GB free"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings object.

    Cached so that the archive ladder runs once and every module sees the same
    answer. Tests that need a different layout construct `Settings(...)`
    directly instead of clearing this cache.
    """
    settings = Settings()
    if settings.archive_root is None:
        archive, rule = resolve_archive_root(settings.project_root)
        settings.archive_root = archive
        logger.info("Archive root resolved to %s (%s)", archive, rule)
    return settings
