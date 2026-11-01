"""The per-source manifest: one parquet row per candidate, and the reason it matters.

This file is the whole of the resumability contract in brief §6.2. A download job
is not a transaction, it is a long sequence of small ones, and it *will* be
interrupted -- by a reboot, by the disk guard, by someone closing the laptop. The
manifest is what makes an interruption cost minutes instead of hours:

1. Every candidate gets a row before anything is fetched, with `status='pending'`.
2. As each candidate resolves, its row is updated in memory.
3. The manifest is flushed to disk atomically every `FLUSH_EVERY` resolutions.
4. On restart, the job loads the manifest and skips every row that is already
   resolved.

The consequence to keep in mind while reading the source modules: re-running a
completed job must download nothing and change nothing. That is asserted by
`tests/test_manifest.py` and, end to end, by the kill-and-resume test.

Why parquet rather than a table in SQLite: the manifest describes *candidates*,
most of which never become images. Keeping them out of the database leaves
`images` meaning exactly "an image we have", and keeps the schema in §7.4 honest.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Flush this often. Small enough that a crash loses seconds of work, large
# enough that we are not rewriting a 30,000-row parquet file per image.
FLUSH_EVERY = 250

CandidateStatus = Literal["pending", "ok", "rejected", "failed"]

# Columns are declared explicitly so an empty manifest has the right dtypes and
# a schema change is a visible edit rather than an emergent surprise.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "source",
    "source_id",
    "url",
    "license_url",
    "source_url",
    "author",
    "author_url",
    "title",
    "expected_md5",
    "upstream_md5",
    "declared_width",
    "declared_height",
    "status",
    "image_id",
    "bytes",
    "reason",
    "attempts",
    "resolved_at",
)


@dataclass
class Candidate:
    """One image we intend to fetch, with everything needed to attribute it.

    `license_url` is not optional by accident. The brief's §7.1 rule is that an
    image with no licence URL is excluded, so a `Candidate` that cannot name its
    licence is never constructed -- the source modules filter first.
    """

    source: str
    source_id: str
    url: str
    license_url: str
    source_url: str | None = None
    author: str | None = None
    author_url: str | None = None
    title: str | None = None
    expected_md5: str | None = None
    # Provenance only, never verified against the bytes we store. Open Images
    # publishes the MD5 of the upstream Flickr file while redistributing its own
    # re-encoded copy, so checking it would reject every image (DECISIONS.md #7).
    upstream_md5: str | None = None
    declared_width: int | None = None
    declared_height: int | None = None

    # Filled in as the candidate resolves.
    status: CandidateStatus = "pending"
    image_id: str | None = None
    bytes: int | None = None
    reason: str | None = None
    attempts: int = 0
    resolved_at: str | None = None

    def __post_init__(self) -> None:
        if not self.license_url:
            raise ValueError(
                f"{self.source}/{self.source_id} has no licence URL; the brief "
                f"excludes such images rather than guessing (§7.1)."
            )

    @property
    def resolved(self) -> bool:
        """True once this candidate needs no further network work."""
        return self.status != "pending"

    def succeed(self, image_id: str, size_bytes: int) -> None:
        self.status = "ok"
        self.image_id = image_id
        self.bytes = size_bytes
        self.reason = None
        self.resolved_at = _now()

    def reject(self, reason: str) -> None:
        """Not wanted, and that is a normal outcome -- too big, wrong shape.

        Distinguished from `fail` so the per-source summary can say "we skipped
        1,200 oversized museum scans" rather than "1,200 errors", and so the
        brief's < 5% failure gate measures real failures.
        """
        self.status = "rejected"
        self.reason = reason
        self.resolved_at = _now()

    def fail(self, reason: str) -> None:
        """Tried and could not get it: a dead URL, a timeout, a 404."""
        self.status = "failed"
        self.reason = reason
        self.resolved_at = _now()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Manifest:
    """A source's candidate list, loadable and flushable, keyed by `source_id`."""

    path: Path
    source: str
    candidates: dict[str, Candidate] = field(default_factory=dict)
    _unflushed: int = 0

    # --- Loading and saving --------------------------------------------------

    @classmethod
    def load(cls, path: Path, source: str) -> Manifest:
        """Read an existing manifest, or return an empty one."""
        manifest = cls(path=path, source=source)
        if not path.exists():
            return manifest

        frame = pd.read_parquet(path)
        for row in frame.to_dict("records"):
            candidate = _candidate_from_row(row)
            manifest.candidates[candidate.source_id] = candidate
        logger.info(
            "loaded %d candidates from %s (%d already resolved)",
            len(manifest.candidates),
            path.name,
            sum(1 for c in manifest.candidates.values() if c.resolved),
        )
        return manifest

    def flush(self, *, force: bool = False) -> None:
        """Write the manifest to disk atomically, if anything has changed.

        Atomic because a manifest truncated by a power cut would be worse than
        no manifest at all: the job would re-download everything it had already
        finished, or worse, believe work was done that was not.
        """
        if not force and self._unflushed < FLUSH_EVERY:
            return
        if not self.candidates:
            return

        frame = pd.DataFrame(
            [asdict(c) for c in self.candidates.values()], columns=list(MANIFEST_COLUMNS)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.part")
        frame.to_parquet(tmp, index=False)
        tmp.replace(self.path)
        self._unflushed = 0

    # --- Mutation ------------------------------------------------------------

    def add_all(self, candidates: list[Candidate]) -> int:
        """Add candidates that are not already known. Returns how many were new.

        Existing rows are left alone, so re-planning a source never discards the
        outcome of work already done -- the point of the whole file.
        """
        added = 0
        for candidate in candidates:
            if candidate.source_id not in self.candidates:
                self.candidates[candidate.source_id] = candidate
                added += 1
        if added:
            self._unflushed += added
        return added

    def mark_resolved(self) -> None:
        """Record that one candidate resolved, flushing periodically."""
        self._unflushed += 1
        self.flush()

    # --- Queries -------------------------------------------------------------

    def pending(self, limit: int | None = None) -> list[Candidate]:
        """Candidates still needing work, in insertion order (so seeds hold)."""
        out = [c for c in self.candidates.values() if not c.resolved]
        return out[:limit] if limit is not None else out

    def count(self, status: CandidateStatus) -> int:
        return sum(1 for c in self.candidates.values() if c.status == status)

    @property
    def succeeded(self) -> int:
        return self.count("ok")

    def image_ids(self) -> list[str]:
        return [c.image_id for c in self.candidates.values() if c.image_id]

    def reason_counts(self, status: CandidateStatus) -> dict[str, int]:
        """How many candidates ended in each reason, for the run summary."""
        counts: dict[str, int] = {}
        for candidate in self.candidates.values():
            if candidate.status == status and candidate.reason:
                counts[candidate.reason] = counts.get(candidate.reason, 0) + 1
        return counts


def _candidate_from_row(row: dict[str, object]) -> Candidate:
    """Rebuild a `Candidate` from a parquet row, repairing pandas' NaN/None mess.

    pandas turns missing strings into `NaN` (a float) and missing integers into
    `NaN` too, so every optional field has to be normalised on the way back in
    or the values would propagate into the database as the string "nan".
    """
    clean: dict[str, object] = {}
    for key in MANIFEST_COLUMNS:
        value = row.get(key)
        clean[key] = None if value is None or pd.isna(value) else value

    return Candidate(
        source=str(clean["source"]),
        source_id=str(clean["source_id"]),
        url=str(clean["url"]),
        license_url=str(clean["license_url"]),
        source_url=_as_str(clean["source_url"]),
        author=_as_str(clean["author"]),
        author_url=_as_str(clean["author_url"]),
        title=_as_str(clean["title"]),
        expected_md5=_as_str(clean["expected_md5"]),
        upstream_md5=_as_str(clean["upstream_md5"]),
        declared_width=_as_int(clean["declared_width"]),
        declared_height=_as_int(clean["declared_height"]),
        status=str(clean["status"] or "pending"),  # type: ignore[arg-type]
        image_id=_as_str(clean["image_id"]),
        bytes=_as_int(clean["bytes"]),
        reason=_as_str(clean["reason"]),
        attempts=_as_int(clean["attempts"]) or 0,
        resolved_at=_as_str(clean["resolved_at"]),
    )


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)  # type: ignore[arg-type]
