"""Turn downloaded originals into rows in the database and WebP derivatives.

Run:  python -m visualtaste.jobs.ingest                 (everything pending)
      python -m visualtaste.jobs.ingest --source pd12m  (one source)
      python -m visualtaste.jobs.ingest --init-only     (just apply migrations)
      python -m visualtaste.jobs.ingest --limit 500     (a trial run)

The stages are brief §7.2, steps 1-6. Steps 7 (safety) and 8 (embedding-level
near-duplicates) need embeddings and so belong to Phase 5; step 9 (selection)
is `jobs/select.py`.

    1. validate      decode twice -- structure, then every pixel
    2. identify      image_id = SHA-256 of the bytes; exact duplicates collapse
    3. licence gate  no licence URL, no image
    4. derivatives   display 1024 and thumb 256 WebP
    5. pHash         near-duplicate groups at Hamming <= 6
    6. colour        a 64-bin CIELAB histogram plus 5 dominant colours

Why a process pool
------------------
Stages 1, 4, 5 and 6 are pure CPU: decode, resize, DCT, histogram. Python's GIL
means threads would not help, so the per-image work happens in worker
processes, `cpu_count - 2` of them, leaving two cores for the OS so the laptop
stays usable. The parent process does all the database writing, which keeps
SQLite single-writer and avoids lock contention entirely.

On Windows those workers are *spawned*, not forked, so each one re-imports this
module. That is why the entry point is guarded by `if __name__ == "__main__"`
and why `_process_one` is a module-level function taking plain data: a closure
or a lambda cannot be pickled across a spawn.

Each worker also pins its numeric libraries to a single thread. NumPy and
scikit-learn default to using every core via OpenMP, so eight worker processes
would each start sixteen threads and the machine would spend its time switching
between 128 of them. The parallelism here is the process pool; the libraries
inside it must stay sequential.

Resumability
------------
Every image already in `images` is skipped, so a run killed at any point and
restarted continues from there. That includes images recorded as `rejected`:
a corrupt file is remembered as unusable rather than being re-decoded on every
future run. Derivatives are skipped individually too, so an image whose row
exists but whose WebP is missing is repaired rather than duplicated.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from visualtaste.config import Settings, check_disk_headroom, get_settings
from visualtaste.data.colour import DESCRIPTOR_SIZE, colour_descriptor, dominant_colours
from visualtaste.data.dedupe import HashedImage, find_near_duplicate_groups, phash_for
from visualtaste.data.derivatives import build_derivatives
from visualtaste.data.manifest import Manifest
from visualtaste.data.validate import ImageRejected, validate_image
from visualtaste.db import repo

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Images per database transaction. Large enough that commits are not the
# bottleneck, small enough that a crash loses little.
COMMIT_EVERY = 200

LOG_EVERY = 2_000


@dataclass
class IngestStats:
    """Counters for the reconciliation the Phase 3 Definition of Done requires."""

    considered: int = 0
    inserted: int = 0
    already_present: int = 0
    exact_duplicates: int = 0
    rejected: dict[str, int] = None  # type: ignore[assignment]
    near_duplicates: int = 0

    def __post_init__(self) -> None:
        if self.rejected is None:
            self.rejected = {}

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected.values())

    def summary(self) -> str:
        parts = [
            f"considered={self.considered}",
            f"inserted={self.inserted}",
            f"already_present={self.already_present}",
            f"exact_dups={self.exact_duplicates}",
            f"rejected={self.total_rejected}",
            f"near_dups={self.near_duplicates}",
        ]
        if self.rejected:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.rejected.items()))
            parts.append(f"reasons({detail})")
        return " ".join(parts)


@dataclass
class ProcessedImage:
    """What a worker produced for one image, ready for the parent to insert."""

    image_id: str
    source: str
    source_id: str
    orig_path: str
    width: int
    height: int
    bytes: int
    phash: str
    dominant: list[str]
    colour: np.ndarray

    # Set instead of the rest when the image was rejected.
    rejected_reason: str | None = None


@dataclass
class WorkItem:
    """The inputs a worker needs. Plain data, because it crosses a process boundary."""

    image_id: str
    source: str
    source_id: str
    original_path: Path
    display_path: Path
    thumb_path: Path


def _init_worker() -> None:
    """Pin numeric libraries to one thread per worker process.

    Set before NumPy and scikit-learn do their thread-pool setup, which happens
    on first use rather than at import, so assigning the variables here is
    effective. Without this, the process pool and OpenMP multiply instead of
    cooperating.
    """
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _process_one(item: WorkItem) -> ProcessedImage:
    """Validate, derive, hash and describe one image. Runs in a worker process.

    Returns a `ProcessedImage` either way -- with `rejected_reason` set if the
    image is unusable -- rather than raising, because an exception crossing a
    process boundary loses its context and one bad file must not end the run.
    """
    try:
        facts = validate_image(item.original_path)
    except ImageRejected as exc:
        return _rejection(item, exc.reason)
    except Exception as exc:  # a surprise in Pillow is still just one bad file
        return _rejection(item, f"error_{type(exc).__name__}")

    try:
        build_derivatives(item.original_path, item.display_path, item.thumb_path)

        from PIL import Image

        with Image.open(item.thumb_path) as thumb:
            thumb.load()
            rgb_thumb = thumb.convert("RGB")

        return ProcessedImage(
            image_id=item.image_id,
            source=item.source,
            source_id=item.source_id,
            orig_path=str(item.original_path),
            width=facts.width,
            height=facts.height,
            bytes=facts.bytes,
            phash=phash_for(rgb_thumb),
            dominant=dominant_colours(rgb_thumb),
            colour=colour_descriptor(rgb_thumb),
        )
    except Exception as exc:
        return _rejection(item, f"derivative_error_{type(exc).__name__}")


def _rejection(item: WorkItem, reason: str) -> ProcessedImage:
    return ProcessedImage(
        image_id=item.image_id,
        source=item.source,
        source_id=item.source_id,
        orig_path=str(item.original_path),
        width=0,
        height=0,
        bytes=0,
        phash="",
        dominant=[],
        colour=np.zeros(DESCRIPTOR_SIZE, dtype=np.float32),
        rejected_reason=reason,
    )


def build_work_items(
    settings: Settings,
    source: str,
    known_ids: set[str],
    limit: int | None,
    *,
    manifest: Manifest | None = None,
) -> list[WorkItem]:
    """List the images in a source's manifest that still need ingestion.

    The manifest is the authority on what was downloaded and what its metadata
    is; the filesystem is only consulted to confirm the file is there.

    `known_ids` is every image already in the database, including ones recorded
    as `rejected` -- which is what stops a re-run from decoding the same corrupt
    file again.
    """
    manifest = manifest or Manifest.load(settings.manifest_path(source), source)
    items: list[WorkItem] = []
    missing_files = 0

    for candidate in manifest.candidates.values():
        if candidate.status != "ok" or not candidate.image_id:
            continue
        if candidate.image_id in known_ids:
            continue

        original = _locate_original(settings, source, candidate.image_id)
        if original is None:
            missing_files += 1
            continue

        items.append(
            WorkItem(
                image_id=candidate.image_id,
                source=source,
                source_id=candidate.source_id,
                original_path=original,
                display_path=settings.display_path(candidate.image_id),
                thumb_path=settings.thumb_path(candidate.image_id),
            )
        )
        if limit is not None and len(items) >= limit:
            break

    if missing_files:
        logger.warning("%s: %d manifest rows marked ok have no file on disk", source, missing_files)
    return items


def _locate_original(settings: Settings, source: str, image_id: str) -> Path | None:
    """Find a stored original whatever extension it was given.

    The manifest records the id, not the suffix, because the suffix comes from
    sniffing the bytes. A glob over one 256-file shard directory is cheap.
    """
    shard = settings.originals_dir / source / image_id[:2]
    if not shard.is_dir():
        return None
    for path in shard.glob(f"{image_id}.*"):
        if path.is_file() and not path.name.endswith(".part"):
            return path
    return None


def ingest_source(
    settings: Settings,
    connection: sqlite3.Connection,
    source: str,
    *,
    limit: int | None = None,
    workers: int | None = None,
) -> tuple[IngestStats, dict[str, np.ndarray]]:
    """Ingest one source. Returns its stats and the colour vectors produced."""
    known = repo.existing_image_ids(connection)
    manifest = Manifest.load(settings.manifest_path(source), source)
    metadata = manifest_metadata(manifest)

    # Before any decoding: this source may have supplied bytes we already have
    # from another source. Those images need no work, but the extra provenance
    # must still be recorded, and `build_work_items` is about to filter them out.
    extra_provenance = _record_known_provenance(connection, manifest, known)

    items = build_work_items(settings, source, known, limit, manifest=manifest)
    stats = IngestStats(exact_duplicates=extra_provenance)

    if not items:
        logger.info("%s: nothing to ingest", source)
        return stats, {}

    logger.info("%s: %d images to ingest", source, len(items))
    colours: dict[str, np.ndarray] = {}
    pool_size = workers if workers is not None else settings.worker_processes
    pending_commit = 0

    # A set of ids seen in *this* run, so two manifest rows pointing at the same
    # bytes do not both try to insert. The database catches it too, but counting
    # it here is what makes the reconciliation add up.
    seen_this_run: set[str] = set()

    for result in _process_all(items, pool_size):
        stats.considered += 1
        row_metadata = metadata.get(result.source_id, {})

        if result.rejected_reason:
            stats.reject(result.rejected_reason)
            _record_rejection(connection, result, row_metadata)
            pending_commit += 1
        elif result.image_id in seen_this_run:
            stats.exact_duplicates += 1
            # Still record the extra provenance: same bytes, another source.
            repo.upsert_image(
                connection,
                image_id=result.image_id,
                source=result.source,
                source_id=result.source_id,
                license_url=row_metadata.get("license_url") or "",
                orig_path=result.orig_path,
            )
            pending_commit += 1
        else:
            seen_this_run.add(result.image_id)
            created = _insert(connection, result, row_metadata)
            if created:
                stats.inserted += 1
                colours[result.image_id] = result.colour
            else:
                stats.already_present += 1
            pending_commit += 1

        if pending_commit >= COMMIT_EVERY:
            connection.commit()
            pending_commit = 0
            check_disk_headroom(settings.data, settings.min_free_gb)

        if stats.considered % LOG_EVERY == 0:
            logger.info("%s: %s", source, stats.summary())

    connection.commit()
    logger.info("%s: done. %s", source, stats.summary())
    return stats, colours


def _record_known_provenance(
    connection: sqlite3.Connection, manifest: Manifest, known_ids: set[str]
) -> int:
    """Record this source against images we already hold from another source.

    Megalith and Open Images both draw on Flickr, and PD12M overlaps Wikimedia
    Commons, so the same photograph genuinely arrives twice. Because `image_id`
    is the hash of the bytes, the second arrival is the same row -- but the
    second source is real provenance and the brief keeps it (`image_sources`).

    Returns how many new provenance rows were written. Pure database work: no
    image is decoded, because there is nothing new to decode.
    """
    already = repo.existing_source_keys(connection)
    written = 0

    for candidate in manifest.candidates.values():
        if candidate.status != "ok" or not candidate.image_id:
            continue
        if candidate.image_id not in known_ids:
            continue  # Not held yet; the normal work pass will handle it.
        if (candidate.source, candidate.source_id) in already:
            continue  # Already recorded on an earlier run.

        repo.upsert_image(
            connection,
            image_id=candidate.image_id,
            source=candidate.source,
            source_id=candidate.source_id,
            license_url=candidate.license_url,
            orig_path="",
        )
        written += 1

    if written:
        connection.commit()
        logger.info(
            "%s: recorded %d images as also supplied by this source",
            manifest.source,
            written,
        )
    return written


def _process_all(items: list[WorkItem], pool_size: int) -> Iterator[ProcessedImage]:
    """Run `_process_one` over every item, in parallel unless asked not to.

    `pool_size <= 1` runs inline, in this process. That is not just an
    optimisation for the tests: when one image in 80,000 crashes a worker, being
    able to re-run the same job single-process puts the traceback in front of
    you instead of inside a dead subprocess.
    """
    if pool_size <= 1:
        _init_worker()
        for item in items:
            yield _process_one(item)
        return

    with ProcessPoolExecutor(max_workers=pool_size, initializer=_init_worker) as pool:
        futures = [pool.submit(_process_one, item) for item in items]
        for done in as_completed(futures):
            yield done.result()


def _record_rejection(
    connection: sqlite3.Connection,
    result: ProcessedImage,
    metadata: dict[str, str | None],
) -> None:
    """Remember that this file is unusable, so later runs do not retry it.

    Two reasons this is a row rather than a log line. The brief's §9
    reconciliation is `downloaded = ok + rejected + near_dup`, which only
    balances if rejections are counted in the database; and without a row,
    every re-run would re-decode every corrupt file forever -- which is exactly
    the bug this replaced.

    The reason goes in `removals`, the schema's audit trail, because `images`
    has a status column but nowhere to say why.
    """
    repo.upsert_image(
        connection,
        image_id=result.image_id,
        source=result.source,
        source_id=result.source_id,
        license_url=metadata.get("license_url") or "",
        orig_path=result.orig_path,
        status="rejected",
    )
    repo.record_removal(connection, result.image_id, f"ingest_rejected: {result.rejected_reason}")


def _insert(
    connection: sqlite3.Connection,
    result: ProcessedImage,
    metadata: dict[str, str | None],
) -> bool:
    return repo.upsert_image(
        connection,
        image_id=result.image_id,
        source=result.source,
        source_id=result.source_id,
        license_url=metadata.get("license_url") or "",
        orig_path=result.orig_path,
        source_url=metadata.get("source_url"),
        author=metadata.get("author"),
        author_url=metadata.get("author_url"),
        title=metadata.get("title"),
        width=result.width,
        height=result.height,
        bytes_=result.bytes,
        phash=result.phash,
        dominant_colours=result.dominant,
    )


def colour_for_thumb(thumb_path: Path) -> np.ndarray | None:
    """Recompute one image's colour descriptor from its thumbnail.

    Module-level and taking a plain path, so it can be sent to a worker process
    on Windows (see `_init_worker`).
    """
    from PIL import Image

    try:
        with Image.open(thumb_path) as thumb:
            thumb.load()
            return colour_descriptor(thumb.convert("RGB"))
    except Exception:
        return None


def repair_colour_features(
    settings: Settings,
    connection: sqlite3.Connection,
    *,
    workers: int | None = None,
) -> int:
    """Recompute colour vectors for usable images that have none. Returns the count.

    This exists because of a real gap. `save_colour_features` writes the whole
    block once, at the end of a run, so an ingestion run that is killed part
    way leaves rows in `images` whose colour vectors were never persisted --
    the images are *ingested* but have no entry in `colour64.npy`. That happened
    here: the first Megalith run was stopped to pick up a performance fix, and
    1,639 of the selected 50,000 were left without a descriptor.

    The symptom was quiet rather than loud: the API found the colour block
    incomplete and dropped it entirely rather than padding it with zeros, so the
    neural ranker would simply have had one fewer feature block than intended.
    `GET /health` said so, which is how it was noticed.

    Repair is cheap because the thumbnails already exist -- no original is
    re-read and no image is re-downloaded.
    """
    have = set()
    ids_path = settings.features_dir / "colour64_ids.npy"
    if ids_path.exists():
        have = set(np.load(ids_path, allow_pickle=False).tolist())

    wanted = repo.image_ids_with_status(connection, "ok")
    missing = [image_id for image_id in wanted if image_id not in have]
    if not missing:
        logger.info("colour features: nothing to repair")
        return 0

    logger.info("colour features: recomputing %d missing descriptors", len(missing))
    pool_size = workers if workers is not None else settings.worker_processes
    paths = [settings.thumb_path(image_id) for image_id in missing]

    recovered: dict[str, np.ndarray] = {}
    failed = 0

    if pool_size <= 1:
        _init_worker()
        vectors = [colour_for_thumb(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=pool_size, initializer=_init_worker) as pool:
            vectors = list(pool.map(colour_for_thumb, paths, chunksize=64))

    for image_id, vector in zip(missing, vectors, strict=True):
        if vector is None:
            failed += 1
            continue
        recovered[image_id] = vector

    if failed:
        logger.warning(
            "colour features: %d thumbnails could not be read; run jobs.ingest "
            "to rebuild their derivatives",
            failed,
        )

    if recovered:
        save_colour_features(settings, connection, recovered)
    logger.info("colour features: repaired %d of %d", len(recovered), len(missing))
    return len(recovered)


def manifest_metadata(manifest: Manifest) -> dict[str, dict[str, str | None]]:
    """Index a manifest by `source_id`, keeping only the attribution fields.

    Built once per source and passed down, rather than cached in a module
    global: a global would go stale the moment two sources (or two runs inside
    one process) were ingested, and the symptom would be attribution from the
    wrong source appearing on a card.
    """
    return {
        candidate.source_id: {
            "license_url": candidate.license_url,
            "source_url": candidate.source_url,
            "author": candidate.author,
            "author_url": candidate.author_url,
            "title": candidate.title,
        }
        for candidate in manifest.candidates.values()
    }


def mark_near_duplicates(connection: sqlite3.Connection, *, threshold: int | None = None) -> int:
    """Run pHash near-duplicate detection over everything and record the groups.

    Done after insertion rather than during it, because a duplicate group can
    only be resolved once every member is known -- "keep the highest resolution"
    is not decidable one image at a time.
    """
    from visualtaste.data.dedupe import PHASH_HAMMING_THRESHOLD

    rows = repo.images_with_phash(connection)
    hashed = [
        HashedImage(
            image_id=row["image_id"],
            phash=row["phash"],
            width=row["width"] or 0,
            height=row["height"] or 0,
        )
        for row in rows
        if row["phash"]
    ]

    groups = find_near_duplicate_groups(hashed, threshold=threshold or PHASH_HAMMING_THRESHOLD)

    folded = 0
    with repo.transaction(connection):
        for group in groups:
            for duplicate in group.duplicates:
                repo.set_image_status(connection, duplicate, "near_dup", dup_of=group.kept)
                folded += 1

    logger.info("marked %d images as near-duplicates of a kept copy", folded)
    return folded


def save_colour_features(
    settings: Settings, connection: sqlite3.Connection, colours: dict[str, np.ndarray]
) -> Path:
    """Write `colour64.npy` plus its id index, merging with anything already there.

    Stored as a plain array with a matching id file, exactly like the embeddings
    in §6.2, so that every feature block in the project is loaded the same way.
    """
    features_dir = settings.features_dir
    features_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = features_dir / "colour64.npy"
    ids_path = features_dir / "colour64_ids.npy"

    existing: dict[str, np.ndarray] = {}
    if vectors_path.exists() and ids_path.exists():
        old_vectors = np.load(vectors_path)
        old_ids = np.load(ids_path, allow_pickle=False)
        existing = dict(zip(old_ids.tolist(), old_vectors, strict=True))

    existing.update(colours)

    # Sorted by id so the file is deterministic: the same corpus produces the
    # same bytes, which makes a diff or a checksum meaningful.
    ordered_ids = sorted(existing)
    if not ordered_ids:
        return vectors_path

    matrix = np.stack([existing[image_id] for image_id in ordered_ids]).astype(np.float32)

    # Written through an open handle rather than by filename: `np.save` appends
    # ".npy" to any path that does not already end in it, which would turn
    # "colour64.npy.part" into "colour64.npy.part.npy" and leave the atomic
    # replace pointing at a file that was never written.
    tmp_vectors = vectors_path.with_name(f".{vectors_path.name}.part")
    tmp_ids = ids_path.with_name(f".{ids_path.name}.part")
    with tmp_vectors.open("wb") as handle:
        np.save(handle, matrix)
    with tmp_ids.open("wb") as handle:
        np.save(handle, np.array(ordered_ids, dtype="<U16"))
    tmp_vectors.replace(vectors_path)
    tmp_ids.replace(ids_path)

    logger.info("colour features: %d x %d -> %s", *matrix.shape, vectors_path)
    return vectors_path


def render_summary(per_source: dict[str, IngestStats], connection: sqlite3.Connection) -> str:
    """The reconciliation table the Phase 3 Definition of Done asks for."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### Ingestion summary ({now})",
        "",
        "| source | considered | inserted | exact dups | rejected | near dups |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source, stats in sorted(per_source.items()):
        lines.append(
            f"| `{source}` | {stats.considered} | {stats.inserted} "
            f"| {stats.exact_duplicates} | {stats.total_rejected} | -- |"
        )

    by_status = repo.count_images_by_status(connection)
    by_source = repo.count_images_by_source(connection)
    considered = sum(s.considered for s in per_source.values())
    rejected = sum(s.total_rejected for s in per_source.values())
    exact = sum(s.exact_duplicates for s in per_source.values())
    in_db = sum(by_status.values())

    lines += [
        "",
        "**Status counts in the database:** "
        + ", ".join(f"`{k}` = {v:,}" for k, v in sorted(by_status.items())),
        "",
        "**Rows per source:** " + ", ".join(f"`{k}` = {v:,}" for k, v in sorted(by_source.items())),
        "",
        "**Reconciliation** (brief §9: downloaded = ok + rejected + near_dup):",
        "",
        f"- considered this run: {considered:,}",
        f"- rejected: {rejected:,}",
        f"- exact duplicates folded into an existing row: {exact:,}",
        f"- distinct images in the database: {in_db:,}",
        f"- balance: {considered:,} considered"
        f" = {in_db:,} in database + {rejected:,} rejected + {exact:,} exact dups"
        f" (difference {considered - in_db - rejected - exact:,}; a non-zero"
        f" difference means rows carried over from an earlier run)",
        "",
    ]

    for source, stats in sorted(per_source.items()):
        if stats.rejected:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(stats.rejected.items()))
            lines.append(f"- `{source}` rejection reasons: {detail}.")

    lines.append("")
    return "\n".join(lines)


def _configure_logging(settings: Settings, verbose: bool) -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / f"ingest_{datetime.now(UTC):%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="source to ingest; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="stop after N images per source")
    parser.add_argument("--init-only", action="store_true", help="apply migrations and exit")
    parser.add_argument(
        "--skip-dedupe", action="store_true", help="do not run pHash near-duplicate marking"
    )
    parser.add_argument(
        "--repair-colour",
        action="store_true",
        help="recompute colour descriptors for usable images that have none, "
        "then exit (see repair_colour_features)",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    log_path = _configure_logging(settings, args.verbose)

    connection = repo.init_db(settings.db_path)
    if args.init_only:
        print(f"database ready at {settings.db_path}")
        connection.close()
        return 0

    if args.repair_colour:
        repaired = repair_colour_features(settings, connection, workers=args.workers)
        print(f"repaired {repaired} colour descriptor(s)")
        connection.close()
        return 0

    sources = args.source or [
        path.stem.replace("manifest_", "")
        for path in sorted(settings.manifests_dir.glob("manifest_*.parquet"))
    ]
    if not sources:
        logger.error("no manifests found in %s -- run jobs.acquire first", settings.manifests_dir)
        return 1

    logger.info("log file: %s", log_path)
    logger.info("ingesting sources: %s", sources)

    per_source: dict[str, IngestStats] = {}
    all_colours: dict[str, np.ndarray] = {}

    for source in sources:
        stats, colours = ingest_source(
            settings, connection, source, limit=args.limit, workers=args.workers
        )
        per_source[source] = stats
        all_colours.update(colours)

    if all_colours:
        save_colour_features(settings, connection, all_colours)

    if not args.skip_dedupe:
        folded = mark_near_duplicates(connection)
        for stats in per_source.values():
            stats.near_duplicates = folded

    summary = render_summary(per_source, connection)
    print("\n" + summary)
    (settings.logs_dir / "ingest_summary.md").write_text(summary, encoding="utf-8")

    connection.close()
    return 0


if __name__ == "__main__":
    # Required on Windows: worker processes are spawned, so they re-import this
    # module, and without the guard each one would start its own ingestion run.
    raise SystemExit(main())
