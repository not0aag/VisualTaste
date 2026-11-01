"""Open Images V7: 15,000 CC BY 2.0 photographs from the validation split.

Google's Open Images publishes two things we need separately:

- an **attribution CSV** per split, served over HTTPS, carrying `ImageID`,
  `License`, `Author`, `AuthorProfileURL`, `OriginalURL` and `OriginalMD5`;
- the **images**, in the public `open-images-dataset` S3 bucket, read with
  unsigned (anonymous) requests, which is what the official downloader does.

The validation split holds 41,620 images, so a 15,000 target needs no second
split.

Attribution is not optional here
--------------------------------
These images are CC BY 2.0, not public domain. The licence requires crediting
the photographer, so `author`, `author_url` and `license_url` are carried
through to the `images` table and the Feed shows them on every card. An Open
Images photo with no author recorded is dropped rather than displayed
uncredited.

A discovered quirk, worth knowing before you read the code
----------------------------------------------------------
The S3 objects are *not* the files the CSV describes. Measured on 2026-10-30:
`validation/fe600639ac5f36c1.jpg` is 86,534 bytes, while the CSV's
`OriginalSize` for it is 242,145 and the `OriginalMD5` matches neither. Open
Images redistributes its own re-encoded copies (long side ~1024 px), and the
CSV describes the Flickr original it was made from.

So integrity is checked against S3's own `ETag`, which for these
single-part objects is the MD5 of the stored bytes, and `OriginalMD5` /
`OriginalURL` are kept purely as provenance of the upstream file. Verifying
against `OriginalMD5` would reject every single image. See DECISIONS.md #7.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from visualtaste.data.http import DownloadRejected, FetchStats, image_id_for
from visualtaste.data.manifest import Candidate

if TYPE_CHECKING:
    from visualtaste.config import Settings
    from visualtaste.data.http import PoliteClient
    from visualtaste.data.manifest import Manifest

logger = logging.getLogger(__name__)

SOURCE = "open_images"
DEFAULT_TARGET = 15_000

ATTRIBUTION_CSV = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
S3_BUCKET = "open-images-dataset"
S3_PREFIX = "validation"


# Downloads run through a thread pool: boto3 is synchronous, and wrapping it in
# threads is both simpler and more reliable than reimplementing unsigned S3
# requests over httpx.
#
# The worker count is tied to `VT_HTTP_CONCURRENCY` rather than set high, for
# two reasons. The brief caps concurrency at 4-8 per source (§7.1), and
# botocore's connection pool must be at least as large as the number of
# threads or urllib3 spends the run discarding and rebuilding sockets.
def _worker_count(settings: Settings) -> int:
    return max(1, settings.http_concurrency)


def _s3_client(settings: Settings):
    """An anonymous S3 client. Unsigned reads of a public bucket only (§5).

    `max_pool_connections` is sized to the worker count: botocore defaults to
    10, and anything above that makes urllib3 discard live connections on every
    request, which shows up as a wall of "Connection pool is full" warnings.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            user_agent_extra="VisualTaste/0.1 (personal research; local use)",
            retries={"max_attempts": 4, "mode": "standard"},
            max_pool_connections=max(10, _worker_count(settings)),
        ),
    )


async def plan(settings: Settings, target: int, *, client: PoliteClient) -> list[Candidate]:
    """Sample `target` images from the validation split, seeded and attributed."""
    rows = _parse_attribution_csv(await client.get_bytes(ATTRIBUTION_CSV))
    logger.info("open_images: %d rows in the validation attribution CSV", len(rows))

    usable = [row for row in rows if _is_usable(row)]
    logger.info(
        "open_images: %d usable after the licence, author and rotation gates",
        len(usable),
    )

    # Plan a modest surplus: the S3 copies are small and uniform, so rejections
    # are rare, but a few objects are missing from the bucket.
    rng = np.random.default_rng(settings.seed)
    wanted = min(len(usable), int(target * 1.1))
    picked = rng.choice(len(usable), size=wanted, replace=False)

    return [_to_candidate(usable[int(i)]) for i in sorted(picked)]


def _parse_attribution_csv(data: bytes) -> list[dict[str, str]]:
    text = io.StringIO(data.decode("utf-8", errors="replace"))
    return list(csv.DictReader(text))


def _is_usable(row: dict[str, str]) -> bool:
    """Keep only rows we can display legally and render correctly.

    - A licence URL is required by §7.1.
    - An author is required by CC BY: we will not show the image uncredited.
    - A non-zero `Rotation` means Open Images knows the stored bytes are rotated
      and expects the consumer to correct it. That correction is not part of
      this project's ingestion rules, and the validation split has far more
      candidates than we need, so those rows are skipped rather than mishandled.
    """
    if not row.get("License") or not row.get("ImageID"):
        return False
    if not (row.get("Author") or "").strip():
        return False
    rotation = (row.get("Rotation") or "").strip()
    return rotation in ("", "0", "0.0")


def _to_candidate(row: dict[str, str]) -> Candidate:
    image_id = row["ImageID"]
    return Candidate(
        source=SOURCE,
        source_id=image_id,
        # The S3 object, not the Flickr original: this is the official
        # redistribution and the only copy guaranteed to still exist.
        url=f"s3://{S3_BUCKET}/{S3_PREFIX}/{image_id}.jpg",
        license_url=row["License"],
        source_url=row.get("OriginalLandingURL") or row.get("OriginalURL") or None,
        author=(row.get("Author") or "").strip() or None,
        author_url=row.get("AuthorProfileURL") or None,
        title=(row.get("Title") or "").strip() or None,
        # Deliberately NOT used for verification -- it describes the upstream
        # Flickr file, not the S3 object. Kept as provenance, which §7.1 asks
        # for. See the module docstring and DECISIONS.md #7.
        expected_md5=None,
        upstream_md5=original_md5_hex(row),
    )


def original_md5_hex(row: dict[str, str]) -> str | None:
    """Decode the CSV's base64 `OriginalMD5` into hex, for the record.

    Open Images encodes the digest as base64 (`0jBpbNION09+r02xkTIBcA==`), not
    as the hex string every other source uses, which is an easy way to produce a
    verifier that rejects everything.
    """
    raw = (row.get("OriginalMD5") or "").strip()
    if not raw:
        return None
    try:
        return base64.b64decode(raw, validate=True).hex()
    except (ValueError, TypeError):
        return None


def _fetch_one_from_s3(
    s3, settings: Settings, candidate: Candidate, max_bytes: int
) -> tuple[str, int]:
    """Read one object, verify it against its ETag, and store it. Synchronous.

    Runs in a worker thread; see `fetch`.
    """
    key = f"{S3_PREFIX}/{candidate.source_id}.jpg"

    head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    size = int(head["ContentLength"])
    if size > max_bytes:
        raise DownloadRejected("too_large", f"{size / 1024**2:.1f}MB")
    if size == 0:
        raise DownloadRejected("empty_response")

    data = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()

    # S3's ETag is the MD5 of the stored bytes for single-part uploads, which
    # these are. It is the only integrity check available for this source, and
    # it does catch a truncated transfer.
    etag = head.get("ETag", "").strip('"')
    if etag and "-" not in etag:
        actual = hashlib.md5(data, usedforsecurity=False).hexdigest()
        if actual != etag:
            raise DownloadRejected("etag_mismatch", f"{actual} != {etag}")

    image_id = image_id_for(data)
    path = settings.original_path(SOURCE, image_id, ".jpg")
    if path.exists():
        return image_id, len(data)

    from visualtaste.data.http import atomic_write

    atomic_write(path, data)
    return image_id, len(data)


async def fetch(
    settings: Settings,
    manifest: Manifest,
    target: int = DEFAULT_TARGET,
    *,
    client: PoliteClient,
) -> FetchStats:
    """Plan (if needed) and download Open Images validation images from S3."""
    import asyncio

    from visualtaste.config import check_disk_headroom
    from visualtaste.data.sources import LOG_EVERY

    from visualtaste.data.sources import needs_planning

    if needs_planning(manifest, target):
        added = manifest.add_all(await plan(settings, target, client=client))
        logger.info("open_images: added %d new candidates to the manifest", added)
        manifest.flush(force=True)

    stats = FetchStats()
    stats.skipped_existing = manifest.succeeded
    stored = manifest.succeeded
    if stored >= target:
        logger.info("open_images: %d images already present, target met", stored)
        return stats

    max_bytes = settings.max_original_mb * 1024**2
    pending = manifest.pending()
    logger.info(
        "open_images: %d already stored, %d pending, target %d", stored, len(pending), target
    )

    s3 = _s3_client(settings)
    loop = asyncio.get_running_loop()
    workers = _worker_count(settings)

    # A dedicated executor, not the default one: the default pool is sized to
    # the CPU count, which would quietly run far more concurrent S3 requests
    # than the politeness budget allows.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oi-s3") as pool:
        for start in range(0, len(pending), workers):
            if stored >= target:
                break
            batch = pending[start : start + workers]
            check_disk_headroom(settings.archive, settings.min_free_gb)

            results = await asyncio.gather(
                *(
                    loop.run_in_executor(
                        pool, _fetch_one_from_s3, s3, settings, candidate, max_bytes
                    )
                    for candidate in batch
                ),
                return_exceptions=True,
            )

            for candidate, result in zip(batch, results, strict=True):
                stats.requested += 1
                candidate.attempts += 1

                if isinstance(result, DownloadRejected):
                    candidate.reject(result.reason)
                    stats.reject(result.reason)
                elif isinstance(result, BaseException):
                    reason = _s3_failure_reason(result)
                    candidate.fail(reason)
                    stats.fail(reason)
                else:
                    image_id, size_bytes = result
                    candidate.succeed(image_id, size_bytes)
                    stats.downloaded += 1
                    stats.bytes_written += size_bytes
                    stored += 1

                manifest.mark_resolved()

            if stats.requested % LOG_EVERY < workers:
                logger.info("open_images: %d/%d stored (%s)", stored, target, stats.summary())

    manifest.flush(force=True)
    logger.info("open_images: finished. %s", stats.summary())
    return stats


def _s3_failure_reason(error: BaseException) -> str:
    """Short stable slugs for botocore's exception zoo.

    A missing object reports as `s3_NoSuchKey`, which is worth distinguishing
    from a network problem when reading the per-source summary.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    from visualtaste.data.sources import failure_reason

    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "unknown")
        return f"s3_{code}"
    if isinstance(error, BotoCoreError):
        return f"s3_{type(error).__name__}"
    return failure_reason(error)
