"""The four image sources, and the download loop they share.

Each source module exposes one coroutine with the same shape:

    async def fetch(settings, manifest, target, *, client) -> FetchStats

so `jobs/acquire.py` can treat them uniformly despite their transports being
completely different: three fetch one image per HTTP request, while Megalith
pulls WebDataset tar shards from a Hugging Face mirror.

What every source must guarantee, because the brief requires it:

- **A licence URL per image, or no image.** Enforced structurally: a
  `Candidate` cannot be constructed without one (`data/manifest.py`).
- **A fixed seed.** Sampling uses `VT_SEED`, so the same 25,000 PD12M images
  are chosen on every machine and every re-run.
- **Resumability.** Planning is deterministic and the manifest records every
  outcome, so re-running downloads nothing new.
- **Politeness.** All HTTP goes through `PoliteClient`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from visualtaste.data.http import (
    DownloadRejected,
    FetchStats,
    PoliteClient,
    SourceBlocked,
)

if TYPE_CHECKING:
    from visualtaste.config import Settings
    from visualtaste.data.manifest import Candidate, Manifest

logger = logging.getLogger(__name__)

# How many downloads to launch per batch. `PoliteClient` caps requests actually
# in flight; this only bounds how many task objects exist at once.
BATCH_SIZE = 64

# Progress is logged this often, so a multi-hour job in a log file is readable.
LOG_EVERY = 500


def dimensions_acceptable(
    width: int | None,
    height: int | None,
    *,
    min_short_side: int = 256,
    max_aspect_ratio: float = 4.0,
) -> bool:
    """Apply the §7.2 shape rules to metadata, before spending a download on it.

    Ingestion re-checks this against the decoded pixels, which is the
    authoritative test. Doing it here as well is purely an economy: PD12M and
    Open Images both publish dimensions, and a 6000x200 panorama we would reject
    later costs nothing to reject now.

    Unknown dimensions pass: absent metadata is not evidence of a bad image.
    """
    if width is None or height is None or width <= 0 or height <= 0:
        return True
    if min(width, height) < min_short_side:
        return False
    return max(width, height) / min(width, height) <= max_aspect_ratio


def needs_planning(manifest: Manifest, target: int) -> bool:
    """True when the manifest cannot reach `target` without more candidates.

    Planning only when the manifest is *empty* would be a trap: a trial run at
    `--target 20` leaves 22 candidates on disk, and a later full run would then
    stop at 22 forever, because it would see a non-empty manifest and never
    plan again.

    Re-planning is safe to do at any time. Planners are deterministic for a
    given seed and `Manifest.add_all` ignores candidates it already knows, so
    this adds the missing rows and touches nothing already resolved.
    """
    reachable = manifest.succeeded + len(manifest.pending())
    return reachable < target


async def download_http_candidates(
    settings: Settings,
    client: PoliteClient,
    manifest: Manifest,
    *,
    source: str,
    target: int,
) -> FetchStats:
    """Fetch pending candidates over HTTP until `target` images are stored.

    The loop is shared by PD12M, Open Images and Unsplash. It stops as soon as
    the target is met, so a planner that over-samples to absorb rejections does
    not download the surplus.

    Rejections (too large, wrong shape) and failures (404, timeout) are both
    recorded in the manifest but counted separately: the brief's "< 5% failure
    per source" gate is about failures only.
    """
    import asyncio

    stats = FetchStats()
    max_bytes = settings.max_original_mb * 1024**2
    already = manifest.succeeded
    stats.skipped_existing = already

    if already >= target:
        logger.info("%s: %d images already present, target %d met", source, already, target)
        return stats

    pending = manifest.pending()
    logger.info(
        "%s: %d already stored, %d candidates pending, target %d",
        source,
        already,
        len(pending),
        target,
    )

    stored = already
    for start in range(0, len(pending), BATCH_SIZE):
        if stored >= target:
            break
        batch = pending[start : start + BATCH_SIZE]

        results = await asyncio.gather(
            *(
                _fetch_one(client, candidate, source=source, max_bytes=max_bytes)
                for candidate in batch
            ),
            return_exceptions=True,
        )

        for candidate, result in zip(batch, results, strict=True):
            stats.requested += 1
            candidate.attempts += 1

            if isinstance(result, SourceBlocked):
                # Flush what we have so the work is not lost, then stop: the
                # brief forbids routing around a block (§0.6).
                manifest.flush(force=True)
                raise result
            if isinstance(result, DownloadRejected):
                candidate.reject(result.reason)
                stats.reject(result.reason)
            elif isinstance(result, BaseException):
                reason = failure_reason(result)
                candidate.fail(reason)
                stats.fail(reason)
            else:
                image_id, size_bytes = result
                candidate.succeed(image_id, size_bytes)
                stats.downloaded += 1
                stats.bytes_written += size_bytes
                stored += 1

            manifest.mark_resolved()

        if stats.requested % LOG_EVERY < BATCH_SIZE:
            logger.info("%s: %d/%d stored (%s)", source, stored, target, stats.summary())

    manifest.flush(force=True)
    logger.info("%s: finished. %s", source, stats.summary())
    return stats


async def _fetch_one(
    client: PoliteClient,
    candidate: Candidate,
    *,
    source: str,
    max_bytes: int,
) -> tuple[str, int]:
    return await client.download_to_archive(
        candidate.url,
        source=source,
        max_bytes=max_bytes,
        expected_md5=candidate.expected_md5,
    )


def failure_reason(error: BaseException) -> str:
    """A short, stable slug per failure cause, for the per-source summary.

    Stable matters: these strings are counted and compared across runs, so
    "http_404" must not become "HTTP 404 Not Found" after a refactor.
    """
    import httpx

    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.TransportError):
        return "transport_error"
    return f"error_{type(error).__name__}"
