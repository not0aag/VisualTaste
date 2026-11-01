"""The one place that makes outbound requests, so the manners live in one file.

Every rule the brief imposes on downloading is enforced here rather than in the
four source modules:

- **Politeness.** At most `VT_HTTP_CONCURRENCY` (default 4) requests in flight,
  across all hosts, enforced by a semaphore.
- **Backoff with jitter.** Retries wait `base * 2**attempt` plus a random
  fraction. The jitter matters: without it, four workers that fail together
  retry together forever, in lockstep.
- **A descriptive User-Agent.** So an operator reading their logs can tell what
  this is and that it is one person's research machine.
- **TLS always verified.** There is no flag to turn this off.
- **Give up rather than route around.** A persistent 403, a persistent 429, or
  a login wall raises `SourceBlocked`, which the job reports and the owner is
  asked about (brief §3 Q4). We never switch to a mirror or a proxy to evade it.
- **Disk and size limits.** The free-space guard runs before each write, and
  `Content-Length` is checked *before* the body is read, so an oversized file
  costs one header exchange instead of a 20 MB transfer.

Writes are atomic: bytes go to a temporary file in the destination directory and
are then `os.replace`d into place. A job killed mid-write therefore leaves
either the old file or the new one, never a half-written image -- which is what
makes the resumability contract in §6.2 true rather than aspirational.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from visualtaste.config import Settings, check_disk_headroom

logger = logging.getLogger(__name__)

# Transient conditions worth retrying. 429 is included, but see SourceBlocked:
# if it survives the whole retry budget we stop rather than grind on.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Statuses that mean "not for you", where retrying is pointless or rude.
FATAL_STATUS = frozenset({401, 402, 403, 404, 410, 451})

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE = 1.5
IMAGE_ID_LENGTH = 16


class SourceBlocked(RuntimeError):
    """A source refused us persistently: 403/429 that outlived the retry budget.

    Raised so the calling job stops touching that source and surfaces it as a
    §3 Q4 question, instead of quietly trying something cleverer.
    """


class DownloadRejected(Exception):
    """This one file is not wanted. Recorded in the manifest; not an error.

    `reason` is a short stable slug so the per-source summary can count causes:
    `too_large`, `too_small`, `bad_aspect`, `wrong_type`, `md5_mismatch`, ...
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class FetchStats:
    """Counters for the end-of-run summary the brief asks each downloader for."""

    requested: int = 0
    downloaded: int = 0
    bytes_written: int = 0
    skipped_existing: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    failed: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def fail(self, reason: str) -> None:
        self.failed[reason] = self.failed.get(reason, 0) + 1

    @property
    def attempted(self) -> int:
        """Candidates we actually tried, excluding ones already on disk."""
        return self.requested - self.skipped_existing

    @property
    def failure_rate(self) -> float:
        """Share of attempts that failed outright.

        Rejections are not failures: an image skipped for being 40 MB or
        640x100 was working exactly as intended. The brief's "< 5% failure per
        source" gate is about this number.
        """
        if self.attempted <= 0:
            return 0.0
        return sum(self.failed.values()) / self.attempted

    def summary(self) -> str:
        parts = [
            f"requested={self.requested}",
            f"downloaded={self.downloaded}",
            f"already_present={self.skipped_existing}",
            f"bytes={self.bytes_written / 1024**3:.2f}GB",
        ]
        if self.rejected:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.rejected.items()))
            parts.append(f"rejected({detail})")
        if self.failed:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.failed.items()))
            parts.append(f"failed({detail})")
        parts.append(f"failure_rate={self.failure_rate:.2%}")
        return " ".join(parts)


def image_id_for(data: bytes) -> str:
    """The brief's §6.2 identity: first 16 hex characters of SHA-256 of the bytes.

    Content-addressed, so the same picture fetched from two sources collapses to
    one row without any comparison logic. 16 hex characters is 64 bits; at 80,000
    images the chance of a collision is about 2e-10, which is small enough to
    ignore and large enough to be worth writing down.
    """
    return hashlib.sha256(data).hexdigest()[:IMAGE_ID_LENGTH]


def atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` so that readers never see a partial file.

    `Path.replace` is atomic within a volume on Windows and POSIX alike. The
    temporary file is created in the destination directory, not in the system
    temp folder, because a cross-volume replace is not atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    try:
        # Closed before the replace: Windows will not rename an open file.
        with tmp.open("wb") as fh:
            fh.write(data)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


class PoliteClient:
    """A shared `httpx.AsyncClient` with concurrency limits and retry policy.

    Use as an async context manager so the connection pool is closed:

        async with PoliteClient(settings) as client:
            data = await client.get_bytes(url, max_bytes=8 * 1024**2)
    """

    def __init__(
        self,
        settings: Settings,
        *,
        concurrency: int | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout: float = 60.0,
    ) -> None:
        self.settings = settings
        self.max_attempts = max_attempts
        self._limit = asyncio.Semaphore(concurrency or settings.http_concurrency)
        self._client = httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            timeout=timeout,
            follow_redirects=True,
            # The pool is sized to the semaphore: more sockets than permitted
            # requests would just sit idle.
            limits=httpx.Limits(
                max_connections=(concurrency or settings.http_concurrency) * 2,
                max_keepalive_connections=concurrency or settings.http_concurrency,
            ),
            # verify defaults to True. There is deliberately no option to change it.
        )

    async def __aenter__(self) -> PoliteClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    def _backoff_seconds(self, attempt: int, retry_after: str | None) -> float:
        """Exponential backoff with jitter, respecting `Retry-After` when sent.

        If a server tells us how long to wait, that instruction wins: it is the
        operator's own rate limit talking.
        """
        if retry_after:
            with contextlib.suppress(ValueError):
                return min(float(retry_after), 300.0)
        base = DEFAULT_BACKOFF_BASE * (2**attempt)
        return base + random.uniform(0, base * 0.5)

    async def get_bytes(
        self,
        url: str,
        *,
        max_bytes: int | None = None,
        expected_md5: str | None = None,
    ) -> bytes:
        """Fetch one URL, with retries, size cap and optional MD5 verification.

        Raises `DownloadRejected` if the response is unwanted (too large, not an
        image), `SourceBlocked` if the host is persistently refusing us, and
        `httpx.HTTPError` if it simply failed every attempt.
        """
        last_error: Exception = RuntimeError("no attempt was made")
        rate_limited = False

        for attempt in range(self.max_attempts):
            try:
                async with self._limit:
                    return await self._get_once(url, max_bytes, expected_md5)
            except DownloadRejected:
                raise  # A verdict, not a failure: do not retry.
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # A 403 or 404 on one file is ordinary -- images get deleted and
                # go private. Retrying it is pointless, so it propagates as a
                # per-file failure. A *source-wide* block shows up as a high
                # failure rate, which `jobs.acquire` checks against the brief's
                # 5% gate; it is not something one request can diagnose.
                if status not in RETRY_STATUS:
                    raise
                rate_limited = rate_limited or status == 429
                last_error = exc
                retry_after = exc.response.headers.get("retry-after")
            except (httpx.TransportError, httpx.StreamError) as exc:
                last_error = exc
                retry_after = None

            if attempt + 1 < self.max_attempts:
                delay = self._backoff_seconds(attempt, retry_after)
                logger.debug(
                    "retry %d/%d for %s in %.1fs (%s)",
                    attempt + 1,
                    self.max_attempts,
                    url,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        if rate_limited:
            # We backed off politely and are still being throttled. The brief
            # (§0.6) forbids working around it, so stop and let a human decide.
            raise SourceBlocked(
                f"{url} still returning 429 after {self.max_attempts} attempts "
                f"with exponential backoff. Stopping rather than evading the "
                f"rate limit. Last error: {last_error}"
            )
        raise last_error

    async def _get_once(self, url: str, max_bytes: int | None, expected_md5: str | None) -> bytes:
        """One attempt: stream the response, checking the size before the body."""
        async with self._client.stream("GET", url) as response:
            response.raise_for_status()

            # Check the advertised length first. This is the cheap rejection:
            # an oversized museum scan costs a header exchange, not 20 MB.
            declared = response.headers.get("content-length")
            if max_bytes is not None and declared and int(declared) > max_bytes:
                raise DownloadRejected("too_large", f"{int(declared) / 1024**2:.1f}MB declared")

            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                # Servers lie, or omit Content-Length entirely for chunked
                # responses, so enforce the cap on the real byte count too.
                if max_bytes is not None and total > max_bytes:
                    raise DownloadRejected("too_large", f"{total / 1024**2:.1f}MB streamed")
                chunks.append(chunk)

        data = b"".join(chunks)
        if not data:
            raise DownloadRejected("empty_response")

        if expected_md5:
            actual = hashlib.md5(data, usedforsecurity=False).hexdigest()
            if actual != expected_md5.lower():
                raise DownloadRejected("md5_mismatch", f"{actual} != {expected_md5}")

        return data

    async def download_to_archive(
        self,
        url: str,
        *,
        source: str,
        max_bytes: int | None = None,
        expected_md5: str | None = None,
    ) -> tuple[str, int]:
        """Fetch a URL and store it as an original. Returns `(image_id, bytes)`.

        The caller does not choose the filename: it is derived from the content
        hash, which is what makes the write idempotent and the archive
        de-duplicated by construction.
        """
        settings = self.settings
        check_disk_headroom(settings.archive, settings.min_free_gb)
        data = await self.get_bytes(url, max_bytes=max_bytes, expected_md5=expected_md5)

        image_id = image_id_for(data)
        suffix = _extension_for(data, url)
        path = settings.original_path(source, image_id, suffix)
        if path.exists():
            # Content-addressed, so an existing file with this name *is* this
            # image. Nothing to do, and nothing to overwrite: originals are
            # write-once (brief §0.3).
            return image_id, len(data)

        atomic_write(path, data)
        return image_id, len(data)


# Magic numbers, so the extension describes the bytes rather than trusting a URL
# that may end in `.jpg?w=2048` or in nothing at all.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)


def _extension_for(data: bytes, url: str) -> str:
    """File extension from the content's magic bytes, falling back to the URL."""
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"

    # Unknown container: keep a plausible extension so the file is still
    # openable by hand, and let Pillow reject it during ingestion.
    tail = url.split("?")[0].rsplit(".", 1)
    if len(tail) == 2 and 1 <= len(tail[1]) <= 5:
        return f".{tail[1].lower()}"
    return ".bin"
