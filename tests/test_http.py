"""Tests for the download layer: politeness, limits, atomicity, identity.

No test here makes a real request. `httpx.MockTransport` serves scripted
responses, which lets us test the retry and backoff behaviour deterministically
and in milliseconds instead of hoping a real server misbehaves on cue.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from visualtaste.config import DiskGuardTripped, Settings
from visualtaste.data import http as http_module
from visualtaste.data.http import (
    DownloadRejected,
    FetchStats,
    PoliteClient,
    SourceBlocked,
    atomic_write,
    image_id_for,
)

# A one-pixel JPEG: real magic bytes, so extension sniffing has something true
# to find.
JPEG_BYTES = bytes.fromhex("ffd8ffe000104a46494600010100000100010000") + b"\x00" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64


@pytest.fixture
def settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        http_concurrency=4,
        min_free_gb=0,
        max_original_mb=8,
    )


def client_with(settings: Settings, handler, **kwargs) -> PoliteClient:
    """A `PoliteClient` whose transport is a scripted handler."""
    client = PoliteClient(settings, **kwargs)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )
    return client


class TestImageId:
    def test_identity_is_the_first_16_hex_of_sha256(self):
        data = b"some image bytes"
        expected = hashlib.sha256(data).hexdigest()[:16]

        assert image_id_for(data) == expected
        assert len(image_id_for(data)) == 16

    def test_the_same_bytes_always_give_the_same_id(self):
        # This is what collapses a duplicate fetched from two sources into one
        # row, with no comparison logic anywhere.
        assert image_id_for(JPEG_BYTES) == image_id_for(bytes(JPEG_BYTES))

    def test_different_bytes_give_different_ids(self):
        assert image_id_for(JPEG_BYTES) != image_id_for(PNG_BYTES)


class TestAtomicWrite:
    def test_writes_the_bytes_and_creates_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.jpg"

        atomic_write(target, JPEG_BYTES)

        assert target.read_bytes() == JPEG_BYTES

    def test_leaves_no_part_file_behind(self, tmp_path):
        target = tmp_path / "c.jpg"

        atomic_write(target, JPEG_BYTES)

        assert list(tmp_path.glob(".*.part")) == []

    def test_a_failed_write_does_not_leave_a_part_file(self, tmp_path, monkeypatch):
        # The guarantee that matters on a laptop that gets closed mid-job: a
        # reader must see the old file or the new one, never a truncated one.
        target = tmp_path / "c.jpg"

        def boom(*args, **kwargs):
            raise OSError("disk went away")

        monkeypatch.setattr(Path, "replace", boom)

        with pytest.raises(OSError, match="disk went away"):
            atomic_write(target, JPEG_BYTES)

        assert not target.exists()
        assert list(tmp_path.glob(".*.part")) == []


class TestSizeCap:
    async def test_rejects_before_reading_the_body_when_length_is_declared(self, settings):
        # The point of the cap: an oversized museum scan should cost a header
        # exchange, not a 20 MB transfer.
        body_reads = 0

        def handler(request):
            nonlocal body_reads
            body_reads += 1
            return httpx.Response(
                200, headers={"content-length": str(20 * 1024**2)}, content=b"x" * 100
            )

        async with client_with(settings, handler) as client:
            with pytest.raises(DownloadRejected) as excinfo:
                await client.get_bytes("https://example.invalid/big.jpg", max_bytes=8 * 1024**2)

        assert excinfo.value.reason == "too_large"
        assert "declared" in excinfo.value.detail

    async def test_enforces_the_cap_when_no_length_is_declared(self, settings):
        # A chunked response carries no content-length at all, so the declared
        # check cannot fire and the running byte count is the only defence.
        async def chunks():
            for _ in range(5):
                yield b"x" * 1000

        def handler(request):
            return httpx.Response(200, content=chunks())

        async with client_with(settings, handler) as client:
            with pytest.raises(DownloadRejected) as excinfo:
                await client.get_bytes("https://example.invalid/big.jpg", max_bytes=1500)

        assert excinfo.value.reason == "too_large"
        assert "streamed" in excinfo.value.detail, "the declared-length path fired instead"

    async def test_accepts_a_file_within_the_cap(self, settings):
        def handler(request):
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            data = await client.get_bytes("https://example.invalid/ok.jpg", max_bytes=8 * 1024**2)

        assert data == JPEG_BYTES

    async def test_an_empty_response_is_rejected(self, settings):
        def handler(request):
            return httpx.Response(200, content=b"")

        async with client_with(settings, handler) as client:
            with pytest.raises(DownloadRejected) as excinfo:
                await client.get_bytes("https://example.invalid/empty.jpg")

        assert excinfo.value.reason == "empty_response"


class TestMd5Verification:
    async def test_accepts_a_matching_digest(self, settings):
        digest = hashlib.md5(JPEG_BYTES, usedforsecurity=False).hexdigest()

        def handler(request):
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            data = await client.get_bytes("https://example.invalid/a.jpg", expected_md5=digest)

        assert data == JPEG_BYTES

    async def test_rejects_a_mismatching_digest(self, settings):
        # PD12M ships an MD5 per image; a mismatch means a corrupted transfer or
        # a changed file, and either way we do not want it in a write-once archive.
        def handler(request):
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            with pytest.raises(DownloadRejected) as excinfo:
                await client.get_bytes("https://example.invalid/a.jpg", expected_md5="0" * 32)

        assert excinfo.value.reason == "md5_mismatch"


class TestRetryPolicy:
    async def test_retries_a_transient_server_error_then_succeeds(self, settings, monkeypatch):
        monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
        attempts = 0

        def handler(request):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(503)
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            data = await client.get_bytes("https://example.invalid/flaky.jpg")

        assert data == JPEG_BYTES
        assert attempts == 3

    async def test_does_not_retry_a_404(self, settings, monkeypatch):
        # A deleted image is not going to come back; retrying is just rude.
        monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
        attempts = 0

        def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(404)

        async with client_with(settings, handler) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_bytes("https://example.invalid/gone.jpg")

        assert attempts == 1

    async def test_a_persistent_429_raises_source_blocked(self, settings, monkeypatch):
        # Brief §0.6: we back off, and if that is not enough we stop. We never
        # switch to a mirror or a proxy to get around a rate limit.
        monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)

        def handler(request):
            return httpx.Response(429, headers={"retry-after": "1"})

        async with client_with(settings, handler, max_attempts=3) as client:
            with pytest.raises(SourceBlocked, match="429"):
                await client.get_bytes("https://example.invalid/limited.jpg")

    async def test_a_rejection_is_never_retried(self, settings, monkeypatch):
        monkeypatch.setattr(http_module.asyncio, "sleep", _no_sleep)
        attempts = 0

        def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(200, content=b"x" * 5000)

        async with client_with(settings, handler) as client:
            with pytest.raises(DownloadRejected):
                await client.get_bytes("https://example.invalid/big.jpg", max_bytes=10)

        assert attempts == 1

    def test_backoff_grows_and_is_jittered(self, settings):
        client = PoliteClient(settings)
        try:
            first = [client._backoff_seconds(0, None) for _ in range(20)]
            later = [client._backoff_seconds(3, None) for _ in range(20)]
        finally:
            pass

        assert min(later) > max(first), "backoff must grow with the attempt number"
        # Jitter exists so that four workers failing together do not retry in
        # lockstep forever.
        assert len(set(first)) > 1, "backoff must be jittered, not constant"

    def test_retry_after_header_overrides_our_own_backoff(self, settings):
        client = PoliteClient(settings)

        assert client._backoff_seconds(0, "7") == pytest.approx(7.0)

    def test_an_absurd_retry_after_is_capped(self, settings):
        # A server asking us to wait an hour should not hang the job.
        client = PoliteClient(settings)

        assert client._backoff_seconds(0, "99999") == pytest.approx(300.0)

    def test_a_malformed_retry_after_falls_back_to_exponential(self, settings):
        client = PoliteClient(settings)

        assert client._backoff_seconds(0, "next tuesday") > 0


class TestExtensionSniffing:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (JPEG_BYTES, ".jpg"),
            (PNG_BYTES, ".png"),
            (WEBP_BYTES, ".webp"),
            (b"GIF89a" + b"\x00" * 10, ".gif"),
            (b"II*\x00" + b"\x00" * 10, ".tif"),
        ],
    )
    def test_extension_comes_from_the_magic_bytes(self, data, expected):
        # Unsplash URLs end in `?w=2048&fm=jpg` and PD12M's end in `.jpeg`
        # regardless of the real container, so the bytes are the authority.
        assert http_module._extension_for(data, "https://example.invalid/x") == expected

    def test_falls_back_to_the_url_for_an_unknown_container(self):
        assert (
            http_module._extension_for(b"\x00\x01\x02\x03", "https://x.invalid/a.heic") == ".heic"
        )

    def test_query_strings_do_not_confuse_the_fallback(self):
        got = http_module._extension_for(b"\x00\x01", "https://x.invalid/a.jpg?w=2048&q=85")

        assert got == ".jpg"

    def test_gives_up_with_bin_rather_than_guessing(self):
        assert http_module._extension_for(b"\x00\x01", "https://x.invalid/noext") == ".bin"


class TestDownloadToArchive:
    async def test_stores_the_file_at_its_content_addressed_path(self, settings):
        def handler(request):
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            image_id, size = await client.download_to_archive(
                "https://example.invalid/a.jpg", source="megalith"
            )

        assert image_id == image_id_for(JPEG_BYTES)
        assert size == len(JPEG_BYTES)
        stored = settings.original_path("megalith", image_id, ".jpg")
        assert stored.read_bytes() == JPEG_BYTES

    async def test_an_already_present_original_is_not_rewritten(self, settings):
        # Originals are write-once (brief §0.3). Re-fetching the same bytes must
        # be a no-op, not an overwrite.
        def handler(request):
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            image_id, _ = await client.download_to_archive(
                "https://example.invalid/a.jpg", source="megalith"
            )
            stored = settings.original_path("megalith", image_id, ".jpg")
            marker = stored.stat().st_mtime_ns

            again, _ = await client.download_to_archive(
                "https://example.invalid/a.jpg", source="megalith"
            )

        assert again == image_id
        assert stored.stat().st_mtime_ns == marker, "the original was rewritten"

    async def test_the_disk_guard_runs_before_any_write(self, settings):
        # A full disk must stop the job at a resumable boundary rather than
        # leaving a half-written archive.
        settings.min_free_gb = 1_000_000

        def handler(request):  # pragma: no cover - must never be reached
            raise AssertionError("the guard should have stopped us before fetching")

        async with client_with(settings, handler) as client:
            with pytest.raises(DiskGuardTripped):
                await client.download_to_archive("https://example.invalid/a.jpg", source="megalith")


class TestPoliteness:
    async def test_never_exceeds_the_configured_concurrency(self, settings):
        settings.http_concurrency = 3
        in_flight = 0
        peak = 0

        async def handler(request):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler, concurrency=3) as client:
            await asyncio.gather(
                *(client.get_bytes(f"https://example.invalid/{i}.jpg") for i in range(12))
            )

        assert peak <= 3, f"{peak} requests were in flight; the limit is 3"

    async def test_sends_the_descriptive_user_agent(self, settings):
        seen: list[str] = []

        def handler(request):
            seen.append(request.headers["user-agent"])
            return httpx.Response(200, content=JPEG_BYTES)

        async with client_with(settings, handler) as client:
            await client.get_bytes("https://example.invalid/a.jpg")

        assert seen == ["VisualTaste/0.1 (personal research; local use)"]
        assert "personal research" in seen[0]

    def test_tls_verification_is_on_and_has_no_off_switch(self, settings):
        # Brief §0.7. Asserted as a property of the module's source so that
        # adding a `verify=False` escape hatch breaks a test.
        import inspect

        source = inspect.getsource(http_module)

        assert "verify=False" not in source


class TestFetchStats:
    def test_rejections_do_not_count_towards_the_failure_rate(self):
        # The brief gates on "<5% failure per source". Skipping 1,200 oversized
        # scans on purpose is not a 1,200-error failure.
        stats = FetchStats(requested=100, downloaded=90)
        for _ in range(9):
            stats.reject("too_large")
        stats.fail("http_404")

        assert stats.failure_rate == pytest.approx(0.01)

    def test_already_present_files_are_not_counted_as_attempts(self):
        stats = FetchStats(requested=100, downloaded=0, skipped_existing=100)

        assert stats.attempted == 0
        assert stats.failure_rate == 0.0

    def test_the_summary_names_every_cause(self):
        stats = FetchStats(requested=10, downloaded=7)
        stats.reject("too_large")
        stats.fail("timeout")
        stats.fail("timeout")

        summary = stats.summary()

        assert "too_large=1" in summary
        assert "timeout=2" in summary
        assert "failure_rate" in summary


async def _no_sleep(_seconds):
    """Collapse backoff waits so retry tests run in milliseconds."""
    return None
