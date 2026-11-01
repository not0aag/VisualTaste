"""Megalith-10M: 30,000 public-domain Flickr photographs, the largest block.

Megalith-10M is a link list of 9.58 million Flickr photos filtered to
"no known copyright restrictions", CC0, Public Domain Mark, or US Government
Work. The original repo (`madebyollin/megalith-10m`) holds only URLs; the
DrawThings community mirror (`drawthingsai/megalith-10m`, confirmed live on
2026-10-30) holds the actual image files as 959 WebDataset shards, each a
~1.81 GB tar of roughly 10,000 images at about 1024 px.

Two paths, and why we take the first
------------------------------------
Pulling three tar shards from the mirror is one resumable transfer of 5.4 GB.
Fetching the same 30,000 images from `farm*.staticflickr.com` would be 30,000
requests to someone else's servers for a personal experiment. The mirror is
both kinder and faster, so `fetch` uses it. `fetch_from_flickr` implements the
direct path the brief also names, for the case where the mirror disappears.

The licence subtlety -- read this before trusting the metadata
--------------------------------------------------------------
Megalith carries **no per-image licence field**. Neither the original parquet
(`url`, `url_highres`, `url_source`) nor the mirror's (`caption`, `url`, `key`,
`sha256`, dimensions) names a licence for an individual photo. The guarantee is
collection-level: the dataset was *built* by filtering Flickr for those four
public-domain statuses.

The brief requires a licence URL per image and excludes images without one. The
honest reading, taken here, is to record the dataset's documented provenance as
the licence URL and to store the Flickr photo page as `source_url`, so any
individual image's status stays checkable by a human in one click. Claiming a
specific licence we have not verified per image -- stamping every file CC0, say
-- would be worse: it would look like evidence while being a guess.

The alternative was querying the Flickr API for each image's true licence,
which needs an API key and 30,000 calls, and is outside what this brief
authorises. See DECISIONS.md #8.
"""

from __future__ import annotations

import logging
import tarfile
from typing import TYPE_CHECKING

import pandas as pd

from visualtaste.config import check_disk_headroom
from visualtaste.data.http import FetchStats, atomic_write, image_id_for
from visualtaste.data.manifest import Candidate
from visualtaste.data.sources import dimensions_acceptable, download_http_candidates

if TYPE_CHECKING:
    from visualtaste.config import Settings
    from visualtaste.data.http import PoliteClient
    from visualtaste.data.manifest import Manifest

logger = logging.getLogger(__name__)

SOURCE = "megalith"
DEFAULT_TARGET = 30_000

MIRROR_REPO = "drawthingsai/megalith-10m"
ORIGINAL_REPO = "madebyollin/megalith-10m"

# The collection-level public-domain statement these images rest on. Not a
# per-image licence: see the module docstring.
LICENCE_STATEMENT_URL = "https://huggingface.co/datasets/madebyollin/megalith-10m"

IMAGES_PER_SHARD = 10_000

# Image members inside a WebDataset tar. img2dataset also writes `.json` and
# `.txt` sidecars per record, which we ignore.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def shards_needed(target: int) -> int:
    """How many tar shards to pull for `target` images, with a little margin.

    Megalith images are pre-resized and pre-filtered, so almost nothing is
    rejected; one extra shard is enough slack.
    """
    return max(1, -(-target // IMAGES_PER_SHARD) + 1)


async def fetch(
    settings: Settings,
    manifest: Manifest,
    target: int = DEFAULT_TARGET,
    *,
    client: PoliteClient,
) -> FetchStats:
    """Download tar shards from the mirror and extract images into the archive.

    Resumable at two levels: `hf_hub_download` resumes a partial tar, and the
    manifest records every image already extracted, so a shard interrupted
    halfway is re-read but nothing is re-written.
    """
    from huggingface_hub import hf_hub_download

    settings.apply_hf_env()

    stats = FetchStats()
    stats.skipped_existing = manifest.succeeded
    stored = manifest.succeeded
    if stored >= target:
        logger.info("megalith: %d images already present, target %d met", stored, target)
        return stats

    max_bytes = settings.max_original_mb * 1024**2

    for shard in range(shards_needed(target)):
        if stored >= target:
            break
        check_disk_headroom(settings.archive, settings.min_free_gb)

        logger.info("megalith: shard %05d -- fetching metadata", shard)
        metadata = _load_shard_metadata(shard)

        logger.info("megalith: shard %05d -- fetching tar (~1.8 GB, resumable)", shard)
        tar_path = hf_hub_download(MIRROR_REPO, f"{shard:05d}.tar", repo_type="dataset")

        extracted = _extract_shard(
            settings,
            manifest,
            stats,
            tar_path=tar_path,
            shard=shard,
            metadata=metadata,
            remaining=target - stored,
            max_bytes=max_bytes,
        )
        stored += extracted
        manifest.flush(force=True)
        logger.info(
            "megalith: shard %05d done, %d extracted, %d/%d stored",
            shard,
            extracted,
            stored,
            target,
        )

        # The cached tar is 1.8 GB and we have already copied out what we want.
        # Deleting it is the difference between 5.4 GB and 10.8 GB of disk on a
        # machine with ~25 GB of headroom. It lives inside `data/cache`, which
        # this project owns, and `hf_hub_download` will re-fetch it if a later
        # run needs it again.
        _discard_cached_tar(tar_path)

    logger.info("megalith: finished. %s", stats.summary())
    return stats


def _load_shard_metadata(shard: int) -> dict[str, dict[str, object]]:
    """Map `key` -> metadata row for one shard, from its sibling parquet.

    The mirror's `caption` column is not a caption: it holds the Flickr photo
    page URL (an artefact of how the mirror was built with img2dataset). That is
    exactly the provenance link we need, so it is read as `source_url` -- with
    this comment, because a future reader will otherwise assume it is text.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MIRROR_REPO, f"{shard:05d}.parquet", repo_type="dataset")
    frame = pd.read_parquet(path, columns=["key", "url", "caption", "status", "width", "height"])
    frame = frame[frame["status"] == "success"]
    return {str(row["key"]): row for row in frame.to_dict("records")}


def _extract_shard(
    settings: Settings,
    manifest: Manifest,
    stats: FetchStats,
    *,
    tar_path: str,
    shard: int,
    metadata: dict[str, dict[str, object]],
    remaining: int,
    max_bytes: int,
) -> int:
    """Stream one tar, writing wanted images into the archive. Returns the count.

    The tar is read member by member rather than extracted wholesale: a 1.8 GB
    archive does not need to exist twice on disk, and we only want the image
    members anyway.
    """
    extracted = 0

    with tarfile.open(tar_path, mode="r|*") as tar:
        # Streaming mode ("r|*") cannot seek, which is what keeps memory flat
        # across a 1.8 GB file.
        for member in tar:
            if extracted >= remaining:
                break
            if not member.isfile():
                continue

            name = member.name
            suffix = _image_suffix(name)
            if suffix is None:
                continue  # A .json or .txt sidecar.

            key = name.rsplit("/", 1)[-1][: -len(suffix)]
            source_id = f"{shard:05d}/{key}"

            existing = manifest.candidates.get(source_id)
            if existing is not None and existing.resolved:
                continue  # Already handled on an earlier run.

            row = metadata.get(key)
            if row is None:
                continue  # Not a successful download in the mirror's own manifest.

            candidate = existing or _to_candidate(source_id, row)
            if existing is None:
                manifest.add_all([candidate])

            stats.requested += 1
            candidate.attempts += 1

            if not dimensions_acceptable(candidate.declared_width, candidate.declared_height):
                candidate.reject("bad_dimensions")
                stats.reject("bad_dimensions")
                continue

            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover - defensive
                candidate.fail("unreadable_tar_member")
                stats.fail("unreadable_tar_member")
                continue
            data = handle.read()

            if len(data) > max_bytes:
                candidate.reject("too_large")
                stats.reject("too_large")
                continue
            if not data:
                candidate.reject("empty_response")
                stats.reject("empty_response")
                continue

            image_id = image_id_for(data)
            path = settings.original_path(SOURCE, image_id, suffix)
            if not path.exists():
                atomic_write(path, data)

            candidate.succeed(image_id, len(data))
            stats.downloaded += 1
            stats.bytes_written += len(data)
            extracted += 1
            manifest.mark_resolved()

    return extracted


def _image_suffix(name: str) -> str | None:
    lowered = name.lower()
    for suffix in IMAGE_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return None


def _to_candidate(source_id: str, row: dict[str, object]) -> Candidate:
    return Candidate(
        source=SOURCE,
        source_id=source_id,
        url=str(row["url"]),
        license_url=LICENCE_STATEMENT_URL,
        # See `_load_shard_metadata`: `caption` holds the Flickr photo page URL.
        source_url=str(row["caption"]) if row.get("caption") else None,
        declared_width=_as_int(row.get("width")),
        declared_height=_as_int(row.get("height")),
    )


def _as_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)  # type: ignore[arg-type]


def _discard_cached_tar(tar_path: str) -> None:
    """Delete a cached shard tar and the symlink pointing at it, if present.

    `hf_hub_download` returns a path inside `<DATA>/cache/hf`, which this
    project owns. Nothing outside `data/` is ever touched (brief §0.3/§0.4).
    """
    from pathlib import Path

    path = Path(tar_path)
    try:
        blob = path.resolve()
        if path.is_symlink():
            path.unlink()
        if blob.exists():
            size = blob.stat().st_size
            blob.unlink()
            logger.info("megalith: freed %.1f GB of shard cache", size / 1024**3)
    except OSError as exc:
        # Not fatal: a leftover tar costs disk, not correctness.
        logger.warning("megalith: could not remove cached tar %s (%s)", tar_path, exc)


async def fetch_from_flickr(
    settings: Settings,
    manifest: Manifest,
    target: int = DEFAULT_TARGET,
    *,
    client: PoliteClient,
) -> FetchStats:
    """Fallback path: fetch from the original Flickr URLs instead of the mirror.

    Only for use if the mirror disappears. It is 30,000 polite requests to
    `staticflickr.com` rather than three tar transfers, so it is slower and
    costs someone else more; `fetch` is the default for that reason.
    """
    if not manifest.candidates:
        manifest.add_all(_plan_from_original_repo(settings, target))
        manifest.flush(force=True)

    logger.warning(
        "megalith: using the Flickr fallback path -- %d individual requests",
        len(manifest.pending()),
    )
    return await download_http_candidates(settings, client, manifest, source=SOURCE, target=target)


def _plan_from_original_repo(settings: Settings, target: int) -> list[Candidate]:
    """Sample candidates from the original link-list parquet (seeded)."""
    import numpy as np
    from huggingface_hub import hf_hub_download

    settings.apply_hf_env()
    path = hf_hub_download(
        ORIGINAL_REPO,
        "data/megalith-00001-of-00010.parquet",
        repo_type="dataset",
    )
    frame = pd.read_parquet(path, columns=["url_highres", "url_source"])

    rng = np.random.default_rng(settings.seed)
    wanted = min(len(frame), int(target * 1.2))
    picked = sorted(int(i) for i in rng.choice(len(frame), size=wanted, replace=False))
    chosen = frame.iloc[picked]

    return [
        Candidate(
            source=SOURCE,
            # The Flickr photo id is the stable part of the static URL.
            source_id=str(row["url_highres"]).rsplit("/", 1)[-1].split("_")[0],
            url=str(row["url_highres"]),
            license_url=LICENCE_STATEMENT_URL,
            source_url=str(row["url_source"]),
        )
        for row in chosen.to_dict("records")
    ]
