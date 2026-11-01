"""Unsplash Lite: 10,000 contemporary photographs, for stylistic range.

The other three sources skew historical and institutional -- museum scans,
Flickr Commons archives, iNaturalist specimens. Unsplash is the counterweight:
deliberately composed, modern, heavily stylised photography. Without it, a
"visual taste" learned from this corpus would have little to say about the kind
of image Alen is most likely to encounter day to day.

How it works: Unsplash distributes a research dataset as a single zip of TSV
files (`unsplash.com/data`, Lite = 25,000 photos, 305 MB measured 2026-10-30,
no account needed). The zip holds metadata only; `photos.tsv000` carries
`photo_image_url`, which is fetched with sizing parameters so we get a 2048 px
JPEG rather than a 6000 px original.

Licence and terms
-----------------
The Unsplash dataset terms permit commercial and non-commercial use but forbid
*redistributing* the images. This project never redistributes anything: the
archive is one folder on one laptop, nothing is served beyond `127.0.0.1`, and
`git` ignores both `data/` and `archive/`. The photographer's name and the photo
page URL are stored and displayed on every card, as the terms ask.

`license_url` is the Unsplash licence page, which is the correct per-image
licence for every photo in the set -- unlike the other sources, there is no
per-image variation to record.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from typing import TYPE_CHECKING

import numpy as np

from visualtaste.data.manifest import Candidate
from visualtaste.data.sources import download_http_candidates, needs_planning

if TYPE_CHECKING:
    from pathlib import Path

    from visualtaste.config import Settings
    from visualtaste.data.http import FetchStats, PoliteClient
    from visualtaste.data.manifest import Manifest

logger = logging.getLogger(__name__)

SOURCE = "unsplash_lite"
DEFAULT_TARGET = 10_000

DATASET_URL = "https://unsplash.com/data/lite/latest"
UNSPLASH_LICENSE_URL = "https://unsplash.com/license"

# Sizing parameters appended to `photo_image_url`. 2048 px on the long edge is
# ample for a 1024 px display derivative, and q=85 keeps the files near 600 KB
# instead of several MB.
IMAGE_PARAMS = "w=2048&q=85&fm=jpg"


def _dataset_zip_path(settings: Settings) -> Path:
    """Where the metadata zip is cached. Inside `data/`, so it is git-ignored."""
    return settings.data / "cache" / "unsplash" / "unsplash-lite-latest.zip"


async def ensure_dataset(settings: Settings, client: PoliteClient) -> Path:
    """Download the metadata zip once, and reuse it on every later run."""
    path = _dataset_zip_path(settings)
    if path.exists() and path.stat().st_size > 0:
        logger.info("unsplash_lite: using cached metadata zip (%s)", path.name)
        return path

    logger.info("unsplash_lite: downloading the Lite metadata zip (~305 MB)")
    # No size cap: this is a metadata archive, not an image, so the
    # `max_original_mb` rule does not apply to it.
    data = await client.get_bytes(DATASET_URL)

    from visualtaste.data.http import atomic_write

    atomic_write(path, data)
    logger.info("unsplash_lite: stored %.0f MB at %s", len(data) / 1024**2, path)
    return path


def read_photos_tsv(zip_path: Path) -> list[dict[str, str]]:
    """Read every `photos.tsv*` member out of the dataset zip.

    The Lite dataset ships one part (`photos.tsv000`); the Full dataset splits
    it. Reading whatever parts are present means this code works for either
    without a flag.
    """
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(n for n in archive.namelist() if n.startswith("photos.tsv"))
        if not members:
            raise RuntimeError(
                f"{zip_path} contains no photos.tsv* member; the dataset layout "
                f"has changed. Members: {archive.namelist()[:10]}"
            )
        for member in members:
            with archive.open(member) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
                # TSV, and captions contain unescaped quotes, so quoting is off:
                # with the default dialect a stray quote swallows later columns.
                rows.extend(csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE))
    logger.info("unsplash_lite: %d photo rows across %d TSV part(s)", len(rows), len(members))
    return rows


def plan(settings: Settings, rows: list[dict[str, str]], target: int) -> list[Candidate]:
    """Pick `target` photos at random (seeded), keeping attribution intact."""
    usable = [row for row in rows if _is_usable(row)]
    logger.info("unsplash_lite: %d rows usable after gates", len(usable))

    rng = np.random.default_rng(settings.seed)
    wanted = min(len(usable), int(target * 1.1))
    picked = rng.choice(len(usable), size=wanted, replace=False)

    return [_to_candidate(usable[int(i)]) for i in sorted(picked)]


def _is_usable(row: dict[str, str]) -> bool:
    """Require an image URL, a photo page and a named photographer.

    The last two are what the Unsplash terms ask us to display, so a row that
    cannot be attributed is not downloaded at all.
    """
    if not (row.get("photo_image_url") or "").strip():
        return False
    if not (row.get("photo_id") or "").strip():
        return False
    return bool((row.get("photographer_username") or "").strip())


def _to_candidate(row: dict[str, str]) -> Candidate:
    photo_id = row["photo_id"].strip()
    username = (row.get("photographer_username") or "").strip()
    separator = "&" if "?" in row["photo_image_url"] else "?"

    width = _as_int(row.get("photo_width"))
    height = _as_int(row.get("photo_height"))

    return Candidate(
        source=SOURCE,
        source_id=photo_id,
        url=f"{row['photo_image_url'].strip()}{separator}{IMAGE_PARAMS}",
        license_url=UNSPLASH_LICENSE_URL,
        source_url=(row.get("photo_url") or "").strip()
        or f"https://unsplash.com/photos/{photo_id}",
        author=(row.get("photographer_name") or "").strip() or username,
        author_url=f"https://unsplash.com/@{username}" if username else None,
        title=(row.get("photo_description") or "").strip() or None,
        # The declared dimensions describe the original upload, not the 2048 px
        # rendition we request, so they are recorded but not used as a gate:
        # every rendition comfortably clears the 256 px minimum.
        declared_width=width,
        declared_height=height,
    )


def _as_int(value: str | None) -> int | None:
    if not value or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


async def fetch(
    settings: Settings,
    manifest: Manifest,
    target: int = DEFAULT_TARGET,
    *,
    client: PoliteClient,
) -> FetchStats:
    """Ensure the metadata zip, plan (if needed), then download the photos."""
    if needs_planning(manifest, target):
        zip_path = await ensure_dataset(settings, client)
        added = manifest.add_all(plan(settings, read_photos_tsv(zip_path), target))
        logger.info("unsplash_lite: added %d new candidates to the manifest", added)
        manifest.flush(force=True)

    return await download_http_candidates(
        settings, client, manifest, source=SOURCE, target=target
    )
