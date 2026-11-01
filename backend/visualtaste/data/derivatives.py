"""Make the two WebP renditions everything else reads.

The originals archive is write-once and is touched exactly twice in this
project's life: here, and never again. Every later stage -- the feed, both
encoders, the colour histogram, the pHash, the embedding map's hover
thumbnails -- reads a derivative instead, for three reasons:

- **Speed.** Decoding a 5504x5504 museum TIFF takes most of a second. Decoding
  the 1024 px WebP made from it takes a few milliseconds, and the embedding job
  does this 80,000 times per model.
- **Uniformity.** Derivatives are always RGB, always upright, always WebP. The
  awkward cases (CMYK, palette, 16-bit, EXIF-rotated, animated) are resolved
  once, here, instead of in five different places.
- **Safety.** The archive cannot be corrupted by a bug in this file, because
  nothing here writes to it.

Two sizes, per brief §6.2:

    display   long edge 1024, quality 82   what the Feed shows
    thumb     long edge  256, quality 80   grids, hover previews, pHash, colour

Why 1024: the Feed shows one image fitted to a laptop viewport, and 1024 covers
that at 2x pixel density on the long edge. Why WebP: at these quality settings
it is roughly 30% smaller than equivalent JPEG, and the whole derivative set has
to live alongside 80 GB of originals on a single drive.

The resize uses Lanczos, which is slower than bilinear and visibly sharper when
shrinking by a factor of five or more -- which is most of this corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from visualtaste.data.validate import load_for_processing

logger = logging.getLogger(__name__)

DISPLAY_LONG_EDGE = 1024
DISPLAY_QUALITY = 82

THUMB_LONG_EDGE = 256
THUMB_QUALITY = 80

# WebP's encoder effort, 0 (fast) to 6 (slow). Measured on 10 real Megalith
# images at 1024 px on this machine:
#
#     method=0   193 ms   139.2 KB
#     method=2   211 ms   125.3 KB
#     method=4   412 ms   122.8 KB   <- Pillow's default
#     method=6   586 ms   118.6 KB
#
# Going from 4 to 2 halves the encode time for 2% more bytes. Across 80,000
# images that is about half an hour of wall clock against 200 MB of disk, and
# encoding is the single largest cost in ingestion. See DECISIONS.md #9.
WEBP_METHOD = 2


@dataclass(frozen=True)
class DerivativePaths:
    """Where the two renditions of one image ended up."""

    display: Path
    thumb: Path


def resize_to_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """Scale so the longer side is `long_edge`, preserving the aspect ratio.

    Images already smaller than the target are returned untouched: upscaling
    would invent detail and cost bytes. Validation has already guaranteed a
    short side of at least 256 px, so a "small" image here is still usable.
    """
    width, height = image.size
    if max(width, height) <= long_edge:
        return image

    scale = long_edge / max(width, height)
    # `round` rather than `int`: truncating can produce an off-by-one that
    # changes the aspect ratio of a nearly-square image.
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _write_webp(image: Image.Image, path: Path, *, quality: int) -> None:
    """Save a WebP atomically, so a killed job leaves no half-written file.

    The same temp-then-replace pattern as `data/http.atomic_write`, but Pillow
    writes the file itself, so it cannot share that function.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    try:
        image.save(tmp, format="WEBP", quality=quality, method=WEBP_METHOD)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def build_derivatives(
    original_path: Path,
    display_path: Path,
    thumb_path: Path,
    *,
    overwrite: bool = False,
) -> DerivativePaths:
    """Write the display and thumb renditions for one original.

    Skips work that is already done unless `overwrite` is set, which is what
    makes `jobs/ingest` resumable: a run killed at image 40,000 re-reads the
    manifest and only builds what is missing.
    """
    paths = DerivativePaths(display=display_path, thumb=thumb_path)

    if not overwrite and paths.display.exists() and paths.thumb.exists():
        return paths

    # One decode of the original, two resizes from it. Decoding twice would
    # double the most expensive step.
    source = load_for_processing(original_path)

    display = resize_to_long_edge(source, DISPLAY_LONG_EDGE)
    _write_webp(display, paths.display, quality=DISPLAY_QUALITY)

    # The thumbnail is made from the display rendition, not the original: it is
    # already the right colour space and orientation, and shrinking 1024 -> 256
    # is four times cheaper than 5504 -> 256 with no visible difference at this
    # size.
    thumb = resize_to_long_edge(display, THUMB_LONG_EDGE)
    _write_webp(thumb, paths.thumb, quality=THUMB_QUALITY)

    return paths
