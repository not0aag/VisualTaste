"""Decide whether a downloaded file is an image we can actually use.

This runs before any expensive work, and it is deliberately strict. Every image
that survives will be decoded twice more (thumbnail, display), embedded by two
neural networks, and then shown to a human; a file that is going to fail should
fail here, once, with a recorded reason.

The two-pass decode, and why it is not redundant
------------------------------------------------
Pillow's `verify()` checks the file's structure without decompressing the pixel
data. It is fast and it catches a corrupt header, but it will happily pass a
JPEG whose bottom half is missing -- the header is fine, the data simply stops.
A truncated file like that then raises halfway through `jobs/embed`, hours into
a run.

So validation opens the file twice:

1. `open()` + `verify()`  -- structure. Cheap, and it invalidates the object,
   which is why Pillow's own documentation says you must reopen afterwards.
2. `open()` + `load()`    -- every pixel. This is what catches truncation.

`ImageFile.LOAD_TRUNCATED_IMAGES` is left at its default of False on purpose:
we want the exception, not a half-grey image quietly entering the corpus.

The rejection rules are brief §7.2 step 1:

    decode error        the file is not a readable image
    short side < 256    too small to show full-screen or to embed at 224 px
    aspect ratio > 4:1  panoramas and banners; they crop badly and they are
                        usually not photographs of anything
    svg / heic / raw     containers Pillow cannot read without extra codecs

Colour modes are *not* a rejection. CMYK, palette, greyscale-with-alpha and
16-bit images are all real photographs; they are converted to RGB when the
derivatives are made (`data/derivatives.py`), and the original is left exactly
as it arrived, because the archive is write-once (§0.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

MIN_SHORT_SIDE = 256
MAX_ASPECT_RATIO = 4.0

# Containers we will not try to read. SVG is not a raster image at all; HEIC and
# the camera raw formats need codecs that are not installed, and adding them
# would mean more to go wrong for a handful of files we do not need.
REJECTED_SUFFIXES = frozenset(
    {
        ".svg",
        ".svgz",
        ".heic",
        ".heif",
        ".avif",
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".dng",
        ".orf",
        ".rw2",
        ".raf",
        ".pef",
        ".psd",
        ".ai",
        ".eps",
        ".pdf",
    }
)

# Pillow formats we accept. Anything else that happens to decode (ICO, PCX, ...)
# is rejected: it would be a surprise in this corpus, and a surprise is worth
# looking at rather than ingesting.
ACCEPTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "MPO"})

# Modes that must be converted before a WebP can be written. Recorded here so
# `derivatives.py` and the tests agree on one list.
MODES_NEEDING_CONVERSION = frozenset({"CMYK", "P", "PA", "LA", "L", "I", "I;16", "F", "1", "RGBa"})


@dataclass(frozen=True)
class ImageFacts:
    """What validation learned about a file that passed.

    `width` and `height` are the dimensions *after* EXIF orientation is applied,
    because that is what the viewer will see. A portrait photograph saved as
    landscape-plus-a-rotate-flag is 768x1024, not 1024x768, and using the raw
    numbers would mis-apply the aspect-ratio rule to it.
    """

    path: Path
    format: str
    mode: str
    width: int
    height: int
    bytes: int
    is_animated: bool
    needs_conversion: bool

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        return max(self.width, self.height) / max(1, min(self.width, self.height))


class ImageRejected(Exception):
    """This file is not usable. `reason` is a short stable slug for counting."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def validate_image(path: Path) -> ImageFacts:
    """Fully validate one file, returning its facts or raising `ImageRejected`.

    Raises rather than returning a flag because every caller either stores the
    facts or records the reason -- there is no third thing to do, and an
    exception carries the reason without a second return value.
    """
    suffix = path.suffix.lower()
    if suffix in REJECTED_SUFFIXES:
        raise ImageRejected("unsupported_format", suffix)

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise ImageRejected("unreadable_file", str(exc)) from exc

    if size_bytes == 0:
        raise ImageRejected("empty_file")

    # Pass 1: structure only. `verify` leaves the object unusable afterwards.
    try:
        with Image.open(path) as probe:
            image_format = probe.format or "UNKNOWN"
            probe.verify()
    except UnidentifiedImageError as exc:
        raise ImageRejected("not_an_image", str(exc)[:200]) from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageRejected("corrupt_header", str(exc)[:200]) from exc

    if image_format not in ACCEPTED_FORMATS:
        raise ImageRejected("unsupported_format", image_format)

    # Pass 2: decode every pixel. This is the one that catches truncation.
    try:
        with Image.open(path) as image:
            is_animated = getattr(image, "n_frames", 1) > 1
            if is_animated:
                # Brief §7.2: an animated image contributes its first frame.
                image.seek(0)

            image.load()
            mode = image.mode

            # EXIF orientation is applied before measuring, so the dimensions
            # describe what a viewer actually sees.
            oriented = ImageOps.exif_transpose(image) or image
            width, height = oriented.size
    except (OSError, SyntaxError, ValueError, EOFError) as exc:
        # A truncated JPEG lands here, having passed `verify()` moments ago.
        raise ImageRejected("truncated_or_corrupt", str(exc)[:200]) from exc

    if width <= 0 or height <= 0:
        raise ImageRejected("zero_dimension", f"{width}x{height}")

    if min(width, height) < MIN_SHORT_SIDE:
        raise ImageRejected("too_small", f"{width}x{height}")

    ratio = max(width, height) / min(width, height)
    if ratio > MAX_ASPECT_RATIO:
        raise ImageRejected("bad_aspect", f"{width}x{height} ({ratio:.1f}:1)")

    return ImageFacts(
        path=path,
        format=image_format,
        mode=mode,
        width=width,
        height=height,
        bytes=size_bytes,
        is_animated=is_animated,
        needs_conversion=mode in MODES_NEEDING_CONVERSION or mode != "RGB",
    )


def load_for_processing(path: Path) -> Image.Image:
    """Open an image as oriented RGB, ready for a derivative or an encoder.

    Everything downstream -- WebP derivatives, pHash, the colour histogram, both
    neural encoders -- wants exactly this: upright, three channels, 8 bits. Doing
    it in one place means a palette image cannot reach the colour histogram as
    an index array and quietly produce nonsense.

    The conversion happens in memory. The original file is never modified.
    """
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)
        image.load()

        oriented = ImageOps.exif_transpose(image) or image

        if oriented.mode == "RGB":
            # `.copy()` so the returned image survives the `with` block closing
            # the underlying file.
            return oriented.copy()

        if oriented.mode in ("RGBA", "LA", "PA", "P"):
            # Flatten transparency onto white rather than letting `convert`
            # discard the alpha channel, which turns transparent pixels black
            # and makes a logo on a transparent background look like a blot.
            rgba = oriented.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, (255, 255, 255))
            flattened.paste(rgba, mask=rgba.split()[-1])
            return flattened

        if oriented.mode in ("I", "I;16", "F"):
            # 16-bit and float images must be scaled into 8 bits; a plain
            # convert("RGB") on mode I clips everything above 255 to white.
            return ImageOps.autocontrast(oriented.convert("L")).convert("RGB")

        return oriented.convert("RGB")
