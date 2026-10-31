"""Synthetic image fixtures for the ingestion tests.

These are *generated*, not committed. Two reasons: a few hundred real JPEGs
would bloat the repository for no gain, and a generated fixture is explicit
about exactly what makes it awkward. When a test says "a truncated JPEG", the
function that built it is right here and shows the truncation.

The set covers every case the Phase 3 Definition of Done names: corrupt,
truncated, tiny, CMYK, palette, LA, I;16, EXIF-rotated, animated, exact
duplicate and near duplicate.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _gradient(width: int, height: int, *, seed: int = 0) -> Image.Image:
    """A smooth, structured image.

    Structure matters: pHash works on the coarse light-and-dark layout, and
    random noise has none, so two noise images can hash identically and make a
    near-duplicate test lie.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)

    red = grid_x
    green = grid_y
    blue = (grid_x + grid_y) / 2
    # A few soft blobs, so the image has recognisable structure rather than
    # being a pure ramp.
    for _ in range(3):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = max(width, height) / 4
        mask = ((grid_x / 255 * width - cx) ** 2 + (grid_y / 255 * height - cy) ** 2) < radius**2
        blue = np.where(mask, 240, blue)

    stacked = np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(stacked, mode="RGB")


def write_valid_jpeg(path: Path, *, width: int = 800, height: int = 600, seed: int = 0) -> Path:
    _gradient(width, height, seed=seed).save(path, format="JPEG", quality=92)
    return path


def write_valid_png(path: Path, *, width: int = 600, height: int = 600) -> Path:
    _gradient(width, height).save(path, format="PNG")
    return path


def write_tiny(path: Path) -> Path:
    """Below the 256 px short-side minimum, so validation must reject it."""
    _gradient(200, 150).save(path, format="JPEG", quality=90)
    return path


def write_panorama(path: Path) -> Path:
    """Aspect ratio past 4:1, so validation must reject it."""
    _gradient(2400, 400).save(path, format="JPEG", quality=90)
    return path


def write_corrupt(path: Path) -> Path:
    """Random bytes with a JPEG extension: fails at the header."""
    path.write_bytes(b"\xff\xd8\xff" + bytes(range(256)) * 4)
    return path


def write_truncated(path: Path) -> Path:
    """A structurally valid JPEG whose pixel data stops half way.

    This is the case that makes the two-pass decode necessary: `verify()` reads
    the header and is satisfied, and only a full `load()` discovers that the
    scan data ends early.
    """
    buffer = io.BytesIO()
    _gradient(800, 600).save(buffer, format="JPEG", quality=92)
    data = buffer.getvalue()
    path.write_bytes(data[: len(data) // 2])
    return path


def write_cmyk(path: Path) -> Path:
    """A CMYK JPEG, as print-oriented archives produce."""
    _gradient(700, 500).convert("CMYK").save(path, format="JPEG", quality=90)
    return path


def write_palette(path: Path) -> Path:
    """A 256-colour palette PNG. Its array is indices, not colours."""
    _gradient(600, 400).convert("P", palette=Image.Palette.ADAPTIVE).save(path, format="PNG")
    return path


def write_greyscale_alpha(path: Path) -> Path:
    """Mode LA: greyscale plus an alpha channel."""
    grey = _gradient(600, 400).convert("L")
    alpha = Image.new("L", grey.size, 128)
    Image.merge("LA", (grey, alpha)).save(path, format="PNG")
    return path


def write_16bit(path: Path) -> Path:
    """Mode I;16, as scientific and archival scanners produce.

    A plain `convert("RGB")` on this clips everything above 255 to white, which
    is why `validate.load_for_processing` rescales instead.
    """
    array = (np.linspace(0, 65535, 600 * 400).reshape(400, 600)).astype(np.uint16)
    # No explicit mode: Pillow infers "I;16" from the uint16 dtype, and passing
    # the mode by hand is rejected as a dtype mismatch in current versions.
    Image.fromarray(array).save(path, format="PNG")
    return path


def write_rgba(path: Path) -> Path:
    """Transparency that must be flattened onto white, not discarded."""
    base = _gradient(600, 400).convert("RGBA")
    pixels = np.asarray(base).copy()
    pixels[:200, :, 3] = 0  # top half fully transparent
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG")
    return path


def write_exif_rotated(path: Path) -> Path:
    """A 600x400 JPEG tagged orientation=6, so it *displays* as 400x600.

    Validation must report the dimensions a viewer would see, because the
    aspect-ratio rule and the derivative sizing both depend on them.
    """
    image = _gradient(600, 400)
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90 degrees clockwise
    image.save(path, format="JPEG", quality=90, exif=exif)
    return path


def write_animated_gif(path: Path) -> Path:
    """A multi-frame GIF. Only the first frame should be ingested."""
    frames = [_gradient(400, 300, seed=i).convert("P") for i in range(4)]
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    return path


def write_near_duplicate(path: Path, original: Path) -> Path:
    """The same photograph re-encoded at low quality.

    Different bytes, so a different `image_id` and a different SHA-256, but
    visually the same picture -- which is exactly what pHash exists to catch.
    """
    with Image.open(original) as image:
        image.load()
        image.convert("RGB").save(path, format="JPEG", quality=35)
    return path


def write_empty(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def build_all(directory: Path) -> dict[str, Path]:
    """Write the whole fixture set into `directory` and return a name -> path map."""
    directory.mkdir(parents=True, exist_ok=True)

    valid = write_valid_jpeg(directory / "valid.jpg", seed=1)
    fixtures = {
        "valid_jpeg": valid,
        "valid_png": write_valid_png(directory / "valid.png"),
        "second_valid": write_valid_jpeg(directory / "valid2.jpg", width=900, height=700, seed=2),
        "tiny": write_tiny(directory / "tiny.jpg"),
        "panorama": write_panorama(directory / "panorama.jpg"),
        "corrupt": write_corrupt(directory / "corrupt.jpg"),
        "truncated": write_truncated(directory / "truncated.jpg"),
        "cmyk": write_cmyk(directory / "cmyk.jpg"),
        "palette": write_palette(directory / "palette.png"),
        "greyscale_alpha": write_greyscale_alpha(directory / "greyscale_alpha.png"),
        "sixteen_bit": write_16bit(directory / "sixteen_bit.png"),
        "rgba": write_rgba(directory / "rgba.png"),
        "exif_rotated": write_exif_rotated(directory / "exif_rotated.jpg"),
        "animated": write_animated_gif(directory / "animated.gif"),
        "empty": write_empty(directory / "empty.jpg"),
        "svg": _write_svg(directory / "vector.svg"),
    }
    fixtures["near_duplicate"] = write_near_duplicate(directory / "near_dup.jpg", valid)
    return fixtures


def _write_svg(path: Path) -> Path:
    """Not a raster image at all; rejected on its extension before any decode."""
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500"></svg>',
        encoding="utf-8",
    )
    return path
