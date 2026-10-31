"""Tests for the WebP derivatives."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests import fixtures
from visualtaste.data.derivatives import (
    DISPLAY_LONG_EDGE,
    THUMB_LONG_EDGE,
    build_derivatives,
    resize_to_long_edge,
)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return fixtures.build_all(tmp_path_factory.mktemp("fixtures"))


class TestResize:
    def test_scales_the_long_edge_to_the_target(self):
        image = Image.new("RGB", (2000, 1000))

        assert resize_to_long_edge(image, 1024).size == (1024, 512)

    def test_works_on_portrait_images_too(self):
        image = Image.new("RGB", (1000, 2000))

        assert resize_to_long_edge(image, 1024).size == (512, 1024)

    def test_preserves_the_aspect_ratio(self):
        image = Image.new("RGB", (1600, 900))

        width, height = resize_to_long_edge(image, 1024).size

        assert width / height == pytest.approx(1600 / 900, rel=1e-2)

    def test_a_small_image_is_left_alone(self):
        # Upscaling would invent detail and cost bytes for nothing.
        image = Image.new("RGB", (600, 400))

        assert resize_to_long_edge(image, 1024).size == (600, 400)

    def test_a_nearly_square_image_does_not_drift(self):
        # `int()` truncation here produces an off-by-one that visibly changes
        # the proportions of an almost-square image; `round()` does not.
        image = Image.new("RGB", (1001, 1000))

        width, height = resize_to_long_edge(image, 500).size

        assert abs(width - height) <= 1

    def test_never_produces_a_zero_dimension(self):
        image = Image.new("RGB", (4000, 100))

        width, height = resize_to_long_edge(image, 10).size

        assert width >= 1 and height >= 1


class TestBuildDerivatives:
    def test_writes_both_renditions_at_the_right_sizes(self, images, tmp_path):
        display = tmp_path / "d" / "x.webp"
        thumb = tmp_path / "t" / "x.webp"

        build_derivatives(images["valid_jpeg"], display, thumb)

        with Image.open(display) as d:
            assert d.format == "WEBP"
            assert max(d.size) == DISPLAY_LONG_EDGE or max(d.size) == 800
        with Image.open(thumb) as t:
            assert t.format == "WEBP"
            assert max(t.size) == THUMB_LONG_EDGE

    def test_derivatives_are_always_rgb(self, images, tmp_path):
        for name in ("cmyk", "palette", "greyscale_alpha", "sixteen_bit", "rgba", "animated"):
            display = tmp_path / f"{name}_d.webp"
            thumb = tmp_path / f"{name}_t.webp"

            build_derivatives(images[name], display, thumb)

            with Image.open(display) as d:
                d.load()
                assert d.convert("RGB").mode == "RGB"
                assert d.size[0] > 0

    def test_an_exif_rotated_original_comes_out_upright(self, images, tmp_path):
        # Stored 600x400 with orientation=6, so the derivative must be portrait.
        display = tmp_path / "d.webp"
        thumb = tmp_path / "t.webp"

        build_derivatives(images["exif_rotated"], display, thumb)

        with Image.open(display) as d:
            assert d.size[1] > d.size[0], "the rotation was not applied"

    def test_a_large_original_is_reduced_to_1024(self, tmp_path):
        big = tmp_path / "big.jpg"
        Image.new("RGB", (4000, 3000), (120, 60, 30)).save(big, format="JPEG")
        display = tmp_path / "d.webp"
        thumb = tmp_path / "t.webp"

        build_derivatives(big, display, thumb)

        with Image.open(display) as d:
            assert max(d.size) == DISPLAY_LONG_EDGE
        with Image.open(thumb) as t:
            assert max(t.size) == THUMB_LONG_EDGE

    def test_existing_derivatives_are_not_rebuilt(self, images, tmp_path):
        # This is what makes ingestion resumable: a run killed at image 40,000
        # must not redo the first 40,000 resizes.
        display = tmp_path / "d.webp"
        thumb = tmp_path / "t.webp"
        build_derivatives(images["valid_jpeg"], display, thumb)
        marker = display.stat().st_mtime_ns

        build_derivatives(images["valid_jpeg"], display, thumb)

        assert display.stat().st_mtime_ns == marker

    def test_overwrite_forces_a_rebuild(self, images, tmp_path):
        display = tmp_path / "d.webp"
        thumb = tmp_path / "t.webp"
        build_derivatives(images["valid_jpeg"], display, thumb)
        display.write_bytes(b"corrupted")

        build_derivatives(images["valid_jpeg"], display, thumb, overwrite=True)

        with Image.open(display) as d:
            assert d.format == "WEBP"

    def test_parent_directories_are_created(self, images, tmp_path):
        display = tmp_path / "a" / "b" / "c" / "d.webp"
        thumb = tmp_path / "x" / "y" / "t.webp"

        build_derivatives(images["valid_jpeg"], display, thumb)

        assert display.exists() and thumb.exists()

    def test_no_part_file_is_left_behind(self, images, tmp_path):
        build_derivatives(images["valid_jpeg"], tmp_path / "d.webp", tmp_path / "t.webp")

        assert list(tmp_path.glob(".*.part")) == []

    def test_the_original_is_never_modified(self, images, tmp_path):
        before = images["valid_jpeg"].read_bytes()

        build_derivatives(images["valid_jpeg"], tmp_path / "d.webp", tmp_path / "t.webp")

        assert images["valid_jpeg"].read_bytes() == before

    def test_the_thumbnail_still_resembles_the_original(self, images, tmp_path):
        # Guards against a resize chain that loses the image entirely -- the
        # thumbnail is built from the display rendition, not the original, so a
        # mistake there would show up as a blank or mismatched thumbnail.
        display = tmp_path / "d.webp"
        thumb = tmp_path / "t.webp"
        build_derivatives(images["valid_jpeg"], display, thumb)

        with Image.open(images["valid_jpeg"]) as original, Image.open(thumb) as t:
            original_small = np.asarray(
                original.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
            thumb_small = np.asarray(
                t.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )

        mean_absolute_error = float(np.abs(original_small - thumb_small).mean())
        assert mean_absolute_error < 12, (
            f"thumbnail diverged from the original ({mean_absolute_error:.1f})"
        )
