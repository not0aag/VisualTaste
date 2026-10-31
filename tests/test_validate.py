"""Tests for validation and the awkward-image cases the Phase 3 DoD names."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests import fixtures
from visualtaste.data.validate import (
    MAX_ASPECT_RATIO,
    MIN_SHORT_SIDE,
    ImageRejected,
    load_for_processing,
    validate_image,
)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    """The whole fixture set, built once for the module."""
    return fixtures.build_all(tmp_path_factory.mktemp("fixtures"))


class TestAccepted:
    def test_an_ordinary_jpeg_passes(self, images):
        facts = validate_image(images["valid_jpeg"])

        assert facts.format == "JPEG"
        assert (facts.width, facts.height) == (800, 600)
        assert facts.bytes > 0
        assert facts.is_animated is False

    def test_a_png_passes(self, images):
        assert validate_image(images["valid_png"]).format == "PNG"

    def test_cmyk_is_accepted_not_rejected(self, images):
        # Print-oriented archives produce these. The mode is converted when the
        # derivative is written; the original is left exactly as it arrived.
        facts = validate_image(images["cmyk"])

        assert facts.mode == "CMYK"
        assert facts.needs_conversion is True

    def test_a_palette_png_is_accepted(self, images):
        facts = validate_image(images["palette"])

        assert facts.mode == "P"
        assert facts.needs_conversion is True

    def test_greyscale_with_alpha_is_accepted(self, images):
        assert validate_image(images["greyscale_alpha"]).mode == "LA"

    def test_a_16_bit_image_is_accepted(self, images):
        facts = validate_image(images["sixteen_bit"])

        assert facts.mode in ("I", "I;16")
        assert facts.needs_conversion is True

    def test_an_animated_gif_is_accepted_and_flagged(self, images):
        # Brief §7.2: animated images contribute their first frame.
        facts = validate_image(images["animated"])

        assert facts.is_animated is True
        assert (facts.width, facts.height) == (400, 300)


class TestRejected:
    def test_a_corrupt_file_is_rejected_at_the_header(self, images):
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(images["corrupt"])

        assert excinfo.value.reason in ("corrupt_header", "not_an_image")

    def test_a_truncated_jpeg_is_rejected(self, images):
        # The case that justifies decoding twice: `verify()` passes this file
        # because its header is intact, and only a full `load()` finds that the
        # pixel data stops half way.
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(images["truncated"])

        assert excinfo.value.reason == "truncated_or_corrupt"

    def test_verify_alone_would_not_catch_the_truncated_file(self, images):
        # Demonstrates why the second pass exists, rather than asserting it.
        with Image.open(images["truncated"]) as probe:
            probe.verify()  # must not raise

        with pytest.raises(ImageRejected):
            validate_image(images["truncated"])

    def test_an_image_below_the_short_side_minimum_is_rejected(self, images):
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(images["tiny"])

        assert excinfo.value.reason == "too_small"
        assert str(MIN_SHORT_SIDE) not in excinfo.value.reason  # detail carries the size

    def test_a_panorama_past_four_to_one_is_rejected(self, images):
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(images["panorama"])

        assert excinfo.value.reason == "bad_aspect"

    def test_an_empty_file_is_rejected(self, images):
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(images["empty"])

        assert excinfo.value.reason == "empty_file"

    def test_an_svg_is_rejected_on_its_extension(self, images):
        # Rejected before any decode attempt: it is not a raster image at all.
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(images["svg"])

        assert excinfo.value.reason == "unsupported_format"

    def test_a_missing_file_is_rejected_rather_than_crashing(self, tmp_path):
        with pytest.raises(ImageRejected) as excinfo:
            validate_image(tmp_path / "does_not_exist.jpg")

        assert excinfo.value.reason == "unreadable_file"

    @pytest.mark.parametrize("suffix", [".heic", ".cr2", ".dng", ".psd", ".pdf"])
    def test_formats_needing_extra_codecs_are_rejected(self, tmp_path, suffix):
        path = tmp_path / f"file{suffix}"
        path.write_bytes(b"\x00" * 1000)

        with pytest.raises(ImageRejected) as excinfo:
            validate_image(path)

        assert excinfo.value.reason == "unsupported_format"


class TestExifOrientation:
    def test_dimensions_are_reported_as_a_viewer_would_see_them(self, images):
        # The file is stored 600x400 with orientation=6 (rotate 90), so what a
        # viewer sees is 400x600. Using the stored numbers would apply the
        # aspect-ratio rule to the wrong shape and size derivatives wrongly.
        facts = validate_image(images["exif_rotated"])

        assert (facts.width, facts.height) == (400, 600)

    def test_loading_for_processing_applies_the_rotation(self, images):
        loaded = load_for_processing(images["exif_rotated"])

        assert loaded.size == (400, 600)


class TestLoadForProcessing:
    def test_always_returns_rgb(self, images):
        for name in (
            "valid_jpeg",
            "valid_png",
            "cmyk",
            "palette",
            "greyscale_alpha",
            "sixteen_bit",
            "rgba",
            "animated",
        ):
            loaded = load_for_processing(images[name])
            assert loaded.mode == "RGB", f"{name} produced mode {loaded.mode}"

    def test_a_16_bit_image_is_rescaled_not_clipped(self, images):
        # convert("RGB") on mode I clips everything above 255 to white, which
        # would turn an archival scan into a white rectangle.
        loaded = load_for_processing(images["sixteen_bit"])
        array = np.asarray(loaded)

        assert array.min() < 60, "the dark end was lost"
        assert array.max() > 200, "the bright end was lost"
        assert len(np.unique(array)) > 20, "the image was flattened to a few values"

    def test_transparency_is_flattened_onto_white_not_black(self, images):
        # Discarding alpha turns transparent pixels black, so a product shot on
        # a transparent background becomes a dark blot.
        loaded = load_for_processing(images["rgba"])
        array = np.asarray(loaded)

        top_half = array[:200]
        assert top_half.mean() > 200, "transparent area did not become white"

    def test_an_animated_gif_yields_its_first_frame(self, images):
        loaded = load_for_processing(images["animated"])

        assert loaded.size == (400, 300)
        assert loaded.mode == "RGB"

    def test_the_returned_image_outlives_the_open_file(self, images):
        # `load_for_processing` opens inside a `with`, so the result has to be a
        # copy or every caller gets "seek of closed file".
        loaded = load_for_processing(images["valid_jpeg"])

        assert np.asarray(loaded).shape == (600, 800, 3)

    def test_the_original_file_is_never_modified(self, images):
        before = images["cmyk"].read_bytes()

        load_for_processing(images["cmyk"])

        assert images["cmyk"].read_bytes() == before, "the write-once archive was touched"


class TestFactsArithmetic:
    def test_short_side_and_aspect_ratio(self, images):
        facts = validate_image(images["valid_jpeg"])

        assert facts.short_side == 600
        assert facts.aspect_ratio == pytest.approx(800 / 600)
        assert facts.aspect_ratio < MAX_ASPECT_RATIO
