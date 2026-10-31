"""Tests for the colour descriptor, including the module's worked example."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visualtaste.data.colour import (
    BINS_PER_AXIS,
    DESCRIPTOR_SIZE,
    colour_descriptor,
    dominant_colours,
    lab_histogram,
    rgb_to_lab,
    srgb_to_linear,
)


class TestSrgbToLinear:
    def test_black_and_white_are_fixed_points(self):
        assert srgb_to_linear(np.array(0.0)) == pytest.approx(0.0)
        assert srgb_to_linear(np.array(1.0)) == pytest.approx(1.0)

    def test_mid_grey_is_pulled_down_by_the_gamma_curve(self):
        # sRGB 0.5 is about 21% of the light of white, not 50%. This is exactly
        # the correction that makes CIELAB arithmetic meaningful, and skipping
        # it is the classic colour-space bug.
        assert float(srgb_to_linear(np.array(0.5))) == pytest.approx(0.2140, abs=1e-3)

    def test_the_curve_is_monotonic(self):
        values = np.linspace(0, 1, 50)

        linear = srgb_to_linear(values)

        assert np.all(np.diff(linear) > 0)


class TestRgbToLab:
    def test_white_maps_to_lightness_100_and_neutral_chroma(self):
        lab = rgb_to_lab(np.array([[255, 255, 255]]))

        assert lab[0, 0] == pytest.approx(100.0, abs=1e-3)
        assert lab[0, 1] == pytest.approx(0.0, abs=1e-2)
        assert lab[0, 2] == pytest.approx(0.0, abs=1e-2)

    def test_black_maps_to_lightness_zero(self):
        lab = rgb_to_lab(np.array([[0, 0, 0]]))

        assert lab[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_mid_grey_lands_near_lightness_54(self):
        # The well-known value for sRGB 128 grey. A wrong gamma step shows up
        # here immediately: skipping it gives about 46 instead.
        lab = rgb_to_lab(np.array([[128, 128, 128]]))

        assert lab[0, 0] == pytest.approx(53.59, abs=0.1)

    def test_red_is_positive_on_the_a_axis(self):
        lab = rgb_to_lab(np.array([[255, 0, 0]]))

        assert lab[0, 1] > 60, "red must be strongly positive a*"
        assert lab[0, 2] > 0, "red is also yellow-ward"

    def test_blue_is_negative_on_the_b_axis(self):
        lab = rgb_to_lab(np.array([[0, 0, 255]]))

        assert lab[0, 2] < -60, "blue must be strongly negative b*"

    def test_green_is_negative_on_the_a_axis(self):
        lab = rgb_to_lab(np.array([[0, 255, 0]]))

        assert lab[0, 1] < -60

    def test_the_shape_is_preserved(self):
        pixels = np.zeros((7, 5, 3), dtype=np.uint8)

        assert rgb_to_lab(pixels).shape == (7, 5, 3)

    def test_lightness_compresses_luminance_by_a_cube_root(self):
        # The property that makes CIELAB perceptually useful: L* is roughly the
        # cube root of luminance, not luminance itself. A surface reflecting
        # half as much light as white does not look half as bright -- it looks
        # about three quarters as bright, and L* says so.
        #
        # The sRGB byte whose *linear* luminance is 0.5:
        half_luminance_byte = round(255 * (1.055 * 0.5 ** (1 / 2.4) - 0.055))

        lab = rgb_to_lab(np.array([[half_luminance_byte] * 3]))

        # 116 * cbrt(0.5) - 16 = 76.07, not 50.
        assert lab[0, 0] == pytest.approx(76.07, abs=0.5)

    def test_equal_byte_steps_are_not_equal_lightness_steps(self):
        # Equal steps in sRGB bytes produce unequal L* steps, which is precisely
        # why colour distance must not be measured in RGB. (Near black both
        # transfer functions are in their linear segment, so that end is
        # compressed rather than expanded -- the point is only that the steps
        # differ, not which direction.)
        steps = [
            abs(rgb_to_lab(np.array([[v + 10] * 3]))[0, 0] - rgb_to_lab(np.array([[v] * 3]))[0, 0])
            for v in (0, 60, 120, 200)
        ]

        assert max(steps) > 1.5 * min(steps)


class TestLabHistogram:
    def test_the_module_worked_example(self):
        """3 pixels in bin 0, 1 pixel in bin 63 -> [0.866, 0, ..., 0, 0.5]."""
        # Bin 0 is dark with minimal a*/b*; bin 63 is light with maximal both.
        # Constructed directly in Lab space so the bins are unambiguous.
        lab = np.array(
            [
                [1.0, -109.0, -109.0],
                [1.0, -109.0, -109.0],
                [1.0, -109.0, -109.0],
                [99.0, 109.0, 109.0],
            ]
        )

        descriptor = lab_histogram(lab)

        assert descriptor[0] == pytest.approx(np.sqrt(0.75), abs=1e-6)
        assert descriptor[0] == pytest.approx(0.8660, abs=1e-4)
        assert descriptor[-1] == pytest.approx(0.5, abs=1e-6)
        assert descriptor[1:-1].sum() == pytest.approx(0.0)

    def test_the_hellinger_transform_makes_every_descriptor_a_unit_vector(self):
        # The identity from the docstring: the squares are the original
        # probabilities, and probabilities sum to 1. So no separate
        # normalisation step is needed anywhere downstream.
        rng = np.random.default_rng(1729)
        lab = np.column_stack(
            [
                rng.uniform(0, 100, 500),
                rng.uniform(-100, 100, 500),
                rng.uniform(-100, 100, 500),
            ]
        )

        descriptor = lab_histogram(lab)

        assert float(np.sum(descriptor**2)) == pytest.approx(1.0, abs=1e-5)
        assert float(np.linalg.norm(descriptor)) == pytest.approx(1.0, abs=1e-5)

    def test_the_descriptor_is_64_numbers(self):
        assert DESCRIPTOR_SIZE == 64
        assert BINS_PER_AXIS**3 == 64
        assert lab_histogram(np.zeros((10, 3))).shape == (64,)

    def test_a_uniform_image_puts_everything_in_one_bin(self):
        lab = np.tile([50.0, 0.0, 0.0], (100, 1))

        descriptor = lab_histogram(lab)

        assert np.count_nonzero(descriptor) == 1
        assert descriptor.max() == pytest.approx(1.0)

    def test_image_size_does_not_change_the_descriptor(self):
        # L1 normalisation means the descriptor is proportions of colour, so a
        # 5504x5504 scan is comparable with a 1024x768 photograph.
        small = np.tile([[20.0, 10.0, 10.0], [80.0, -10.0, -10.0]], (50, 1))
        large = np.tile([[20.0, 10.0, 10.0], [80.0, -10.0, -10.0]], (5000, 1))

        np.testing.assert_allclose(lab_histogram(small), lab_histogram(large), atol=1e-6)

    def test_out_of_range_values_are_clipped_not_wrapped(self):
        # A value past the a*/b* range must land in the edge bin, not index
        # out of bounds or wrap around to the opposite colour.
        lab = np.array([[50.0, 500.0, -500.0]])

        descriptor = lab_histogram(lab)

        assert np.count_nonzero(descriptor) == 1
        assert float(np.sum(descriptor**2)) == pytest.approx(1.0)

    def test_an_empty_input_returns_zeros_rather_than_raising(self):
        descriptor = lab_histogram(np.zeros((0, 3)))

        assert descriptor.shape == (64,)
        assert not descriptor.any()

    def test_the_result_is_float32(self):
        # The feature blocks are stored as float32; returning float64 here would
        # silently double the size of colour64.npy.
        assert lab_histogram(np.zeros((5, 3))).dtype == np.float32


class TestColourDescriptor:
    def test_a_solid_red_image_concentrates_in_few_bins(self):
        image = Image.new("RGB", (64, 64), (220, 30, 30))

        descriptor = colour_descriptor(image)

        assert np.count_nonzero(descriptor) == 1
        assert float(np.linalg.norm(descriptor)) == pytest.approx(1.0, abs=1e-5)

    def test_different_colours_give_different_descriptors(self):
        red = colour_descriptor(Image.new("RGB", (32, 32), (200, 20, 20)))
        blue = colour_descriptor(Image.new("RGB", (32, 32), (20, 20, 200)))

        assert float(red @ blue) == pytest.approx(0.0)

    def test_similar_colours_give_similar_descriptors(self):
        # Two nearby reds should land in the same coarse bin, which is the
        # point of using only four bins per axis.
        one = colour_descriptor(Image.new("RGB", (32, 32), (200, 20, 20)))
        two = colour_descriptor(Image.new("RGB", (32, 32), (210, 30, 25)))

        assert float(one @ two) == pytest.approx(1.0, abs=1e-5)

    def test_a_non_rgb_image_is_converted_rather_than_misread(self):
        # A palette image's array holds indices. Read as colours it would
        # produce a descriptor describing the palette order, which is nonsense.
        palette = Image.new("RGB", (32, 32), (100, 150, 200)).convert("P")

        descriptor = colour_descriptor(palette)

        assert float(np.linalg.norm(descriptor)) == pytest.approx(1.0, abs=1e-5)


class TestDominantColours:
    def test_returns_hex_strings(self):
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")

        colours = dominant_colours(image)

        assert len(colours) >= 1
        for colour in colours:
            assert colour.startswith("#")
            assert len(colour) == 7
            int(colour[1:], 16)  # must parse as hex

    def test_a_solid_image_reports_that_colour(self):
        image = Image.new("RGB", (64, 64), (18, 52, 86))

        colours = dominant_colours(image)

        assert colours[0] == "#123456"

    def test_a_two_colour_image_reports_the_larger_area_first(self):
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :] = (200, 30, 30)
        pixels[:16, :] = (30, 30, 200)  # a quarter of the image
        image = Image.fromarray(pixels, mode="RGB")

        colours = dominant_colours(image, count=2)

        red_first = int(colours[0][1:3], 16) > int(colours[0][5:7], 16)
        assert red_first, f"expected the dominant red first, got {colours}"

    def test_the_same_image_gives_the_same_answer(self):
        # k-means is seeded, so the Profile page does not reshuffle its swatches
        # every time it is opened.
        rng = np.random.default_rng(3)
        image = Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8), "RGB")

        assert dominant_colours(image) == dominant_colours(image)

    def test_asking_for_more_colours_than_exist_does_not_raise(self):
        image = Image.new("RGB", (32, 32), (10, 10, 10))

        colours = dominant_colours(image, count=5)

        assert len(colours) >= 1
