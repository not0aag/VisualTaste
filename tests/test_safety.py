"""Tests for the safety screen's arithmetic and calibration.

No model is loaded: the screen's logic is margin arithmetic over unit vectors,
and testing it with constructed vectors is both faster and far more precise
about what the thresholds actually do.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.data.safety import (
    CALIBRATION_FALSE_FLAG_RATE,
    NEUTRAL_PROMPTS,
    PROMPT_TEMPLATE,
    TOP_FRACTION_FLAGGED,
    UNSAFE_PROMPTS,
    SafetyScreen,
    calibrate,
    sample_reference_vectors,
    unsafe_margin,
)


def unit(rows) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


# Two prompt directions and two neutral ones, in a 4-D space, so that an
# image's similarity to each is easy to reason about.
UNSAFE = unit([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
NEUTRAL = unit([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


class TestMargin:
    def test_the_docstring_worked_example(self):
        # An image whose best unsafe cosine is 0.18 and best neutral is 0.11
        # has a margin of 0.07.
        image = np.array([[0.18, 0.05, 0.11, 0.02]], dtype=np.float32)

        margin = unsafe_margin(image, UNSAFE, NEUTRAL)

        assert margin[0] == pytest.approx(0.07, abs=1e-6)

    def test_the_second_worked_example_is_clearly_safe(self):
        # 0.05 unsafe against 0.20 neutral gives -0.15: safe, even though the
        # raw unsafe similarity is not zero. This is the whole reason a margin
        # is used rather than a raw similarity.
        image = np.array([[0.05, 0.01, 0.20, 0.03]], dtype=np.float32)

        assert unsafe_margin(image, UNSAFE, NEUTRAL)[0] == pytest.approx(-0.15, abs=1e-6)

    def test_the_margin_takes_the_best_of_each_set(self):
        # max over unsafe prompts, max over neutral prompts -- not a mean, so
        # one strongly matching unsafe prompt is enough.
        image = np.array([[0.02, 0.30, 0.05, 0.09]], dtype=np.float32)

        assert unsafe_margin(image, UNSAFE, NEUTRAL)[0] == pytest.approx(0.21, abs=1e-6)

    def test_a_uniform_offset_cancels_out(self):
        # The point of subtracting the neutral match: a black-and-white
        # photograph scores lower against *every* prompt, and that offset must
        # not look like safety (or danger).
        base = np.array([[0.10, 0.08, 0.06, 0.04]], dtype=np.float32)
        dimmed = base - 0.03

        assert unsafe_margin(base, UNSAFE, NEUTRAL)[0] == pytest.approx(
            unsafe_margin(dimmed, UNSAFE, NEUTRAL)[0], abs=1e-6
        )

    def test_handles_a_batch(self):
        images = np.array([[0.2, 0.0, 0.1, 0.0], [0.0, 0.0, 0.3, 0.0]], dtype=np.float32)

        margins = unsafe_margin(images, UNSAFE, NEUTRAL)

        assert margins.shape == (2,)
        assert margins[0] > 0 > margins[1]

    def test_a_single_flat_vector_is_accepted(self):
        margins = unsafe_margin(np.array([0.2, 0.0, 0.1, 0.0], np.float32), UNSAFE, NEUTRAL)

        assert margins.shape == (1,)

    def test_an_empty_batch_returns_an_empty_array(self):
        assert unsafe_margin(np.zeros((0, 4), np.float32), UNSAFE, NEUTRAL).shape == (0,)


class TestCalibration:
    def _reference(self, n=200, *, seed=0):
        """`n` ordinary images: clearly closer to neutral than to unsafe."""
        rng = np.random.default_rng(seed)
        rows = np.zeros((n, 4), dtype=np.float32)
        rows[:, 2] = rng.uniform(0.15, 0.25, n)  # neutral-leaning
        rows[:, 0] = rng.uniform(0.00, 0.08, n)  # a little unsafe similarity
        return rows

    def test_the_threshold_lands_above_almost_every_safe_image(self):
        screen = calibrate(UNSAFE, NEUTRAL, self._reference())

        margins = unsafe_margin(self._reference(), UNSAFE, NEUTRAL)
        flagged = (margins > screen.threshold).sum()

        assert flagged <= max(1, int(len(margins) * CALIBRATION_FALSE_FLAG_RATE))

    def test_it_records_what_it_was_calibrated_on(self):
        screen = calibrate(UNSAFE, NEUTRAL, self._reference(150))

        assert screen.calibration_size == 150
        assert screen.threshold < 0, "ordinary images should have negative margins"

    def test_an_obviously_unsafe_image_is_flagged(self):
        screen = calibrate(UNSAFE, NEUTRAL, self._reference())
        unsafe_image = np.array([[0.40, 0.05, 0.05, 0.02]], dtype=np.float32)

        flagged, margins = screen.flag(unsafe_image)

        assert flagged[0]
        assert margins[0] > screen.threshold

    def test_calibrating_on_nothing_fails_loudly(self):
        with pytest.raises(ValueError, match="no reference images"):
            calibrate(UNSAFE, NEUTRAL, np.zeros((0, 4), np.float32))

    def test_a_tiny_reference_set_still_produces_a_threshold(self):
        screen = calibrate(UNSAFE, NEUTRAL, self._reference(5))

        assert np.isfinite(screen.threshold)


class TestFlagging:
    def _screen(self, threshold=0.0):
        return SafetyScreen(
            unsafe_vectors=UNSAFE,
            neutral_vectors=NEUTRAL,
            threshold=threshold,
            calibration_size=200,
            calibration_flagged=1,
        )

    def test_the_top_fraction_is_always_flagged(self):
        # Brief §7.2 step 7 flags the top 1% regardless of the threshold, so the
        # screen always surfaces something for review even if the threshold is
        # too permissive for this corpus.
        rng = np.random.default_rng(1729)
        images = np.zeros((1000, 4), dtype=np.float32)
        images[:, 2] = 0.3  # everything strongly neutral
        images[:, 0] = rng.uniform(0.0, 0.05, 1000)
        screen = self._screen(threshold=10.0)  # nothing can exceed this

        flagged, _ = screen.flag(images)

        expected = max(1, int(1000 * TOP_FRACTION_FLAGGED))
        assert flagged.sum() == expected

    def test_images_above_the_threshold_are_flagged_too(self):
        images = np.array(
            [
                [0.40, 0.0, 0.05, 0.0],  # margin +0.35
                [0.05, 0.0, 0.30, 0.0],  # margin -0.25
                [0.06, 0.0, 0.30, 0.0],  # margin -0.24
            ],
            dtype=np.float32,
        )
        screen = self._screen(threshold=0.0)

        flagged, margins = screen.flag(images)

        assert flagged[0]
        assert margins[0] == pytest.approx(0.35, abs=1e-6)

    def test_the_two_rules_are_unioned_not_intersected(self):
        images = np.array(
            [[0.40, 0.0, 0.05, 0.0]] + [[0.01, 0.0, 0.30, 0.0]] * 199,
            dtype=np.float32,
        )
        screen = self._screen(threshold=0.0)

        flagged, _ = screen.flag(images)

        # Flagged by the threshold and in the top 1%: counted once.
        assert flagged.sum() >= 1
        assert flagged[0]

    def test_an_empty_corpus_flags_nothing(self):
        flagged, margins = self._screen().flag(np.zeros((0, 4), np.float32))

        assert flagged.shape == (0,)
        assert margins.shape == (0,)

    def test_a_single_image_corpus_does_not_crash(self):
        flagged, _ = self._screen().flag(np.array([[0.1, 0.0, 0.2, 0.0]], np.float32))

        assert flagged.shape == (1,)


class TestReferenceSampling:
    def test_sampling_is_seeded_and_reproducible(self):
        vectors = np.arange(400 * 4, dtype=np.float32).reshape(400, 4)

        first = sample_reference_vectors(vectors, seed=1729, size=50)
        second = sample_reference_vectors(vectors, seed=1729, size=50)

        np.testing.assert_array_equal(first, second)

    def test_a_different_seed_samples_differently(self):
        vectors = np.arange(400 * 4, dtype=np.float32).reshape(400, 4)

        first = sample_reference_vectors(vectors, seed=1729, size=50)
        other = sample_reference_vectors(vectors, seed=7, size=50)

        assert not np.array_equal(first, other)

    def test_a_corpus_smaller_than_the_sample_is_used_whole(self):
        vectors = np.zeros((10, 4), dtype=np.float32)

        assert len(sample_reference_vectors(vectors, seed=1, size=200)) == 10

    def test_the_sample_is_the_requested_size(self):
        vectors = np.zeros((1000, 4), dtype=np.float32)

        assert len(sample_reference_vectors(vectors, seed=1, size=200)) == 200


class TestPromptSets:
    def test_the_six_unsafe_prompts_are_the_ones_the_brief_names(self):
        assert UNSAFE_PROMPTS == (
            "explicit nudity",
            "pornography",
            "graphic violence",
            "gore",
            "a corpse",
            "a severe medical injury",
        )

    def test_the_neutral_set_spans_the_corpus_rather_than_one_subject(self):
        # A neutral set that was all landscapes would make every portrait look
        # relatively unsafe. These cover what the four sources actually hold.
        assert len(NEUTRAL_PROMPTS) >= 12
        joined = " ".join(NEUTRAL_PROMPTS)
        for expected in ("portrait", "landscape", "animal", "painting", "artefact"):
            assert expected in joined

    def test_the_template_is_recorded_because_it_changes_every_score(self):
        assert "{}" in PROMPT_TEMPLATE
        assert PROMPT_TEMPLATE.format("gore") == "a photo of gore."
