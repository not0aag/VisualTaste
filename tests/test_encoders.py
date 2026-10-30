"""Tests for the encoder helpers.

Nothing here loads a model: these tests must run in a second, offline, on any
machine. The models themselves are exercised by the embedding tests in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.vision.encoders import (
    DINOV2_KEY,
    ENCODER_SPECS,
    SIGLIP2_KEY,
    l2_normalise,
    load_encoder,
)


class TestL2Normalise:
    """Maths unit M1. The numbers are the brief's section 8 worked example."""

    def test_m1_worked_example_cosine_of_a_and_b(self):
        # A = (1,2,2) and B = (2,1,2) both have norm 3, and A.B = 2+2+4 = 8,
        # so cos(A,B) = 8/9 = 0.8889.
        A = np.array([1.0, 2.0, 2.0])
        B = np.array([2.0, 1.0, 2.0])

        A_hat = l2_normalise(A)
        B_hat = l2_normalise(B)

        np.testing.assert_allclose(A_hat, [1 / 3, 2 / 3, 2 / 3], rtol=1e-12)
        assert float(A_hat @ B_hat) == pytest.approx(8 / 9, abs=1e-10)
        assert float(A_hat @ B_hat) == pytest.approx(0.8889, abs=1e-4)

    def test_m1_worked_example_negative_cosine(self):
        # C = (-2,-1,0): cos(A,C) = (-2-2+0)/(3*sqrt(5)) = -4/6.708 = -0.596.
        A = l2_normalise(np.array([1.0, 2.0, 2.0]))
        C = l2_normalise(np.array([-2.0, -1.0, 0.0]))

        assert float(A @ C) == pytest.approx(-0.596, abs=1e-3)

    def test_m1_worked_example_squared_distance_identity(self):
        # For unit vectors, ||a-b||^2 = 2 - 2cos(a,b). With cos = 8/9 that is
        # 2 - 16/9 = 0.222, which is why a cosine index can answer nearest-
        # neighbour queries: the two orderings are the same.
        A = l2_normalise(np.array([1.0, 2.0, 2.0]))
        B = l2_normalise(np.array([2.0, 1.0, 2.0]))

        squared_distance = float(np.sum((A - B) ** 2))
        assert squared_distance == pytest.approx(2 - 2 * (8 / 9), abs=1e-10)
        assert squared_distance == pytest.approx(0.222, abs=1e-3)

    def test_rows_are_normalised_independently(self):
        x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, -2.0]])

        normalised = l2_normalise(x)

        np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), [1.0, 1.0, 1.0], rtol=1e-12)
        np.testing.assert_allclose(normalised[0], [0.6, 0.8], rtol=1e-12)

    def test_a_zero_vector_does_not_divide_by_zero(self):
        # A real encoder never emits one, but a fixture or a failed decode might,
        # and a NaN would silently poison every similarity downstream.
        result = l2_normalise(np.zeros(4))

        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, np.zeros(4))

    def test_normalising_twice_changes_nothing(self):
        x = np.array([[1.0, 2.0, 2.0], [5.0, 0.0, 0.0]])

        once = l2_normalise(x)
        twice = l2_normalise(once)

        np.testing.assert_allclose(once, twice, rtol=1e-12)


class TestEncoderSpecs:
    def test_both_models_are_registered_at_768_dimensions(self):
        # The neural ranker's first layer is sized 768 + 768 + 64 + 8 = 1608,
        # so a dimension change here would silently break R6.
        assert set(ENCODER_SPECS) == {SIGLIP2_KEY, DINOV2_KEY}
        assert all(spec.dim == 768 for spec in ENCODER_SPECS.values())

    def test_only_siglip2_claims_a_text_encoder(self):
        # Concepts, "taste in words" and the safety screen all need text.
        assert ENCODER_SPECS[SIGLIP2_KEY].has_text_encoder is True
        assert ENCODER_SPECS[DINOV2_KEY].has_text_encoder is False

    def test_checkpoints_are_the_tags_verified_against_the_live_registries(self):
        # Read from open_clip.list_pretrained() on 2026-10-30. Unsuffixed means
        # 224 px; the -256/-384/-512 variants are different checkpoints.
        assert ENCODER_SPECS[SIGLIP2_KEY].checkpoint == "ViT-B-16-SigLIP2/webli"
        assert ENCODER_SPECS[DINOV2_KEY].checkpoint == "facebook/dinov2-base"

    def test_specs_are_frozen_so_a_caller_cannot_mutate_the_registry(self):
        with pytest.raises(AttributeError):
            ENCODER_SPECS[SIGLIP2_KEY].dim = 512  # type: ignore[misc]

    def test_an_unknown_model_key_fails_loudly_and_lists_the_known_ones(self):
        with pytest.raises(KeyError) as excinfo:
            load_encoder("clip_vit_l14")

        message = str(excinfo.value)
        assert SIGLIP2_KEY in message
        assert DINOV2_KEY in message
