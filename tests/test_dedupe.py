"""Tests for near-duplicate detection, by pHash and by embedding."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tests import fixtures
from visualtaste.data.dedupe import (
    EMBEDDING_COSINE_THRESHOLD,
    PHASH_HAMMING_THRESHOLD,
    HashedImage,
    find_embedding_duplicate_groups,
    find_near_duplicate_groups,
    hamming_distance,
    phash_for,
)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return fixtures.build_all(tmp_path_factory.mktemp("fixtures"))


def thumb(path, size=(256, 256)):
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


class TestHammingDistance:
    def test_identical_hashes_are_zero_apart(self):
        assert hamming_distance("a1b2c3d4e5f60718", "a1b2c3d4e5f60718") == 0

    def test_counts_differing_bits_not_characters(self):
        # 0x0 vs 0x1 is one bit; 0x0 vs 0xf is four, though both are one
        # character apart.
        assert hamming_distance("0000000000000000", "0000000000000001") == 1
        assert hamming_distance("0000000000000000", "000000000000000f") == 4

    def test_a_fully_inverted_hash_is_64_apart(self):
        assert hamming_distance("0" * 16, "f" * 16) == 64


class TestPhashOnRealImages:
    def test_a_re_encoded_copy_hashes_close_to_the_original(self):
        # The whole point of a perceptual hash: quality 35 JPEG of the same
        # photograph has entirely different bytes but nearly the same hash.
        import io

        original = Image.new("RGB", (512, 512))
        pixels = np.asarray(original).copy()
        pixels[:256, :] = (200, 120, 40)
        pixels[256:, :] = (30, 70, 160)
        structured = Image.fromarray(pixels, "RGB")

        buffer = io.BytesIO()
        structured.save(buffer, format="JPEG", quality=30)
        buffer.seek(0)
        with Image.open(buffer) as reloaded:
            reloaded.load()
            degraded = reloaded.convert("RGB")

        distance = hamming_distance(phash_for(structured), phash_for(degraded))

        assert distance <= PHASH_HAMMING_THRESHOLD

    def test_the_fixture_near_duplicate_is_detected(self, images):
        original = thumb(images["valid_jpeg"])
        duplicate = thumb(images["near_duplicate"])

        distance = hamming_distance(phash_for(original), phash_for(duplicate))

        assert distance <= PHASH_HAMMING_THRESHOLD

    def test_different_images_hash_far_apart(self, images):
        one = phash_for(thumb(images["valid_jpeg"]))
        other = phash_for(thumb(images["second_valid"]))

        assert hamming_distance(one, other) > PHASH_HAMMING_THRESHOLD

    def test_the_hash_is_16_hex_characters(self, images):
        value = phash_for(thumb(images["valid_jpeg"]))

        assert len(value) == 16
        int(value, 16)


class TestGrouping:
    def test_the_highest_resolution_copy_is_kept(self):
        # Brief §7.2 step 5. The big copy makes the better derivative and its
        # embedding is computed from more real detail.
        items = [
            HashedImage("small", "0000000000000000", 400, 300),
            HashedImage("huge", "0000000000000001", 4000, 3000),
            HashedImage("medium", "0000000000000003", 1200, 900),
        ]

        groups = find_near_duplicate_groups(items)

        assert len(groups) == 1
        assert groups[0].kept == "huge"
        assert set(groups[0].duplicates) == {"small", "medium"}

    def test_images_further_apart_than_the_threshold_are_left_alone(self):
        items = [
            HashedImage("a", "0000000000000000", 800, 600),
            HashedImage("b", "00000000000000ff", 800, 600),  # 8 bits away
        ]

        assert find_near_duplicate_groups(items) == []

    def test_a_pair_exactly_at_the_threshold_is_grouped(self):
        # 6 bits apart, which the brief's <= 6 rule includes.
        items = [
            HashedImage("a", "0000000000000000", 800, 600),
            HashedImage("b", "000000000000003f", 400, 300),
        ]

        groups = find_near_duplicate_groups(items)

        assert len(groups) == 1
        assert groups[0].kept == "a"

    def test_an_image_belongs_to_at_most_one_group(self):
        items = [
            HashedImage("a", "0000000000000000", 4000, 3000),
            HashedImage("b", "0000000000000001", 800, 600),
            HashedImage("c", "0000000000000003", 400, 300),
        ]

        groups = find_near_duplicate_groups(items)

        claimed = [d for g in groups for d in g.duplicates] + [g.kept for g in groups]
        assert len(claimed) == len(set(claimed)), "an image was claimed twice"

    def test_no_duplicates_means_no_groups(self):
        items = [
            HashedImage("a", "0000000000000000", 800, 600),
            HashedImage("b", "ffffffffffffffff", 800, 600),
            HashedImage("c", "aaaaaaaaaaaaaaaa", 800, 600),
        ]

        assert find_near_duplicate_groups(items) == []

    def test_an_empty_input_is_handled(self):
        assert find_near_duplicate_groups([]) == []

    def test_the_result_is_deterministic(self):
        # Ties are broken by image_id, so two runs over the same corpus fold the
        # same copies away.
        items = [
            HashedImage("z", "0000000000000000", 800, 600),
            HashedImage("a", "0000000000000001", 800, 600),
        ]

        first = find_near_duplicate_groups(items)
        second = find_near_duplicate_groups(list(reversed(items)))

        assert first[0].kept == second[0].kept == "a"


class TestEmbeddingDuplicates:
    def _unit(self, vectors):
        array = np.asarray(vectors, dtype=np.float32)
        return array / np.linalg.norm(array, axis=1, keepdims=True)

    def test_near_identical_vectors_are_grouped(self):
        vectors = self._unit([[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0]])
        pixels = {"a": 4000 * 3000, "b": 800 * 600, "c": 800 * 600}

        groups = find_embedding_duplicate_groups(["a", "b", "c"], vectors, pixels=pixels)

        assert len(groups) == 1
        assert groups[0].kept == "a"
        assert groups[0].duplicates == ["b"]

    def test_distinct_vectors_are_not_grouped(self):
        vectors = self._unit([[1.0, 0.0], [0.0, 1.0]])

        assert find_embedding_duplicate_groups(["a", "b"], vectors) == []

    def test_the_largest_image_is_kept(self):
        vectors = self._unit([[1.0, 0.0], [0.999, 0.001]])
        pixels = {"small": 100, "big": 10_000}

        groups = find_embedding_duplicate_groups(["small", "big"], vectors, pixels=pixels)

        assert groups[0].kept == "big"
        assert groups[0].duplicates == ["small"]

    def test_the_threshold_is_respected(self):
        # Cosine 0.9 is similar but below the 0.95 rule, so it must survive:
        # different photographs of the same subject are not duplicates.
        angle = np.arccos(0.90)
        vectors = self._unit([[1.0, 0.0], [np.cos(angle), np.sin(angle)]])

        assert find_embedding_duplicate_groups(["a", "b"], vectors) == []
        assert len(find_embedding_duplicate_groups(["a", "b"], vectors, threshold=0.85)) == 1

    def test_mismatched_lengths_fail_loudly(self):
        # Silently misaligned ids and vectors would fold away the wrong images,
        # which is unrecoverable once statuses are written.
        with pytest.raises(ValueError, match="must align"):
            find_embedding_duplicate_groups(["a", "b", "c"], np.zeros((2, 4)))

    def test_blocking_does_not_change_the_result(self):
        # The implementation processes 512 rows at a time to keep memory flat;
        # a corpus larger than one block must give the same answer.
        rng = np.random.default_rng(1729)
        base = self._unit(rng.normal(size=(600, 8)))
        # Make rows 0 and 599 near-identical, across a block boundary.
        base[599] = base[0] + 0.001
        base = self._unit(base)
        ids = [f"i{n:04d}" for n in range(600)]

        groups = find_embedding_duplicate_groups(ids, base)

        folded = {d for g in groups for d in g.duplicates}
        assert "i0599" in folded or "i0000" in folded

    def test_the_default_threshold_matches_the_brief(self):
        assert EMBEDDING_COSINE_THRESHOLD == 0.95
        assert PHASH_HAMMING_THRESHOLD == 6
