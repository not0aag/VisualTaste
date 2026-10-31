"""Tests for clustering, diversity selection and PCA.

The property under test throughout: the selection must be varied *by
construction*, because the corpus is the space in which Alen's taste can be
expressed, and an unbalanced corpus produces a recommender that has learned
the corpus rather than the person.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.data.select import (
    CLUSTER_CAP_MULTIPLIER,
    CLUSTER_COUNT,
    COLDSTART_CLUSTER_COUNT,
    SELECTION_TARGET,
    fit_clusters,
    fit_pca,
    select_diverse,
)


def blobbed_vectors(
    cluster_sizes: list[int], *, dim: int = 16, spread: float = 0.05, seed: int = 1729
):
    """Vectors drawn from well-separated blobs, one per entry in `cluster_sizes`.

    Returns `(vectors, ids, true_cluster_of_id)`. Separated blobs mean a test
    can assert which cluster an image *should* land in.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(cluster_sizes), dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    vectors, ids, truth = [], [], {}
    counter = 0
    for blob, size in enumerate(cluster_sizes):
        for _ in range(size):
            point = centres[blob] + rng.normal(scale=spread, size=dim).astype(np.float32)
            vectors.append(point / np.linalg.norm(point))
            image_id = f"{counter:016x}"
            ids.append(image_id)
            truth[image_id] = blob
            counter += 1
    return np.asarray(vectors, dtype=np.float32), ids, truth


class TestFitClusters:
    def test_recovers_well_separated_blobs(self):
        vectors, ids, truth = blobbed_vectors([60, 60, 60])

        labels, centroids = fit_clusters(vectors, k=3, seed=1729)

        assert centroids.shape == (3, 16)
        # Each true blob should map onto a single predicted label.
        for blob in range(3):
            blob_labels = {labels[i] for i, image_id in enumerate(ids) if truth[image_id] == blob}
            assert len(blob_labels) == 1, f"blob {blob} was split across clusters"

    def test_the_same_seed_gives_the_same_partition(self):
        # cluster64 is stored per image and read by the cold-start rung and the
        # exploration quota, so a re-run that renumbered clusters would
        # silently invalidate the exploration statistics.
        vectors, _, _ = blobbed_vectors([40, 40, 40])

        first, _ = fit_clusters(vectors, k=3, seed=1729)
        second, _ = fit_clusters(vectors, k=3, seed=1729)

        np.testing.assert_array_equal(first, second)

    def test_every_vector_gets_a_label(self):
        vectors, _, _ = blobbed_vectors([30, 30])

        labels, _ = fit_clusters(vectors, k=2, seed=1729)

        assert len(labels) == len(vectors)
        assert labels.dtype == np.int32

    def test_k_is_capped_at_the_number_of_vectors(self):
        vectors, _, _ = blobbed_vectors([3])

        labels, centroids = fit_clusters(vectors, k=50, seed=1)

        assert len(centroids) <= 3
        assert labels.max() < len(centroids)

    def test_clustering_nothing_fails_loudly(self):
        with pytest.raises(ValueError, match="empty"):
            fit_clusters(np.zeros((0, 8), np.float32), k=4, seed=1)

    def test_a_corpus_larger_than_the_fit_sample_still_labels_everything(self):
        # Centroids are fitted on a sample for speed, but assignment is exact
        # for every image -- so no image may be left unlabelled.
        vectors, _, _ = blobbed_vectors([400, 400])

        labels, _ = fit_clusters(vectors, k=2, seed=1729, sample_for_fit=100)

        assert len(labels) == 800
        assert set(labels.tolist()) == {0, 1}


class TestSelectDiverse:
    def _sources(self, ids, source="pd12m"):
        return dict.fromkeys(ids, source)

    def test_a_dominant_cluster_is_capped(self):
        # The case this whole module exists for: one cluster holding most of the
        # corpus must not contribute most of the selection.
        _, ids, truth = blobbed_vectors([900, 50, 50])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=150, seed=1729)

        counts = result.per_cluster
        cap = int(CLUSTER_CAP_MULTIPLIER * 150 / 3)
        assert counts[0] <= cap + 150, "the huge cluster was not capped"
        assert result.capped_clusters >= 1

    def test_small_clusters_are_fully_represented(self):
        _, ids, truth = blobbed_vectors([900, 50, 50])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=150, seed=1729)

        assert set(result.per_cluster) == {0, 1, 2}

    def test_the_target_is_met_exactly_when_the_corpus_allows(self):
        _, ids, truth = blobbed_vectors([300, 300, 300])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=200, seed=1729)

        assert len(result.selected) == 200

    def test_a_shortfall_after_capping_is_filled_at_random(self):
        # With three clusters and a target of 150, the cap is 75 per cluster, so
        # capping alone cannot reach the target from two tiny clusters.
        _, ids, truth = blobbed_vectors([900, 10, 10])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=150, seed=1729)

        assert len(result.selected) == 150
        assert result.filled_at_random > 0

    def test_a_corpus_smaller_than_the_target_returns_everything(self):
        _, ids, truth = blobbed_vectors([20, 20])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=500, seed=1729)

        assert len(result.selected) == 40

    def test_no_image_is_selected_twice(self):
        _, ids, truth = blobbed_vectors([200, 200, 200])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=300, seed=1729)

        assert len(result.selected) == len(set(result.selected))

    def test_the_same_seed_selects_the_same_images(self):
        _, ids, truth = blobbed_vectors([200, 200])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        first = select_diverse(ids, labels, self._sources(ids), target=100, seed=1729)
        second = select_diverse(ids, labels, self._sources(ids), target=100, seed=1729)

        assert first.selected == second.selected

    def test_a_different_seed_selects_differently(self):
        _, ids, truth = blobbed_vectors([200, 200])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        first = select_diverse(ids, labels, self._sources(ids), target=100, seed=1729)
        other = select_diverse(ids, labels, self._sources(ids), target=100, seed=7)

        assert first.selected != other.selected

    def test_per_source_counts_are_reported(self):
        _, ids, truth = blobbed_vectors([100, 100])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)
        sources = {image_id: ("megalith" if i % 2 else "pd12m") for i, image_id in enumerate(ids)}

        result = select_diverse(ids, labels, sources, target=100, seed=1729)

        assert set(result.per_source) == {"megalith", "pd12m"}
        assert sum(result.per_source.values()) == 100

    def test_a_dominant_source_is_warned_about(self, caplog):
        # Brief §7.2 step 9: if one source exceeds 50% of the selection, say so
        # rather than letting it pass unremarked.
        _, ids, truth = blobbed_vectors([100, 100])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        with caplog.at_level("WARNING"):
            select_diverse(ids, labels, self._sources(ids), target=100, seed=1729)

        assert "above the 50% threshold" in caplog.text

    def test_misaligned_labels_fail_loudly(self):
        with pytest.raises(ValueError, match="ids but"):
            select_diverse(["a", "b", "c"], np.array([0, 1]), {}, target=2, seed=1)

    def test_the_result_is_sorted_so_downstream_order_is_stable(self):
        _, ids, truth = blobbed_vectors([50, 50])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=50, seed=1729)

        assert result.selected == sorted(result.selected)

    def test_every_candidate_gets_a_cluster_even_if_not_selected(self):
        # cluster400 is stored for every image, not only the selected ones, so
        # an image promoted into the pool later already has its cluster.
        _, ids, truth = blobbed_vectors([100, 100])
        labels = np.array([truth[image_id] for image_id in ids], dtype=np.int32)

        result = select_diverse(ids, labels, self._sources(ids), target=20, seed=1729)

        assert len(result.cluster_of) == 200


class TestFitPca:
    def test_reduces_to_the_requested_components(self):
        rng = np.random.default_rng(1729)
        vectors = rng.normal(size=(300, 64)).astype(np.float32)

        reduced, _, cumulative = fit_pca(vectors, components=16, seed=1729)

        assert reduced.shape == (300, 16)
        assert len(cumulative) == 16

    def test_cumulative_variance_rises_to_at_most_one(self):
        rng = np.random.default_rng(1729)
        vectors = rng.normal(size=(200, 32)).astype(np.float32)

        _, _, cumulative = fit_pca(vectors, components=8, seed=1729)

        assert np.all(np.diff(cumulative) >= -1e-6), "cumulative variance must not fall"
        assert 0 < cumulative[-1] <= 1.0 + 1e-6

    def test_structure_in_the_data_concentrates_the_variance(self):
        # Data that genuinely lives on a 3-dimensional subspace should have
        # almost all its variance in the first three components. This is the
        # property that makes reducing 768 -> 128 safe.
        rng = np.random.default_rng(1729)
        basis = rng.normal(size=(3, 40)).astype(np.float32)
        coefficients = rng.normal(size=(400, 3)).astype(np.float32)
        vectors = coefficients @ basis + rng.normal(scale=0.01, size=(400, 40)).astype(np.float32)

        _, _, cumulative = fit_pca(vectors, components=10, seed=1729)

        assert cumulative[2] > 0.95

    def test_the_output_is_float32(self):
        vectors = np.random.default_rng(0).normal(size=(50, 16)).astype(np.float32)

        reduced, _, _ = fit_pca(vectors, components=4, seed=1)

        assert reduced.dtype == np.float32

    def test_components_are_capped_by_the_data_shape(self):
        vectors = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)

        reduced, _, _ = fit_pca(vectors, components=128, seed=1)

        assert reduced.shape[1] <= 8

    def test_the_fitted_model_can_project_a_new_image(self):
        # Saved as a joblib so the API can project one newly embedded image at
        # request time without re-fitting anything.
        rng = np.random.default_rng(1729)
        vectors = rng.normal(size=(200, 32)).astype(np.float32)

        _, model, _ = fit_pca(vectors, components=8, seed=1729)
        projected = model.transform(rng.normal(size=(1, 32)).astype(np.float32))

        assert projected.shape == (1, 8)


class TestConstants:
    def test_they_match_the_brief(self):
        assert SELECTION_TARGET == 50_000
        assert CLUSTER_COUNT == 400
        assert COLDSTART_CLUSTER_COUNT == 64
        assert CLUSTER_CAP_MULTIPLIER == 1.5

    def test_the_cap_works_out_to_the_number_in_the_brief(self):
        # §7.2 step 9: "a cap of 1.5 x (50,000/400) per cluster" = 187.
        assert int(CLUSTER_CAP_MULTIPLIER * SELECTION_TARGET / CLUSTER_COUNT) == 187
