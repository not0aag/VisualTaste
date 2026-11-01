"""R5 interests, R3+R5's time-respecting features, and the leakage guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_logreg import labels_from
from visualtaste.rec.base import FeatureBundle
from visualtaste.rec.features import time_respecting_features
from visualtaste.rec.r3_logreg import LogisticRecommender, scale_interest_features
from visualtaste.rec.r5_interests import (
    InterestRecommender,
    InterestTracker,
    interest_count,
    interest_features,
    lloyd,
    logsumexp_score,
    match_ids,
)


class TestM8:
    def test_logsumexp_with_a_small_temperature_is_the_max(self):
        assert logsumexp_score(np.array([0.40, 0.10, 0.05]), 0.05) == pytest.approx(0.400, abs=1e-3)

    def test_logsumexp_with_temperature_one(self):
        assert logsumexp_score(np.array([0.40, 0.10, 0.05]), 1.0) == pytest.approx(1.294, abs=1e-3)

    def test_it_never_overflows(self):
        assert np.isfinite(logsumexp_score(np.array([0.99, 0.98]), 1e-4))

    def test_one_dimensional_kmeans(self):
        centroids = lloyd(np.array([1.0, 2.0, 9.0, 10.0]), np.array([1.0, 2.0]))
        assert sorted(centroids.ravel().tolist()) == pytest.approx([1.5, 9.5])

    def test_one_interest_per_25_likes_up_to_8(self):
        assert [interest_count(n) for n in (24, 50, 75, 199, 500)] == [0, 2, 3, 7, 8]


def unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def three_tastes(n_per: int = 40, dim: int = 12, seed: int = 0):
    """Liked vectors around three well-separated directions."""
    rng = np.random.default_rng(seed)
    centres = unit(rng.normal(size=(3, dim)))
    points = np.vstack([unit(c + 0.15 * rng.normal(size=(n_per, dim))) for c in centres])
    return centres, points


class TestStableIds:
    def test_shuffled_clusters_keep_their_ids(self):
        rng = np.random.default_rng(1)
        old = unit(rng.normal(size=(4, 8)))
        order = rng.permutation(4)
        new = old[order] + 0.01 * rng.normal(size=(4, 8))

        ids, born, retired, _ = match_ids(old, [10, 11, 12, 13], unit(new), next_id=14)

        assert ids == [[10, 11, 12, 13][i] for i in order]
        assert born == [] and retired == []

    def test_a_new_cluster_is_born_and_a_vanished_one_retired(self):
        rng = np.random.default_rng(2)
        old = unit(rng.normal(size=(2, 8)))
        stranger = -old[1]  # nothing like either old interest
        ids, born, retired, next_id = match_ids(old, [0, 1], np.vstack([old[0], stranger]), 2)
        assert ids == [0, 2] and born == [2] and retired == [1] and next_id == 3

    def test_the_tracker_finds_the_tastes_and_keeps_ids_across_a_recluster(self):
        centres, points = three_tastes(n_per=40)
        tracker = InterestTracker(seed=0)
        order = np.random.default_rng(3).permutation(len(points))
        tracker.update(points[order[:75]])  # 75 likes -> k = 3
        first_ids = list(tracker.state.ids)
        first = tracker.state.centroids.copy()

        tracker.update(points[order])  # 120 likes: nudges, then later reclusters
        tracker.update(np.vstack([points[order], points[order[:60]]]))  # 180: recluster due

        assert tracker.state.k >= 3
        # Every original interest survives under its own id, still near its centre.
        for interest_id, centroid in zip(first_ids, first, strict=True):
            index = tracker.state.ids.index(interest_id)
            assert float(tracker.state.centroids[index] @ centroid) > 0.9
        assert max(float((tracker.state.centroids @ c).max()) for c in centres) > 0.9

    def test_features_are_padded_to_eight(self):
        _centres, points = three_tastes()
        tracker = InterestTracker(seed=0)
        tracker.update(points[:75])
        block = interest_features(tracker.state, points[:4])
        assert block.shape == (4, 10)
        assert np.all(block[:, 1 + tracker.state.k : -1] == 0)
        assert np.all(block[:, -1] == tracker.state.k)


class TestNoLeakage:
    """Part 2 §4.6: a future event cannot influence an earlier example's features."""

    def test_the_generic_builder(self):
        history = list(np.arange(100, dtype=float))

        def state_for_prefix(b):
            return sum(history[:b])

        def featurise(state, indices):
            return np.full((len(indices), 1), state)

        before = time_respecting_features(100, state_for_prefix, featurise)
        history[60] = 1e9  # plant a "future" event at position 60
        after = time_respecting_features(100, state_for_prefix, featurise)

        assert np.array_equal(before[:61], after[:61])  # nothing at or before 60 moved
        assert not np.array_equal(before[61:], after[61:])  # later rows do see it

    def test_r3_plus_r5_features_for_early_examples_ignore_later_likes(self):
        features, labels = interest_corpus()
        model = r35()
        model.fit(labels, features)
        early = model._train_X[:50, -10:].copy()

        # Rewrite the *future*: every label after position 75 becomes a like of
        # a single, new kind of image.
        flipped = labels_from(labels.image_ids, labels.actions[:75] + ["like"] * (len(labels) - 75))
        model.fit(flipped, features)

        np.testing.assert_array_equal(model._train_X[:50, -10:], early)

    def test_the_first_examples_have_no_interests_at_all(self):
        features, labels = interest_corpus()
        model = r35()
        model.fit(labels, features)
        assert np.all(model._train_X[:25, -10:-1] == 0)  # nothing liked before them
        assert np.all(model._train_X[:25, -1] == 0)


def interest_corpus(n: int = 600, seed: int = 4):
    rng = np.random.default_rng(seed)
    _centres, liked_points = three_tastes(n_per=60, dim=24, seed=seed)
    others = unit(rng.normal(size=(n - len(liked_points), 24)))
    vectors = np.vstack([liked_points, others]).astype(np.float32)
    features = FeatureBundle(image_ids=[f"{i:016x}" for i in range(n)], siglip=vectors)
    order = rng.permutation(n)[:300]
    actions = ["like" if i < len(liked_points) else "dislike" for i in order]
    return features, labels_from([features.image_ids[i] for i in order], actions)


def r35() -> LogisticRecommender:
    model = LogisticRecommender(blocks=("sig", "int"), lam=0.1)
    model.rung = "r3+r5"
    return model


class TestInterestRungs:
    def test_r5_scores_by_the_nearest_interest(self):
        features, labels = interest_corpus()
        model = InterestRecommender(seed=0).fit(labels, features)
        scores = model.score(features, np.arange(len(features)))
        assert model.is_fitted and model.interests.k >= 2
        assert scores[:180].mean() > scores[180:].mean()
        assert model.explain(features, 0).reason.startswith("interest:")

    def test_r5_needs_two_interests(self):
        features, labels = interest_corpus()
        few = labels_from(labels.image_ids[:60], ["dislike"] * 30 + ["like"] * 30)
        model = InterestRecommender().fit(few, features)
        assert not model.is_fitted and "r5 needs 50" in model.unfit_reason

    def test_r3_plus_r5_fits_scores_and_reloads(self, tmp_path):
        features, labels = interest_corpus()
        model = r35().fit(labels, features)
        candidates = np.arange(len(features))
        scores = model.score(features, candidates)
        assert model._train_X.shape[1] == 24 + 10
        assert scores[:180].mean() > scores[180:].mean()

        model.save(tmp_path / "v")
        restored = LogisticRecommender(blocks=("sig", "int")).load(tmp_path / "v")
        np.testing.assert_allclose(restored.score(features, candidates), scores, atol=1e-6)

    def test_r4_plus_r5_samples_over_the_interest_columns_too(self):
        from visualtaste.rec.r4_thompson import ThompsonRecommender

        features, labels = interest_corpus()
        model = ThompsonRecommender(blocks=("sig", "int"), lam=0.1).fit(labels, features)
        assert model.L.shape == (34, 34)

    def test_the_scaled_columns(self):
        tracker = InterestTracker(seed=0)
        _, points = three_tastes()
        tracker.update(points[:75])
        scaled = scale_interest_features(tracker.state, points[:3])
        raw = interest_features(tracker.state, points[:3])
        assert scaled[0, 1] == pytest.approx((raw[0, 1] - 0.5) / 0.1, rel=1e-5)
        assert scaled[0, -1] == pytest.approx(tracker.state.k / 8)
