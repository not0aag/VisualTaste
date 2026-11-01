"""The per-fit score caches in R2 and R3, and the cached Beta interval width.

Each cache exists only for speed, so the contract is that it is invisible:
every score equals what the uncached computation gives, bit for bit; a refit
or a change of weights recomputes; and no caller can reach into the cache
and change the next batch's scores.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import beta

from tests.test_logreg import separable_labels, toy_bundle
from tests.test_r1_r2 import bundle, labels_for, unit
from visualtaste.rec.r2_knn import KnnRecommender
from visualtaste.rec.r3_logreg import LogisticRecommender
from visualtaste.rec.rerank import CREDIBILITY, beta_interval_width


def knn_setup(liked: int = 20, disliked: int = 10, pool: int = 300, dim: int = 16):
    rng = np.random.default_rng(1729)
    features = bundle(unit(rng.normal(size=(pool, dim))))
    ids = features.image_ids
    actions = ["like"] * liked + ["dislike"] * disliked
    labels = labels_for(list(zip(ids[: liked + disliked], actions, strict=True)))
    return features, labels


def uncached_knn(model: KnnRecommender, features) -> np.ndarray:
    """The pre-cache scoring path, written out."""
    pool = np.ascontiguousarray(features.siglip, dtype=np.float32)
    positive = model._top_k_similarity(model._liked_index, pool)
    if model._disliked_index is None:
        return positive
    return positive - model.gamma * model._top_k_similarity(model._disliked_index, pool)


class TestKnnCache:
    def test_cached_scores_are_bit_identical_to_uncached(self):
        features, labels = knn_setup()
        model = KnnRecommender().fit(labels, features)
        everything = np.arange(len(features))

        first = model.score(features, everything)
        second = model.score(features, everything)  # served from the cache

        np.testing.assert_array_equal(first, uncached_knn(model, features))
        np.testing.assert_array_equal(second, first)

    def test_the_pool_is_searched_once_per_fit(self, monkeypatch):
        features, labels = knn_setup()
        model = KnnRecommender().fit(labels, features)
        calls = []
        original = model._top_k_similarity
        monkeypatch.setattr(model, "_top_k_similarity", lambda *a: calls.append(1) or original(*a))

        for _ in range(4):  # four batches between refits
            model.score(features, np.arange(5))

        assert len(calls) == 2  # one search for likes, one for dislikes

    def test_a_refit_recomputes(self):
        features, labels = knn_setup()
        model = KnnRecommender().fit(labels, features)
        before = model.score(features, np.arange(len(features)))

        _, more_labels = knn_setup(liked=40, disliked=20)
        model.fit(more_labels, features)
        after = model.score(features, np.arange(len(features)))

        np.testing.assert_array_equal(after, uncached_knn(model, features))
        assert not np.array_equal(before, after)

    def test_changing_gamma_by_hand_recomputes(self):
        features, labels = knn_setup()
        model = KnnRecommender().fit(labels, features)
        model.score(features, np.arange(len(features)))

        model.gamma = 0.9
        np.testing.assert_array_equal(
            model.score(features, np.arange(len(features))), uncached_knn(model, features)
        )

    def test_another_pool_is_not_served_the_old_scores(self):
        features, labels = knn_setup()
        model = KnnRecommender().fit(labels, features)
        model.score(features, np.arange(len(features)))

        rng = np.random.default_rng(7)
        other = bundle(unit(rng.normal(size=(len(features), 16))), features.image_ids)
        np.testing.assert_array_equal(
            model.score(other, np.arange(len(other))), uncached_knn(model, other)
        )

    def test_callers_cannot_corrupt_the_cache(self):
        features, labels = knn_setup()
        model = KnnRecommender().fit(labels, features)
        expected = model.score(features, np.arange(len(features))).copy()

        sliced = model.score(features, slice(0, 10))  # a slice would be a view
        sliced[:] = 99.0
        picked = model.score(features, np.arange(10))
        picked[:] = 99.0

        np.testing.assert_array_equal(model.score(features, np.arange(len(features))), expected)


def uncached_logreg(model: LogisticRecommender, features) -> np.ndarray:
    return expit(model.logits(features)).astype(np.float32)


class TestLogisticCache:
    def test_cached_scores_are_bit_identical_to_uncached(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        everything = np.arange(len(features))

        first = model.score(features, everything)
        second = model.score(features, everything)

        np.testing.assert_array_equal(first, uncached_logreg(model, features))
        np.testing.assert_array_equal(second, first)

    def test_new_weights_recompute(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        before = model.score(features, np.arange(len(features)))

        model.w = model.w * 2.0  # replaced, as a refit or a load does
        after = model.score(features, np.arange(len(features)))

        np.testing.assert_array_equal(after, uncached_logreg(model, features))
        assert not np.array_equal(before, after)

    def test_weights_edited_in_place_also_recompute(self):
        # Compared by value, not identity, so even an in-place edit is seen.
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        model.score(features, np.arange(len(features)))

        model.w[0] += 1.0
        np.testing.assert_array_equal(
            model.score(features, np.arange(len(features))), uncached_logreg(model, features)
        )

    def test_a_new_bias_recomputes(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        model.score(features, np.arange(len(features)))

        model.b += 0.5
        np.testing.assert_array_equal(
            model.score(features, np.arange(len(features))), uncached_logreg(model, features)
        )

    def test_a_refit_recomputes(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features, n=60), features)
        model.score(features, np.arange(len(features)))

        model.fit(separable_labels(features, n=120), features)
        np.testing.assert_array_equal(
            model.score(features, np.arange(len(features))), uncached_logreg(model, features)
        )

    def test_callers_cannot_corrupt_the_cache(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        expected = model.score(features, np.arange(len(features))).copy()

        model.score(features, slice(0, 10))[:] = 99.0
        model.score(features, np.arange(10))[:] = 99.0

        np.testing.assert_array_equal(model.score(features, np.arange(len(features))), expected)


class TestBetaWidthCache:
    def test_cached_widths_equal_scipy_exactly(self):
        tail = (1.0 - CREDIBILITY) / 2.0
        for likes in range(4):
            for dislikes in range(4):
                a, b = 1 + likes, 1 + dislikes
                direct = float(beta.ppf(1 - tail, a, b) - beta.ppf(tail, a, b))
                assert beta_interval_width(likes, dislikes) == direct
                assert beta_interval_width(likes, dislikes) == direct  # cached call

    def test_numpy_integers_hit_the_same_value(self):
        assert beta_interval_width(np.int64(1), np.int64(2)) == beta_interval_width(1, 2)
