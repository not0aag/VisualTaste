"""Tests for R1 Rocchio and R2 kNN, anchored on the brief's M1 and M2 examples."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from visualtaste.rec.base import FeatureBundle, UnfittedModelError
from visualtaste.rec.labels import LabelSet
from visualtaste.rec.r1_rocchio import (
    ALPHA,
    BETA,
    RocchioRecommender,
    rocchio_vector,
)
from visualtaste.rec.r2_knn import GAMMA, NEIGHBOURS, KnnRecommender, top_k_mean


def unit(rows) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def bundle(vectors: np.ndarray, ids: list[str] | None = None) -> FeatureBundle:
    ids = ids or [f"{n:016x}" for n in range(len(vectors))]
    return FeatureBundle(image_ids=ids, siglip=np.asarray(vectors, dtype=np.float32))


def labels_for(pairs: list[tuple[str, str]]) -> LabelSet:
    """`[(image_id, action)]` -> a LabelSet with the standard weights."""
    from visualtaste.rec.labels import ACTION_TO_TARGET, DEFAULT_ACTION_WEIGHTS

    ids = [image_id for image_id, _ in pairs]
    actions = [action for _, action in pairs]
    return LabelSet(
        image_ids=ids,
        targets=np.array([ACTION_TO_TARGET[a] for a in actions], dtype=np.float32),
        weights=np.array([DEFAULT_ACTION_WEIGHTS[a] for a in actions], dtype=np.float32),
        actions=actions,
        timestamps=["2026-10-31T00:00:00"] * len(pairs),
    )


# ---------------------------------------------------------------------------
# M1 and M2: the brief's pre-computed examples
# ---------------------------------------------------------------------------


class TestM1Cosine:
    """Maths unit M1. The same numbers as `tests/test_encoders.py`, applied here
    to the scoring path rather than to the normaliser, because M1 is what makes
    "score = dot product" legitimate."""

    def test_cosine_of_the_worked_vectors(self):
        a = np.array([1.0, 2.0, 2.0])
        b = np.array([2.0, 1.0, 2.0])

        cosine = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

        assert cosine == pytest.approx(8 / 9, abs=1e-10)
        assert cosine == pytest.approx(0.8889, abs=1e-4)

    def test_the_negative_case(self):
        a = np.array([1.0, 2.0, 2.0])
        c = np.array([-2.0, -1.0, 0.0])

        cosine = (a @ c) / (np.linalg.norm(a) * np.linalg.norm(c))

        assert cosine == pytest.approx(-0.596, abs=1e-3)

    def test_unit_vectors_make_the_dot_product_the_cosine(self):
        # Why every rung can score with a matrix multiply and no division.
        a = unit([[1.0, 2.0, 2.0]])[0]
        b = unit([[2.0, 1.0, 2.0]])[0]

        assert float(a @ b) == pytest.approx(8 / 9, abs=1e-6)


class TestM2Rocchio:
    """Maths unit M2, exactly as the brief pre-computed it."""

    LIKED = np.array([[1.0, 0.0], [0.8, 0.6]])
    DISLIKED = np.array([[0.0, 1.0]])

    def test_the_preference_vector(self):
        vector = rocchio_vector(self.LIKED, self.DISLIKED)

        assert vector[0] == pytest.approx(0.976, abs=1e-3)
        assert vector[1] == pytest.approx(-0.217, abs=1e-3)

    def test_the_intermediate_arithmetic(self):
        # mean(liked) = (0.9, 0.3); u = (0.9, 0.3) - 0.5 (0, 1) = (0.9, -0.2);
        # ||u|| = sqrt(0.85) = 0.92195.
        liked_mean = self.LIKED.mean(axis=0)
        raw = ALPHA * liked_mean - BETA * self.DISLIKED.mean(axis=0)

        np.testing.assert_allclose(liked_mean, [0.9, 0.3])
        np.testing.assert_allclose(raw, [0.9, -0.2])
        assert np.linalg.norm(raw) == pytest.approx(0.92195, abs=1e-5)

    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [((0.6, 0.8), 0.412), ((1.0, 0.0), 0.976), ((0.0, 1.0), -0.217)],
    )
    def test_the_three_scores(self, candidate, expected):
        vector = rocchio_vector(self.LIKED, self.DISLIKED)

        assert float(np.array(candidate) @ vector) == pytest.approx(expected, abs=1e-3)

    def test_the_alpha_and_beta_match_the_brief(self):
        assert ALPHA == 1.0
        assert BETA == 0.5

    def test_beta_below_one_keeps_dislikes_from_steering(self):
        # At beta = 1 the dislike dominates and flips the vector's second
        # component much further; the brief's 0.5 lets it push without steering.
        half = rocchio_vector(self.LIKED, self.DISLIKED, beta=0.5)
        full = rocchio_vector(self.LIKED, self.DISLIKED, beta=1.0)

        assert full[1] < half[1] < 0


# ---------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------


class TestRocchioFormula:
    def test_a_save_counts_double_through_the_weights(self):
        # "A save is worth two likes" is implemented as a weighted mean, so the
        # vector should sit twice as close to the saved image.
        liked = np.array([[1.0, 0.0], [0.0, 1.0]])

        plain = rocchio_vector(liked, np.zeros((0, 2)))
        weighted = rocchio_vector(liked, np.zeros((0, 2)), weights=np.array([2.0, 1.0]))

        assert weighted[0] > plain[0]
        assert plain[0] == pytest.approx(plain[1])

    def test_no_dislikes_gives_the_plain_liked_mean(self):
        liked = np.array([[1.0, 0.0], [0.6, 0.8]])

        vector = rocchio_vector(liked, np.zeros((0, 2)))
        expected = liked.mean(axis=0)

        np.testing.assert_allclose(vector, expected / np.linalg.norm(expected), atol=1e-6)

    def test_the_result_is_unit_length(self):
        vector = rocchio_vector(np.array([[3.0, 4.0]]), np.zeros((0, 2)))

        assert np.linalg.norm(vector) == pytest.approx(1.0)

    def test_no_likes_fails_loudly(self):
        # A vector built from dislikes alone would score "anything unlike what
        # he rejected" highest, which is not a taste.
        with pytest.raises(ValueError, match="at least one liked"):
            rocchio_vector(np.zeros((0, 2)), np.array([[1.0, 0.0]]))

    def test_exactly_cancelling_labels_fail_loudly(self):
        # A zero vector would score everything 0 and look broken rather than
        # uncertain.
        with pytest.raises(ValueError, match="cancelled"):
            rocchio_vector(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0]]))


class TestRocchioRecommender:
    def _fitted(self, decay: float = 1.0):
        vectors = unit([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]])
        features = bundle(vectors)
        ids = features.image_ids
        labels = labels_for(
            [
                (ids[0], "like"),
                (ids[1], "like"),
                (ids[2], "dislike"),
                (ids[3], "like"),
                (ids[4], "dislike"),
            ]
        )
        model = RocchioRecommender(decay=decay).fit(labels, features)
        return model, features, ids

    def test_it_fits_and_scores(self):
        model, features, _ = self._fitted()

        scores = model.score(features, np.arange(len(features)))

        assert scores.shape == (5,)
        assert scores.dtype == np.float32

    def test_images_like_the_likes_outscore_images_like_the_dislikes(self):
        model, features, _ = self._fitted()

        scores = model.score(features, np.arange(len(features)))

        # Index 0 is a liked direction, index 4 is the opposite.
        assert scores[0] > scores[4]

    def test_it_refuses_to_score_before_being_fitted(self):
        model = RocchioRecommender()
        features = bundle(unit([[1.0, 0.0]]))

        with pytest.raises(UnfittedModelError):
            model.score(features, np.arange(1))

    def test_too_few_labels_leaves_it_unfitted(self):
        vectors = unit([[1.0, 0.0], [0.0, 1.0]])
        features = bundle(vectors)
        labels = labels_for([(features.image_ids[0], "like")])

        model = RocchioRecommender().fit(labels, features)

        assert not model.is_fitted

    def test_skips_do_not_pull_the_vector(self):
        # Rocchio subtracts a *mean*, which has no notion of weight, so skips
        # are excluded rather than down-weighted.
        vectors = unit([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.7, 0.7], [0.5, 0.5]])
        features = bundle(vectors)
        ids = features.image_ids

        with_skip = RocchioRecommender().fit(
            labels_for(
                [
                    (ids[0], "like"),
                    (ids[1], "like"),
                    (ids[2], "dislike"),
                    (ids[3], "skip"),
                    (ids[4], "skip"),
                ]
            ),
            features,
        )
        without_skip = RocchioRecommender().fit(
            labels_for(
                [
                    (ids[0], "like"),
                    (ids[1], "like"),
                    (ids[2], "dislike"),
                    (ids[3], "like"),
                    (ids[4], "like"),
                ]
            ),
            features,
        )

        assert not np.allclose(with_skip.preference, without_skip.preference)

    def test_recency_decay_moves_the_vector_towards_recent_likes(self):
        vectors = unit([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.6, 0.8]])
        features = bundle(vectors)
        ids = features.image_ids
        # Oldest like is (1,0); newest is (0,1).
        pairs = [
            (ids[0], "like"),
            (ids[2], "dislike"),
            (ids[3], "dislike"),
            (ids[4], "dislike"),
            (ids[1], "like"),
        ]

        flat = RocchioRecommender(decay=1.0).fit(labels_for(pairs), features)
        decayed = RocchioRecommender(decay=0.5).fit(labels_for(pairs), features)

        # With a steep decay, the newest like (0,1) should dominate.
        assert decayed.preference[1] > flat.preference[1]

    def test_the_decay_halves_at_the_documented_age(self):
        # 0.995 ** 139 = 0.50, which is the claim in the docstring.
        assert pytest.approx(0.5, abs=0.005) == 0.995**139

    def test_it_explains_itself_with_nearest_likes(self):
        model, features, ids = self._fitted()

        explanation = model.explain(features, 0)

        assert explanation.reason == "similar_likes"
        assert explanation.detail["neighbours"]
        assert len(explanation.detail["neighbours"]) <= 3
        assert all(n in ids for n in explanation.detail["neighbours"])

    def test_the_nearest_like_is_the_image_itself_when_it_was_liked(self):
        model, features, ids = self._fitted()

        explanation = model.explain(features, 0)

        assert explanation.detail["neighbours"][0] == ids[0]

    def test_it_round_trips_through_disk(self, tmp_path):
        model, features, _ = self._fitted()
        before = model.score(features, np.arange(len(features)))

        model.save(tmp_path / "v1")
        restored = RocchioRecommender().load(tmp_path / "v1")

        np.testing.assert_allclose(
            restored.score(features, np.arange(len(features))), before, atol=1e-6
        )

    def test_its_params_record_the_hyperparameters(self):
        params = RocchioRecommender(alpha=1.0, beta=0.3, decay=0.99).params

        assert params["beta"] == 0.3
        assert params["decay"] == 0.99
        assert params["rung"] == "r1"


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------


class TestTopKMean:
    def test_it_averages_only_the_largest_values(self):
        similarity = np.array([[0.1, 0.9, 0.5, 0.8, 0.2]])

        assert top_k_mean(similarity, 2)[0] == pytest.approx((0.9 + 0.8) / 2)

    def test_with_fewer_values_than_k_it_uses_them_all(self):
        # What makes the rung usable from its tenth label rather than its fiftieth.
        similarity = np.array([[0.4, 0.6]])

        assert top_k_mean(similarity, 5)[0] == pytest.approx(0.5)

    def test_an_empty_label_set_is_zero_rather_than_an_error(self):
        assert top_k_mean(np.zeros((3, 0)), 5).tolist() == [0.0, 0.0, 0.0]

    def test_it_works_row_wise(self):
        similarity = np.array([[1.0, 0.0], [0.5, 0.5]])

        np.testing.assert_allclose(top_k_mean(similarity, 1), [1.0, 0.5])


class TestKnnRecommender:
    #: Index 10 is exactly between the two interests; index 11 is opposite both.
    TWO_INTEREST_VECTORS: ClassVar[list[list[float]]] = [
        [1.0, 0.0],  # 0-4: interest A
        [0.98, 0.2],
        [0.96, 0.28],
        [0.99, 0.14],
        [0.97, 0.24],
        [0.0, 1.0],  # 5-9: interest B
        [0.2, 0.98],
        [0.28, 0.96],
        [0.14, 0.99],
        [0.24, 0.97],
        [0.7071, 0.7071],  # 10: the midpoint
        [-0.7071, -0.7071],  # 11: opposite both
    ]

    def _two_interests(self):
        """Ten likes in two separate directions, nothing liked between them.

        This is the case that separates R2 from R1: their mean points at the
        midpoint, which neither interest resembles. Deliberately *no* dislikes,
        so the comparison isolates how each rung aggregates likes.
        """
        features = bundle(unit(self.TWO_INTEREST_VECTORS))
        ids = features.image_ids
        labels = labels_for([(ids[i], "like") for i in range(10)])
        return features, ids, labels

    def _with_dislikes(self):
        """The same two interests, plus two dislikes at the opposite corner."""
        features = bundle(unit(self.TWO_INTEREST_VECTORS))
        ids = features.image_ids
        pairs = [(ids[i], "like") for i in range(10)]
        pairs.append((ids[11], "dislike"))
        return features, ids, labels_for(pairs)

    MIDPOINT = 10
    OPPOSITE = 11
    INTEREST_A = 0
    INTEREST_B = 5

    def test_it_fits_and_scores(self):
        features, _, labels = self._two_interests()

        model = KnnRecommender().fit(labels, features)
        scores = model.score(features, np.arange(len(features)))

        assert model.is_fitted
        assert scores.shape == (len(self.TWO_INTEREST_VECTORS),)

    def test_it_beats_rocchio_on_two_separate_interests(self):
        # The point of the rung. R1 compresses two tastes into their midpoint;
        # R2 keeps them apart, so the midpoint image should score *lower* than
        # the images at either interest -- and under R1 it scores higher.
        features, _, labels = self._two_interests()

        knn = KnnRecommender(neighbours=2).fit(labels, features)
        rocchio = RocchioRecommender().fit(labels, features)

        knn_scores = knn.score(features, np.arange(len(features)))
        rocchio_scores = rocchio.score(features, np.arange(len(features)))

        interests = min(knn_scores[self.INTEREST_A], knn_scores[self.INTEREST_B])
        assert knn_scores[self.MIDPOINT] < interests

        # Rocchio's vector lands almost exactly on the midpoint, which is
        # exactly its limitation: it prefers an image neither interest wants.
        assert rocchio_scores[self.MIDPOINT] > max(
            rocchio_scores[self.INTEREST_A], rocchio_scores[self.INTEREST_B]
        )

    def test_dislikes_lower_the_score_of_images_resembling_them(self):
        features, _, labels = self._with_dislikes()
        model = KnnRecommender().fit(labels, features)

        with_gamma = model.score(features, np.arange(len(features)))
        model.gamma = 0.0
        without_gamma = model.score(features, np.arange(len(features)))

        # The image sitting on the dislike is the one the term should punish.
        assert with_gamma[self.OPPOSITE] < without_gamma[self.OPPOSITE]

    def test_being_unlike_the_dislikes_raises_the_score(self):
        """A consequence of the formula worth knowing about.

        The dislike term is `- gamma * mean top-k cosine to dislikes`, and a
        cosine can be negative, so an image pointing *away* from the dislikes
        gets a positive contribution rather than zero. "Far from what he
        rejected" is therefore treated as mild evidence in favour.

        That is defensible in a space where opposite directions mean opposite
        content, and it is bounded by gamma, but it is not what the one-line
        description of kNN suggests -- so it is asserted here rather than left
        for a future reader to rediscover from a surprising ranking.
        """
        features, _, labels = self._with_dislikes()
        model = KnnRecommender(neighbours=2).fit(labels, features)

        with_gamma = model.score(features, np.arange(len(features)))
        model.gamma = 0.0
        without_gamma = model.score(features, np.arange(len(features)))

        # The midpoint points away from the dislike at (-0.707, -0.707).
        assert with_gamma[self.MIDPOINT] > without_gamma[self.MIDPOINT]

    def test_the_defaults_match_the_brief(self):
        assert NEIGHBOURS == 5
        assert GAMMA == 0.5

    def test_no_dislikes_still_scores(self):
        vectors = unit([[1.0, 0.0]] * 12)
        features = bundle(vectors)
        labels = labels_for([(i, "like") for i in features.image_ids])

        model = KnnRecommender().fit(labels, features)
        scores = model.score(features, np.arange(len(features)))

        assert np.all(scores > 0)

    def test_no_likes_fails_loudly(self):
        vectors = unit([[1.0, 0.0]] * 12)
        features = bundle(vectors)
        labels = labels_for([(i, "dislike") for i in features.image_ids])

        with pytest.raises(ValueError, match="at least one liked"):
            KnnRecommender().fit(labels, features)

    def test_its_explanation_names_the_terms_of_its_own_sum(self):
        # In R2 the top-k liked images literally *are* the score, so the
        # explanation is exact rather than approximate.
        features, ids, labels = self._two_interests()
        model = KnnRecommender(neighbours=2).fit(labels, features)

        explanation = model.explain(features, 0)

        assert explanation.reason == "similar_likes"
        assert explanation.detail["neighbours"][0] == ids[0]
        assert explanation.detail["liked_similarity"] > 0
        combined = (
            explanation.detail["liked_similarity"]
            - GAMMA * explanation.detail["disliked_similarity"]
        )
        assert explanation.detail["score"] == pytest.approx(combined, abs=1e-3)

    def test_it_round_trips_through_disk(self, tmp_path):
        features, _, labels = self._two_interests()
        model = KnnRecommender().fit(labels, features)
        before = model.score(features, np.arange(len(features)))

        model.save(tmp_path / "v1")
        restored = KnnRecommender().load(tmp_path / "v1")

        np.testing.assert_allclose(
            restored.score(features, np.arange(len(features))), before, atol=1e-6
        )

    def test_scoring_scales_with_labels_not_the_corpus(self):
        # Documented behaviour: the matrix is (candidates x labels), so the
        # label count is the axis that grows. Asserted so the shape contract is
        # explicit rather than assumed.
        features, _, labels = self._two_interests()
        model = KnnRecommender().fit(labels, features)

        assert model._liked is not None
        assert model._liked.shape[0] == len(labels.positives)


class TestCandidateSubsetting:
    """Both rungs score every row and index afterwards (see `r1_rocchio.score`).

    That optimisation is only valid if scoring a subset gives exactly what
    scoring everything and then subsetting gives. It would be easy to break by
    returning the full array, or by indexing with positions instead of offsets,
    and the symptom -- cards ranked by another image's score -- would be subtle
    enough to live for months.
    """

    def _fitted_models(self):
        vectors = unit(
            [[1.0, 0.0], [0.9, 0.3], [0.0, 1.0], [0.3, 0.9], [-1.0, 0.0], [0.5, 0.5]] * 3
        )
        features = bundle(vectors)
        ids = features.image_ids
        actions = ["like", "like", "like", "like", "dislike", "like"] * 3
        labels = labels_for(list(zip(ids, actions, strict=True)))
        return features, [
            RocchioRecommender().fit(labels, features),
            KnnRecommender().fit(labels, features),
        ]

    def test_a_subset_scores_exactly_as_the_full_pool_does(self):
        features, models = self._fitted_models()
        subset = np.array([1, 4, 7, 11], dtype=np.int64)

        for model in models:
            everything = model.score(features, np.arange(len(features)))
            just_those = model.score(features, subset)

            np.testing.assert_allclose(just_those, everything[subset], atol=1e-6)

    def test_the_returned_length_follows_the_candidates(self):
        features, models = self._fitted_models()
        subset = np.array([2, 3], dtype=np.int64)

        for model in models:
            assert model.score(features, subset).shape == (2,)

    def test_order_follows_the_candidate_array_not_the_pool(self):
        features, models = self._fitted_models()
        forwards = np.array([0, 4], dtype=np.int64)
        backwards = np.array([4, 0], dtype=np.int64)

        for model in models:
            first = model.score(features, forwards)
            second = model.score(features, backwards)

            np.testing.assert_allclose(first, second[::-1], atol=1e-6)

    def test_an_empty_candidate_array_scores_nothing(self):
        features, models = self._fitted_models()

        for model in models:
            assert model.score(features, np.array([], dtype=np.int64)).shape == (0,)


class TestKnnMatchesItsOwnFormula:
    """FAISS must compute exactly what the docstring says it computes.

    R2 scores through `IndexFlatIP.search`, which returns the k largest inner
    products directly. That is a different *implementation* of the top-k mean
    than `top_k_mean`, and the whole argument for using it is that it is the
    same *computation* -- 2.5x faster and equal to 4e-8 at the real pool size.
    If that ever stops being true, every result in `docs/EVALUATION.md` for this
    rung is wrong, so it is asserted directly against the reference formula.
    """

    def _setup(self, *, liked: int, disliked: int, dim: int = 16, pool: int = 300):
        rng = np.random.default_rng(1729)
        vectors = unit(rng.normal(size=(pool, dim)))
        features = bundle(vectors)
        ids = features.image_ids
        actions = ["like"] * liked + ["dislike"] * disliked
        labels = labels_for(list(zip(ids[: liked + disliked], actions, strict=True)))
        return features, KnnRecommender().fit(labels, features)

    def _reference(self, model, features) -> np.ndarray:
        """The formula written out with NumPy, as the docstring states it."""
        positive = top_k_mean(features.siglip @ model._liked.T, model.neighbours)
        if model._disliked is None or len(model._disliked) == 0:
            return positive
        negative = top_k_mean(features.siglip @ model._disliked.T, model.neighbours)
        return positive - model.gamma * negative

    @pytest.mark.parametrize(
        ("liked", "disliked"),
        [(20, 10), (50, 0), (6, 4), (200, 50)],
    )
    def test_faiss_agrees_with_the_written_formula(self, liked, disliked):
        features, model = self._setup(liked=liked, disliked=disliked)

        scores = model.score(features, np.arange(len(features)))

        np.testing.assert_allclose(scores, self._reference(model, features), atol=1e-5)

    def test_it_agrees_when_there_are_fewer_labels_than_k(self):
        # The `min(k, len(index))` path, where FAISS returns every vector it
        # has rather than padding with -1 sentinels. Three likes and seven
        # dislikes: enough labels for the rung, fewer likes than k.
        features, model = self._setup(liked=3, disliked=7)

        scores = model.score(features, np.arange(len(features)))

        np.testing.assert_allclose(scores, self._reference(model, features), atol=1e-5)

    def test_the_indexes_survive_a_save_and_load(self, tmp_path):
        # They are rebuilt on load rather than serialised, so this is the test
        # that the rebuild actually happens.
        features, model = self._setup(liked=20, disliked=10)
        before = model.score(features, np.arange(len(features)))

        model.save(tmp_path / "v1")
        restored = KnnRecommender().load(tmp_path / "v1")

        np.testing.assert_allclose(
            restored.score(features, np.arange(len(features))), before, atol=1e-6
        )
