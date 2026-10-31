"""Tests for the planted-taste simulator.

The simulator is the instrument every learned rung is measured with, so a bug
here would not fail loudly -- it would produce plausible numbers for every
model and quietly invalidate the whole study. Hence the detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.db import repo
from visualtaste.eval.sim import (
    INTEREST_COUNT,
    LOGIT_OFFSET,
    LOGIT_SLOPE,
    SKIP_PROBABILITY,
    SimulatedUser,
    create_simulated_profile,
    make_user,
    plant_interests,
)


def unit_rows(count: int, dim: int = 32, *, seed: int = 1729) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, dim)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def user_with_one_interest(direction: np.ndarray, *, seed: int = 0) -> SimulatedUser:
    """A single-interest user, so `max_k` cannot mask the probability curve."""
    return SimulatedUser(centroids=direction.reshape(1, -1), rng=np.random.default_rng(seed))


def vector_at_cosine(centroid: np.ndarray, cosine: float) -> np.ndarray:
    """A unit vector with exactly `cosine` similarity to `centroid`."""
    basis = np.zeros_like(centroid)
    basis[0] = 1.0
    perpendicular = basis - (basis @ centroid) * centroid
    perpendicular /= np.linalg.norm(perpendicular)
    return (cosine * centroid + np.sqrt(1 - cosine**2) * perpendicular).astype(np.float32)


class TestProbabilityCurve:
    def test_the_specified_curve(self):
        # p(like) = sigma(10 (cos - 0.25)), so cos = 0.25 is a coin flip.
        # Tested with a single interest, because with three the max over
        # interests floors the probability and hides the curve.
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid)

        for cosine, expected in (
            (0.25, 0.500),
            (0.35, 0.731),
            (0.45, 0.881),
            (0.15, 0.269),
            (0.05, 0.119),
        ):
            vector = vector_at_cosine(centroid, cosine)
            assert user.like_probability(vector.reshape(1, -1))[0] == pytest.approx(
                expected, abs=1e-3
            ), f"cosine {cosine}"

    def test_the_boundary_is_soft_not_a_step(self):
        # A step function would let a model reach AUC 1.0, which would say
        # nothing about the model.
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid)

        just_below = user.like_probability(vector_at_cosine(centroid, 0.24).reshape(1, -1))[0]
        just_above = user.like_probability(vector_at_cosine(centroid, 0.26).reshape(1, -1))[0]

        assert 0.4 < just_below < 0.5 < just_above < 0.6

    def test_the_constants_match_the_brief(self):
        assert LOGIT_SLOPE == 10.0
        assert LOGIT_OFFSET == 0.25
        assert SKIP_PROBABILITY == 0.1
        assert INTEREST_COUNT == 3

    def test_affinity_is_the_max_over_interests_not_the_mean(self):
        # A user with three interests likes an image matching *any* of them.
        centroids = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        user = SimulatedUser(centroids=centroids, rng=np.random.default_rng(0))

        affinity = user.affinity(np.array([[0.0, 0.9, 0.0]], dtype=np.float32))

        assert affinity[0] == pytest.approx(0.9)

    def test_it_reports_which_interest_matched(self):
        centroids = np.eye(3)
        user = SimulatedUser(centroids=centroids, rng=np.random.default_rng(0))

        winners = user.best_interest(np.array([[0.1, 0.1, 0.9]], dtype=np.float32))

        assert winners[0] == 2


class TestReactions:
    def test_skips_occur_at_the_specified_rate(self):
        user = make_user(unit_rows(50), seed=1729)

        actions = user.react(unit_rows(4000, seed=7))

        skip_share = actions.count("skip") / len(actions)
        assert skip_share == pytest.approx(SKIP_PROBABILITY, abs=0.02)

    def test_a_skip_carries_no_information_about_liking(self):
        # The skip check runs first, independent of p. That is what makes the
        # Phase 9 skip-weight comparison a real question.
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid, seed=5)

        # 2,000 images the user would almost certainly like.
        loved = np.repeat(vector_at_cosine(centroid, 0.9).reshape(1, -1), 2000, axis=0)
        actions = user.react(loved)

        assert actions.count("skip") / len(actions) == pytest.approx(SKIP_PROBABILITY, abs=0.03)

    def test_images_it_loves_are_mostly_liked_or_saved(self):
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid, seed=3)
        loved = np.repeat(vector_at_cosine(centroid, 0.9).reshape(1, -1), 500, axis=0)

        actions = user.react(loved)
        positive = sum(1 for a in actions if a in ("like", "save"))

        # p(like) ~ 1 at cosine 0.9, so ~90% positive after the 10% skip rate.
        assert positive / len(actions) > 0.85

    def test_images_it_dislikes_are_mostly_disliked(self):
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid, seed=3)
        hated = np.repeat(vector_at_cosine(centroid, -0.2).reshape(1, -1), 500, axis=0)

        actions = user.react(hated)

        assert actions.count("dislike") / len(actions) > 0.85

    def test_saves_only_happen_for_images_it_loves(self):
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid, seed=11)
        lukewarm = np.repeat(vector_at_cosine(centroid, 0.3).reshape(1, -1), 500, axis=0)

        actions = user.react(lukewarm)

        # p = 0.62 here, below the 0.9 save threshold, so no saves at all.
        assert actions.count("save") == 0

    def test_saves_appear_at_roughly_the_specified_rate_when_loved(self):
        centroid = unit_rows(1)[0]
        user = user_with_one_interest(centroid, seed=13)
        loved = np.repeat(vector_at_cosine(centroid, 0.95).reshape(1, -1), 2000, axis=0)

        actions = user.react(loved)
        positives = actions.count("like") + actions.count("save")

        # Of the positives, about 30% should be saves.
        assert actions.count("save") / positives == pytest.approx(0.3, abs=0.05)

    def test_every_action_is_one_of_the_four(self):
        user = make_user(unit_rows(50), seed=1)

        actions = user.react(unit_rows(200, seed=2))

        assert set(actions) <= {"like", "save", "dislike", "skip"}

    def test_the_same_seed_gives_the_same_reactions(self):
        vectors = unit_rows(200, seed=4)

        first = make_user(unit_rows(50), seed=99).react(vectors)
        second = make_user(unit_rows(50), seed=99).react(vectors)

        assert first == second

    def test_a_different_seed_gives_different_reactions(self):
        vectors = unit_rows(200, seed=4)

        first = make_user(unit_rows(50), seed=99).react(vectors)
        other = make_user(unit_rows(50), seed=7).react(vectors)

        assert first != other


class TestPlantedInterests:
    def test_it_plants_the_specified_number(self):
        centroids = plant_interests(unit_rows(500), seed=1729)

        assert len(centroids) == INTEREST_COUNT

    def test_the_centroids_are_unit_length(self):
        centroids = plant_interests(unit_rows(500), seed=1729)

        np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)

    def test_interests_are_drawn_from_real_clusters_when_available(self):
        # So the planted taste is for things the corpus contains, rather than a
        # direction nothing points in -- which would make every rung look bad.
        vectors = unit_rows(300)
        labels = np.repeat(np.arange(6), 50)

        centroids = plant_interests(vectors, seed=1729, cluster_labels=labels)

        # Each centroid should be close to some cluster mean.
        cluster_means = np.stack([vectors[labels == c].mean(axis=0) for c in range(6)])
        cluster_means /= np.linalg.norm(cluster_means, axis=1, keepdims=True)
        for centroid in centroids:
            assert (cluster_means @ centroid).max() > 0.99

    def test_the_chosen_interests_are_pushed_apart(self):
        # Three interests that were the same kind of image would make "did it
        # find all three" meaningless.
        vectors = unit_rows(600, dim=16)
        labels = np.repeat(np.arange(12), 50)

        centroids = plant_interests(vectors, seed=1729, cluster_labels=labels)
        pairwise = centroids @ centroids.T
        off_diagonal = pairwise[np.triu_indices(len(centroids), k=1)]

        assert off_diagonal.max() < 0.5

    def test_it_falls_back_to_individual_images_without_clusters(self):
        centroids = plant_interests(unit_rows(10), seed=1, cluster_labels=None)

        assert len(centroids) == INTEREST_COUNT

    def test_unlabelled_images_are_ignored(self):
        # -1 marks "no cluster"; those rows must not become a centroid.
        vectors = unit_rows(200)
        labels = np.concatenate([np.full(100, -1), np.repeat(np.arange(4), 25)])

        centroids = plant_interests(vectors, seed=1729, cluster_labels=labels)

        assert len(centroids) == INTEREST_COUNT

    def test_a_corpus_smaller_than_the_interest_count_still_works(self):
        centroids = plant_interests(unit_rows(2), seed=1, cluster_labels=None)

        assert 1 <= len(centroids) <= 2


class TestInterestRecall:
    def test_nothing_liked_means_nothing_found(self):
        user = make_user(unit_rows(50), seed=1)

        assert user.recall_of_planted_interests() == 0.0

    def test_liking_images_from_one_interest_scores_a_third(self):
        centroids = np.eye(3)
        user = SimulatedUser(centroids=centroids, rng=np.random.default_rng(0))
        # Images pointing hard at interest 0 only.
        user.react(np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), 50, axis=0))

        assert user.recall_of_planted_interests() == pytest.approx(1 / 3)

    def test_liking_images_from_all_three_scores_one(self):
        centroids = np.eye(3)
        user = SimulatedUser(centroids=centroids, rng=np.random.default_rng(0))
        for interest in range(3):
            vector = np.zeros((40, 3), dtype=np.float32)
            vector[:, interest] = 1.0
            user.react(vector)

        assert user.recall_of_planted_interests() == pytest.approx(1.0)

    def test_a_handful_of_likes_does_not_count_as_found(self):
        # The threshold keeps a single lucky like from claiming an interest.
        centroids = np.eye(3)
        user = SimulatedUser(centroids=centroids, rng=np.random.default_rng(0))
        user.interest_hits = {0: 1}

        assert user.recall_of_planted_interests(minimum_hits=3) == 0.0


class TestDrift:
    def test_without_drift_the_interests_stay_put(self):
        user = make_user(unit_rows(100), seed=1, drift_per_step=0.0)
        before = user.centroids.copy()

        user.react(unit_rows(100, seed=2))

        np.testing.assert_allclose(user.centroids, before)

    def test_with_drift_the_interests_move(self):
        user = make_user(unit_rows(100), seed=1, drift_per_step=0.01)
        before = user.centroids.copy()

        user.react(unit_rows(100, seed=2))

        assert not np.allclose(user.centroids, before)
        # And they stay unit length, or the probability curve would shift.
        np.testing.assert_allclose(np.linalg.norm(user.centroids, axis=1), 1.0, atol=1e-5)

    def test_drift_goes_somewhere_rather_than_cancelling_out(self):
        # A fresh random direction each step would mostly cancel; a fixed
        # direction actually moves the taste, which is the thing to test.
        user = make_user(unit_rows(100), seed=1, drift_per_step=0.02)
        start = user.centroids.copy()

        for _ in range(10):
            user.react(unit_rows(20, seed=3))

        moved = float((start * user.centroids).sum(axis=1).min())
        assert moved < 0.99, "the taste did not actually move"


class TestSimulatedProfile:
    def test_the_profile_is_flagged_simulated(self, tmp_path):
        connection = repo.init_db(tmp_path / "t.db")

        profile_id = create_simulated_profile(connection, note="study")
        connection.commit()

        row = repo.query(connection, "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,))[
            0
        ]
        assert row["is_simulated"] == 1
        connection.close()

    def test_a_simulated_profile_never_becomes_the_active_one(self, tmp_path):
        # The flag is load-bearing: simulator events must not reach Alen's
        # training data or dashboards (brief §7.4).
        connection = repo.init_db(tmp_path / "t.db")
        real = repo.create_profile(connection, note="real")
        create_simulated_profile(connection, note="study")
        connection.commit()

        assert repo.active_profile_id(connection) == real
        connection.close()
