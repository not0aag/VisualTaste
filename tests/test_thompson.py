"""R4: the Laplace posterior, MMR, the exploration quota, and propensities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cholesky
from scipy.stats import norm

from tests.test_logreg import separable_labels, toy_bundle
from visualtaste.rec import rerank
from visualtaste.rec.base import FeatureBundle
from visualtaste.rec.r4_thompson import (
    ThompsonRecommender,
    precision_matrix,
    sample_weights,
)


class TestM7:
    def test_thompson_gives_the_uncertain_image_a_37_percent_chance(self):
        # A: mean 0.60, sd 0.05. B: mean 0.50, sd 0.30.
        p_b_beats_a = norm.cdf((0.50 - 0.60) / np.sqrt(0.05**2 + 0.30**2))
        assert p_b_beats_a == pytest.approx(0.37, abs=0.005)

        rng = np.random.default_rng(0)
        a = rng.normal(0.60, 0.05, 200_000)
        b = rng.normal(0.50, 0.30, 200_000)
        assert (b > a).mean() == pytest.approx(p_b_beats_a, abs=0.005)

    def test_mmr_worked_example(self):
        assert rerank.mmr_value(0.88, 0.95) == pytest.approx(0.331, abs=1e-3)
        assert rerank.mmr_value(0.80, 0.30) == pytest.approx(0.470, abs=1e-3)

    def test_mmr_picks_c_after_a(self):
        # Vectors built so that sim(a, b) = 0.95 and sim(a, c) = 0.30.
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.95, np.sqrt(1 - 0.95**2), 0.0])
        c = np.array([0.30, 0.0, np.sqrt(1 - 0.30**2)])
        picks = rerank.mmr_select(np.array([0.90, 0.88, 0.80]), np.stack([a, b, c]), 2)
        assert picks == [0, 2]


class TestPosterior:
    def test_the_sampled_covariance_is_h_inverse(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4))
        s = rng.choice([0.25, 1.0, 2.0], size=200)
        w, b, lam = np.array([0.5, -0.3, 0.2, 0.0]), 0.1, 0.05

        H = precision_matrix(X, s, w, b, lam)
        L = cholesky(H, lower=True)
        samples = sample_weights(w, L, np.random.default_rng(2), draws=200_000)

        np.testing.assert_allclose(samples.mean(axis=1), w, atol=3e-3)
        # Tolerance relative to the variance scale: with 200,000 draws an
        # entry's standard error is about var / sqrt(200,000), 0.2% of the
        # diagonal, so 1% is about four standard errors.
        expected = np.linalg.inv(H)
        np.testing.assert_allclose(np.cov(samples), expected, atol=0.01 * expected.diagonal().max())

    def test_the_prior_precision_is_lambda_times_s(self):
        X = np.zeros((10, 3))  # no data curvature at all
        H = precision_matrix(X, np.full(10, 2.0), np.zeros(3), 0.0, lam=0.1)
        np.testing.assert_allclose(H, 0.1 * 20.0 * np.eye(3))

    def test_draws_tighten_as_labels_accumulate(self):
        features = toy_bundle(n=400)
        spreads = []
        for count in (40, 300):
            model = ThompsonRecommender(lam=0.1).fit(separable_labels(features, count), features)
            draws = sample_weights(model.w, model.L, np.random.default_rng(0), draws=2_000)
            spreads.append(draws.std(axis=1).mean())
        assert spreads[1] < spreads[0]

    def test_temperature_scales_the_deviation(self):
        w, L = np.zeros(2), np.eye(2)
        cold = sample_weights(w, L, np.random.default_rng(0), draws=50_000, tau=0.25).std()
        hot = sample_weights(w, L, np.random.default_rng(0), draws=50_000, tau=4.0).std()
        assert hot / cold == pytest.approx(4.0, rel=0.02)


class TestQuota:
    def test_one_in_five_then_one_in_ten(self):
        assert rerank.explore_slots(0, 5, labels=100) == 1
        early = sum(rerank.explore_slots(5 * k, 5, labels=100) for k in range(20))
        late = sum(rerank.explore_slots(5 * k, 5, labels=2_500) for k in range(20))
        assert (early, late) == (20, 10)
        assert rerank.explore_slots(0, 5, labels=100, every=None) == 0

    def test_the_widest_interval_wins_and_rated_clusters_are_skipped(self):
        ratings = {0: (1, 1), 1: (0, 0), 2: (2, 1)}  # cluster 2 has 3 ratings
        chosen = rerank.choose_explore_cluster(
            ratings, np.array([0, 1, 2]), np.random.default_rng(0)
        )
        assert chosen == 1
        assert rerank.beta_interval_width(0, 0) == pytest.approx(0.9)

    def test_no_eligible_cluster_means_no_probe(self):
        ratings = {0: (2, 1)}
        assert (
            rerank.choose_explore_cluster(ratings, np.array([0]), np.random.default_rng(0)) is None
        )


def clustered_bundle(n: int = 400, clusters: int = 50) -> FeatureBundle:
    # 50 clusters, so 80 labels leave every cluster with at most 2 ratings and
    # the quota has somewhere to explore.
    features = toy_bundle(n=n)
    return FeatureBundle(
        image_ids=features.image_ids,
        siglip=features.siglip,
        cluster64=(np.arange(n) % clusters).astype(np.int32),
    )


class TestBatch:
    def _model(self, features, count=80):
        return ThompsonRecommender().fit(separable_labels(features, count), features)

    def test_a_batch_has_four_mmr_cards_and_one_probe(self):
        features = clustered_bundle()
        model = self._model(features)
        candidates = np.arange(80, len(features))

        chosen, sampled, propensities = model.rerank(
            features, candidates, model.score(features, candidates), 5
        )
        reasons = [model.explain(features, int(candidates[c])).reason for c in chosen]

        assert len(set(chosen.tolist())) == 5
        assert reasons.count("explore") == 1
        assert propensities is None  # estimated later, off the request path
        assert sampled.shape == candidates.shape

    def test_the_probe_comes_from_an_unrated_cluster(self):
        # 40 labels on rows 0-39 rate clusters 0-39 once each; 40-49 are unrated.
        # Unrated clusters have the widest interval (0.9), so the probe is one.
        features = clustered_bundle()
        model = self._model(features, count=40)
        candidates = np.arange(40, len(features))
        model.rerank(features, candidates, model.score(features, candidates), 5)

        probe = next(iter(model._explore_rows.values()))
        assert probe["ratings"] == [0, 0]
        assert probe["cluster64"] >= 40

    def test_batches_are_reproducible_from_the_seed(self):
        features = clustered_bundle()
        candidates = np.arange(80, len(features))
        first = self._model(features)
        second = self._model(features)
        a = first.rerank(features, candidates, first.score(features, candidates), 5)[0]
        b = second.rerank(features, candidates, second.score(features, candidates), 5)[0]
        np.testing.assert_array_equal(a, b)

    def test_inclusion_probabilities_are_probabilities(self):
        features = clustered_bundle()
        model = self._model(features)
        candidates = np.arange(80, len(features))
        chosen, _, _ = model.rerank(features, candidates, model.score(features, candidates), 5)

        n, n_explore = model.last_batch
        p = model.inclusion_probabilities(
            features, candidates, chosen, n=n, n_explore=n_explore, draws=100
        )
        assert p.shape == (5,)
        assert np.all((p >= 0) & (p <= 1))

    def test_with_no_posterior_spread_the_mmr_cards_are_certain(self):
        # tau_post -> 0: every draw is w*, so the MMR cards are nearly certain
        # and only the probe is left to chance. "Nearly": now and then a
        # replay's random probe lands on an image MMR would have picked, which
        # takes it out of the shortlist and shifts MMR's later picks.
        features = clustered_bundle()
        model = ThompsonRecommender(tau_post=1e-12).fit(separable_labels(features, 80), features)
        candidates = np.arange(80, len(features))
        chosen, _, _ = model.rerank(features, candidates, model.score(features, candidates), 5)
        explore_rows = set(model._explore_rows)

        n, n_explore = model.last_batch
        p = model.inclusion_probabilities(
            features, candidates, chosen, n=n, n_explore=n_explore, draws=50
        )
        for offset, probability in zip(chosen, p, strict=True):
            if int(candidates[offset]) in explore_rows:
                assert probability < 0.5
            else:
                assert probability >= 0.9

    def test_save_and_load_keep_the_posterior(self, tmp_path):
        features = clustered_bundle()
        model = self._model(features)
        model.save(tmp_path / "v")
        restored = ThompsonRecommender().load(tmp_path / "v")
        np.testing.assert_array_equal(restored.L, model.L)
        assert restored.cluster_ratings == model.cluster_ratings


class TestPhase15Options:
    """ROADMAP D1: a fading temperature and a narrower MMR shortlist.

    Both default to the original behaviour exactly, so every cached study is
    unaffected; only specs that name them change.
    """

    def test_without_a_half_life_the_temperature_is_tau_post_itself(self):
        model = ThompsonRecommender(tau_post=0.1)
        model.labels_at_fit = 5_000
        assert model.effective_tau() == 0.1

    def test_the_temperature_halves_at_the_half_life(self):
        model = ThompsonRecommender(tau_post=0.1, tau_halflife=200)
        model.labels_at_fit = 0
        assert model.effective_tau() == pytest.approx(0.1)
        model.labels_at_fit = 200
        assert model.effective_tau() == pytest.approx(0.05)
        model.labels_at_fit = 600
        assert model.effective_tau() == pytest.approx(0.025)

    def test_the_default_shortlist_is_the_original_constant(self):
        assert ThompsonRecommender().mmr_shortlist == rerank.SHORTLIST

    def test_a_short_list_keeps_mmr_among_the_top_scores(self):
        features = clustered_bundle()
        candidates = np.arange(len(features))
        relevance = np.linspace(1.0, 0.0, len(candidates))  # row 0 is best
        rng = np.random.default_rng(0)

        picks, explore = rerank.explore_batch(
            features, candidates, relevance, 5, 0, rng, cluster_ratings={}, shortlist=8
        )

        assert explore == []
        assert len(picks) == 5
        assert max(picks) < 8  # every pick from the top eight

    def test_the_options_parse_onto_the_model(self):
        from visualtaste.config import Settings
        from visualtaste.eval.study import RungSpec

        spec = RungSpec.parse("r4:tau_halflife=200,mmr_top=50")
        model = spec.build(Settings(), seed=1729)

        assert model.tau_halflife == 200.0
        assert model.mmr_shortlist == 50
