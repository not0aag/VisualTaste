"""Tests for the metrics, anchored on the brief's §8 M10 worked examples.

Every hand-implemented metric that has a scikit-learn equivalent is also
checked against it, so the version written for readability is a verified
version rather than a plausible one.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.eval.metrics import (
    ACTION_GAIN,
    beta_interval,
    calibration_bins,
    coverage,
    dcg,
    expected_calibration_error,
    exploration_yield,
    gains_from_actions,
    hit_rate,
    intra_batch_diversity,
    like_rate,
    log_loss,
    ndcg_at_k,
    novelty,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    roc_auc,
    rolling_like_rate,
    sigmoid,
    sign_test,
)


class TestSigmoid:
    def test_the_m5_worked_example(self):
        # w = (2, -1), x = (0.8, 0.6), b = 0, so z = 1.6 - 0.6 = 1.0.
        z = 2 * 0.8 + (-1) * 0.6

        assert sigmoid(np.array([z])).item() == pytest.approx(0.7311, abs=1e-4)

    def test_zero_maps_to_a_half(self):
        assert sigmoid(np.array([0.0])).item() == pytest.approx(0.5)

    def test_it_is_symmetric_about_zero(self):
        assert sigmoid(np.array([2.0])).item() == pytest.approx(
            1 - sigmoid(np.array([-2.0])).item()
        )

    def test_it_does_not_overflow_at_extremes(self):
        # The naive 1/(1+exp(-z)) overflows below about -745 in float64.
        values = sigmoid(np.array([-1000.0, -745.0, 0.0, 745.0, 1000.0]))

        assert np.all(np.isfinite(values))
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(1.0)

    def test_a_scalar_is_accepted(self):
        # Callers pass both scalars and arrays.
        assert float(sigmoid(1.0)) == pytest.approx(0.7311, abs=1e-4)


class TestLogLoss:
    def test_the_m5_worked_example(self):
        # y = 0 with p = sigmoid(1.0): -ln(1 - 0.73106) = 1.31326.
        # Computed from the exact probability rather than the brief's rounded
        # 0.7311, which would give 1.31342 -- the brief's 1.313 is the exact
        # value to three places.
        p = sigmoid(np.array([1.0]))

        assert log_loss(np.array([0.0]), p) == pytest.approx(1.313, abs=1e-3)

    def test_agrees_with_sklearn(self):
        from sklearn.metrics import log_loss as sk_log_loss

        rng = np.random.default_rng(1729)
        y = rng.integers(0, 2, 200).astype(float)
        p = rng.uniform(0.02, 0.98, 200)

        assert log_loss(y, p) == pytest.approx(sk_log_loss(y, p), rel=1e-9)

    def test_predicting_a_half_for_everything_scores_ln_two(self):
        # The honest baseline: 0.693 is what refusing to guess costs, so a
        # model above it is worse than useless.
        y = np.array([1.0, 0.0, 1.0, 0.0])

        assert log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2), abs=1e-9)

    def test_a_perfect_confident_prediction_scores_near_zero(self):
        assert log_loss(np.array([1.0, 0.0]), np.array([0.999, 0.001])) < 0.002

    def test_weights_change_the_mean(self):
        y = np.array([1.0, 0.0])
        p = np.array([0.9, 0.9])

        unweighted = log_loss(y, p)
        # Weighting the badly-predicted second example more must worsen it.
        weighted = log_loss(y, p, np.array([1.0, 3.0]))

        assert weighted > unweighted

    def test_a_confident_wrong_prediction_is_finite(self):
        # log(0) is -inf; one such prediction would make the average useless.
        assert np.isfinite(log_loss(np.array([1.0]), np.array([0.0])))

    def test_an_empty_input_is_nan_not_zero(self):
        assert np.isnan(log_loss(np.array([]), np.array([])))

    def test_mismatched_shapes_fail_loudly(self):
        with pytest.raises(ValueError, match="probabilities"):
            log_loss(np.array([1.0, 0.0]), np.array([0.5]))


class TestRocAuc:
    def test_the_m10_worked_example(self):
        # Likes score 0.9 and 0.6; dislikes 0.7 and 0.2. Three of four pairs
        # are ordered correctly, so AUC = 0.75.
        targets = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.7, 0.2])

        assert roc_auc(targets, scores) == pytest.approx(0.75)

    def test_agrees_with_sklearn_on_random_data(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(1729)
        for _ in range(5):
            targets = rng.integers(0, 2, 300)
            scores = rng.normal(size=300)
            if len(set(targets.tolist())) < 2:
                continue
            assert roc_auc(targets, scores) == pytest.approx(
                roc_auc_score(targets, scores), rel=1e-9
            )

    def test_agrees_with_sklearn_when_scores_are_tied(self):
        # Ties should count a half, which is where a naive implementation
        # silently diverges from the standard definition.
        from sklearn.metrics import roc_auc_score

        targets = np.array([1, 1, 0, 0, 1, 0])
        scores = np.array([0.5, 0.5, 0.5, 0.2, 0.9, 0.5])

        assert roc_auc(targets, scores) == pytest.approx(roc_auc_score(targets, scores), rel=1e-9)

    def test_perfect_separation_is_one(self):
        assert roc_auc(np.array([1, 1, 0, 0]), np.array([1.0, 0.9, 0.1, 0.0])) == 1.0

    def test_reversed_separation_is_zero(self):
        assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.0, 0.1, 0.9, 1.0])) == 0.0

    def test_all_tied_scores_give_a_half(self):
        assert roc_auc(np.array([1, 0, 1, 0]), np.full(4, 0.5)) == pytest.approx(0.5)

    def test_one_class_only_is_nan_rather_than_a_half(self):
        # Returning 0.5 would look like a measurement; there is no question to
        # answer without both classes.
        assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.9, 0.5, 0.1])))
        assert np.isnan(roc_auc(np.array([0, 0]), np.array([0.9, 0.1])))

    def test_it_only_depends_on_the_ordering(self):
        # AUC is rank-based, so any monotonic rescaling of the scores must
        # leave it unchanged. This is also why log loss is reported beside it.
        targets = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.7, 0.2])

        assert roc_auc(targets, scores) == pytest.approx(roc_auc(targets, scores * 100))
        assert roc_auc(targets, scores) == pytest.approx(roc_auc(targets, scores + 5))


class TestNdcg:
    def test_the_m10_worked_example(self):
        # gains [1, 0, 2, 0, 1] -> 0.699 with exponential gain.
        assert ndcg_at_k([1, 0, 2, 0, 1], 5) == pytest.approx(0.699, abs=1e-3)

    def test_the_intermediate_dcg_values_from_the_worked_example(self):
        gains = [1, 0, 2, 0, 1]

        assert dcg(gains, 5) == pytest.approx(2.8869, abs=1e-4)
        assert dcg(sorted(gains, reverse=True), 5) == pytest.approx(4.1309, abs=1e-4)

    def test_exponential_gain_is_what_reproduces_the_brief(self):
        # Linear gain would give 0.762 for the same list. Recorded as a test so
        # that switching convention breaks loudly rather than silently shifting
        # every reported NDCG.
        gains = np.asarray([1, 0, 2, 0, 1], dtype=float)
        discounts = np.log2(np.arange(2, 7))
        linear = (gains / discounts).sum() / (np.sort(gains)[::-1] / discounts).sum()

        assert linear == pytest.approx(0.762, abs=1e-3)
        assert ndcg_at_k(gains, 5) == pytest.approx(0.699, abs=1e-3)

    def test_a_perfectly_ordered_list_scores_one(self):
        assert ndcg_at_k([2, 1, 1, 0, 0], 5) == pytest.approx(1.0)

    def test_order_matters(self):
        assert ndcg_at_k([2, 0, 0], 3) > ndcg_at_k([0, 0, 2], 3)

    def test_a_list_with_nothing_wanted_scores_zero(self):
        assert ndcg_at_k([0, 0, 0], 3) == 0.0

    def test_it_deliberately_disagrees_with_sklearn_on_gain_convention(self):
        # `sklearn.metrics.ndcg_score` uses *linear* gain, so it reports 0.762
        # for the brief's example while this module reports 0.699. Neither is
        # wrong; they are different conventions, and the brief's figure pins
        # ours to the exponential one. Asserted so that nobody "fixes" this
        # module to agree with sklearn and silently shifts every NDCG in
        # docs/EVALUATION.md.
        from sklearn.metrics import ndcg_score

        gains = [1, 0, 2, 0, 1]
        scores = [5, 4, 3, 2, 1]  # any strictly descending scores keep this order

        assert ndcg_score([gains], [scores], k=5) == pytest.approx(0.762, abs=1e-3)
        assert ndcg_at_k(gains, 5) == pytest.approx(0.699, abs=1e-3)

    def test_k_truncates_but_the_ideal_still_knows_about_the_save(self):
        # Gains [1, 1, 2] at k = 2. The achieved DCG@2 uses the two likes
        # (1 + 1/log2(3) = 1.631), but the *ideal* top two are the save and a
        # like (3 + 1/log2(3) = 3.631), because the ideal ordering is over the
        # whole relevance set. So burying a save at rank 3 is penalised:
        # 1.631 / 3.631 = 0.449. That penalty is the point of the metric.
        assert ndcg_at_k([1, 1, 2], 2) == pytest.approx(0.449, abs=1e-3)
        assert ndcg_at_k([2, 1, 1], 2) == pytest.approx(1.0)

    def test_gains_come_from_the_action_table(self):
        assert ACTION_GAIN == {"save": 2, "like": 1, "dislike": 0, "skip": 0}
        assert gains_from_actions(["save", "like", "skip", "dislike"]) == [2, 1, 0, 0]

    def test_an_unknown_action_contributes_nothing(self):
        assert gains_from_actions(["shrug"]) == [0]


class TestTopKMetrics:
    def test_precision_divides_by_k_not_by_what_was_shown(self):
        # Three good results when ten were asked for is 30%, not 100%.
        assert precision_at_k([True, True, True], 10) == pytest.approx(0.3)

    def test_precision_of_a_full_list(self):
        assert precision_at_k([True, False, True, False], 4) == pytest.approx(0.5)

    def test_precision_with_k_of_zero_is_zero_not_an_error(self):
        assert precision_at_k([True], 0) == 0.0

    def test_recall_needs_the_full_truth_set(self):
        assert recall_at_k([True, False, True], total_relevant=10, k=3) == pytest.approx(0.2)

    def test_recall_without_a_denominator_is_nan(self):
        # Alen's real preferences have no denominator; only the simulator does.
        assert np.isnan(recall_at_k([True], total_relevant=0, k=1))

    def test_hit_rate_is_one_if_anything_landed(self):
        assert hit_rate([False, False, True], 3) == 1.0
        assert hit_rate([False, False, False], 3) == 0.0

    def test_hit_rate_respects_k(self):
        assert hit_rate([False, False, True], 2) == 0.0


class TestDiversity:
    def test_orthogonal_vectors_are_maximally_diverse(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])

        assert intra_batch_diversity(vectors) == pytest.approx(1.0)

    def test_identical_vectors_have_no_diversity(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        assert intra_batch_diversity(vectors) == pytest.approx(0.0)

    def test_opposite_vectors_exceed_one(self):
        # cos = -1 gives 1 - (-1) = 2. Not clipped: the metric is a mean of
        # 1 - cos, and clamping it would hide a genuinely contrasting batch.
        vectors = np.array([[1.0, 0.0], [-1.0, 0.0]])

        assert intra_batch_diversity(vectors) == pytest.approx(2.0)

    def test_a_single_image_has_no_pairs(self):
        assert intra_batch_diversity(np.array([[1.0, 0.0]])) == 0.0

    def test_an_empty_batch_is_zero(self):
        assert intra_batch_diversity(np.zeros((0, 4))) == 0.0

    def test_it_averages_over_distinct_pairs_only(self):
        # Three vectors give three pairs, not nine: the diagonal and the
        # mirror image would both drag the mean towards zero.
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        # pairs: (1,2) cos 0 -> 1 ; (1,3) cos -1 -> 2 ; (2,3) cos 0 -> 1
        assert intra_batch_diversity(vectors) == pytest.approx((1 + 2 + 1) / 3)


class TestCoverage:
    def test_counts_distinct_clusters(self):
        assert coverage([0, 0, 1, 2, 2], total_clusters=10) == pytest.approx(0.3)

    def test_full_coverage(self):
        assert coverage([0, 1, 2, 3], total_clusters=4) == pytest.approx(1.0)

    def test_an_empty_window_is_zero(self):
        assert coverage([], total_clusters=64) == 0.0

    def test_no_clusters_is_nan(self):
        assert np.isnan(coverage([0], total_clusters=0))


class TestNovelty:
    def test_an_unseen_direction_is_novel(self):
        candidates = np.array([[0.0, 1.0]], dtype=np.float32)
        seen = np.array([[1.0, 0.0]], dtype=np.float32)

        assert novelty(candidates, seen)[0] == pytest.approx(1.0, abs=1e-6)

    def test_something_already_shown_has_no_novelty(self):
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)

        assert novelty(vectors, vectors)[0] == pytest.approx(0.0, abs=1e-6)

    def test_with_nothing_seen_everything_is_maximally_novel(self):
        candidates = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        np.testing.assert_allclose(novelty(candidates, np.zeros((0, 2), np.float32)), [1.0, 1.0])

    def test_it_uses_the_nearest_seen_image_not_the_mean(self):
        # A candidate identical to one seen image must score 0 even if it is
        # far from the others -- "have I shown this already" is a max, not a mean.
        candidates = np.array([[1.0, 0.0]], dtype=np.float32)
        seen = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)

        assert novelty(candidates, seen)[0] == pytest.approx(0.0, abs=1e-6)

    def test_a_flat_vector_is_accepted(self):
        assert novelty(np.array([1.0, 0.0], np.float32), np.zeros((0, 2), np.float32)).shape == (1,)


class TestBetaInterval:
    def test_no_data_is_maximally_uncertain(self):
        # This is the behaviour exploration needs: never sampled means
        # uncertain, not bad. A plain k/n would divide by zero.
        interval = beta_interval(0, 0)

        assert interval.mean == pytest.approx(0.5)
        assert interval.low == pytest.approx(0.05, abs=0.01)
        assert interval.high == pytest.approx(0.95, abs=0.01)

    def test_the_mean_is_the_laplace_rule_not_the_raw_rate(self):
        # 8 of 10 gives (1+8)/(2+10) = 0.75, not 0.8.
        assert beta_interval(8, 10).mean == pytest.approx(0.75)

    def test_more_data_narrows_the_interval(self):
        few = beta_interval(8, 10)
        many = beta_interval(80, 100)

        assert many.width < few.width
        assert many.mean == pytest.approx(81 / 102)

    def test_an_impossible_count_fails_loudly(self):
        with pytest.raises(ValueError, match="not possible"):
            beta_interval(5, 2)

    def test_the_interval_brackets_the_mean(self):
        interval = beta_interval(3, 20)

        assert interval.low < interval.mean < interval.high


class TestExplorationYield:
    def test_reports_both_rates_with_intervals(self):
        result = exploration_yield([True, False, False, False], [True, True, True, False])

        assert result.explore.mean < result.exploit.mean
        assert result.difference > 0

    def test_small_samples_overlap_which_is_the_honest_answer(self):
        # Two overlapping intervals communicate "we cannot tell yet" far better
        # than a significance test does.
        result = exploration_yield([True, False], [True, True])

        assert result.intervals_overlap

    def test_a_large_clear_difference_does_not_overlap(self):
        result = exploration_yield([False] * 200, [True] * 200)

        assert not result.intervals_overlap


class TestCalibration:
    def test_bins_compare_the_claim_to_the_outcome(self):
        targets = np.array([0.0, 0.0, 1.0, 1.0])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])

        lower, upper = calibration_bins(targets, probabilities, bins=2)

        assert lower.mean_predicted == pytest.approx(0.15)
        assert lower.observed_rate == pytest.approx(0.0)
        assert upper.mean_predicted == pytest.approx(0.85)
        assert upper.observed_rate == pytest.approx(1.0)

    def test_the_gap_sign_says_which_way_the_model_is_wrong(self):
        # Positive means over-confident: it claimed more than it delivered.
        targets = np.array([0.0, 0.0])
        probabilities = np.array([0.9, 0.9])

        over = calibration_bins(targets, probabilities, bins=2)[-1]

        assert over.gap > 0

    def test_a_perfectly_calibrated_model_has_no_error(self):
        # 30 of 100 at p = 0.3, and 70 of 100 at p = 0.7.
        targets = np.concatenate([np.repeat([1.0, 0.0], [30, 70]), np.repeat([1.0, 0.0], [70, 30])])
        probabilities = np.concatenate([np.full(100, 0.3), np.full(100, 0.7)])

        assert expected_calibration_error(targets, probabilities, bins=10) < 0.01

    def test_every_prediction_lands_in_exactly_one_bin(self):
        probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        targets = np.zeros(5)

        buckets = calibration_bins(targets, probabilities, bins=4)

        assert sum(b.count for b in buckets) == 5

    def test_an_empty_bin_is_nan_rather_than_zero(self):
        buckets = calibration_bins(np.array([1.0]), np.array([0.95]), bins=10)

        assert buckets[0].count == 0
        assert np.isnan(buckets[0].mean_predicted)

    def test_ece_weights_bins_by_population(self):
        # One badly-calibrated prediction among 100 good ones should barely
        # move the number; a plain average over bins would let it dominate.
        targets = np.concatenate([np.ones(99), np.zeros(1)])
        probabilities = np.concatenate([np.full(99, 0.99), np.array([0.99])])

        assert expected_calibration_error(targets, probabilities, bins=10) < 0.03


class TestLikeRate:
    def test_counts_likes_and_saves_as_positive(self):
        assert like_rate(["like", "save", "skip", "dislike"]) == pytest.approx(0.5)

    def test_an_empty_history_is_nan(self):
        assert np.isnan(like_rate([]))

    def test_the_rolling_version_shows_a_step_a_cumulative_one_would_hide(self):
        # 20 misses then 20 hits. Cumulatively that ends at 0.5; with a window
        # of 10 the final value should be 1.0, which is what a learning-curve
        # plot needs to show.
        actions = ["skip"] * 20 + ["like"] * 20

        curve = rolling_like_rate(actions, window=10)

        assert curve[-1] == pytest.approx(1.0)
        assert like_rate(actions) == pytest.approx(0.5)

    def test_the_rolling_curve_has_one_point_per_action(self):
        assert len(rolling_like_rate(["like"] * 7, window=3)) == 7

    def test_a_non_positive_window_fails_loudly(self):
        with pytest.raises(ValueError, match="positive"):
            rolling_like_rate(["like"], window=0)


class TestPairedBootstrap:
    def test_a_clear_difference_excludes_zero(self):
        first = np.array([0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71])
        second = np.array([0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.51])

        mean, low, _high = paired_bootstrap(first, second, seed=1729)

        assert mean == pytest.approx(0.20, abs=0.01)
        assert low > 0

    def test_no_difference_straddles_zero(self):
        # The most likely outcome of most comparisons in this project, and it
        # should be reported as "no detectable difference".
        rng = np.random.default_rng(1729)
        values = rng.normal(0.6, 0.05, 20)

        _, low, high = paired_bootstrap(values, values + rng.normal(0, 0.05, 20), seed=1729)

        assert low < 0 < high

    def test_it_is_deterministic_for_a_seed(self):
        first = np.array([0.6, 0.7, 0.8])
        second = np.array([0.5, 0.6, 0.7])

        assert paired_bootstrap(first, second, seed=7) == paired_bootstrap(first, second, seed=7)

    def test_mismatched_lengths_fail_loudly(self):
        with pytest.raises(ValueError, match="pairs must align"):
            paired_bootstrap(np.array([1.0, 2.0]), np.array([1.0]))

    def test_an_empty_comparison_is_nan(self):
        mean, low, high = paired_bootstrap(np.array([]), np.array([]))

        assert np.isnan(mean) and np.isnan(low) and np.isnan(high)


class TestSignTest:
    def test_the_worked_example(self):
        # 8 wins, 2 losses: suggestive, not conclusive.
        assert sign_test(8, 2) == pytest.approx(0.109, abs=1e-3)

    def test_an_even_split_is_not_significant(self):
        assert sign_test(5, 5) == pytest.approx(1.0)

    def test_a_unanimous_result_is_significant(self):
        assert sign_test(10, 0) < 0.01

    def test_it_is_symmetric(self):
        assert sign_test(8, 2) == pytest.approx(sign_test(2, 8))

    def test_no_comparisons_is_nan(self):
        assert np.isnan(sign_test(0, 0))

    def test_agrees_with_scipy(self):
        from scipy.stats import binomtest

        for wins, losses in ((8, 2), (3, 9), (15, 10), (1, 1)):
            expected = binomtest(wins, wins + losses, 0.5).pvalue
            assert sign_test(wins, losses) == pytest.approx(expected, rel=1e-9)
