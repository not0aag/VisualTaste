"""R3: the hand-written solver, the scikit-learn path, and the agreement between them.

The agreement test is the centrepiece. Two independent implementations of the
same objective, one written out from the maths and one from a library, landing
on the same weights to 1e-3: that is the evidence that both the derivation in
`logreg_numpy.py` and the C mapping in `r3_logreg.py` are right.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import expit

from visualtaste.eval.sim import SimulatedUser
from visualtaste.rec import logreg_numpy
from visualtaste.rec.base import FeatureBundle, UnfittedModelError
from visualtaste.rec.features import design_matrix, standardise
from visualtaste.rec.labels import ACTION_TO_TARGET, DEFAULT_ACTION_WEIGHTS, LabelSet
from visualtaste.rec.r3_logreg import (
    LAMBDA_GRID,
    LogisticRecommender,
    choose_lambda,
    fit_sklearn,
    sklearn_c,
    tune_lambda,
)

AGREEMENT = 1e-3


def labels_from(ids: list[str], actions: list[str]) -> LabelSet:
    return LabelSet(
        image_ids=list(ids),
        targets=np.array([ACTION_TO_TARGET[a] for a in actions], dtype=np.float32),
        weights=np.array([DEFAULT_ACTION_WEIGHTS[a] for a in actions], dtype=np.float32),
        actions=list(actions),
        timestamps=["2026-10-31T00:00:00"] * len(ids),
    )


def synthetic_problem(n: int = 300, d: int = 8, seed: int = 1729):
    """A seeded, non-separable logistic problem with mixed example weights."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=d)
    y = (rng.random(n) < expit(X @ true_w - 0.3)).astype(np.float64)
    s = rng.choice([0.25, 1.0, 2.0], size=n)
    return X, y, s


# ---------------------------------------------------------------------------
# M5: the brief's worked example
# ---------------------------------------------------------------------------


class TestM5:
    w = np.array([2.0, -1.0])
    b = 0.0
    X = np.array([[0.8, 0.6]])
    y = np.array([0.0])
    s = np.array([1.0])

    def test_prediction_and_loss(self):
        p = logreg_numpy.sigmoid(self.X @ self.w + self.b)[0]
        loss = logreg_numpy.objective(self.w, self.b, self.X, self.y, self.s, lam=0.0)

        assert p == pytest.approx(0.731, abs=1e-3)
        assert loss == pytest.approx(1.313, abs=1e-3)

    def test_one_gradient_step(self):
        w, b = logreg_numpy.gd_step(self.w, self.b, self.X, self.y, self.s, lam=0.0, eta=0.5)

        assert w == pytest.approx([1.708, -1.219], abs=1e-3)
        assert b == pytest.approx(-0.366, abs=1e-3)
        assert logreg_numpy.sigmoid(self.X @ w + b)[0] == pytest.approx(0.567, abs=1e-3)

    def test_the_gradient_matches_finite_differences(self):
        X, y, s = synthetic_problem(n=40, d=3)
        w = np.array([0.3, -0.2, 0.5])
        b, lam, h = 0.1, 0.05, 1e-6

        grad_w, grad_b = logreg_numpy.gradient(w, b, X, y, s, lam)

        for j in range(3):
            step = np.zeros(3)
            step[j] = h
            numeric = (
                logreg_numpy.objective(w + step, b, X, y, s, lam)
                - logreg_numpy.objective(w - step, b, X, y, s, lam)
            ) / (2 * h)
            assert grad_w[j] == pytest.approx(numeric, abs=1e-6)
        numeric_b = (
            logreg_numpy.objective(w, b + h, X, y, s, lam)
            - logreg_numpy.objective(w, b - h, X, y, s, lam)
        ) / (2 * h)
        assert grad_b == pytest.approx(numeric_b, abs=1e-6)

    def test_the_sigmoid_does_not_overflow(self):
        assert logreg_numpy.sigmoid(np.array([-1000.0, 1000.0])) == pytest.approx([0.0, 1.0])


# ---------------------------------------------------------------------------
# The C mapping, and agreement
# ---------------------------------------------------------------------------


class TestCMapping:
    def test_the_docstring_example(self):
        assert sklearn_c(0.01, 500.0) == pytest.approx(0.2)

    def test_it_is_not_one_over_lambda(self):
        # The blueprint's C = 1/lambda. Fit both ways; only 1/(lambda S)
        # reproduces the hand-written minimiser of the §5.2 objective.
        X, y, s = synthetic_problem()
        lam = 0.05
        reference = logreg_numpy.fit(X, y, s, lam=lam)

        from sklearn.linear_model import LogisticRegression

        wrong = LogisticRegression(C=1 / lam, tol=1e-10, max_iter=10_000).fit(X, y, sample_weight=s)
        right_w, _ = fit_sklearn(X, y, s, lam)

        assert np.abs(right_w - reference.w).max() < AGREEMENT
        assert np.abs(wrong.coef_[0] - reference.w).max() > 0.05

    def test_it_rejects_a_non_positive_lambda(self):
        with pytest.raises(ValueError):
            sklearn_c(0.0, 10.0)


class TestAgreement:
    """Part 2 §5.3: NumPy and scikit-learn agree within 1e-3 after convergence."""

    @pytest.mark.parametrize("lam", [1e-3, 1e-2, 0.3])
    def test_on_a_synthetic_problem(self, lam):
        X, y, s = synthetic_problem()

        hand = logreg_numpy.fit(X, y, s, lam=lam)
        w, b = fit_sklearn(X, y, s, lam)

        assert hand.converged
        assert np.abs(hand.w - w).max() < AGREEMENT
        assert abs(hand.b - b) < AGREEMENT

    def test_on_a_500_label_simulated_profile(self):
        # A simulated user over a 128-dimensional standardised corpus: the
        # shape of R3's real problem, with the real weight mix (saves 2.0,
        # skips 0.25), and lambda chosen by R3's own tuning.
        rng = np.random.default_rng(7)
        corpus = rng.normal(size=(2_000, 128)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        user = SimulatedUser(
            centroids=corpus[rng.choice(2_000, 3, replace=False)],
            rng=np.random.default_rng(8),
            offset=0.2,
        )
        shown = rng.choice(2_000, 500, replace=False)
        actions = user.react(corpus[shown])

        X, _, _ = standardise(corpus)
        X = X[shown].astype(np.float64)
        y = np.array([ACTION_TO_TARGET[a] for a in actions], dtype=np.float64)
        s = np.array([DEFAULT_ACTION_WEIGHTS[a] for a in actions], dtype=np.float64)
        lam = tune_lambda(X, y, s).lam

        hand = logreg_numpy.fit(X, y, s, lam=lam)
        w, b = fit_sklearn(X, y, s, lam)

        # At the optimum, not merely past an iteration cap. On this problem
        # gradient descent reaches |g| ~ 1e-8, and whether it crosses its own
        # 1e-8 stopping rule depends on the last bits of floating-point
        # rounding, which differ between CPUs (it failed once in CI at
        # 1.67e-8). The claim under test is agreement, so test the gradient
        # at a meaningful scale and then the agreement itself.
        assert hand.gradient_norm < 1e-6
        assert np.abs(hand.w - w).max() < AGREEMENT
        assert abs(hand.b - b) < AGREEMENT

    def test_the_loss_curve_only_goes_down(self):
        # Backtracking guarantees a decrease every step; notebook 03 plots this.
        X, y, s = synthetic_problem()
        history = logreg_numpy.fit(X, y, s, lam=0.01).loss_history
        assert all(b <= a + 1e-12 for a, b in itertools.pairwise(history))


# ---------------------------------------------------------------------------
# Tuning lambda
# ---------------------------------------------------------------------------


class TestLambda:
    def test_ties_go_to_the_larger_lambda(self):
        curve = {0.01: 0.5000, 0.1: 0.5008, 1.0: 0.5020}
        assert choose_lambda(curve) == 0.1

    def test_a_clear_winner_wins(self):
        assert choose_lambda({0.01: 0.40, 0.1: 0.45, 1.0: 0.60}) == 0.01

    def test_tuning_covers_the_whole_grid_on_the_latest_fifth(self):
        X, y, s = synthetic_problem(n=200)
        result = tune_lambda(X, y, s)

        assert set(result.curve) == set(LAMBDA_GRID)
        assert result.validation_size == 40
        assert result.train_size == 160
        assert result.lam in LAMBDA_GRID

    def test_a_one_class_training_split_falls_back_without_raising(self):
        X, _, s = synthetic_problem(n=50)
        y = np.zeros(50)
        y[-5:] = 1.0  # positives only in the validation fifth
        result = tune_lambda(X, y, s)
        assert result.note and not result.curve


# ---------------------------------------------------------------------------
# The recommender
# ---------------------------------------------------------------------------


def toy_bundle(n: int = 200, d: int = 16, seed: int = 3) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return FeatureBundle(image_ids=[f"{i:016x}" for i in range(n)], siglip=vectors)


def separable_labels(features: FeatureBundle, n: int = 60) -> LabelSet:
    # Like whatever points along the first axis.
    ids = features.image_ids[:n]
    actions = ["like" if features.siglip[i, 0] > 0 else "dislike" for i in range(n)]
    return labels_from(ids, actions)


class TestRecommender:
    def test_it_learns_a_planted_direction(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        scores = model.score(features, np.arange(len(features)))

        liked_side = features.siglip[:, 0] > 0
        assert model.is_fitted
        assert scores[liked_side].mean() > scores[~liked_side].mean() + 0.2
        assert np.all((scores > 0) & (scores < 1))

    def test_default_blocks_drop_dino_when_the_bundle_has_none(self):
        features = toy_bundle()  # SigLIP only
        model = LogisticRecommender().fit(separable_labels(features), features)
        assert model.is_fitted
        assert model.blocks == ("sig",)

    def test_default_blocks_use_dino_when_present(self):
        base = toy_bundle()
        rng = np.random.default_rng(5)
        dino = rng.normal(size=(len(base), 6)).astype(np.float32)
        features = FeatureBundle(image_ids=base.image_ids, siglip=base.siglip, dinov2=dino)
        model = LogisticRecommender().fit(separable_labels(features), features)
        assert model.blocks == ("sig", "dino")
        assert model.w.shape == (16 + 6,)

    def test_too_few_labels_leave_it_unfitted_with_a_reason(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features, n=20), features)
        assert not model.is_fitted
        assert "needs 30" in model.unfit_reason

    def test_one_class_leaves_it_unfitted_rather_than_raising(self):
        # Part 2 §4.4: 30 likes and 0 dislikes is an ordinary young profile.
        features = toy_bundle()
        labels = labels_from(features.image_ids[:30], ["like"] * 30)
        model = LogisticRecommender().fit(labels, features)

        assert not model.is_fitted
        assert "0 negative" in model.unfit_reason
        with pytest.raises(UnfittedModelError):
            model.score(features, np.arange(5))

    def test_explain_reports_the_probability(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        explanation = model.explain(features, 7)

        assert explanation.reason == "model_confident"
        assert explanation.detail["probability"] == pytest.approx(
            float(model.score(features, np.array([7]))[0]), abs=1e-4
        )
        assert "%" in explanation.sentence

    def test_save_and_load_give_identical_scores(self, tmp_path):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        model.save(tmp_path / "v1")

        restored = LogisticRecommender().load(tmp_path / "v1")
        candidates = np.arange(len(features))

        assert restored.is_fitted
        assert restored.lam == model.lam
        np.testing.assert_array_equal(
            restored.score(features, candidates), model.score(features, candidates)
        )

    def test_a_fitted_version_missing_its_weights_raises_oserror(self, tmp_path):
        # So the API falls back to cold start instead of serving zeros.
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        model.save(tmp_path / "v1")
        (tmp_path / "v1" / "weights.npy").unlink()

        with pytest.raises(OSError):
            LogisticRecommender().load(tmp_path / "v1")

    def test_fit_report_carries_the_validation_curve(self):
        features = toy_bundle()
        model = LogisticRecommender().fit(separable_labels(features), features)
        report = model.fit_report()
        assert report["lambda"] == model.lam
        assert len(report["validation_curve"]) == len(LAMBDA_GRID)


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_the_standardisation_worked_example(self):
        z, mean, std = standardise(np.array([[1.0], [2.0], [3.0], [6.0]]))
        assert mean[0] == pytest.approx(3.0)
        assert std[0] == pytest.approx(np.sqrt(3.5))
        assert z[:, 0] == pytest.approx([-1.069, -0.535, 0.0, 1.604], abs=1e-3)

    def test_a_constant_column_stays_zero(self):
        z, _, _ = standardise(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]))
        assert np.all(z[:, 1] == 0.0)

    def test_blocks_are_concatenated_and_cached(self):
        rng = np.random.default_rng(0)
        features = FeatureBundle(
            image_ids=[f"{i:016x}" for i in range(10)],
            siglip=rng.normal(size=(10, 4)).astype(np.float32),
            colour=rng.random(size=(10, 3)).astype(np.float32),
        )
        first = design_matrix(features, ("sig", "col"))
        assert first.shape == (10, 7)
        assert design_matrix(features, ("sig", "col")) is first

    def test_an_unavailable_block_is_a_clear_error(self):
        features = toy_bundle(n=5)
        with pytest.raises(KeyError, match="dino"):
            design_matrix(features, ("dino",))
