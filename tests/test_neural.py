"""R6: backprop by hand vs autograd (M9), BPR, training, checkpoints, no leakage."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_logreg import labels_from
from visualtaste.rec.base import FeatureBundle
from visualtaste.rec.r5_interests import InterestTracker
from visualtaste.rec.r6_neural import (
    NeuralRecommender,
    bpr_loss,
    build_profile_state,
    profile_features,
    session_pairs,
)


class TestM9:
    """x = 1, w1 = 0.5 (ReLU), w2 = 2, y = 1."""

    def test_forward_and_loss(self):
        h = max(0.0, 0.5 * 1.0)
        p = 1 / (1 + np.exp(-(2.0 * h)))
        assert p == pytest.approx(0.731, abs=1e-3)
        assert -np.log(p) == pytest.approx(0.313, abs=1e-3)

    def test_hand_gradients_match_autograd(self):
        x = torch.tensor(1.0)
        w1 = torch.tensor(0.5, requires_grad=True)
        w2 = torch.tensor(2.0, requires_grad=True)
        p = torch.sigmoid(w2 * torch.relu(w1 * x))
        loss = -torch.log(p)
        loss.backward()

        # By hand: dL/dz = p - y; dL/dw2 = (p - y) h; dL/dw1 = (p - y) w2 relu'(.) x.
        p_hand, h = 0.7310586, 0.5
        assert w2.grad.item() == pytest.approx((p_hand - 1) * h, abs=1e-6)
        assert w1.grad.item() == pytest.approx((p_hand - 1) * 2.0 * 1.0 * 1.0, abs=1e-6)
        assert w2.grad.item() == pytest.approx(-0.134, abs=1e-3)
        assert w1.grad.item() == pytest.approx(-0.538, abs=1e-3)

    def test_one_step_raises_p(self):
        w1, w2 = 0.5 + 0.1 * 0.538, 2.0 + 0.1 * 0.134
        p = 1 / (1 + np.exp(-(w2 * max(0.0, w1))))
        assert p == pytest.approx(0.753, abs=1e-3)

    def test_bpr(self):
        assert bpr_loss(torch.tensor([1.2]), torch.tensor([0.4])).item() == pytest.approx(
            0.371, abs=1e-3
        )


def neural_corpus(n: int = 600, seed: int = 0):
    """A non-linear taste: like only when two separate directions agree (an AND)."""
    rng = np.random.default_rng(seed)
    siglip = rng.normal(size=(n, 16)).astype(np.float32)
    siglip /= np.linalg.norm(siglip, axis=1, keepdims=True)
    colour = rng.random(size=(n, 8)).astype(np.float32)
    features = FeatureBundle(
        image_ids=[f"{i:016x}" for i in range(n)],
        siglip=siglip,
        colour=colour,
        cluster64=(np.arange(n) % 20).astype(np.int32),
    )
    liked = (siglip[:, 0] > 0) & (colour[:, 0] > 0.5)
    order = rng.permutation(n)[:400]
    actions = ["like" if liked[i] else "dislike" for i in order]
    return features, labels_from([features.image_ids[i] for i in order], actions), liked


def small_r6(**kwargs) -> NeuralRecommender:
    return NeuralRecommender(min_labels=30, members=2, device="cpu", max_epochs=60, **kwargs)


class TestTraining:
    def test_it_learns_a_conjunctive_taste(self):
        features, labels, liked = neural_corpus()
        model = small_r6().fit(labels, features)
        scores = model.score(features, np.arange(len(features)))
        assert model.is_fitted
        assert scores[liked].mean() > scores[~liked].mean() + 0.15
        report = model.fit_report()
        assert len(report["members"]) == 2
        assert all(m["best_epoch"] < m["epochs_run"] for m in report["members"])

    def test_bpr_variant_trains_too(self):
        features, labels, _ = neural_corpus()
        model = small_r6(loss="bpr").fit(labels, features)
        assert model.is_fitted and model.loss == "bpr"

    def test_it_refuses_below_its_minimum(self):
        features, labels, _ = neural_corpus()
        model = NeuralRecommender(min_labels=1500, device="cpu").fit(labels, features)
        assert not model.is_fitted and "needs 1500" in model.unfit_reason

    def test_checkpoints_reload_with_identical_outputs(self, tmp_path):
        features, labels, _ = neural_corpus()
        model = small_r6().fit(labels, features)
        before = model.score(features, np.arange(len(features)))
        model.save(tmp_path / "v")

        restored = NeuralRecommender(min_labels=30, device="cpu").load(tmp_path / "v")
        np.testing.assert_allclose(
            restored.score(features, np.arange(len(features))), before, atol=1e-6
        )

    def test_a_batch_explores_with_one_member(self):
        features, labels, _ = neural_corpus()
        model = small_r6().fit(labels, features)
        candidates = np.arange(400, len(features))
        chosen, _sampled, _ = model.rerank(
            features, candidates, model.score(features, candidates), 5
        )
        assert len(set(chosen.tolist())) == 5
        p = model.inclusion_probabilities(
            features, candidates, chosen, n=5, n_explore=model.last_batch[1], draws=20
        )
        assert np.all((p >= 0) & (p <= 1))

    def test_session_pairs_stay_inside_their_block(self):
        targets = np.array([1, 0, 0] * 70, dtype=float)
        actions = ["like", "dislike", "skip"] * 70
        pairs = session_pairs(targets, actions, np.random.default_rng(0))
        assert len(pairs) and np.all(pairs[:, 0] // 100 == pairs[:, 1] // 100)
        assert all(actions[n] == "dislike" for n in pairs[:, 1])


class TestProfileFeatures:
    def test_no_history_means_neutral_features(self):
        features, _, _ = neural_corpus()
        state = build_profile_state(features, np.array([], int), [], InterestTracker())
        block = profile_features(state, features, np.arange(5))
        assert block.shape == (5, 8)
        assert np.all(block[:, [0, 1, 2, 3, 4, 6, 7]] == 0) and np.all(block[:, 5] == 1)

    def test_an_images_own_like_is_not_in_its_features(self):
        # Leakage (Part 2 §4.6): the first labelled image's "max similarity to
        # likes" must not be its similarity to itself (1.0).
        features, labels, _ = neural_corpus()
        model = small_r6()
        captured = {}
        import visualtaste.rec.r6_neural as r6

        original = r6.time_respecting_features

        def spy(*args, **kwargs):
            captured["profile"] = original(*args, **kwargs)
            return captured["profile"]

        r6.time_respecting_features = spy
        try:
            model.fit(labels, features)
        finally:
            r6.time_respecting_features = original
        assert np.all(captured["profile"][:25, 2] == 0)  # nothing liked before them
        assert np.all(captured["profile"][:, 2] < 0.999)
