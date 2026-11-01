"""The doubly robust estimator on synthetic bandits, where the truth is known."""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.eval.propensity import doubly_robust, snips

ARMS = 5


def bandit(rng: np.random.Generator, rounds: int):
    """Contexts, true like-probabilities per arm, a logging and a target policy."""
    contexts = rng.normal(size=(rounds, 3))
    weights = rng.normal(size=(3, ARMS))
    truth = 1 / (1 + np.exp(-(contexts @ weights)))  # p(like | x, a)
    logging = np.full((rounds, ARMS), 1 / ARMS)  # uniform logger
    target = np.zeros((rounds, ARMS))
    target[np.arange(rounds), truth.argmax(axis=1)] = 1.0  # greedy target
    actions = np.array([rng.choice(ARMS, p=p) for p in logging])
    rewards = (rng.random(rounds) < truth[np.arange(rounds), actions]).astype(float)
    value = float((target * truth).sum(axis=1).mean())
    return truth, logging, target, actions, rewards, value


def estimates(rng, rounds, model_noise: float | None):
    truth, logging, target, actions, rewards, value = bandit(rng, rounds)
    idx = np.arange(rounds)
    if model_noise is None:  # a useless reward model: a constant
        r_hat = np.full_like(truth, 0.5)
    else:
        r_hat = np.clip(truth + rng.normal(0, model_noise, truth.shape), 0, 1)
    dr = doubly_robust(
        rewards,
        logging[idx, actions],
        target[idx, actions],
        r_hat[idx, actions],
        (target * r_hat).sum(axis=1),
        weight_cap=1e9,
    )
    ips = snips(rewards, logging[idx, actions], target[idx, actions], weight_cap=1e9)
    return dr, ips, value


class TestDoublyRobust:
    def test_unbiased_with_a_good_reward_model(self):
        rng = np.random.default_rng(1729)
        values = [estimates(rng, 2_000, 0.05) for _ in range(60)]
        bias = np.mean([dr.value - truth for dr, _, truth in values])
        assert abs(bias) < 0.01

    def test_still_unbiased_with_a_useless_reward_model(self):
        # Correct propensities rescue a wrong model: the "doubly" in the name.
        rng = np.random.default_rng(7)
        values = [estimates(rng, 2_000, None) for _ in range(60)]
        bias = np.mean([dr.value - truth for dr, _, truth in values])
        assert abs(bias) < 0.02

    def test_a_good_reward_model_is_no_noisier_than_snips(self):
        # With binary likes most of the variance is each like's own coin flip,
        # which no reward model removes, so the gain over SNIPS is small here
        # (about 6% lower spread with a perfect model). It must not be worse.
        rng = np.random.default_rng(11)
        values = [estimates(rng, 1_000, 0.05) for _ in range(80)]
        dr_spread = np.std([dr.value - truth for dr, _, truth in values])
        ips_spread = np.std([ips.value - truth for _, ips, truth in values])
        assert dr_spread <= ips_spread

    def test_a_useless_reward_model_is_noisier_than_snips(self):
        # The price of the correction term: with a constant model DR is still
        # unbiased but noisier than SNIPS. Worth knowing before trusting it.
        rng = np.random.default_rng(11)
        values = [estimates(rng, 1_000, None) for _ in range(80)]
        dr_spread = np.std([dr.value - truth for dr, _, truth in values])
        ips_spread = np.std([ips.value - truth for _, ips, truth in values])
        assert dr_spread > ips_spread

    def test_reports_a_sensible_standard_error(self):
        rng = np.random.default_rng(3)
        values = [estimates(rng, 1_000, 0.05) for _ in range(80)]
        observed = np.std([dr.value - truth for dr, _, truth in values])
        reported = np.mean([dr.standard_error for dr, _, _ in values])
        assert reported == pytest.approx(observed, rel=0.5)

    def test_drops_implausible_propensities_and_counts_them(self):
        estimate = doubly_robust(
            np.array([1.0, 0.0, 1.0]),
            np.array([0.5, 1e-6, np.nan]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.6, 0.4, 0.5]),
            np.array([0.6, 0.4, 0.5]),
        )
        assert estimate.samples == 1
        assert estimate.dropped == 2

    def test_misaligned_inputs_are_an_error(self):
        with pytest.raises(ValueError, match="align"):
            doubly_robust(np.ones(3), np.ones(3), np.ones(3), np.ones(2), np.ones(3))
