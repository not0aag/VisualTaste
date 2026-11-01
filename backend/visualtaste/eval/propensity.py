"""Estimating how a *different* policy would have done, from logged data.

Clearly labelled as noisy and educational, because it is
--------------------------------------------------------
The brief asks for inverse propensity scoring and asks for it to be labelled
honestly (§7.7). This module takes that seriously, so read this before
believing any number it produces.

The question IPS answers
------------------------
"The logged feed showed these images and Alen liked 34% of them. What would a
*new* model have achieved, without deploying it?"

The idea is reweighting. If the old policy showed an image with probability
0.5 and the new policy would show it with probability 0.9, that observation
counts 1.8 times as much, because the new policy would have encountered it
more often:

    V_IPS = (1/n) * sum_i ( pi_new(a_i | x_i) / pi_old(a_i | x_i) ) * r_i

and the self-normalised variant, which is what this module uses:

    V_SNIPS = sum_i w_i r_i / sum_i w_i,    w_i = pi_new / pi_old

Why self-normalised: plain IPS is unbiased but its variance explodes when any
propensity is small, and it can return estimates outside the range the reward
can actually take -- a "like rate" of 1.4. SNIPS introduces a small bias and
in exchange is bounded by the observed rewards and vastly more stable. For a
few thousand logged impressions that trade is clearly worth making.

Why these numbers are weak here, specifically
---------------------------------------------
1. **The propensities are approximate.** Only R4 onwards has a stochastic
   policy with a computable probability, and even there it is estimated by
   Monte Carlo (200 replays per batch, §7.5). R0-R3 are deterministic: their
   true propensity is 1 for what they showed and 0 for everything else, which
   makes the importance weights degenerate.

2. **Support is tiny.** IPS can only learn about images the logged policy
   actually showed. A new model whose favourite images the old one never served
   gets no credit for them -- the estimate is silent about exactly the
   difference you most want to measure.

3. **n is small.** A few thousand impressions, and the weights have high
   variance, so the confidence interval is wide enough to contain most
   hypotheses.

So this is a sanity check and a teaching device, not evidence. The brief's
preferred tool for a real comparison is interleaving (`eval/interleave.py`),
which measures two policies against each other on live traffic and does not
need propensities at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Importance weights are clipped here. Without a cap, one impression logged at
# propensity 0.001 would carry a weight of 1,000 and dominate the whole
# estimate -- the single worst failure mode of IPS.
DEFAULT_WEIGHT_CAP = 10.0

# Propensities below this are treated as missing rather than tiny: a recorded
# probability of 1e-6 is far more likely to be an estimation artefact than a
# real one-in-a-million event.
MINIMUM_PROPENSITY = 1e-3


@dataclass
class IpsEstimate:
    """A self-normalised IPS estimate, with the diagnostics needed to judge it."""

    value: float
    logged_value: float
    samples: int
    effective_sample_size: float
    max_weight: float
    clipped: int
    dropped: int

    @property
    def effective_fraction(self) -> float:
        """ESS as a share of the sample count. Below ~0.1 the estimate is noise.

        ESS = (sum w)^2 / sum(w^2) is the classic importance-sampling
        diagnostic: the number of equally-weighted observations this weighted
        sample is worth. If 2,000 impressions have an ESS of 40, the estimate
        rests on about forty of them.
        """
        return self.effective_sample_size / self.samples if self.samples else 0.0

    @property
    def trustworthy(self) -> bool:
        """A deliberately strict gate, used to decide whether to print a number.

        Not a statistical test. It encodes "would a careful person quote this",
        and the answer is no when the estimate leans on a handful of
        observations.
        """
        return self.samples >= 100 and self.effective_fraction >= 0.1

    def summarise(self) -> str:
        if not self.trustworthy:
            return (
                f"SNIPS estimate {self.value:.3f} against a logged {self.logged_value:.3f}, "
                f"but from {self.samples} impressions with an effective sample size of "
                f"{self.effective_sample_size:.0f} ({self.effective_fraction:.1%}). "
                f"Too few effective samples to quote -- treat as unmeasured."
            )
        return (
            f"SNIPS estimate {self.value:.3f} against a logged {self.logged_value:.3f} "
            f"(n={self.samples}, effective {self.effective_sample_size:.0f}, "
            f"max weight {self.max_weight:.1f}, {self.clipped} clipped). "
            f"Noisy by construction -- see the module docstring."
        )


def snips(
    rewards: np.ndarray,
    logged_propensities: np.ndarray,
    new_propensities: np.ndarray,
    *,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
) -> IpsEstimate:
    """Self-normalised inverse propensity estimate of a new policy's reward.

    `rewards` is typically 1 for a like or save and 0 otherwise.
    `logged_propensities` is the probability the deployed policy had of showing
    each impression; `new_propensities` the same for the candidate policy.

    Impressions whose logged propensity is missing or implausibly small are
    dropped and counted, rather than retained with an enormous weight.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    logged = np.asarray(logged_propensities, dtype=np.float64)
    proposed = np.asarray(new_propensities, dtype=np.float64)

    if not (len(rewards) == len(logged) == len(proposed)):
        raise ValueError(
            f"{len(rewards)} rewards, {len(logged)} logged and "
            f"{len(proposed)} new propensities must all align"
        )
    if len(rewards) == 0:
        return IpsEstimate(float("nan"), float("nan"), 0, 0.0, 0.0, 0, 0)

    keep = np.isfinite(logged) & (logged >= MINIMUM_PROPENSITY)
    dropped = int((~keep).sum())

    rewards, logged, proposed = rewards[keep], logged[keep], proposed[keep]
    if len(rewards) == 0:
        logger.warning("every impression lacked a usable propensity")
        return IpsEstimate(float("nan"), float("nan"), 0, 0.0, 0.0, 0, dropped)

    raw_weights = proposed / logged
    weights = np.minimum(raw_weights, weight_cap)
    clipped = int((raw_weights > weight_cap).sum())

    total_weight = weights.sum()
    if total_weight <= 0:
        return IpsEstimate(
            float("nan"), float(rewards.mean()), len(rewards), 0.0, 0.0, clipped, dropped
        )

    value = float((weights * rewards).sum() / total_weight)
    ess = float(total_weight**2 / np.square(weights).sum())

    estimate = IpsEstimate(
        value=value,
        logged_value=float(rewards.mean()),
        samples=len(rewards),
        effective_sample_size=ess,
        max_weight=float(weights.max()),
        clipped=clipped,
        dropped=dropped,
    )
    logger.info("%s", estimate.summarise())
    return estimate


def rewards_from_actions(actions: list[str]) -> np.ndarray:
    """1 for a like or save, 0 otherwise -- the reward IPS estimates."""
    return np.array(
        [1.0 if action in ("like", "save") else 0.0 for action in actions],
        dtype=np.float64,
    )


def monte_carlo_propensity(
    sampler,
    *,
    slots: int,
    trials: int = 200,
) -> dict[tuple[int, int], float]:
    """Estimate p(image at slot) by repeating a stochastic selection.

    R4's selection is a Thompson draw followed by MMR re-ranking and an
    exploration quota. There is no closed form for "the probability this image
    lands in slot 2", so it is estimated by running the whole selection
    `trials` times and counting (brief §7.5).

    `sampler()` must return a list of candidate indices, one per slot, drawn
    afresh. Returns `{(slot, candidate index): probability}`.

    The approximation and its cost are both worth stating. 200 trials gives a
    standard error of about 0.035 on a probability of 0.5 and about 0.007 on
    one of 0.01 -- fine for reweighting, useless for distinguishing 0.002 from
    0.004. And it costs 200 re-rankings per batch, which is why it runs offline
    rather than on the serving path.
    """
    counts: dict[tuple[int, int], int] = {}
    for _ in range(trials):
        chosen = sampler()
        for slot, candidate in enumerate(chosen[:slots]):
            key = (slot, int(candidate))
            counts[key] = counts.get(key, 0) + 1

    return {key: count / trials for key, count in counts.items()}


@dataclass
class DrEstimate:
    """A doubly robust estimate of a new policy's reward, with its standard error."""

    value: float
    standard_error: float
    samples: int
    dropped: int
    clipped: int


def doubly_robust(
    rewards: np.ndarray,
    logged_propensities: np.ndarray,
    new_propensities: np.ndarray,
    predicted_reward_logged: np.ndarray,
    predicted_value_new: np.ndarray,
    *,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
) -> DrEstimate:
    """Doubly robust off-policy estimate (Dudik, Langford and Li, 2011).

    Per logged round t, with a reward model r_hat:

        DR_t = V_hat(x_t) + (pi(a_t|x_t) / mu(a_t|x_t)) * (r_t - r_hat(x_t, a_t))

    where V_hat(x_t) = sum_a pi(a|x_t) r_hat(x_t, a) is the reward model's
    value of the new policy in that context (`predicted_value_new`), and
    r_hat(x_t, a_t) its prediction for the logged action
    (`predicted_reward_logged`).

    It is unbiased if either the propensities or the reward model are right.
    How much it gains over SNIPS depends on where the variance comes from: a
    good reward model removes the variance due to contexts, but not each
    binary like's own coin-flip noise. On synthetic bandits with yes/no
    rewards the gain is small (about 6% lower spread with a perfect model),
    and a poor reward model makes DR noisier than SNIPS. Same propensity floor
    and weight cap as `snips`.

    Applying it to VisualTaste's own logs needs `r_hat` for every card the new
    policy might have shown, which the simulator caches do not store; the
    estimator is validated on synthetic bandits (tests/test_propensity_dr.py).
    """
    arrays = [
        np.asarray(a, dtype=np.float64)
        for a in (
            rewards,
            logged_propensities,
            new_propensities,
            predicted_reward_logged,
            predicted_value_new,
        )
    ]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("rewards, propensities and predictions must all align")
    r, logged, proposed, r_hat, v_hat = arrays
    if len(r) == 0:
        return DrEstimate(float("nan"), float("nan"), 0, 0, 0)

    keep = np.isfinite(logged) & (logged >= MINIMUM_PROPENSITY)
    dropped = int((~keep).sum())
    r, logged, proposed, r_hat, v_hat = (a[keep] for a in (r, logged, proposed, r_hat, v_hat))
    if len(r) == 0:
        return DrEstimate(float("nan"), float("nan"), 0, dropped, 0)

    raw_weights = proposed / logged
    weights = np.minimum(raw_weights, weight_cap)
    clipped = int((raw_weights > weight_cap).sum())

    per_round = v_hat + weights * (r - r_hat)
    standard_error = float(per_round.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
    return DrEstimate(float(per_round.mean()), standard_error, len(r), dropped, clipped)
