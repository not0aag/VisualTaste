"""R4: R3 with a sense of its own uncertainty, used to explore (Thompson sampling).

Intuition
---------
R3 gives one weight vector, its best guess. But with 200 labels many weight
vectors explain the history almost equally well, and they disagree most about
images unlike anything rated so far. Thompson sampling uses that disagreement:
for each batch, draw one plausible weight vector from the posterior and serve
what *that* vector likes best. Where the model is sure, every draw agrees and
the feed exploits. Where it is unsure, different batches favour different
regions, so the feed explores exactly where it is ignorant, in proportion to
how ignorant it is.

The posterior: a Laplace approximation (Part 2 §6.1)
----------------------------------------------------
Approximate the posterior over w by a Gaussian centred at R3's solution w*,
with precision (inverse covariance) equal to the curvature of the loss there.
In the summed form of R3's objective, the L2 term is a Gaussian prior with
precision alpha = lam * S, so

    H = alpha I + sum_i s_i p_i (1 - p_i) x_i x_i^T        the precision matrix

Each labelled example adds curvature s_i p_i (1 - p_i) along its own feature
direction: most where the model was unsure (p near 0.5), least where it was
confident. As labels accumulate, H grows, its inverse shrinks, and the draws
tighten around w*. That is "the model becomes sure".

Sampling without inverting H
----------------------------
Factor H = L L^T (Cholesky; L lower-triangular). Draw z ~ N(0, I) and solve

    w~ = w* + sqrt(tau_post) * solve(L^T, z)

Then w~ - w* has covariance (L^T)^-1 L^-1 = (L L^T)^-1 = H^-1, as required.
`tests/test_thompson.py` checks the sample covariance against H^-1 directly.
Solving a triangular system costs O(d^2) and is numerically stable, where
forming H^-1 would cost O(d^3) and lose precision.

`tau_post` is a temperature (default 1): above 1 explores more, below 1 less.
The bias stays at b*. Adding the same number to every image's logit cannot
change their order, so sampling it would add noise to `sampled_score` and
nothing to the ranking.

Worked example (M7, asserted in `tests/test_thompson.py`)
---------------------------------------------------------
Image A: predicted like probability mean 0.60, sd 0.05. Image B: mean 0.50, sd
0.30. A greedy ranker always shows A. Under Thompson sampling, B wins a draw
when its sample exceeds A's:

    P(B > A) = Phi((0.50 - 0.60) / sqrt(0.05^2 + 0.30^2)) = Phi(-0.329) = 0.37

so B, the image the model knows little about, is shown about 37% of the time.

The batch (Part 2 §6.2)
-----------------------
1. one posterior draw, then sampled probabilities for the whole pool (one
   matrix product, F2);
2. the top 500 by sampled probability;
3. MMR (lam 0.7) over those, for four cards;
4. the exploration slot (`rec/rerank.py`);
5. a seeded shuffle, so the probe is not always in the same position. Its
   pre-shuffle slot is kept in the explanation.

Propensities
------------
The probability that the policy would have shown an image is needed for
inverse-propensity estimates (Phase 10 IPS check), and here it has no closed
form. `inclusion_probabilities` estimates it by re-running steps 1-4 for 200
posterior draws at once and counting how often each shown image made the
batch. That is a Monte-Carlo estimate, accurate to about +-0.035 at p = 0.5
with 200 draws. It runs after the response is sent, never on the request path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import expit

from visualtaste.rec.base import Explanation, FeatureBundle, require_fitted
from visualtaste.rec.r3_logreg import LogisticRecommender
from visualtaste.rec.rerank import (
    EXPLORE_EVERY,
    MMR_LAMBDA,
    SHORTLIST,
    cluster_ratings_from,
    explore_batch,
    explore_slots,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from visualtaste.rec.labels import LabelSet

logger = logging.getLogger(__name__)

PROPENSITY_DRAWS = 200
PROPENSITY_FLOOR = 0.005


def precision_matrix(
    X: np.ndarray, s: np.ndarray, w: np.ndarray, b: float, lam: float
) -> np.ndarray:
    """H = (lam S) I + sum_i s_i p_i (1 - p_i) x_i x_i^T, at the MAP solution."""
    p = expit(X @ w + b)
    curvature = s * p * (1.0 - p)
    alpha = lam * float(s.sum())
    return alpha * np.eye(X.shape[1]) + (X * curvature[:, None]).T @ X


def sample_weights(
    w: np.ndarray, L: np.ndarray, rng: np.random.Generator, *, draws: int = 1, tau: float = 1.0
) -> np.ndarray:
    """`draws` samples from N(w, tau H^-1), with H = L L^T. Returns (d, draws)."""
    z = rng.standard_normal((len(w), draws))
    deviation = solve_triangular(L.T, z, lower=False)
    return w[:, None] + np.sqrt(tau) * deviation


class ThompsonRecommender(LogisticRecommender):
    """R4: R3's model, a Laplace posterior over it, and an exploring batch builder."""

    rung = "r4"
    #: Its probabilities are sampled, so propensities are estimated afterwards
    #: rather than set to 1 (rec/feed.py).
    stochastic = True

    def __init__(
        self,
        *,
        tau_post: float = 1.0,
        mmr_lambda: float = MMR_LAMBDA,
        explore_every: int | None = EXPLORE_EVERY,
        seed: int = 1729,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tau_post = tau_post
        self.mmr_lambda = mmr_lambda
        self.explore_every = explore_every
        self.seed = seed

        self.L: np.ndarray | None = None  # Cholesky factor of H, lower-triangular
        #: cluster -> (likes, dislikes) at fit time, for the exploration quota.
        self.cluster_ratings: dict[int, tuple[int, int]] = {}
        self.labels_at_fit = 0
        self.batches_served = 0
        self.cards_served = 0
        #: Pool rows of the last batch's exploration cards, for `explain`.
        self._explore_rows: dict[int, dict[str, Any]] = {}
        #: (batch size, exploration cards) of the last batch, which the
        #: propensity estimate must replay with the same shape.
        self.last_batch: tuple[int, int] = (0, 0)

    # --- Fitting -------------------------------------------------------------

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        super()._fit(labels, features)
        assert self.w is not None and self.lam is not None
        assert self._train_X is not None and self._train_s is not None

        # The curvature of exactly the matrix R3 was fitted on, interest
        # columns included for R4+R5.
        H = precision_matrix(self._train_X, self._train_s, self.w, self.b, self.lam)
        self.L = cholesky(H, lower=True)

        self.cluster_ratings = cluster_ratings_from(labels, features)
        self.labels_at_fit = len(labels)

    # --- The batch -------------------------------------------------------------

    def _batch_rng(self) -> np.random.Generator:
        """A generator per batch: reproducible from (seed, batch number)."""
        return np.random.default_rng((self.seed, self.batches_served))

    def sampled_probabilities(self, features: FeatureBundle, W: np.ndarray) -> np.ndarray:
        """sigma(X w~ + b*) for every pool row, for each column of W: (pool, draws)."""
        return expit(self.pool_logits(features, W))

    def _one_batch(
        self,
        features: FeatureBundle,
        candidates: np.ndarray,
        sampled: np.ndarray,
        n: int,
        n_explore: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[int]]:
        """Steps 2-4 for one draw. Returns (mmr offsets, explore offsets) into candidates."""
        return explore_batch(
            features,
            candidates,
            sampled,
            n,
            n_explore,
            rng,
            cluster_ratings=self.cluster_ratings,
            mmr_lambda=self.mmr_lambda,
        )

    def rerank(
        self, features: FeatureBundle, candidates: np.ndarray, scores: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray, None]:
        """The feed's re-rank hook: one Thompson batch (steps 1-5)."""
        require_fitted(self)
        assert self.w is not None and self.L is not None
        rng = self._batch_rng()

        w_tilde = sample_weights(self.w, self.L, rng, tau=self.tau_post)
        sampled = self.sampled_probabilities(features, w_tilde)[:, 0][candidates]

        n_explore = explore_slots(
            self.cards_served, n, self.labels_at_fit, every=self.explore_every
        )
        mmr, explore = self._one_batch(features, candidates, sampled, n, n_explore, rng)

        slots = mmr + explore
        order = rng.permutation(len(slots))
        chosen = np.array([slots[i] for i in order], dtype=np.int64)

        self._explore_rows = {}
        for pre_shuffle_slot, offset in enumerate(slots):
            if offset in explore:
                row = int(candidates[offset])
                cluster = int(np.asarray(features.cluster64)[row])
                self._explore_rows[row] = {
                    "cluster64": cluster,
                    "ratings": list(self.cluster_ratings.get(cluster, (0, 0))),
                    "slot_before_shuffle": pre_shuffle_slot,
                }

        self.batches_served += 1
        self.cards_served += len(chosen)
        self.last_batch = (n, n_explore)
        return chosen, sampled.astype(np.float32), None

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        probe = self._explore_rows.get(int(image_index))
        if probe is not None:
            return Explanation(reason="explore", detail=dict(probe))
        return super().explain(features, image_index)

    # --- Propensities, off the request path ---------------------------------

    def inclusion_probabilities(
        self,
        features: FeatureBundle,
        candidates: np.ndarray,
        shown: Sequence[int],
        *,
        n: int,
        n_explore: int,
        draws: int = PROPENSITY_DRAWS,
        seed: int = 0,
    ) -> np.ndarray:
        """Monte-Carlo P(each shown image is in the batch), over `draws` posterior draws.

        `shown` are offsets into `candidates`. All draws share one matrix
        product with the pool, (pool x d) @ (d x draws); then each draw runs its
        own shortlist, MMR and exploration choice, exactly as `rerank` does.
        """
        require_fitted(self)
        assert self.w is not None and self.L is not None
        rng = np.random.default_rng((self.seed, seed, 1))
        W = sample_weights(self.w, self.L, rng, draws=draws, tau=self.tau_post)
        sampled = self.sampled_probabilities(features, W)[candidates]

        shown = [int(s) for s in shown]
        counts = np.zeros(len(shown))
        for j in range(draws):
            mmr, explore = self._one_batch(features, candidates, sampled[:, j], n, n_explore, rng)
            batch = set(mmr) | set(explore)
            counts += [offset in batch for offset in shown]
        return counts / draws

    # --- Reporting and persistence -------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        return {
            **super().params,
            "rung": self.rung,
            "tau_post": self.tau_post,
            "mmr_lambda": self.mmr_lambda,
            "explore_every": self.explore_every,
            "shortlist": SHORTLIST,
            "seed": self.seed,
        }

    def _save(self, path: Path) -> None:
        super()._save(path)
        if self.L is None:
            return
        with (path / "cholesky.npy").open("wb") as handle:
            np.save(handle, self.L)
        (path / "exploration.json").write_text(
            json.dumps(
                {
                    "cluster_ratings": {str(k): v for k, v in self.cluster_ratings.items()},
                    "labels_at_fit": self.labels_at_fit,
                }
            ),
            encoding="utf-8",
        )

    def _load(self, path: Path) -> None:
        super()._load(path)
        if not self.is_fitted:
            return
        self.L = np.load(path / "cholesky.npy")
        meta = json.loads((path / "exploration.json").read_text(encoding="utf-8"))
        self.cluster_ratings = {int(k): tuple(v) for k, v in meta["cluster_ratings"].items()}
        self.labels_at_fit = int(meta["labels_at_fit"])
