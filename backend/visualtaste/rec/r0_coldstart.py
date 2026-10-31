"""R0: the first thirty images, chosen without knowing anything.

The problem
-----------
At the very start there are no labels, so there is nothing to learn from and
nothing to rank by. Something still has to choose what to show, and that choice
determines what every later model gets to learn from -- which makes it more
consequential than it looks.

Why not just show random images
-------------------------------
Uniform random sampling from 50,000 images sounds neutral and is not. The
corpus is not uniform: some regions of it are much denser than others, so a
random sample over-represents whatever there is most of. Thirty random images
might be twenty landscapes, and the first preference vector would then be built
almost entirely from opinions about landscapes.

What R0 does instead
--------------------
Round-robin over the 64 `cluster64` clusters, taking one random unseen image
from each in turn. In thirty swipes that means thirty *different kinds* of
image. The resulting label set covers the space rather than the density, which
is the most informative thirty labels available before anything is known.

This is the cheapest possible form of the same idea R4 spends a Bayesian
posterior on: spend early interactions on learning where the interesting
regions are, not on exploiting a preference you do not yet have.

It learns nothing
-----------------
R0 has no parameters and ignores the labels it collects. Its `score` is a
round-robin position, and its only state is which clusters have been used in
the current batch. At 30 labels the feed hands over to R1.
"""

from __future__ import annotations

import logging

import numpy as np

from visualtaste.rec.base import Explanation, FeatureBundle, Recommender

logger = logging.getLogger(__name__)

# Brief §7.5: R0 runs until there are this many labels.
COLDSTART_LABEL_TARGET = 30


class ColdStartRecommender(Recommender):
    """Round-robin over `cluster64`, one random unseen image per cluster.

    `seed` makes the choice reproducible, which matters for the simulator study:
    comparing R1 against R2 is only meaningful if both started from the same
    thirty cold-start labels.
    """

    rung = "r0"
    min_labels = 0  # It is the rung that exists precisely when there are none.

    def __init__(self, *, seed: int = 1729) -> None:
        super().__init__()
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # How many times each cluster has been drawn from, so the round-robin
        # survives across requests rather than restarting every batch.
        self._draws_per_cluster: dict[int, int] = {}

    def _fit(self, labels, features) -> None:
        """R0 has nothing to learn. Present because the interface requires it."""

    def fit(self, labels, features) -> ColdStartRecommender:
        """Always `is_fitted`: a rule with no parameters is always ready."""
        self.trained_on_events = len(labels)
        self.is_fitted = True
        return self

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """Score so that one image per least-used cluster comes out on top.

        The score has two parts:

        - the **cluster's turn**: candidates in clusters drawn from least often
          score highest, which is the round-robin;
        - a **random tiebreak** within a cluster, so the particular image is an
          arbitrary choice rather than always the lowest id.

        Expressed as a score rather than as a selection loop so that R0 fits the
        same interface as every other rung, and so MMR and the exploration quota
        in `rec/rerank.py` apply to it unchanged.
        """
        if features.cluster64 is None:
            # Before `jobs.select` has run there are no clusters, so the honest
            # fallback is random -- with a warning, because a feed quietly
            # serving random images looks the same as one that is working.
            logger.warning(
                "no cluster64 assignments available; R0 is falling back to "
                "random order. Run `python -m visualtaste.jobs.select`."
            )
            return self._rng.random(len(candidates)).astype(np.float32)

        clusters = np.asarray(features.cluster64)[candidates]
        draws = np.array(
            [self._draws_per_cluster.get(int(c), 0) for c in clusters], dtype=np.float32
        )

        # Negated so fewer draws means a higher score. The random term is scaled
        # to stay well below one full draw, so it only ever breaks ties *within*
        # a cluster and never reorders clusters.
        tiebreak = self._rng.random(len(candidates)).astype(np.float32) * 0.5
        return -draws + tiebreak

    def note_shown(self, cluster: int | None) -> None:
        """Record that an image from this cluster was served.

        Called by the feed for each card it sends. Without this the round-robin
        would restart on every request and the first few clusters would be shown
        over and over.
        """
        if cluster is None:
            return
        key = int(cluster)
        self._draws_per_cluster[key] = self._draws_per_cluster.get(key, 0) + 1

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        cluster = None
        if features.cluster64 is not None:
            cluster = int(np.asarray(features.cluster64)[image_index])

        return Explanation(
            reason="cold_start",
            sentence=(
                "Still learning what you like -- this is one of a spread of "
                "different kinds of image, chosen to cover the collection."
            ),
            detail={"cluster64": cluster, "draws_so_far": sum(self._draws_per_cluster.values())},
        )

    @property
    def params(self) -> dict[str, object]:
        return {
            "rung": self.rung,
            "min_labels": self.min_labels,
            "seed": self.seed,
            "label_target": COLDSTART_LABEL_TARGET,
        }


def coldstart_is_active(label_count: int) -> bool:
    """Whether the feed should still be using R0.

    One function rather than an inline comparison, because the feed, the Model
    Lab and the simulator all need to agree on the answer.
    """
    return label_count < COLDSTART_LABEL_TARGET
