"""A fake user with a known taste, so the machinery can be tested before Alen uses it.

The problem this solves
-----------------------
Nothing in the recommendation ladder can be validated without labels, and the
only source of real labels is Alen swiping. That is a chicken-and-egg problem
with a long feedback loop: ship R3, wait for him to rate 150 images, discover
the feature pipeline was misaligned.

So we invent a user whose preferences we already know, let the models try to
discover them, and check whether they do.

How the simulated user works (brief §7.7)
-----------------------------------------
Three hidden interest centroids, drawn from the real `cluster64` centroids so
that the planted taste corresponds to actual regions of the real corpus rather
than to arbitrary directions in a 768-dimensional space.

    p(like | x) = sigma( 10 * ( max_k cos(c_k, x) - 0.25 ) )

Reading that: an image whose best match to any interest has cosine 0.25 is a
coin flip. The slope of 10 means cosine 0.35 is already 73% likely and cosine
0.15 is 27% -- a soft boundary, not a step, so the labels contain genuine noise
and a model cannot reach AUC 1.0 by luck.

On top of that:

- an image with p > 0.9 is **saved** rather than liked, with probability 0.3;
- any image is **skipped** with probability 0.1, regardless of p, modelling
  inattention -- which is what makes the skip-weight question in Phase 9 a real
  question rather than a formality;
- interests can **drift** over time, if asked, to test whether a model that has
  settled can still move.

What this can and cannot establish
----------------------------------
**Can:** that the feed serves, that labels land, that models train and improve,
that a rung which should recover a planted taste does recover it, and that the
metrics compute. Those are real failures it catches.

**Cannot:** which rung will serve Alen best. The truth here is "cosine to one of
three centroids", which is almost exactly the inductive bias of Rocchio and of
logistic regression on embeddings. R1 and R3 are being graded on an exam they
wrote. A neural ranker's advantage -- learning interactions between colour,
appearance and content that no cosine captures -- is invisible to a simulator
whose truth has no such interactions.

That asymmetry is not a flaw to be fixed by a better simulator; it is why
`docs/EVALUATION.md` leads with the warning, and why the brief asks for the
hypotheses to be reported honestly rather than confirmed.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from visualtaste.eval.metrics import sigmoid

logger = logging.getLogger(__name__)

# Brief §7.7.
INTEREST_COUNT = 3
LOGIT_SLOPE = 10.0
LOGIT_OFFSET = 0.25
SAVE_PROBABILITY = 0.3
SAVE_THRESHOLD = 0.9
SKIP_PROBABILITY = 0.1


@dataclass
class SimulatedUser:
    """A user whose taste is three hidden directions in embedding space.

    `centroids` are L2-normalised, so `cos(c, x)` is a dot product.
    """

    centroids: np.ndarray
    rng: np.random.Generator
    drift_per_step: float = 0.0
    steps_taken: int = 0

    #: Per-interest count of how many images it has been the best match for.
    #: Used by `recall_of_planted_interests` to say whether a model found all
    #: three or settled on one.
    interest_hits: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.centroids = _normalise(np.asarray(self.centroids, dtype=np.float32))

    @property
    def interest_count(self) -> int:
        return len(self.centroids)

    def affinity(self, vectors: np.ndarray) -> np.ndarray:
        """`max_k cos(c_k, x)` for each row, plus which interest won.

        Returns the best cosine only; `best_interest` gives the index.
        """
        return self._similarities(vectors).max(axis=1)

    def best_interest(self, vectors: np.ndarray) -> np.ndarray:
        return self._similarities(vectors).argmax(axis=1)

    def _similarities(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors @ self.centroids.T

    def like_probability(self, vectors: np.ndarray) -> np.ndarray:
        """p(like | x) = sigma(10 (max_k cos(c_k, x) - 0.25))."""
        return sigmoid(LOGIT_SLOPE * (self.affinity(vectors) - LOGIT_OFFSET))

    def react(self, vectors: np.ndarray) -> list[str]:
        """Draw an action for each image: like, save, dislike or skip.

        The order of the three rules matters and is worth stating. The skip
        check comes *first*, so a skip genuinely carries no information about
        whether the image was liked -- that is what makes it an honest model of
        inattention, and what makes "should skips be labelled 0 at weight 0.25"
        a question the simulator can actually answer.
        """
        probabilities = self.like_probability(vectors)
        draws = self.rng.random(len(probabilities))
        skip_draws = self.rng.random(len(probabilities))
        save_draws = self.rng.random(len(probabilities))

        winners = self.best_interest(vectors)
        actions: list[str] = []

        for index, probability in enumerate(probabilities):
            if skip_draws[index] < SKIP_PROBABILITY:
                actions.append("skip")
                continue

            if draws[index] < probability:
                if probability > SAVE_THRESHOLD and save_draws[index] < SAVE_PROBABILITY:
                    actions.append("save")
                else:
                    actions.append("like")
                interest = int(winners[index])
                self.interest_hits[interest] = self.interest_hits.get(interest, 0) + 1
            else:
                actions.append("dislike")

        self.steps_taken += len(actions)
        if self.drift_per_step:
            self._drift(len(actions))
        return actions

    def _drift(self, steps: int) -> None:
        """Nudge the centroids in a fixed random direction and renormalise.

        A *fixed* direction per interest rather than a fresh random walk each
        step: a random walk of mean zero mostly cancels out and the taste would
        not actually move anywhere. This models a preference genuinely
        shifting, which is the thing worth testing a model's ability to follow.
        """
        if not hasattr(self, "_drift_directions"):
            directions = self.rng.normal(size=self.centroids.shape).astype(np.float32)
            self._drift_directions = _normalise(directions)
        self.centroids = _normalise(
            self.centroids + self.drift_per_step * steps * self._drift_directions
        )

    def recall_of_planted_interests(self, minimum_hits: int = 3) -> float:
        """Share of the planted interests that produced at least some likes.

        The headline question for R4 (brief §7.7): does exploration find *all
        three* interests, or does the model lock onto the first one it finds and
        never discover the others? A model that only ever serves images from
        one centroid scores 1/3 here while possibly having a perfectly good like
        rate -- which is exactly the failure exploration exists to prevent.
        """
        found = sum(1 for count in self.interest_hits.values() if count >= minimum_hits)
        return found / self.interest_count


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def plant_interests(
    vectors: np.ndarray,
    *,
    seed: int,
    count: int = INTEREST_COUNT,
    cluster_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Choose `count` hidden interest centroids from the corpus itself.

    Drawn from real `cluster64` centroids where available, so the planted taste
    is for things the corpus actually contains -- "Japanese woodblock prints"
    rather than a direction that no image points in. A taste nothing satisfies
    would make every rung look equally bad.

    The chosen clusters are forced apart: three interests that were all the
    same kind of image would make "did it find all three" meaningless.
    """
    rng = np.random.default_rng(seed)
    vectors = np.asarray(vectors, dtype=np.float32)

    if cluster_labels is None:
        # No clusters available (a synthetic test corpus): fall back to
        # individual images as centroids.
        chosen = rng.choice(len(vectors), size=min(count, len(vectors)), replace=False)
        return _normalise(vectors[chosen])

    labels = np.asarray(cluster_labels)
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        chosen = rng.choice(len(vectors), size=min(count, len(vectors)), replace=False)
        return _normalise(vectors[chosen])

    centroids = _normalise(np.stack([vectors[labels == label].mean(axis=0) for label in unique]))

    # Greedy farthest-point selection: start at random, then repeatedly take
    # the cluster least similar to everything already chosen.
    picked = [int(rng.integers(len(centroids)))]
    while len(picked) < min(count, len(centroids)):
        similarity_to_picked = (centroids @ centroids[picked].T).max(axis=1)
        similarity_to_picked[picked] = np.inf
        picked.append(int(np.argmin(similarity_to_picked)))

    logger.info(
        "planted %d interests from clusters %s (pairwise cosine at most %.3f)",
        len(picked),
        [int(unique[i]) for i in picked],
        float((centroids[picked] @ centroids[picked].T)[np.triu_indices(len(picked), k=1)].max())
        if len(picked) > 1
        else 0.0,
    )
    return centroids[picked]


def make_user(
    vectors: np.ndarray,
    *,
    seed: int,
    cluster_labels: np.ndarray | None = None,
    drift_per_step: float = 0.0,
) -> SimulatedUser:
    """Build a simulated user with interests planted in this corpus."""
    return SimulatedUser(
        centroids=plant_interests(vectors, seed=seed, cluster_labels=cluster_labels),
        rng=np.random.default_rng(seed + 1),
        drift_per_step=drift_per_step,
    )


def create_simulated_profile(connection: sqlite3.Connection, *, note: str) -> int:
    """A profile flagged `is_simulated = 1`.

    The flag is load-bearing, not documentation. `repo.active_profile_id`
    ignores simulated profiles, so a simulator run can never become the profile
    the app serves, and its events can never be mixed into Alen's training data
    or dashboards by default (brief §7.4).
    """
    from visualtaste.db import repo

    return repo.create_profile(connection, note=note, is_simulated=True)
