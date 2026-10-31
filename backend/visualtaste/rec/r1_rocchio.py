"""R1: one vector that is "what Alen likes", minus "what he doesn't".

Intuition
---------
If every image is a point in a 768-dimensional space where similar pictures sit
close together, then a taste is a *direction* in that space. Rocchio builds it
in the most direct way imaginable: average the liked images, subtract some of
the average of the disliked ones, and score everything by how closely it points
the same way.

That is the whole model. It has no parameters to learn, no optimiser, and no
training loop. It works from about ten labels, which is why it is the rung
directly after cold start.

The formula (maths unit M2)
---------------------------
    u = alpha * weighted_mean(liked) - beta * mean(disliked)
    u_hat = u / ||u||
    score(x) = cos(u_hat, x) = u_hat . x        (x is already unit length)

with alpha = 1 and beta = 0.5 (brief §7.5).

**Why beta is smaller than alpha.** A like says "more like this", which is a
direction. A dislike says "not that" -- but "not that" does not point anywhere
in particular, because there are vastly more ways to be wrong than right. At
beta = 1 a handful of dislikes of very different images would drag the
preference vector towards whatever is left, which is nowhere useful. Half
weight lets dislikes push without steering.

**Why the liked mean is weighted but the disliked mean is not.** Likes carry
per-example weights (a save counts double, §7.5), and averaging with those
weights is exactly what "a save is worth two likes" means. Dislikes are drawn
from `strong_negatives`, which excludes skips -- a mean has no notion of weight,
so including ambiguous skips at full strength would distort the direction
rather than inform it.

Worked example (M2, asserted in `tests/test_r1_rocchio.py`)
-----------------------------------------------------------
Likes (1, 0) and (0.8, 0.6); dislike (0, 1); alpha = 1, beta = 0.5.

    mean(liked)    = ((1 + 0.8)/2, (0 + 0.6)/2) = (0.9, 0.3)
    mean(disliked) = (0, 1)
    u              = (0.9, 0.3) - 0.5 * (0, 1)  = (0.9, -0.2)
    ||u||          = sqrt(0.81 + 0.04) = 0.92195
    u_hat          = (0.976, -0.217)

and then:

    score((0.6, 0.8)) = 0.6(0.976) + 0.8(-0.217) = 0.412
    score((1, 0))     = 0.976
    score((0, 1))     = -0.217

Optional recency decay
----------------------
A taste can move. With `decay = 0.995`, an example `age` labels old is weighted
by `0.995^age`, so a label 139 swipes ago counts half as much as a fresh one
(0.995^139 = 0.5). Off by default: with a few hundred labels, down-weighting the
early ones mostly costs information, and the brief marks it optional.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from visualtaste.rec.base import Explanation, FeatureBundle, Recommender, require_fitted
from visualtaste.rec.labels import LabelSet

logger = logging.getLogger(__name__)

# Brief §7.5.
ALPHA = 1.0
BETA = 0.5
DEFAULT_DECAY = 1.0  # 1.0 means no decay; 0.995 is the brief's suggested value.

# Below this, a preference vector is built from too little to mean anything and
# the feed is better off still spreading across clusters.
MIN_LABELS = 5

# How many nearest liked images the explanation names (brief §7.6).
EXPLANATION_NEIGHBOURS = 3


class RocchioRecommender(Recommender):
    """A single preference vector, scored by cosine similarity."""

    rung = "r1"
    min_labels = MIN_LABELS

    def __init__(
        self,
        *,
        alpha: float = ALPHA,
        beta: float = BETA,
        decay: float = DEFAULT_DECAY,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

        self.preference: np.ndarray | None = None
        #: Liked vectors, kept so `explain` can name the nearest ones.
        self._liked_vectors: np.ndarray | None = None
        self._liked_ids: list[str] = []

    # --- Fitting -------------------------------------------------------------

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        liked_ids = labels.positives
        disliked_ids = labels.strong_negatives

        liked = features.rows("siglip", liked_ids)
        disliked = features.rows("siglip", disliked_ids)

        if len(liked) == 0:
            # Nothing positive to point at. A vector built only from dislikes
            # would score "anything unlike what he rejected" highest, which is
            # not a taste -- so refuse rather than invent one.
            raise ValueError(
                "Rocchio needs at least one liked image; it cannot build a "
                "preference vector from dislikes alone."
            )

        weights = self._example_weights(labels, liked_ids)
        liked_mean = np.average(liked, axis=0, weights=weights)

        vector = self.alpha * liked_mean
        if len(disliked):
            vector = vector - self.beta * disliked.mean(axis=0)

        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            # The likes and dislikes cancelled exactly. Vanishingly unlikely
            # with real data, but a zero vector would make every score 0 and
            # the feed would look broken rather than uncertain.
            raise ValueError(
                "the liked and disliked means cancelled; no preference "
                "direction can be formed from these labels"
            )

        self.preference = (vector / norm).astype(np.float32)
        self._liked_vectors = liked
        self._liked_ids = [i for i in liked_ids if features.position(i) is not None]

        logger.info(
            "r1: preference vector from %d liked and %d disliked images",
            len(liked),
            len(disliked),
        )

    def _example_weights(self, labels: LabelSet, liked_ids: list[str]) -> np.ndarray:
        """Per-like weights, combining the label weight with optional decay.

        The label weight is what makes a save count double. The decay term, if
        enabled, discounts older examples -- `age` counts backwards from the
        most recent label, so the newest has age 0 and weight 1.
        """
        by_id = dict(zip(labels.image_ids, labels.weights, strict=True))
        positions = {image_id: index for index, image_id in enumerate(labels.image_ids)}
        newest = len(labels.image_ids) - 1

        weights = []
        for image_id in liked_ids:
            weight = float(by_id.get(image_id, 1.0))
            if self.decay < 1.0:
                age = newest - positions.get(image_id, newest)
                weight *= self.decay**age
            weights.append(weight)

        array = np.asarray(weights, dtype=np.float64)
        # `np.average` rejects weights summing to zero; a floor keeps a
        # fully-decayed history usable rather than fatal.
        return array if array.sum() > 0 else np.ones_like(array)

    # --- Scoring -------------------------------------------------------------

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """Cosine similarity between each candidate and the preference vector.

        Note what this does *not* do: it never builds `features.siglip
        [candidates]`. Scoring every row and then discarding the few that are
        not candidates is faster than selecting the candidates first, by a
        factor of eight, and the reason is worth understanding because it
        recurs everywhere in this project.

        Measured on the 50,000 x 768 pool (2026-10-31, `docs/BENCHMARKS.md`):

            siglip[candidates] @ u      58.5 ms
            (siglip @ u)[candidates]     7.0 ms
            siglip[candidates] alone    93.3 ms   <- the whole cost

        The fancy index is a *copy*: it allocates 153 MB and moves every byte of
        it, scattered, before any arithmetic happens. The dot product itself
        touches the same 153 MB once, sequentially, inside BLAS. So the version
        that does strictly more arithmetic wins, because this operation is
        bound by memory bandwidth rather than by floating-point throughput --
        38 million multiply-adds is nothing for a modern CPU, and 153 MB of
        gather is not.

        The wasted work is proportional to the number of images already seen, a
        few thousand at most, so it never grows into a problem.
        """
        require_fitted(self)
        assert self.preference is not None
        return (features.siglip @ self.preference).astype(np.float32)[candidates]

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        """Name the liked images this one is closest to (reason `similar_likes`).

        The three nearest liked images are the honest answer to "why this one":
        the score *is* similarity to their weighted mean, so the nearest
        contributors are what actually produced it.
        """
        require_fitted(self)
        assert self.preference is not None

        vector = features.siglip[image_index]
        score = float(vector @ self.preference)

        neighbours: list[str] = []
        if self._liked_vectors is not None and len(self._liked_vectors):
            similarity = self._liked_vectors @ vector
            order = np.argsort(-similarity)[:EXPLANATION_NEIGHBOURS]
            neighbours = [self._liked_ids[int(i)] for i in order if int(i) < len(self._liked_ids)]

        return Explanation(
            reason="similar_likes",
            detail={
                "score": round(score, 4),
                "neighbours": neighbours,
                "liked_count": len(self._liked_ids),
            },
        )

    # --- Persistence ---------------------------------------------------------

    @property
    def params(self) -> dict[str, object]:
        return {
            "rung": self.rung,
            "min_labels": self.min_labels,
            "alpha": self.alpha,
            "beta": self.beta,
            "decay": self.decay,
        }

    def _save(self, path: Path) -> None:
        if self.preference is None:
            return
        with (path / "preference.npy").open("wb") as handle:
            np.save(handle, self.preference)
        if self._liked_vectors is not None:
            with (path / "liked.npy").open("wb") as handle:
                np.save(handle, self._liked_vectors)
        (path / "liked_ids.json").write_text(json.dumps(self._liked_ids), encoding="utf-8")

    def _load(self, path: Path) -> None:
        preference_path = path / "preference.npy"
        if preference_path.exists():
            self.preference = np.load(preference_path)
        liked_path = path / "liked.npy"
        if liked_path.exists():
            self._liked_vectors = np.load(liked_path)
        ids_path = path / "liked_ids.json"
        if ids_path.exists():
            self._liked_ids = json.loads(ids_path.read_text(encoding="utf-8"))


def rocchio_vector(
    liked: np.ndarray,
    disliked: np.ndarray,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """The Rocchio formula on its own, for tests and notebooks.

    Separated from the class so maths unit M2 can be checked against exactly
    the arithmetic in the docstring, with no object construction in the way.
    """
    liked = np.asarray(liked, dtype=np.float64)
    disliked = np.asarray(disliked, dtype=np.float64)

    if len(liked) == 0:
        raise ValueError("Rocchio needs at least one liked vector")

    liked_mean = np.average(liked, axis=0, weights=weights)
    vector = alpha * liked_mean
    if len(disliked):
        vector = vector - beta * disliked.mean(axis=0)

    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("the liked and disliked means cancelled exactly")
    return vector / norm
