"""R2: judge an image by the labelled images it most resembles.

Intuition, and what it fixes about R1
-------------------------------------
Rocchio compresses a taste into one direction. That is a real limitation, and
it is easiest to see with an example: suppose Alen likes both stark black-and-
white architecture and lush botanical illustration. Their mean points
*somewhere between* the two -- at nothing in particular, probably a grey plant.
A single vector cannot represent two tastes.

kNN does not compress at all. It keeps every labelled image and asks a local
question:

    score(x) = mean of the top-k cosines to liked images
             - gamma * mean of the top-k cosines to disliked images

with k = 5 and gamma = 0.5 (brief §7.5).

Because the comparison is to the *nearest* likes rather than to their average,
an image near either of two separate interests scores well, and an image in the
middle of both scores badly. That is the right answer, and R1 cannot give it.

Why top-k rather than the single nearest
----------------------------------------
k = 1 would make the score a maximum, and a maximum over a few hundred labels
is noisy: one unusual like creates a small region of high scores around itself
that may reflect a mood rather than a taste. Averaging the five nearest asks
"is this in a neighbourhood he liked" instead of "is this near one thing he
liked once", which is a more stable question and the one worth answering.

Why gamma is still less than one
--------------------------------
Same reason as Rocchio's beta. "Near things he rejected" is weaker evidence
than "near things he liked", because rejection has many more causes -- the
image was dull, or he had already seen three like it, or he was moving fast.

One consequence worth knowing
-----------------------------
A cosine can be negative, so the dislike term does not only subtract. An image
pointing *away* from everything he rejected gets a positive contribution from
`- gamma * (negative number)`. In effect, "far from what he rejected" counts as
mild evidence in favour.

In a space where opposite directions mean opposite content that is defensible,
and gamma bounds how much it can matter. But it is not what the one-line
description of kNN suggests, and it is the explanation for an occasional
surprise: a bland image far from every dislike can out-rank a vivid one that
sits near both a like and a dislike. `tests/test_r1_r2.py` asserts the
behaviour so it stays a known property rather than a mystery.

What this rung costs, which turned out to matter
------------------------------------------------
Nothing to train: `fit` is bookkeeping. Scoring, though, is the most expensive
rung on the ladder, because its cost is `pool x labels` rather than `pool`:
every candidate must be compared against every label before the top five can be
averaged. Measured on the real 50,000 x 768 pool (2026-10-31):

    labels     NumPy matmul + partition     FAISS IndexFlatIP
        50                     64 ms                   32 ms
       100                     94 ms                   46 ms
       200                    165 ms                   65 ms
       500                    323 ms                  145 ms
     1,000                    608 ms                  268 ms
     2,000                  1,181 ms                  536 ms

The two agree to 4e-8, so FAISS is the same computation done better: it blocks
the multiply and keeps a running top-k heap instead of materialising a
50,000 x 1,000 similarity matrix and sorting it. The brief asks for this rung
"via FAISS" (§7.5) and this is why.

It is still over budget. `/feed/next` is allowed a 50 ms p95, which FAISS buys
up to roughly 100 labels and no further -- so R2 is honest to serve for the
first few dozen swipes and becomes unservable at about the point it gets good.
That is reported rather than hidden (`docs/EVALUATION.md` and
`docs/BENCHMARKS.md` carry the numbers), and it is a real argument
for the rung above: R3 scores on 128 PCA dimensions with a single weight
vector, so its cost is constant in the number of labels and about a millisecond.

A kNN score *can* be made cheap -- shortlist candidates from each like's
neighbours, or move the multiply to the GPU -- but both change what is being
computed or add machinery the brief does not ask for, so neither is here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from visualtaste.rec.base import Explanation, FeatureBundle, Recommender, require_fitted
from visualtaste.rec.labels import LabelSet
from visualtaste.search.index import VectorIndex

logger = logging.getLogger(__name__)

# Brief §7.5.
NEIGHBOURS = 5
GAMMA = 0.5

# Below this there are not enough labels for a top-5 mean to differ from a
# top-everything mean, so the rung has nothing to add over R1.
MIN_LABELS = 10

EXPLANATION_NEIGHBOURS = 3


def top_k_mean(similarity: np.ndarray, k: int) -> np.ndarray:
    """Mean of the `k` largest values along the last axis.

    `argpartition` finds the top k without sorting the rest, which matters
    because the array here is (candidates x labels) and sorting 50,000 rows of
    500 would dominate the request.

    With fewer than k labels, every label is used -- the honest behaviour, and
    what makes the rung usable from its tenth label rather than its fiftieth.
    """
    if similarity.shape[-1] == 0:
        return np.zeros(similarity.shape[:-1], dtype=np.float32)

    k = min(k, similarity.shape[-1])
    partitioned = np.partition(similarity, -k, axis=-1)[..., -k:]
    return partitioned.mean(axis=-1).astype(np.float32)


class KnnRecommender(Recommender):
    """Scores by similarity to the nearest liked and disliked images."""

    rung = "r2"
    min_labels = MIN_LABELS

    def __init__(self, *, neighbours: int = NEIGHBOURS, gamma: float = GAMMA) -> None:
        super().__init__()
        self.neighbours = neighbours
        self.gamma = gamma

        self._liked: np.ndarray | None = None
        self._disliked: np.ndarray | None = None
        self._liked_ids: list[str] = []

        # FAISS indexes over the *labelled* vectors, not the corpus: the query
        # side is the 50,000-image pool and the indexed side is a few hundred
        # labels. Rebuilt whenever the vectors change, which is cheap -- an
        # IndexFlatIP is a copy of the vectors plus a dimension check.
        self._liked_index: VectorIndex | None = None
        self._disliked_index: VectorIndex | None = None

        # The whole-pool score depends only on the held vectors, k, gamma and
        # the pool, so it is computed once per fit and reused by every batch
        # until the next one. The feed asks for a batch of 5 and refits every
        # 20 labels, so this removes 3 of every 4 pool searches. Cleared in
        # `_rebuild_indexes`, the one place the held vectors change.
        self._pool_scores: tuple[tuple, np.ndarray, np.ndarray] | None = None

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        liked_ids = [i for i in labels.positives if features.position(i) is not None]
        self._liked = features.rows("siglip", labels.positives)
        self._disliked = features.rows("siglip", labels.strong_negatives)
        self._liked_ids = liked_ids

        if len(self._liked) == 0:
            raise ValueError(
                "kNN needs at least one liked image; similarity to dislikes "
                "alone gives every candidate the same negative score."
            )

        self._rebuild_indexes()
        logger.info(
            "r2: holding %d liked and %d disliked vectors (k=%d, gamma=%.2f)",
            len(self._liked),
            len(self._disliked),
            self.neighbours,
            self.gamma,
        )

    def _rebuild_indexes(self) -> None:
        """Index whatever vectors are currently held. Safe to call repeatedly."""
        self._pool_scores = None
        self._liked_index = (
            VectorIndex(self._liked, self._liked_ids or [""] * len(self._liked))
            if self._liked is not None and len(self._liked)
            else None
        )
        self._disliked_index = (
            VectorIndex(self._disliked, [""] * len(self._disliked))
            if self._disliked is not None and len(self._disliked)
            else None
        )

    def _top_k_similarity(self, index: VectorIndex | None, pool: np.ndarray) -> np.ndarray:
        """Mean of the `k` largest similarities between each pool row and `index`.

        `search_many` already returns the k largest, sorted, so the mean over
        its columns *is* the top-k mean -- and with fewer than k indexed
        vectors it returns all of them, which matches `top_k_mean`.
        """
        if index is None:
            return np.zeros(len(pool), dtype=np.float32)
        similarities, _ = index.search_many(pool, self.neighbours)
        if similarities.shape[1] == 0:
            return np.zeros(len(pool), dtype=np.float32)
        return similarities.mean(axis=1).astype(np.float32)

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        require_fitted(self)
        scores = self._score_pool(features.siglip)[candidates]
        # A slice would be a view of the cache; hand out a copy instead.
        return scores.copy() if np.shares_memory(scores, self._pool_scores[2]) else scores

    def _score_pool(self, vectors: np.ndarray) -> np.ndarray:
        """The score of every pool row, cached until the next fit (see __init__).

        Scored over every row and indexed afterwards, never gathered first:
        the gather costs more than the arithmetic. `r1_rocchio.score` has the
        measurements and the explanation.
        """
        # Keyed on the pool array itself (held by the cache, so its id cannot
        # be reused) and on the parameters that shape the score, so changing k
        # or gamma by hand also recomputes.
        key = (id(vectors), vectors.shape, self.neighbours, self.gamma)
        cached = self._pool_scores
        if cached is not None and cached[0] == key and cached[1] is vectors:
            return cached[2]

        pool = np.ascontiguousarray(vectors, dtype=np.float32)
        positive = self._top_k_similarity(self._liked_index, pool)
        if self._disliked_index is None:
            scores = positive
        else:
            negative = self._top_k_similarity(self._disliked_index, pool)
            scores = positive - self.gamma * negative
        scores.setflags(write=False)
        self._pool_scores = (key, vectors, scores)
        return scores

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        """Name the nearest liked images, which here *are* the reason.

        Unlike R1, where the score comes from a mean and the neighbours only
        approximate the cause, in R2 the top-k liked images are literally the
        terms of the sum. The explanation is exact.
        """
        require_fitted(self)
        assert self._liked is not None and self._disliked is not None

        vector = features.siglip[image_index]
        liked_similarity = self._liked @ vector
        order = np.argsort(-liked_similarity)[:EXPLANATION_NEIGHBOURS]
        neighbours = [self._liked_ids[int(i)] for i in order if int(i) < len(self._liked_ids)]

        positive = float(top_k_mean(liked_similarity.reshape(1, -1), self.neighbours)[0])
        negative = 0.0
        if len(self._disliked):
            negative = float(
                top_k_mean((self._disliked @ vector).reshape(1, -1), self.neighbours)[0]
            )

        return Explanation(
            reason="similar_likes",
            detail={
                "score": round(positive - self.gamma * negative, 4),
                "liked_similarity": round(positive, 4),
                "disliked_similarity": round(negative, 4),
                "neighbours": neighbours,
            },
        )

    @property
    def params(self) -> dict[str, object]:
        return {
            "rung": self.rung,
            "min_labels": self.min_labels,
            "neighbours": self.neighbours,
            "gamma": self.gamma,
        }

    def _save(self, path: Path) -> None:
        for name, array in (("liked", self._liked), ("disliked", self._disliked)):
            if array is not None:
                with (path / f"{name}.npy").open("wb") as handle:
                    np.save(handle, array)
        (path / "liked_ids.json").write_text(json.dumps(self._liked_ids), encoding="utf-8")

    def _load(self, path: Path) -> None:
        for name in ("liked", "disliked"):
            array_path = path / f"{name}.npy"
            if array_path.exists():
                setattr(self, f"_{name}", np.load(array_path))
        ids_path = path / "liked_ids.json"
        if ids_path.exists():
            self._liked_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        # The indexes are derived from the vectors, so they are rebuilt rather
        # than serialised: a saved FAISS index would be a second copy of the
        # same numbers, able to disagree with them.
        self._rebuild_indexes()
