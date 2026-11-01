"""R5: several interests instead of one taste vector.

Intuition
---------
R1's single preference vector is an average, and averaging two tastes produces
a third thing that is neither. Someone who likes brutalist architecture *and*
botanical prints has a Rocchio vector pointing at "grey concrete plants",
which nobody likes. R5 instead keeps up to eight centroids, one per cluster of
liked images, and scores an image by how close it is to its *nearest* interest:

    m(x) = tau * log sum_k exp( cos(c_k, x) / tau )        tau = 0.05

That is a soft maximum (maths unit M8). With tau small it is almost exactly
max_k cos(c_k, x): an image only has to match one interest to score well. With
tau large it becomes an average-like sum and the "several tastes" advantage
disappears. The worked examples, asserted in `tests/test_interests.py`:

    sims (0.40, 0.10, 0.05), tau = 0.05  ->  m = 0.400   (just the max)
    sims (0.40, 0.10, 0.05), tau = 1     ->  m = 1.294   (log of a sum of three)

Computed stably by subtracting the max before exponentiating: e^(0.40/0.05)
is e^8, harmless, but cosines of 0.9 at tau 0.05 give e^18, and the general
form must never overflow.

Finding the interests: k-means (M8)
-----------------------------------
Liked images are clustered with k-means on L2-normalised SigLIP 2 vectors.
For unit vectors ||a - b||^2 = 2 - 2 cos(a, b), so Euclidean k-means on them
*is* cosine clustering. k = min(8, likes // 25): one interest per 25 likes,
because a cluster of a dozen images is a coincidence, not a taste. R5 needs
k >= 2 (50 likes) to differ from R1 at all.

The 1-D example: points 1, 2, 9, 10 with initial centroids 1 and 2. Assign:
{1} to 1, {2, 9, 10} to 2. Update: 1 and 7. Assign: {1, 2} and {9, 10}.
Update: 1.5 and 9.5. Stable. `lloyd` reproduces it.

Keeping interest ids stable
---------------------------
Between full reclusters (every 100 likes) each new like nudges its nearest
centroid, c <- normalise(c + 0.05 (x - c)), so interests drift with the
person. A full recluster then starts afresh, and k-means numbers clusters
arbitrarily: "interest 2" yesterday may be "interest 0" today. The Hungarian
algorithm (`scipy.optimize.linear_sum_assignment`) pairs new clusters with old
ones to minimise total cosine distance, and each new cluster inherits its
partner's id. A pair further apart than `MATCH_MAX_DISTANCE` is not the same
interest: the new cluster gets a fresh id and the old one is retired. Each
change is recorded in `history`, which the Profile page shows.

Labels: the top three concepts by z-score for each centroid (rec/concepts.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from visualtaste.rec.base import (
    Explanation,
    FeatureBundle,
    InsufficientSignal,
    Recommender,
    require_fitted,
)
from visualtaste.rec.explain import interest_reason

if TYPE_CHECKING:
    from visualtaste.rec.labels import LabelSet

logger = logging.getLogger(__name__)

# Part 2 §7.1.
MAX_INTERESTS = 8
LIKES_PER_INTEREST = 25
MIN_INTERESTS = 2
TAU = 0.05
NUDGE = 0.05
RECLUSTER_EVERY = 100
N_INIT = 10
#: A recluster pair further apart than this (1 - cos) is a different interest.
MATCH_MAX_DISTANCE = 0.5


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def logsumexp_score(similarities: np.ndarray, tau: float = TAU) -> np.ndarray:
    """m = tau * log sum_k exp(s_k / tau), along the last axis, without overflow.

    log sum_k e^(a_k) = a_max + log sum_k e^(a_k - a_max): every exponent is
    then <= 0, so nothing can overflow, and the largest term is exactly e^0 = 1.
    """
    a = np.asarray(similarities, dtype=np.float64) / tau
    a_max = a.max(axis=-1, keepdims=True)
    return tau * (a_max[..., 0] + np.log(np.exp(a - a_max).sum(axis=-1)))


def lloyd(points: np.ndarray, centroids: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Plain Lloyd's k-means from given centroids, for the M8 example and notebook 06.

    Alternate: assign each point to its nearest centroid, then move each
    centroid to the mean of its points. Stops when nothing moves.
    """
    points = np.asarray(points, dtype=np.float64).reshape(len(points), -1)
    centroids = np.asarray(centroids, dtype=np.float64).reshape(len(centroids), -1).copy()
    for _ in range(iterations):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        assignment = distances.argmin(axis=1)
        updated = np.array(
            [
                points[assignment == k].mean(axis=0) if np.any(assignment == k) else centroids[k]
                for k in range(len(centroids))
            ]
        )
        if np.allclose(updated, centroids):
            break
        centroids = updated
    return centroids


def interest_count(likes: int) -> int:
    """k = min(8, likes // 25)."""
    return min(MAX_INTERESTS, likes // LIKES_PER_INTEREST)


def cluster_likes(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k-means centroids of liked vectors, L2-normalised. scikit-learn, n_init=10."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
    model.fit(_normalise(np.asarray(vectors, dtype=np.float64)))
    return _normalise(model.cluster_centers_)


def match_ids(
    old: np.ndarray, old_ids: list[int], new: np.ndarray, next_id: int
) -> tuple[list[int], list[int], list[int], int]:
    """Give each new centroid an id: its Hungarian partner's, or a fresh one.

    Returns (ids for `new`, ids born, ids retired, the next unused id). The
    cost is cosine distance, 1 - cos; a pairing costing more than
    `MATCH_MAX_DISTANCE` is refused, because two interests that far apart are
    not the same interest renamed.
    """
    ids: list[int | None] = [None] * len(new)
    matched_old: set[int] = set()
    if len(old) and len(new):
        cost = 1.0 - _normalise(new) @ _normalise(old).T
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            if cost[r, c] <= MATCH_MAX_DISTANCE:
                ids[r] = old_ids[c]
                matched_old.add(c)

    born: list[int] = []
    for index, value in enumerate(ids):
        if value is None:
            ids[index] = next_id
            born.append(next_id)
            next_id += 1
    retired = [old_ids[c] for c in range(len(old_ids)) if c not in matched_old]
    return [int(i) for i in ids], born, retired, next_id  # type: ignore[arg-type]


@dataclass
class InterestState:
    """The interests at one moment: centroids, stable ids, likes per interest."""

    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    ids: list[int] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    next_id: int = 0
    likes_at_recluster: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.ids)

    def similarities(self, vectors: np.ndarray) -> np.ndarray:
        """cos(c_k, x) for every row and interest: (rows, k)."""
        return np.asarray(vectors, dtype=np.float32) @ self.centroids.T.astype(np.float32)


class InterestTracker:
    """Maintains interests over a growing list of likes: nudge, then recluster.

    Shared by R5 and by R3+R5's time-respecting features, so both see the
    same interests for the same likes.
    """

    def __init__(self, seed: int = 1729) -> None:
        self.seed = seed
        self.state = InterestState()
        self.likes_seen = 0

    def update(self, liked: np.ndarray, *, concept_labeller=None) -> InterestState:
        """Bring the interests up to date with `liked` (all likes so far, in order).

        Reclusters when the like count has crossed a 100-like boundary since
        the last recluster, or when k should grow; otherwise nudges with each
        like not seen before.
        """
        liked = _normalise(np.asarray(liked, dtype=np.float32))
        total = len(liked)
        k = interest_count(total)
        state = self.state

        due = (
            state.k == 0
            or total - state.likes_at_recluster >= RECLUSTER_EVERY
            or (k > state.k and total - state.likes_at_recluster >= LIKES_PER_INTEREST)
        )
        if k >= MIN_INTERESTS and due:
            new = cluster_likes(liked, k, self.seed)
            ids, born, retired, next_id = match_ids(state.centroids, state.ids, new, state.next_id)
            state.history.append({"likes": total, "born": born, "retired": retired, "ids": ids})
            state.centroids, state.ids, state.next_id = new.astype(np.float32), ids, next_id
            state.likes_at_recluster = total
        elif state.k:
            for x in liked[self.likes_seen :]:
                nearest = int(np.argmax(state.centroids @ x))
                c = state.centroids[nearest]
                state.centroids[nearest] = _normalise(c + NUDGE * (x - c))

        if state.k:
            assignment = np.argmax(liked @ state.centroids.T, axis=1)
            state.counts = [int((assignment == j).sum()) for j in range(state.k)]
            if concept_labeller is not None:
                state.labels = [concept_labeller(c) for c in state.centroids]
        self.likes_seen = total
        return state


class InterestRecommender(Recommender):
    """R5 standalone: rank by m(x), the soft-max similarity to any interest.

    Kept as a reference rung (Part 2 §7.1). The serving form is R3+R5, which
    feeds m(x) and the per-interest similarities to R3 as extra features.
    """

    rung = "r5"
    min_labels = MIN_INTERESTS * LIKES_PER_INTEREST  # 50, all of which would have to be likes

    def __init__(self, *, seed: int = 1729, concept_labeller=None) -> None:
        super().__init__()
        self.tracker = InterestTracker(seed)
        self.concept_labeller = concept_labeller

    @property
    def interests(self) -> InterestState:
        return self.tracker.state

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        liked = features.rows("siglip", labels.positives)
        if interest_count(len(liked)) < MIN_INTERESTS:
            raise InsufficientSignal(
                f"{len(liked)} likes; r5 needs {MIN_INTERESTS * LIKES_PER_INTEREST} "
                f"for {MIN_INTERESTS} interests"
            )
        self.tracker.update(liked, concept_labeller=self.concept_labeller)

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        require_fitted(self)
        sims = self.interests.similarities(features.siglip)
        return logsumexp_score(sims).astype(np.float32)[candidates]

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        require_fitted(self)
        sims = self.interests.similarities(features.siglip[image_index : image_index + 1])[0]
        best = int(np.argmax(sims))
        state = self.interests
        label = state.labels[best] if best < len(state.labels) else ""
        return Explanation(
            reason=interest_reason(state.ids[best]),
            detail={
                "interest_id": state.ids[best],
                "label": label,
                "similarity": round(float(sims[best]), 4),
            },
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "min_labels": self.min_labels,
            "tau": TAU,
            "max_interests": MAX_INTERESTS,
            "likes_per_interest": LIKES_PER_INTEREST,
            "recluster_every": RECLUSTER_EVERY,
            "seed": self.tracker.seed,
        }

    def _save(self, path: Path) -> None:
        save_interests(path, self.tracker)

    def _load(self, path: Path) -> None:
        if self.is_fitted:
            load_interests(path, self.tracker)


def save_interests(path: Path, tracker: InterestTracker) -> None:
    state = tracker.state
    if not state.k:
        return
    with (path / "interests.npy").open("wb") as handle:
        np.save(handle, state.centroids)
    (path / "interests.json").write_text(
        json.dumps(
            {
                "ids": state.ids,
                "counts": state.counts,
                "labels": state.labels,
                "next_id": state.next_id,
                "likes_at_recluster": state.likes_at_recluster,
                "likes_seen": tracker.likes_seen,
                "history": state.history,
            }
        ),
        encoding="utf-8",
    )


def load_interests(path: Path, tracker: InterestTracker) -> None:
    meta = json.loads((path / "interests.json").read_text(encoding="utf-8"))
    tracker.state = InterestState(
        centroids=np.load(path / "interests.npy"),
        ids=meta["ids"],
        counts=meta["counts"],
        labels=meta["labels"],
        next_id=meta["next_id"],
        likes_at_recluster=meta["likes_at_recluster"],
        history=meta["history"],
    )
    tracker.likes_seen = meta["likes_seen"]


def interest_features(state: InterestState, vectors: np.ndarray) -> np.ndarray:
    """R3+R5's extra columns: [m(x), cos(c_1..c_8, x) padded with 0, n_interests].

    Zero interests (too few likes yet) gives m = 0 and all-zero similarities:
    the columns exist from the first fit, so R3's weight vector never changes
    shape, and they carry no signal until there is something to say.
    """
    n = len(vectors)
    out = np.zeros((n, MAX_INTERESTS + 2), dtype=np.float32)
    if state.k:
        sims = state.similarities(vectors)
        out[:, 0] = logsumexp_score(sims)
        out[:, 1 : 1 + state.k] = sims
    out[:, -1] = state.k
    return out


def tracker_of(model) -> InterestTracker | None:
    """The interest tracker inside R5, R3+R5 or R4+R5, or None for other rungs."""
    tracker = getattr(model, "tracker", None) or getattr(model, "interest_tracker", None)
    return tracker if isinstance(tracker, InterestTracker) else None


def persist_interests(connection, profile_id: int, version: str, tracker: InterestTracker) -> int:
    """Write one `interests` row per current interest for this model version.

    Rows are keyed (profile, interest id, version), so a version's interests
    stay recorded after later versions move on, and the Profile page can show
    when each id was born or retired across versions. Returns rows written.
    """
    state = tracker.state
    rows = [
        (
            profile_id,
            int(interest_id),
            version,
            np.asarray(state.centroids[index], dtype=np.float32).tobytes(),
            state.labels[index] if index < len(state.labels) else None,
            int(state.counts[index]) if index < len(state.counts) else 0,
        )
        for index, interest_id in enumerate(state.ids)
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO interests
            (profile_id, interest_id, model_version, centroid, label, n_likes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        rows,
    )
    connection.commit()
    return len(rows)
