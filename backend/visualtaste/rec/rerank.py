"""Re-ranking for exploration: MMR for variety, and a quota of deliberate probes.

Phase 8 found the problem this file exists to fix (F5): R1 and R2 raise the like
rate by narrowing what they show. Coverage collapses from 100% of the clusters
to about 40%, and one planted interest in three is never found. A ranking that
only ever serves its top scores cannot discover that it was wrong about what it
does not show.

Two mechanisms, both applied after scoring:

**MMR** (maximal marginal relevance, maths unit M7). Pick cards one at a time;
each next pick maximises

    mmr(c) = lam * relevance(c) - (1 - lam) * max over picked j of sim(c, j)

so a candidate nearly identical to one already picked pays for the
duplication. `lam` = 0.7 keeps relevance in charge: MMR breaks up clumps
rather than overriding the model.

Worked example (M7, asserted in `tests/test_rerank.py`), lam = 0.7: candidate
a (relevance 0.90) is picked first. Then
b (0.88, similarity to a 0.95): 0.7(0.88) - 0.3(0.95) = 0.331
c (0.80, similarity to a 0.30): 0.7(0.80) - 0.3(0.30) = 0.470,
so c is picked next. b scores higher on its own, but it is a near-copy of a.

**The exploration quota.** One card in five (one in ten once there are 2,000
labels) is not chosen by the model at all. It comes from the cluster64 cluster
whose like rate is most *uncertain*: among clusters rated fewer than 3 times,
the one with the widest 90% interval of Beta(1 + likes, 1 + dislikes). An
unrated cluster is Beta(1, 1), the uniform distribution, with interval width
0.9. So the quota systematically visits the parts of the collection the model
knows nothing about, which is where a missed interest would be hiding.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import beta as beta_distribution

#: Part 2 §6.2.
MMR_LAMBDA = 0.7
SHORTLIST = 500
EXPLORE_EVERY = 5
EXPLORE_EVERY_LATE = 10
LATE_AFTER_LABELS = 2_000
MAX_RATINGS_TO_EXPLORE = 3
CREDIBILITY = 0.9


def mmr_value(relevance: float, max_similarity: float, lam: float = MMR_LAMBDA) -> float:
    """lam * relevance - (1 - lam) * max_similarity: one candidate's MMR score."""
    return lam * relevance - (1.0 - lam) * max_similarity


def mmr_select(
    relevance: np.ndarray, vectors: np.ndarray, k: int, *, lam: float = MMR_LAMBDA
) -> list[int]:
    """Greedy MMR: `k` indices into `relevance`, in pick order.

    `vectors` are unit-length rows aligned with `relevance`, so similarity is a
    dot product. The running `max_similarity` is updated with one product per
    pick (shortlist x dim) rather than recomputed, so k picks over a 500-row
    shortlist cost k small matrix-vector products.
    """
    relevance = np.asarray(relevance, dtype=np.float64)
    k = min(k, len(relevance))
    if k <= 0:
        return []

    picked = [int(np.argmax(relevance))]
    max_similarity = np.asarray(vectors @ vectors[picked[0]], dtype=np.float64)
    available = np.ones(len(relevance), dtype=bool)
    available[picked[0]] = False

    while len(picked) < k:
        value = lam * relevance - (1.0 - lam) * max_similarity
        value[~available] = -np.inf
        choice = int(np.argmax(value))
        picked.append(choice)
        available[choice] = False
        max_similarity = np.maximum(max_similarity, vectors @ vectors[choice])
    return picked


@lru_cache(maxsize=1024)
def beta_interval_width(likes: int, dislikes: int, credibility: float = CREDIBILITY) -> float:
    """Width of the central `credibility` interval of Beta(1 + likes, 1 + dislikes).

    Cached: exploration only asks about clusters rated fewer than three times,
    so there are six distinct inputs, and each `ppf` call costs ~1 ms of
    scipy argument checking. The cached value is the same float.
    """
    tail = (1.0 - credibility) / 2.0
    a, b = 1 + likes, 1 + dislikes
    return float(beta_distribution.ppf(1 - tail, a, b) - beta_distribution.ppf(tail, a, b))


def choose_explore_cluster(
    ratings: dict[int, tuple[int, int]],
    available: np.ndarray,
    rng: np.random.Generator,
) -> int | None:
    """The cluster to probe: widest Beta interval among clusters rated < 3 times.

    `ratings` maps cluster -> (likes, dislikes); clusters absent from it have
    never been rated. `available` holds the clusters that still have unseen
    images. Ties, which are the rule early on (every unrated cluster has width
    0.9), are broken at random with the batch's generator. None when every
    available cluster has been rated enough.
    """
    eligible = [
        int(c)
        for c in np.unique(available)
        if sum(ratings.get(int(c), (0, 0))) < MAX_RATINGS_TO_EXPLORE
    ]
    if not eligible:
        return None
    widths = np.array([beta_interval_width(*ratings.get(c, (0, 0))) for c in eligible])
    widest = [c for c, w in zip(eligible, widths, strict=True) if w >= widths.max() - 1e-12]
    return int(rng.choice(widest))


def explore_slots(
    served_so_far: int, batch_size: int, labels: int, *, every: int | None = EXPLORE_EVERY
) -> int:
    """How many exploration cards this batch gets.

    One in `every` cards overall, counted across batches: with five-card
    batches, one in five is one per batch, and one in ten (after 2,000 labels)
    is one every other batch. `every=None` turns exploration off, which is one
    arm of the Phase 10 sensitivity study.
    """
    if every is None:
        return 0
    if labels >= LATE_AFTER_LABELS:
        every = max(every, EXPLORE_EVERY_LATE)
    return (served_so_far + batch_size) // every - served_so_far // every


def explore_batch(
    features,
    candidates: np.ndarray,
    relevance: np.ndarray,
    n: int,
    n_explore: int,
    rng: np.random.Generator,
    *,
    cluster_ratings: dict[int, tuple[int, int]],
    mmr_lambda: float = MMR_LAMBDA,
    shortlist: int = SHORTLIST,
) -> tuple[list[int], list[int]]:
    """One exploring batch from one set of (sampled) relevance scores.

    Shared by R4 (relevance = one posterior draw) and R6 (one ensemble
    member's scores). Returns (MMR offsets, probe offsets) into `candidates`:
    first the probes, from the widest-interval barely-rated clusters; then
    MMR over the top `SHORTLIST` of what is left.
    """
    explore: list[int] = []
    if n_explore and features.cluster64 is not None:
        candidate_clusters = np.asarray(features.cluster64)[candidates]
        for _ in range(n_explore):
            cluster = choose_explore_cluster(cluster_ratings, candidate_clusters, rng)
            if cluster is None:
                break
            members = np.flatnonzero(candidate_clusters == cluster)
            members = members[~np.isin(members, explore)]
            if len(members):
                explore.append(int(rng.choice(members)))

    relevance = np.array(relevance, dtype=np.float64)
    if explore:
        relevance[explore] = -np.inf
    take = min(shortlist, len(relevance))
    shortlist = np.argpartition(-relevance, take - 1)[:take]
    shortlist = shortlist[np.isfinite(relevance[shortlist])]
    picks = mmr_select(
        relevance[shortlist],
        features.siglip[candidates[shortlist]],
        n - len(explore),
        lam=mmr_lambda,
    )
    return [int(shortlist[i]) for i in picks], explore


def cluster_ratings_from(labels, features) -> dict[int, tuple[int, int]]:
    """cluster64 -> (likes, dislikes) over a label set, for the exploration quota."""
    ratings: dict[int, tuple[int, int]] = {}
    if features.cluster64 is None:
        return ratings
    clusters = np.asarray(features.cluster64)
    for image_id, target in zip(labels.image_ids, labels.targets, strict=True):
        row = features.position(image_id)
        if row is None:
            continue
        likes, dislikes = ratings.get(int(clusters[row]), (0, 0))
        ratings[int(clusters[row])] = (
            (likes + 1, dislikes) if target == 1 else (likes, dislikes + 1)
        )
    return ratings
