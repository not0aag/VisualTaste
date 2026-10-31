"""Every metric this project uses, written out rather than imported.

Why hand-implement what scikit-learn already has
------------------------------------------------
Because the point of this file is to be read. `roc_auc_score(y, p)` is one line
and tells you nothing about what AUC *is*; the twelve lines below say that it
is the probability a randomly chosen liked image outranks a randomly chosen
disliked one, and then compute exactly that. The test suite checks these
implementations against scikit-learn where an equivalent exists, so the
teaching version is also a verified version.

Each function below states its formula in plain text and carries a worked
numeric example whose value a unit test asserts. The examples marked **M10**
are the pre-computed ones from the brief's §8.

Two conventions chosen once, here
---------------------------------
**Logarithms are natural.** Log loss uses `ln`, which is what every library
means by log loss and what makes the derivative in `rec/logreg_numpy.py` come
out as `(p - y)x` with no stray constant.

**NDCG uses exponential gain**, `(2^g - 1) / log2(rank + 1)`. That is the
standard formulation, and it is the one that reproduces the brief's figure:
gains `[1, 0, 2, 0, 1]` give 0.699 exponentially and 0.762 linearly. The
difference matters because a save (gain 2) is worth 3 under exponential gain
and only 2 under linear -- exponential gain rewards getting the *best* item
high, which is what a five-card feed is for.

What these numbers can and cannot tell you
------------------------------------------
Until Alen swipes, every one of them is computed from the simulator, whose
"truth" is cosine distance to three planted centroids. That flatters any model
whose inductive bias is also cosine distance -- Rocchio and logistic regression
-- and tells you very little about a neural ranker. `docs/EVALUATION.md` says so
at the top, and so does this paragraph, because a metric detached from its
caveat is how a project talks itself into believing something.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Gains per action for ranking metrics (brief §7.7).
ACTION_GAIN: dict[str, int] = {"save": 2, "like": 1, "dislike": 0, "skip": 0}

# Clipping bound for probabilities in log loss. log(0) is -inf, and one
# over-confident wrong prediction would otherwise make the whole average
# infinite -- which is true but useless.
_EPSILON = 1e-15


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """The logistic function, 1 / (1 + e^-z), computed without overflowing.

    The naive form overflows for z < -745 in float64. Splitting on the sign
    keeps both branches' exponentials in [0, 1]:

        z >= 0:  1 / (1 + e^-z)
        z <  0:  e^z / (1 + e^z)

    Worked example (M5): z = 2(0.8) + (-1)(0.6) = 1.0, so
    sigmoid(1.0) = 1/(1 + e^-1) = 0.7311.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def log_loss(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted mean cross-entropy, in nats.

        L = -(1/W) * sum_i w_i [ y_i ln p_i + (1 - y_i) ln(1 - p_i) ]

    where W = sum of weights. Lower is better; 0 is perfect.

    What the number means: a log loss of 0.693 is exactly what you get by
    predicting 0.5 for everything (-ln 0.5), so anything above that is worse
    than refusing to guess. That makes it the honest baseline to compare
    against, and the reason log loss is reported alongside AUC -- AUC only sees
    the *ordering*, so a model can rank perfectly while being badly calibrated.

    Worked example (M5): y = 0, p = 0.7311 gives
    -ln(1 - 0.7311) = -ln(0.2689) = 1.3133.
    """
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), _EPSILON, 1 - _EPSILON)

    if targets.shape != probabilities.shape:
        raise ValueError(f"{targets.shape} targets but {probabilities.shape} probabilities")
    if len(targets) == 0:
        return float("nan")

    weights = np.ones_like(targets) if weights is None else np.asarray(weights, dtype=np.float64)
    total_weight = weights.sum()
    if total_weight <= 0:
        return float("nan")

    per_example = targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)
    return float(-(weights * per_example).sum() / total_weight)


def roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed as its probabilistic definition.

    AUC is the probability that a randomly chosen positive scores higher than a
    randomly chosen negative, with ties counting a half:

        AUC = (1/(P*N)) * sum over positive i, negative j of
                  1   if s_i > s_j
                  0.5 if s_i = s_j
                  0   if s_i < s_j

    Computed here via rank sums (the Mann-Whitney U identity) rather than by
    enumerating all P*N pairs, which would be 2.5 billion comparisons at
    50,000 items:

        AUC = (sum of positive ranks - P(P+1)/2) / (P * N)

    Average ranks are used for ties, which is what makes a tie count a half.

    Worked example (M10): likes score 0.9 and 0.6, dislikes 0.7 and 0.2. The
    four pairs are 0.9>0.7 (win), 0.9>0.2 (win), 0.6<0.7 (loss), 0.6>0.2 (win),
    so AUC = 3/4 = 0.75.

    Returns NaN when one class is absent -- with no negatives there is no
    question to answer, and returning 0.5 would look like a measurement.
    """
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.shape != scores.shape:
        raise ValueError(f"{targets.shape} targets but {scores.shape} scores")

    positives = targets == 1
    n_pos = int(positives.sum())
    n_neg = int(len(targets) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    positive_rank_sum = ranks[positives].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, with tied values sharing their average rank.

    Ties matter: a model that scores two items identically has expressed no
    preference between them, and the metric should treat that as half a win
    rather than an arbitrary one.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def dcg(gains: list[int] | np.ndarray, k: int | None = None) -> float:
    """Discounted cumulative gain with exponential gain.

        DCG@k = sum_{i=1..k} (2^g_i - 1) / log2(i + 1)

    The discount `1 / log2(i + 1)` says a result at rank 1 is worth about 1.6
    times one at rank 3 and twice one at rank 7 -- a model of how attention
    falls off a list. The exponential gain says a save is worth three likes,
    not two, so getting the single best image to the top matters more than
    filling the list with merely-good ones.
    """
    gains = np.asarray(gains, dtype=np.float64)
    if k is not None:
        gains = gains[:k]
    if len(gains) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(((2**gains - 1) / discounts).sum())


def ndcg_at_k(gains: list[int] | np.ndarray, k: int = 10) -> float:
    """DCG@k divided by the best DCG@k achievable from the same gains.

        NDCG@k = DCG@k / IDCG@k,  where IDCG is DCG of the gains sorted
                                  in descending order

    Normalising makes the number comparable across batches: a batch containing
    one save and four skips cannot score as highly in raw DCG as one containing
    five saves, but both can be perfectly *ranked*.

    Worked example (M10): gains [1, 0, 2, 0, 1] at k = 5.

        DCG  = 1/log2(2) + 0 + 3/log2(4) + 0 + 1/log2(6)
             = 1 + 1.5 + 0.3869 = 2.8869
        IDCG = gains sorted to [2, 1, 1, 0, 0]
             = 3/log2(2) + 1/log2(3) + 1/log2(4)
             = 3 + 0.6309 + 0.5 = 4.1309
        NDCG = 2.8869 / 4.1309 = 0.6988

    Returns 0.0 when every gain is zero: nothing in the list was wanted, so
    there was no ranking to get right.
    """
    gains = np.asarray(gains, dtype=np.float64)
    ideal = np.sort(gains)[::-1]
    best = dcg(ideal, k)
    if best == 0:
        return 0.0
    return dcg(gains, k) / best


def gains_from_actions(actions: list[str]) -> list[int]:
    """Map a batch's actions to ranking gains: save 2, like 1, otherwise 0."""
    return [ACTION_GAIN.get(action, 0) for action in actions]


def precision_at_k(relevant: list[bool] | np.ndarray, k: int = 10) -> float:
    """Share of the top k that was wanted.

        P@k = (number of relevant items in the top k) / k

    Divided by `k` rather than by the number shown, deliberately: a model that
    returns three good results when asked for ten has not achieved 100%.
    """
    if k <= 0:
        return 0.0
    relevant = np.asarray(relevant, dtype=bool)[:k]
    return float(relevant.sum() / k)


def recall_at_k(relevant: list[bool] | np.ndarray, total_relevant: int, k: int = 10) -> float:
    """Share of everything wanted that appeared in the top k.

        R@k = (relevant in top k) / (total relevant available)

    Only computable when the full truth set is known, which in this project
    means the simulator alone -- Alen's real preferences have no denominator.
    """
    if total_relevant <= 0:
        return float("nan")
    relevant = np.asarray(relevant, dtype=bool)[:k]
    return float(relevant.sum() / total_relevant)


def hit_rate(relevant: list[bool] | np.ndarray, k: int = 10) -> float:
    """1.0 if anything in the top k was wanted, else 0.0.

    Crude, and useful precisely because it is: for a five-card feed, "was there
    at least one image worth stopping on" is close to what the experience
    actually feels like.
    """
    relevant = np.asarray(relevant, dtype=bool)[:k]
    return 1.0 if relevant.any() else 0.0


def intra_batch_diversity(vectors: np.ndarray) -> float:
    """Mean `1 - cosine` over all distinct pairs in a batch. Higher is varied.

        diversity = mean over pairs (i<j) of (1 - cos(x_i, x_j))

    Vectors arrive L2-normalised, so each cosine is a dot product.

    What it is for: a model that has learned Alen likes misty landscapes can
    serve five nearly identical misty landscapes, which scores well on every
    accuracy metric and makes a terrible feed. This is the number that notices,
    and it is why MMR exists in `rec/rerank.py`.

    Worked example: two orthogonal unit vectors have cosine 0, so diversity is
    1.0. Two identical vectors give 0.0. A batch of one has no pairs, so the
    answer is 0.0 -- there is no variety in a single image.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) < 2:
        return 0.0

    similarity = vectors @ vectors.T
    # The upper triangle, excluding the diagonal: each unordered pair once.
    upper = np.triu_indices(len(vectors), k=1)
    return float(np.mean(1.0 - similarity[upper]))


def coverage(shown_clusters: list[int] | np.ndarray, total_clusters: int) -> float:
    """Share of clusters represented among the images shown.

        coverage = |distinct clusters shown| / total clusters

    Measured over a recent window (the last 500 impressions, per §7.7) rather
    than all time, because the question is "is the feed still exploring", not
    "did it ever". A model that collapses onto one region of the corpus shows
    up here long before it shows up in the like rate.
    """
    if total_clusters <= 0:
        return float("nan")
    return float(len({int(c) for c in shown_clusters}) / total_clusters)


def novelty(candidate_vectors: np.ndarray, seen_vectors: np.ndarray) -> np.ndarray:
    """Distance from each candidate to the nearest already-seen image.

        novelty(x) = 1 - max over seen s of cos(x, s)

    0 means "identical to something already shown", 1 means "orthogonal to
    everything shown". With nothing seen yet every candidate is maximally
    novel, which is the right answer rather than an error.
    """
    candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)
    if candidate_vectors.ndim == 1:
        candidate_vectors = candidate_vectors.reshape(1, -1)
    seen_vectors = np.asarray(seen_vectors, dtype=np.float32)

    if len(seen_vectors) == 0:
        return np.ones(len(candidate_vectors), dtype=np.float64)

    nearest = (candidate_vectors @ seen_vectors.T).max(axis=1)
    return (1.0 - nearest).astype(np.float64)


@dataclass
class BetaInterval:
    """A Beta posterior over a rate, summarised as mean and an interval.

    Used for two things in this project: the like rate of an exploration slot
    against an exploitation slot (§7.7), and the uncertainty of a `cluster64`
    that R4's quota draws from (§7.5).

    The prior is Beta(1, 1) -- uniform, "any rate is equally plausible" -- so
    the posterior after `k` successes in `n` trials is Beta(1 + k, 1 + n - k).
    That choice is what makes a cluster rated 0 out of 0 times have a *wide*
    interval rather than an undefined one, which is exactly the behaviour
    exploration needs: never sampled means maximally uncertain, not bad.
    """

    successes: int
    trials: int
    mean: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low


def beta_interval(successes: int, trials: int, *, credibility: float = 0.9) -> BetaInterval:
    """The Beta(1+k, 1+n-k) posterior interval for a rate.

    Worked example: 0 successes in 0 trials gives mean 0.5 and the interval
    [0.05, 0.95] at 90% -- maximally uncertain. 8 in 10 gives mean 0.75, and
    80 in 100 gives mean 0.794 with a far narrower interval. The mean is
    (1+k)/(2+n), the Laplace rule of succession, not k/n: with no data it says
    0.5 rather than dividing by zero.
    """
    from scipy.stats import beta as beta_dist

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"{successes} successes in {trials} trials is not possible")

    alpha = 1 + successes
    beta_param = 1 + trials - successes
    tail = (1 - credibility) / 2

    return BetaInterval(
        successes=successes,
        trials=trials,
        mean=alpha / (alpha + beta_param),
        low=float(beta_dist.ppf(tail, alpha, beta_param)),
        high=float(beta_dist.ppf(1 - tail, alpha, beta_param)),
    )


@dataclass
class ExplorationYield:
    """Like rates in explore and exploit slots, each with its interval.

    The comparison the brief asks for (§7.7). Reported as two intervals rather
    than one difference and a p-value, because with a few hundred exploration
    slots the honest answer is usually "these overlap" -- and two overlapping
    intervals communicate that far better than a significance test does.
    """

    explore: BetaInterval
    exploit: BetaInterval

    @property
    def difference(self) -> float:
        return self.exploit.mean - self.explore.mean

    @property
    def intervals_overlap(self) -> bool:
        return not (self.explore.low > self.exploit.high or self.exploit.low > self.explore.high)


def exploration_yield(
    explore_outcomes: list[bool] | np.ndarray,
    exploit_outcomes: list[bool] | np.ndarray,
    *,
    credibility: float = 0.9,
) -> ExplorationYield:
    """Compare the like rate of exploration slots against exploitation slots.

    Exploration is expected to score *worse* on immediate like rate -- that is
    what paying for information looks like. The question is whether the cost is
    small, and whether the corpus coverage it buys shows up later.
    """
    explore_outcomes = np.asarray(explore_outcomes, dtype=bool)
    exploit_outcomes = np.asarray(exploit_outcomes, dtype=bool)

    return ExplorationYield(
        explore=beta_interval(
            int(explore_outcomes.sum()), len(explore_outcomes), credibility=credibility
        ),
        exploit=beta_interval(
            int(exploit_outcomes.sum()), len(exploit_outcomes), credibility=credibility
        ),
    )


@dataclass
class CalibrationBin:
    """One bucket of a calibration curve."""

    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Signed error: positive means the model was over-confident."""
        return self.mean_predicted - self.observed_rate


def calibration_bins(
    targets: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> list[CalibrationBin]:
    """Group predictions into probability bands and compare claim to outcome.

    A model is calibrated when, among the images it gave a 70% chance of being
    liked, about 70% were liked. This is a different question from AUC: a model
    can rank perfectly and still claim 0.95 for everything, and in this project
    the claim is shown to the user ("model_confident", §7.6), so it had better
    mean something.

    Worked example: predictions 0.1, 0.2 both disliked and 0.8, 0.9 both liked,
    in two bins, gives a lower bin predicting 0.15 and observing 0.0, and an
    upper bin predicting 0.85 and observing 1.0 -- slightly under-confident at
    both ends, which is the usual shape for a small, regularised model.
    """
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must align")

    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[CalibrationBin] = []

    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        # The last bin includes 1.0; the others are half-open, so each
        # prediction lands in exactly one bin.
        in_bin = (
            (probabilities >= low) & (probabilities <= high)
            if index == bins - 1
            else (probabilities >= low) & (probabilities < high)
        )
        count = int(in_bin.sum())
        out.append(
            CalibrationBin(
                low=float(low),
                high=float(high),
                count=count,
                mean_predicted=float(probabilities[in_bin].mean()) if count else float("nan"),
                observed_rate=float(targets[in_bin].mean()) if count else float("nan"),
            )
        )
    return out


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Mean |predicted - observed| across bins, weighted by bin population.

        ECE = sum_b (n_b / N) * |mean_predicted_b - observed_rate_b|

    One number for the Model Lab's table. Empty bins contribute nothing, which
    is why the weighting is by count rather than a plain average over bins.
    """
    buckets = calibration_bins(targets, probabilities, bins)
    total = sum(b.count for b in buckets)
    if total == 0:
        return float("nan")
    return float(sum(b.count / total * abs(b.gap) for b in buckets if b.count > 0))


def like_rate(actions: list[str]) -> float:
    """Share of actions that were a like or a save.

    The headline number for the simulator study, and the one to be most careful
    about: it is measured on whatever the model chose to show, so a model that
    shows only safe, obvious images can score well while being useless. That is
    exactly why diversity, coverage and novelty are reported beside it.
    """
    if not actions:
        return float("nan")
    positives = sum(1 for action in actions if action in ("like", "save"))
    return positives / len(actions)


def rolling_like_rate(actions: list[str], window: int = 50) -> list[float]:
    """Like rate over a sliding window, for the learning-curve plot.

    A cumulative rate hides exactly what the plot is for: a model that improved
    sharply at label 200 shows as a gentle upward drift cumulatively, and as a
    step with a window.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    out: list[float] = []
    for end in range(1, len(actions) + 1):
        start = max(0, end - window)
        out.append(like_rate(actions[start:end]))
    return out


def paired_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    samples: int = 2_000,
    credibility: float = 0.9,
    seed: int = 1729,
) -> tuple[float, float, float]:
    """Bootstrap interval for the mean *difference* between paired measurements.

    Returns `(mean difference, low, high)`.

    Paired, because the two models are evaluated on the *same* cut points, so
    the differences are what carries the signal -- comparing two independent
    intervals would throw away that pairing and widen the result for nothing.

    How it works: resample the per-cut differences with replacement, take the
    mean each time, and read the percentiles of those means. No assumption that
    the differences are normally distributed, which with twenty cut points is a
    relief.

    If the interval straddles zero, the honest statement is "no detectable
    difference at this sample size" -- which is the most likely outcome of most
    comparisons in this project, and should be reported as such.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"{first.shape} vs {second.shape}: pairs must align")
    if len(first) == 0:
        return float("nan"), float("nan"), float("nan")

    differences = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)

    tail = (1 - credibility) / 2 * 100
    return (
        float(differences.mean()),
        float(np.percentile(means, tail)),
        float(np.percentile(means, 100 - tail)),
    )


def sign_test(wins: int, losses: int) -> float:
    """Two-sided sign-test p-value for `wins` against `losses`, ties discarded.

    Used for team-draft interleaving (§7.7), where each batch yields a winner.
    Under the null hypothesis the two versions are equally good, so each
    comparison is a fair coin and the count of wins is Binomial(n, 0.5).

    Worked example: 8 wins and 2 losses gives p = 0.109 -- suggestive and not
    conclusive, which at ten comparisons is the correct amount of confidence.
    """
    n = wins + losses
    if n == 0:
        return float("nan")

    # P(X >= max(wins, losses)) under Binomial(n, 0.5), doubled for two sides.
    extreme = max(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return float(min(1.0, 2 * tail))
