"""Time-ordered replay: what would this model have known, and how well did it do?

The mistake this file exists to avoid
-------------------------------------
The obvious way to evaluate a recommender is a random train/test split. For
this project that would be straightforwardly wrong, and in a way that
*flatters* the model rather than breaking it.

Alen's labels arrive in time order, and the feed's choices depend on earlier
labels. So a random split lets a model train on label 400 and be tested on
label 200 -- predicting a verdict that, in part, *caused* the data it trained
on. The reported AUC would be higher than anything achievable in reality, and
nothing would look broken.

So: cut the history at increasing points, train on everything before the cut,
evaluate on the next block, and never let information flow backwards.

    cut at 200:  train on events 1-200,   test on 201-300
    cut at 300:  train on events 1-300,   test on 301-400
    cut at 400:  train on events 1-400,   test on 401-500
    ...

Brief §7.7: cuts every 100 events after the first 200. The first 200 are the
burn-in -- a model trained on 50 labels is not being tested, it is being
guessed at.

Comparing two models
--------------------
Both are replayed over the *same* cut points, which makes the per-cut results
paired. `paired_bootstrap` then gives an interval for the mean difference. That
pairing matters: the cuts differ enormously in difficulty (early cuts have few
labels; some test blocks are mostly skips), and comparing two unpaired
intervals would drown the model difference in cut-to-cut variance.

The honest default outcome
--------------------------
With twenty cut points and a hundred test events each, most comparisons in this
project will produce an interval that straddles zero. That is a real result --
"no detectable difference at this sample size" -- and `summarise` says so in
those words rather than reporting the sign of a noisy mean.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from visualtaste.eval.metrics import (
    gains_from_actions,
    log_loss,
    ndcg_at_k,
    paired_bootstrap,
    roc_auc,
)

logger = logging.getLogger(__name__)

# Brief §7.7: the first cut is here, and every `CUT_EVERY` events after it.
FIRST_CUT = 200
CUT_EVERY = 100

POSITIVE_ACTIONS = frozenset({"like", "save"})


@dataclass
class ReplayPoint:
    """Results at one cut: trained on `train_size`, tested on the next block."""

    cut: int
    train_size: int
    test_size: int
    auc: float
    log_loss: float
    ndcg_at_10: float
    positive_rate: float

    @property
    def usable(self) -> bool:
        """Whether this cut produced a meaningful AUC.

        A test block containing only likes, or only dislikes, has no AUC -- and
        that happens genuinely, when a well-performing model serves a hundred
        images the user likes. Those cuts are excluded from comparisons rather
        than counted as 0.5.
        """
        return not np.isnan(self.auc)


@dataclass
class ReplayResult:
    """One model replayed across every cut."""

    label: str
    points: list[ReplayPoint] = field(default_factory=list)

    def metric(self, name: str) -> np.ndarray:
        """A metric across usable cuts, as an array for the bootstrap."""
        return np.array(
            [getattr(point, name) for point in self.points if point.usable],
            dtype=np.float64,
        )

    def mean(self, name: str) -> float:
        values = self.metric(name)
        return float(values.mean()) if len(values) else float("nan")

    @property
    def usable_cuts(self) -> int:
        return sum(1 for point in self.points if point.usable)


#: A model factory plus its scoring function, as replay needs them.
#:
#: `fit_and_score(train_indices, test_indices) -> probabilities for the test
#: block`. Passed as a callable rather than a `Recommender` so that replay can
#: also evaluate things that are not rungs -- a constant baseline, say, or a
#: feature ablation -- without inventing a class for each.
FitAndScore = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cut_points(total_events: int) -> list[int]:
    """Where to cut a history of `total_events`.

    Each cut needs a test block after it, so the last cut is at least
    `CUT_EVERY` events before the end. A history shorter than
    `FIRST_CUT + CUT_EVERY` yields no cuts at all, which is the correct answer
    rather than a degenerate single cut.
    """
    cuts = []
    cut = FIRST_CUT
    while cut + CUT_EVERY <= total_events:
        cuts.append(cut)
        cut += CUT_EVERY
    return cuts


def replay(
    actions: Sequence[str],
    fit_and_score: FitAndScore,
    *,
    label: str,
    block: int = CUT_EVERY,
    cuts: Sequence[int] | None = None,
) -> ReplayResult:
    """Replay one model over a history, returning per-cut metrics.

    `actions` is the full history in time order; `fit_and_score` is given the
    training and test index arrays and returns a probability per test event.

    `cuts` overrides the §7.7 schedule. The Phase 9 H2 learning curve uses it
    to cut from 50 labels, because "R3 overtakes R2 at about 150" cannot be seen
    from a schedule that starts at 200.
    """
    if cuts is not None:
        schedule = [cut for cut in cuts if cut + 1 <= len(actions)]
    else:
        schedule = cut_points(len(actions))
    result = ReplayResult(label=label)
    targets = np.array(
        [1.0 if action in POSITIVE_ACTIONS else 0.0 for action in actions],
        dtype=np.float64,
    )

    for cut in schedule:
        train_index = np.arange(cut)
        test_index = np.arange(cut, min(cut + block, len(actions)))

        probabilities = np.asarray(fit_and_score(train_index, test_index), dtype=np.float64)
        if len(probabilities) != len(test_index):
            raise ValueError(
                f"{label} returned {len(probabilities)} scores for "
                f"{len(test_index)} test events at cut {cut}"
            )

        test_targets = targets[test_index]

        # NDCG over the test block ranked by predicted probability, which is
        # the order the feed would actually have shown them in.
        order = np.argsort(-probabilities)
        ranked_actions = [actions[int(test_index[i])] for i in order]

        result.points.append(
            ReplayPoint(
                cut=cut,
                train_size=len(train_index),
                test_size=len(test_index),
                auc=roc_auc(test_targets.astype(int), probabilities),
                log_loss=log_loss(test_targets, probabilities),
                ndcg_at_10=ndcg_at_k(gains_from_actions(ranked_actions), 10),
                positive_rate=float(test_targets.mean()),
            )
        )

    logger.info(
        "%s: replayed %d cuts (%d usable), mean AUC %.3f",
        label,
        len(result.points),
        result.usable_cuts,
        result.mean("auc"),
    )
    return result


@dataclass
class Comparison:
    """Two models compared across shared cut points."""

    metric: str
    first: str
    second: str
    first_mean: float
    second_mean: float
    difference: float
    low: float
    high: float
    cuts: int

    @property
    def detectable(self) -> bool:
        """True when the interval excludes zero."""
        return not (self.low <= 0.0 <= self.high)

    def summarise(self) -> str:
        """A sentence that states the result including when there is none."""
        direction = "higher" if self.difference > 0 else "lower"
        if not self.detectable:
            return (
                f"{self.first} vs {self.second} on {self.metric}: "
                f"{self.first_mean:.3f} vs {self.second_mean:.3f}, difference "
                f"{self.difference:+.3f} with a 90% interval of "
                f"[{self.low:+.3f}, {self.high:+.3f}] over {self.cuts} cuts. "
                f"The interval includes zero, so there is no detectable "
                f"difference at this sample size."
            )
        return (
            f"{self.first} vs {self.second} on {self.metric}: "
            f"{self.first_mean:.3f} vs {self.second_mean:.3f}. {self.first} is "
            f"{direction} by {abs(self.difference):.3f} "
            f"(90% interval [{self.low:+.3f}, {self.high:+.3f}], {self.cuts} cuts)."
        )


def compare(
    first: ReplayResult, second: ReplayResult, *, metric: str = "auc", seed: int = 1729
) -> Comparison:
    """Paired bootstrap comparison of two replays on one metric.

    Only cuts where *both* models produced a usable result are compared, which
    is what keeps the pairing valid: dropping a cut from one side and not the
    other would silently compare different sets of cuts.
    """
    usable = [
        index
        for index, (a, b) in enumerate(zip(first.points, second.points, strict=True))
        if a.usable and b.usable
    ]
    if not usable:
        return Comparison(
            metric,
            first.label,
            second.label,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )

    a_values = np.array([getattr(first.points[i], metric) for i in usable])
    b_values = np.array([getattr(second.points[i], metric) for i in usable])
    difference, low, high = paired_bootstrap(a_values, b_values, seed=seed)

    return Comparison(
        metric=metric,
        first=first.label,
        second=second.label,
        first_mean=float(a_values.mean()),
        second_mean=float(b_values.mean()),
        difference=difference,
        low=low,
        high=high,
        cuts=len(usable),
    )


def learning_curve(result: ReplayResult, metric: str = "auc") -> list[tuple[int, float]]:
    """`(training set size, metric)` pairs, for the Model Lab's curve.

    Plotted against training size rather than cut index, because the question
    the curve answers is "how much does another hundred labels buy", and that
    is only legible on an x-axis of labels.
    """
    return [(point.train_size, getattr(point, metric)) for point in result.points if point.usable]
