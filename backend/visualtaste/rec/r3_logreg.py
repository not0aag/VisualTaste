"""R3: a personal logistic-regression classifier, the workhorse of the ladder.

Intuition
---------
R1 says "images like the average of what he liked". R3 *learns* which directions
in feature space separate his likes from his dislikes, giving each of 128
feature dimensions its own weight. Two images equally close to his liked mean
can score very differently if they differ along a dimension that, for him,
matters. And because the output is a calibrated probability, "73%" means
something checkable, which the Model Lab's calibration plot does check.

The maths is in `rec/logreg_numpy.py`, written out by hand. This file is the
production path: the same objective, solved by scikit-learn's L-BFGS, which is
faster and more robust than plain gradient descent, plus the parts a real
model needs -- choosing lambda, refusing to serve without enough signal, and
explaining itself.

The objective, and the one setting that is easy to get wrong
------------------------------------------------------------
Part 2 §5.2 fixes the objective:

    L(w, b) = (1/S) sum_i s_i CE(y_i, p_i) + (lam/2) ||w||^2,   S = sum_i s_i

scikit-learn's `LogisticRegression` minimises a different-looking thing:

    (1/2) ||w||^2 + C sum_i s_i CE(y_i, p_i)

Divide ours by lam and the two match term for term:

    L / lam = (1/(lam S)) sum_i s_i CE_i + (1/2) ||w||^2

so the equivalent setting is **C = 1 / (lam * S)** -- *not* `C = 1/lam`, which is
what the original blueprint said. The difference is not cosmetic: with
`C = 1/lam`, the effective penalty shrinks as labels accumulate (the data term
grows with S, the penalty does not), so the same lambda would mean "strong
regularisation" at 50 labels and "almost none" at 5,000. `sklearn_c` implements
the mapping and `tests/test_logreg.py` checks it against the hand-written
solver.

scikit-learn 1.8 deprecated `penalty="l2"`; L2 is now the default via
`l1_ratio=0`, so it is not passed (docs/DECISIONS.md #17).

Choosing lambda: time-ordered validation
----------------------------------------
lambda trades fitting the labels against keeping `w` short. It is chosen on
every retrain from the grid {1e-4, 3e-4, ..., 3, 10} by fitting on the earliest
80% of labels and measuring log loss on the latest 20% -- *the latest*, not a
random 20%, because the model will always be predicting the future from the
past, and taste drifts. A random split would let it peek. Losses within 1e-3 of
the best count as a tie, broken towards the larger lambda: when the data cannot
tell two values apart, prefer the simpler model.

When R3 refuses to serve
------------------------
Fewer than 30 labels, or fewer than 5 of either class (Part 2 §4.4). A
classifier trained on thirty likes and no dislikes has learned nothing but the
bias. The feed then falls back to R0 and says why in the impression.

Worked example
--------------
The M5 example (one gradient step, w = (2, -1) to (1.708, -1.219)) lives in
`logreg_numpy.py`, where the arithmetic is visible. Here, the mapping:
S = 500 labels, lambda = 0.01 -> C = 1 / (0.01 x 500) = 0.2.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import expit

from visualtaste.rec.base import (
    Explanation,
    FeatureBundle,
    InsufficientSignal,
    Recommender,
    UnfittedModelError,
    require_fitted,
)
from visualtaste.rec.features import (
    DEFAULT_BLOCKS,
    OPTIONAL_BLOCKS,
    block_available,
    design_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from visualtaste.rec.labels import LabelSet

logger = logging.getLogger(__name__)

# Part 2 §5.4.
LAMBDA_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
VALIDATION_SHARE = 0.2
TIE_TOLERANCE = 1e-3

# Part 2 §4.4 / §5.4: the minimum to serve.
MIN_LABELS = 30
MIN_PER_CLASS = 5

#: Used when the training part of the split has one class only, so no lambda
#: can be validated. Mid-grid and on the strong side: with that little signal,
#: a heavily regularised model is the safe guess.
FALLBACK_LAMBDA = 1.0

# Part 2 §5.3: the production solver settings.
SOLVER_TOL = 1e-10
SOLVER_MAX_ITER = 10_000

#: Probability clipping for log loss, so a single 0 or 1 cannot make it infinite.
EPSILON = 1e-12


#: The pseudo-block naming R5's interest features, e.g. blocks=("sig", "int").
INTEREST_BLOCK = "int"

#: Fixed affine scale for interest similarities: SigLIP image-to-centroid
#: cosines sit near 0.5 with a spread of about 0.1 on this pool, so
#: (cos - 0.5) / 0.1 is roughly standard. Fixed rather than fitted, because the
#: interests change every fit and a scale that moved with them would make the
#: same cosine mean different things in different training rows.
INTEREST_CENTRE, INTEREST_SCALE = 0.5, 0.1


def scale_interest_features(state, vectors: np.ndarray) -> np.ndarray:
    """[m(x), cos(c_1..c_8, x), n_interests], scaled for R3's L2 penalty.

    Similarity columns that exist are centred and scaled; padding for absent
    interests stays exactly 0, so an interest that does not exist yet has no
    effect. n_interests is divided by 8.
    """
    from visualtaste.rec.r5_interests import MAX_INTERESTS, interest_features

    raw = interest_features(state, vectors)
    out = np.zeros_like(raw)
    k = state.k
    if k:
        out[:, 0] = (raw[:, 0] - INTEREST_CENTRE) / INTEREST_SCALE
        out[:, 1 : 1 + k] = (raw[:, 1 : 1 + k] - INTEREST_CENTRE) / INTEREST_SCALE
    out[:, -1] = raw[:, -1] / MAX_INTERESTS
    return out


def sklearn_c(lam: float, total_weight: float) -> float:
    """scikit-learn's C equivalent to our lambda: C = 1 / (lam * S)."""
    if lam <= 0 or total_weight <= 0:
        raise ValueError("lambda and the total example weight must both be positive")
    return 1.0 / (lam * total_weight)


def fit_sklearn(
    X: np.ndarray, y: np.ndarray, s: np.ndarray, lam: float, *, estimator=None
) -> tuple[np.ndarray, float]:
    """Minimise the §5.2 objective with L-BFGS. Returns (w, b).

    `estimator` lets `tune_lambda` reuse one object with `warm_start`, so each
    lambda starts from the previous solution instead of from zero -- the same
    answer, several times faster along the grid.
    """
    from sklearn.linear_model import LogisticRegression

    if estimator is None:
        estimator = LogisticRegression(solver="lbfgs", tol=SOLVER_TOL, max_iter=SOLVER_MAX_ITER)
    estimator.set_params(C=sklearn_c(lam, float(np.sum(s))))
    with warnings.catch_warnings():
        # At tol 1e-10 a tiny residual sometimes trips the warning; the
        # agreement test is the real check on convergence.
        warnings.simplefilter("ignore", category=UserWarning)
        from sklearn.exceptions import ConvergenceWarning

        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(X, y, sample_weight=s)
    return estimator.coef_[0].astype(np.float64), float(estimator.intercept_[0])


def weighted_log_loss(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> float:
    p = np.clip(p, EPSILON, 1 - EPSILON)
    cross_entropy = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float((s * cross_entropy).sum() / s.sum())


@dataclass
class TuningResult:
    """Which lambda won, and the validation curve it won on."""

    lam: float
    curve: dict[float, float] = field(default_factory=dict)
    train_size: int = 0
    validation_size: int = 0
    note: str = ""


def choose_lambda(curve: dict[float, float], tolerance: float = TIE_TOLERANCE) -> float:
    """The largest lambda whose loss is within `tolerance` of the best."""
    best = min(curve.values())
    return max(lam for lam, loss in curve.items() if loss <= best + tolerance)


def tune_lambda(
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    *,
    grid: Sequence[float] = LAMBDA_GRID,
    validation_share: float = VALIDATION_SHARE,
) -> TuningResult:
    """Time-ordered validation over `grid`. Rows must be in time order.

    The grid is walked from the largest lambda down, warm-starting each fit
    from the last: strongly regularised solutions are close to zero and easy
    to find, and each weaker one is a small step from the previous.
    """
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    n_val = max(1, round(validation_share * n))
    train, val = slice(0, n - n_val), slice(n - n_val, n)

    if len(np.unique(y[train])) < 2:
        return TuningResult(
            lam=FALLBACK_LAMBDA,
            train_size=n - n_val,
            validation_size=n_val,
            note="training split has one class; lambda not tuned",
        )

    estimator = LogisticRegression(
        solver="lbfgs", tol=SOLVER_TOL, max_iter=SOLVER_MAX_ITER, warm_start=True
    )
    curve: dict[float, float] = {}
    for lam in sorted(grid, reverse=True):
        w, b = fit_sklearn(X[train], y[train], s[train], lam, estimator=estimator)
        p = expit(X[val] @ w + b)
        curve[lam] = weighted_log_loss(y[val], p, s[val])

    return TuningResult(
        lam=choose_lambda(curve),
        curve=dict(sorted(curve.items())),
        train_size=n - n_val,
        validation_size=n_val,
    )


def decay_weights(count: int, halflife: float) -> np.ndarray:
    """0.5 ** (age / halflife), age counted in labels, newest label age 0."""
    age = np.arange(count - 1, -1, -1, dtype=np.float64)
    return np.power(0.5, age / halflife)


def blend_ranks(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """(1 - weight) * rank(first) + weight * rank(second), ranks scaled to [0, 1].

    Ranks rather than raw scores, because a probability and a kNN similarity
    live on unrelated scales.
    """
    n = len(first)
    scale = 1.0 / max(1, n - 1)
    rank_first = np.argsort(np.argsort(first, kind="stable"), kind="stable") * scale
    rank_second = np.argsort(np.argsort(second, kind="stable"), kind="stable") * scale
    return ((1.0 - weight) * rank_first + weight * rank_second).astype(np.float32)


#: Liked images named in an explanation, and how many recent likes are searched.
NEAREST_LIKED = 3
NEAREST_LIKED_POOL = 500


class LogisticRecommender(Recommender):
    """R3: weighted, L2-regularised logistic regression on standardised features."""

    rung = "r3"
    min_labels = MIN_LABELS

    def __init__(
        self,
        *,
        blocks: Sequence[str] = DEFAULT_BLOCKS,
        lam: float | None = None,
        grid: Sequence[float] = LAMBDA_GRID,
        min_per_class: int = MIN_PER_CLASS,
    ) -> None:
        super().__init__()
        self.blocks = tuple(blocks)
        #: A fixed lambda, or None to tune on every fit (the default).
        self.fixed_lam = lam
        self.grid = tuple(grid)
        self.min_per_class = min_per_class
        #: ROADMAP D2. 0 (the default) is plain R3. Above 0, a kNN model is fitted
        #: on the same labels and the pool is ranked by
        #: (1 - knn_blend) * rank(R3) + knn_blend * rank(kNN): H2 found R3 the
        #: better-calibrated model and kNN the better ranker.
        self.knn_blend: float = 0.0
        #: ROADMAP D6. None (the default) weighs every label alike. A number H
        #: halves a label's weight for every H newer labels, so a taste that
        #: has drifted away fades from the fit (S5). Applied before the
        #: posterior, so R4's uncertainty matches what R3 fits.
        self.label_halflife: float | None = None
        #: ROADMAP E1. Tune lambda on every k-th fit and reuse the last choice
        #: in between. 1 (the default) tunes every fit, as every recorded study
        #: did; tuning is most of R3's fitting time, so k = 3 is roughly 3x
        #: faster at the cost of a lambda up to 2 refits stale.
        self.tune_every: int = 1
        self._fits = 0
        self._knn = None

        # The fitted parameters, named after the maths (see logreg_numpy.py).
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.lam: float | None = None
        self.tuning: TuningResult | None = None
        self.train_log_loss: float | None = None

        # R3+R5 only (blocks include "int"): the interests, their pool columns.
        self.seed_for_interests = 1729
        self.concept_labeller = None
        self.interest_tracker = None
        self._pool_interests: np.ndarray | None = None
        self._train_X: np.ndarray | None = None
        self._train_s: np.ndarray | None = None
        #: Liked images at fit time, newest last, for "close to these likes".
        self._liked_ids: list[str] = []
        #: ((image, verdict) history of the last fit, {boundary: InterestTracker}).
        self._interest_snapshots: tuple[list[tuple[str, str]], dict] = ([], {})

    # --- Fitting -------------------------------------------------------------

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        positives, negatives = labels.class_balance()
        if positives < self.min_per_class or negatives < self.min_per_class:
            raise InsufficientSignal(
                f"{positives} positive and {negatives} negative labels; r3 needs "
                f"at least {self.min_per_class} of each"
            )

        rows = features.positions(labels.image_ids)
        self._liked_ids = [i for i in labels.positives if features.position(i) is not None]
        # Labels on images no longer in the pool (quarantined later) are
        # dropped, so targets, weights and actions are dropped with them.
        keep = np.array([features.position(i) is not None for i in labels.image_ids], dtype=bool)
        y = labels.targets[keep].astype(np.float64)
        s = labels.weights[keep].astype(np.float64)
        self._drop_unavailable_optional_blocks(features)
        X = design_matrix(features, self.static_blocks)[rows].astype(np.float64)

        if self.uses_interests:
            actions = [a for a, k in zip(labels.actions, keep, strict=True) if k]
            X = np.hstack([X, self._training_interest_features(features, rows, actions)])
            self._refresh_pool_interests(features, rows, actions)

        # Kept for R4, whose posterior curvature needs exactly this matrix.
        if self.label_halflife is not None:
            s = s * decay_weights(len(s), self.label_halflife)

        self._train_X, self._train_s = X, s

        if len(np.unique(y)) < 2:
            raise InsufficientSignal("all usable labels are one class")

        if self.fixed_lam is None:
            reuse = self.tune_every > 1 and self._fits % self.tune_every != 0
            if not (reuse and self.lam is not None):
                self.tuning = tune_lambda(X, y, s, grid=self.grid)
                self.lam = self.tuning.lam
            self._fits += 1
        else:
            self.tuning = None
            self.lam = self.fixed_lam

        self.w, self.b = fit_sklearn(X, y, s, self.lam)
        if self.knn_blend > 0:
            from visualtaste.rec.r2_knn import KnnRecommender

            knn = KnnRecommender().fit(labels, features)
            self._knn = knn if knn.is_fitted else None
        self.train_log_loss = weighted_log_loss(y, expit(X @ self.w + self.b), s)
        logger.info(
            "r3 [%s]: lambda %.0e on %d labels, train log loss %.4f",
            "+".join(self.blocks),
            self.lam,
            len(y),
            self.train_log_loss,
        )

    # --- Scoring -------------------------------------------------------------

    def pool_logits(self, features: FeatureBundle, W: np.ndarray) -> np.ndarray:
        """X W + b for every pool row and every column of W.

        With interests, X = [static | interest] and the product is split,
        static @ W_static + interest @ W_interest, so the 50,000-row static
        matrix is never copied into a wider one on the request path.
        """
        W = np.asarray(W, dtype=np.float32)
        static = design_matrix(features, self.static_blocks)
        width = static.shape[1]
        out = static @ W[:width]
        if self.uses_interests:
            out = out + self._interest_columns(features) @ W[width:]
        return out + np.float32(self.b)

    def row_vector(self, features: FeatureBundle, image_index: int) -> np.ndarray:
        """One image's full feature row, for explanations."""
        x = design_matrix(features, self.static_blocks)[image_index]
        if self.uses_interests:
            x = np.concatenate([x, self._interest_columns(features)[image_index]])
        return x.astype(np.float64)

    def _interest_columns(self, features: FeatureBundle) -> np.ndarray:
        """The pool's interest columns, built from the tracker on first use after a load."""
        if self._pool_interests is None:
            if self.interest_tracker is None:
                raise UnfittedModelError(f"{self.rung} uses interests but has none loaded")
            self._pool_interests = scale_interest_features(
                self.interest_tracker.state, features.siglip
            ).astype(np.float32)
        return self._pool_interests

    def logits(self, features: FeatureBundle) -> np.ndarray:
        """w . x + b for every pool row: one matrix-vector product (F2, #13)."""
        require_fitted(self)
        assert self.w is not None
        return self.pool_logits(features, self.w[:, None])[:, 0]

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """p(like) for each candidate: sigma(w . x + b), computed over the whole pool.

        The whole-pool result is cached until anything it depends on changes:
        the weights (compared by value), the bias, the blocks, the design
        matrix and interest columns (by identity; both are replaced, never
        edited, when they change). Between refits every batch reuses it, which
        takes a 50,000-row product off each feed request. R4 overrides the
        batch path with fresh Thompson draws, so this only serves plain R3.
        """
        require_fitted(self)
        assert self.w is not None
        static = design_matrix(features, self.static_blocks)
        interests = self._interest_columns(features) if self.uses_interests else None
        key = (self.w.tobytes(), self.b, self.blocks, self.knn_blend)
        held = (features, static, interests, self._knn)
        cached = getattr(self, "_pool_scores", None)
        if (
            cached is None
            or cached[0] != key
            or any(a is not b for a, b in zip(cached[1], held, strict=True))
        ):
            scores = expit(self.logits(features)).astype(np.float32)
            if self.knn_blend > 0 and self._knn is not None:
                # Ranks, not probabilities: the blend is for ordering only.
                knn = self._knn._score_pool(features.siglip)
                scores = blend_ranks(scores, knn, self.knn_blend)
            scores.setflags(write=False)
            cached = self._pool_scores = (key, held, scores)
        out = cached[2][candidates]
        # A slice would be a view of the cache; hand out a copy instead.
        return out.copy() if np.shares_memory(out, cached[2]) else out

    # --- Interest features (R3+R5) -------------------------------------------

    @property
    def static_blocks(self) -> tuple[str, ...]:
        return tuple(b for b in self.blocks if b != INTEREST_BLOCK)

    @property
    def uses_interests(self) -> bool:
        return INTEREST_BLOCK in self.blocks

    def _drop_unavailable_optional_blocks(self, features: FeatureBundle) -> None:
        """Fit without `dino`/`col` when the bundle lacks them (a corpus not yet
        embedded with DINOv2, or a small test bundle); `sig` is never optional."""
        missing = [
            b
            for b in self.static_blocks
            if b in OPTIONAL_BLOCKS and not block_available(features, b)
        ]
        if missing:
            logger.warning("%s: blocks %s unavailable; fitting without them", self.rung, missing)
            self.blocks = tuple(b for b in self.blocks if b not in missing)

    def _training_interest_features(
        self, features: FeatureBundle, rows: np.ndarray, actions: list[str]
    ) -> np.ndarray:
        """Interest columns for each training example, from strictly earlier likes.

        The tracker's state at each 25-label boundary is a snapshot. Retraining
        every 20 labels only appends to the history, so snapshots for
        boundaries whose prefix is unchanged are reused from the previous fit,
        and only new boundaries cost a k-means. Without this, every retrain
        would recluster the whole history from the start (DECISIONS #24).
        """
        import copy

        from visualtaste.rec.features import time_respecting_features
        from visualtaste.rec.r5_interests import InterestTracker

        # The cache key is (image, verdict): a changed verdict on the same image
        # changes the history even though the ids do not.
        ids = [(features.image_ids[int(r)], a) for r, a in zip(rows, actions, strict=True)]
        liked_rows = [int(r) for r, a in zip(rows, actions, strict=True) if a in ("like", "save")]
        like_index = np.cumsum([a in ("like", "save") for a in actions])  # likes up to and incl. i

        previous_ids, previous = self._interest_snapshots
        snapshots: dict[int, InterestTracker] = {
            b: tracker
            for b, tracker in previous.items()
            if b <= len(previous_ids) and ids[:b] == previous_ids[:b]
        }

        def state_for_prefix(b: int):
            if b not in snapshots:
                earlier = [k for k in snapshots if k < b]
                tracker = (
                    copy.deepcopy(snapshots[max(earlier)])
                    if earlier
                    else InterestTracker(self.seed_for_interests)
                )
                likes_before = int(like_index[b - 1]) if b > 0 else 0
                if likes_before:
                    tracker.update(features.siglip[liked_rows[:likes_before]])
                snapshots[b] = tracker
            return snapshots[b].state

        def featurise(state, indices: np.ndarray) -> np.ndarray:
            return scale_interest_features(state, features.siglip[rows[indices]])

        out = time_respecting_features(len(rows), state_for_prefix, featurise).astype(np.float64)
        self._interest_snapshots = (ids, snapshots)
        return out

    def _refresh_pool_interests(
        self, features: FeatureBundle, rows: np.ndarray, actions: list[str]
    ) -> None:
        """Interests from *all* likes, for serving; pool columns cached until the next fit.

        Continues from the latest training snapshot rather than starting over,
        so the interests served are the same lineage, with the same ids, as the
        ones the training features were built from.
        """
        import copy

        from visualtaste.rec.r5_interests import InterestTracker

        liked_rows = [int(r) for r, a in zip(rows, actions, strict=True) if a in ("like", "save")]
        _, snapshots = self._interest_snapshots
        latest = max(snapshots) if snapshots else None
        self.interest_tracker = (
            copy.deepcopy(snapshots[latest])
            if latest is not None
            else InterestTracker(self.seed_for_interests)
        )
        if liked_rows:
            self.interest_tracker.update(
                features.siglip[liked_rows], concept_labeller=self.concept_labeller
            )
        self._pool_interests = scale_interest_features(
            self.interest_tracker.state, features.siglip
        ).astype(np.float32)

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        """The predicted probability (reason `model_confident`).

        Also names the three feature dimensions that pushed hardest towards
        "like" for this image -- weight times value -- which is what the
        Model Lab shows when a card is inspected.
        """
        require_fitted(self)
        assert self.w is not None
        x = self.row_vector(features, image_index)
        contribution = self.w * x
        top = np.argsort(-contribution)[:3]
        probability = float(expit(x @ self.w + self.b))
        return Explanation(
            reason="model_confident",
            detail={
                "probability": round(probability, 4),
                "top_dimensions": [
                    {"dim": int(i), "contribution": round(float(contribution[i]), 4)} for i in top
                ],
                "blocks": list(self.blocks),
                "nearest_liked": self.nearest_liked(features, image_index),
            },
        )

    def nearest_liked(self, features: FeatureBundle, image_index: int) -> list[dict[str, Any]]:
        """The liked images most like this one (SigLIP cosine), for the explanation.

        Feature dimensions explain the arithmetic; three liked images explain
        the decision to a person. Searched over the most recent likes only, so
        the cost stays flat however many labels there are.
        """
        # The recent likes and their vectors change only on a refit, but a
        # batch asks five times: gather them once per (fit, bundle).
        key = (id(features), id(self._liked_ids), len(self._liked_ids))
        cached = self.__dict__.get("_nearest_cache")
        if cached is None or cached[0] != key:
            recent = [
                (image_id, position)
                for image_id in self._liked_ids[-NEAREST_LIKED_POOL:]
                if (position := features.position(image_id)) is not None
            ]
            positions = np.array([position for _, position in recent], dtype=np.int64)
            vectors = features.siglip[positions] if recent else None
            cached = (key, recent, positions, vectors)
            self._nearest_cache = cached
        _, recent, positions, vectors = cached
        keep = positions != image_index
        if not keep.any():
            return []
        ids = [image_id for (image_id, _), k in zip(recent, keep, strict=True) if k]
        similarity = (vectors @ features.siglip[image_index])[keep]
        best = np.argsort(-similarity)[:NEAREST_LIKED]
        return [{"image_id": ids[i], "similarity": round(float(similarity[i]), 3)} for i in best]

    # --- Reporting and persistence -------------------------------------------

    def fit_report(self) -> dict[str, Any]:
        """The metrics recorded in `rec_models.metrics_json` for this version."""
        report: dict[str, Any] = {
            "lambda": self.lam,
            "train_log_loss": self.train_log_loss,
            "trained_on_events": self.trained_on_events,
        }
        if self.tuning is not None:
            curve = self.tuning.curve
            report["validation_curve"] = {f"{lam:g}": loss for lam, loss in curve.items()}
            report["validation_log_loss"] = self.tuning.curve.get(self.lam) if self.lam else None
            report["validation_size"] = self.tuning.validation_size
            if self.tuning.note:
                report["tuning_note"] = self.tuning.note
        return report

    @property
    def params(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "min_labels": self.min_labels,
            "min_per_class": self.min_per_class,
            "blocks": list(self.blocks),
            "lambda": self.lam,
            "lambda_tuned": self.fixed_lam is None,
            "grid": list(self.grid),
            "solver": f"sklearn lbfgs tol={SOLVER_TOL:g}",
        }

    def _save(self, path: Path) -> None:
        if self.w is None:
            return
        with (path / "weights.npy").open("wb") as handle:
            np.save(handle, self.w)
        (path / "model.json").write_text(
            json.dumps(
                {
                    "b": self.b,
                    "lambda": self.lam,
                    "blocks": list(self.blocks),
                    "liked_ids": self._liked_ids[-NEAREST_LIKED_POOL:],
                    "knn_blend": self.knn_blend,
                }
            ),
            encoding="utf-8",
        )
        if self.uses_interests and self.interest_tracker is not None:
            from visualtaste.rec.r5_interests import save_interests

            save_interests(path, self.interest_tracker)
        if self._knn is not None:
            (path / "knn").mkdir(exist_ok=True)
            self._knn._save(path / "knn")

    def _load(self, path: Path) -> None:
        if not self.is_fitted:
            return
        # A fitted version without its weights is a broken artefact; raising
        # OSError lets the API fall back to cold start rather than serving zeros.
        self.w = np.load(path / "weights.npy")
        meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
        self.b = float(meta["b"])
        self.lam = meta.get("lambda")
        self.blocks = tuple(meta.get("blocks", self.blocks))
        # Absent from artefacts saved before explanations named liked images.
        self._liked_ids = list(meta.get("liked_ids", []))
        self.knn_blend = float(meta.get("knn_blend", 0.0))
        if self.knn_blend > 0 and (path / "knn").exists():
            from visualtaste.rec.r2_knn import KnnRecommender

            self._knn = KnnRecommender()
            self._knn._load(path / "knn")
        if self.uses_interests:
            from visualtaste.rec.r5_interests import InterestTracker, load_interests

            self.interest_tracker = InterestTracker(self.seed_for_interests)
            if (path / "interests.json").exists():
                load_interests(path, self.interest_tracker)
            self._pool_interests = None  # rebuilt from the tracker on first score
