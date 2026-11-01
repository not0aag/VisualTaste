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
    require_fitted,
)
from visualtaste.rec.features import DEFAULT_BLOCKS, design_matrix

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

        # The fitted parameters, named after the maths (see logreg_numpy.py).
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.lam: float | None = None
        self.tuning: TuningResult | None = None
        self.train_log_loss: float | None = None

    # --- Fitting -------------------------------------------------------------

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        positives, negatives = labels.class_balance()
        if positives < self.min_per_class or negatives < self.min_per_class:
            raise InsufficientSignal(
                f"{positives} positive and {negatives} negative labels; r3 needs "
                f"at least {self.min_per_class} of each"
            )

        X_pool = design_matrix(features, self.blocks)
        rows = features.positions(labels.image_ids)
        if len(rows) != len(labels):
            # Labels on images no longer in the pool (quarantined later) are
            # dropped, so targets and weights must be dropped with them.
            keep = [features.position(i) is not None for i in labels.image_ids]
            y = labels.targets[keep].astype(np.float64)
            s = labels.weights[keep].astype(np.float64)
        else:
            y = labels.targets.astype(np.float64)
            s = labels.weights.astype(np.float64)
        X = X_pool[rows].astype(np.float64)

        if len(np.unique(y)) < 2:
            raise InsufficientSignal("all usable labels are one class")

        if self.fixed_lam is None:
            self.tuning = tune_lambda(X, y, s, grid=self.grid)
            self.lam = self.tuning.lam
        else:
            self.tuning = None
            self.lam = self.fixed_lam

        self.w, self.b = fit_sklearn(X, y, s, self.lam)
        self.train_log_loss = weighted_log_loss(y, expit(X @ self.w + self.b), s)
        logger.info(
            "r3 [%s]: lambda %.0e on %d labels, train log loss %.4f",
            "+".join(self.blocks),
            self.lam,
            len(y),
            self.train_log_loss,
        )

    # --- Scoring -------------------------------------------------------------

    def logits(self, features: FeatureBundle) -> np.ndarray:
        """w . x + b for every pool row: one matrix-vector product (F2, #13)."""
        require_fitted(self)
        assert self.w is not None
        X = design_matrix(features, self.blocks)
        return X @ self.w.astype(np.float32) + np.float32(self.b)

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """p(like) for each candidate: sigma(w . x + b), computed over the whole pool."""
        return expit(self.logits(features)).astype(np.float32)[candidates]

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        """The predicted probability (reason `model_confident`).

        Also names the three feature dimensions that pushed hardest towards
        "like" for this image -- weight times value -- which is what the
        Model Lab shows when a card is inspected.
        """
        require_fitted(self)
        assert self.w is not None
        x = design_matrix(features, self.blocks)[image_index].astype(np.float64)
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
            },
        )

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
            json.dumps({"b": self.b, "lambda": self.lam, "blocks": list(self.blocks)}),
            encoding="utf-8",
        )

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
