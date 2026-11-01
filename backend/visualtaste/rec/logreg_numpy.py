"""Logistic regression by hand: the loss, its gradient, and gradient descent.

This is the teaching implementation of R3 (maths unit M5). Production uses
scikit-learn (`rec/r3_logreg.py`); a test asserts the two find the same
weights to within 1e-3, which is the evidence that this file is correct and
that scikit-learn is being called with the right settings.

Intuition
---------
A logistic regression draws a flat boundary -- a hyperplane -- through feature
space and says "the further an image is on the liked side, the more likely he
likes it". The weight vector `w` points across the boundary, towards "liked";
the bias `b` slides it back and forth. Training means finding the boundary that
makes the observed likes and dislikes as unsurprising as possible, while
keeping `w` short so that a few lucky examples cannot make it extreme.

The model
---------
    z_i = w . x_i + b                 the logit: signed distance from the boundary
    p_i = sigma(z_i) = 1 / (1 + e^(-z_i))    probability of a like

The objective (Part 2 §5.2 -- use exactly this)
-----------------------------------------------
    L(w, b) = (1/S) sum_i s_i CE(y_i, p_i) + (lam/2) ||w||^2

    CE(y, p) = -[ y ln p + (1 - y) ln(1 - p) ]      cross-entropy
    s_i      the example's weight (save 2.0, like 1.0, dislike 1.0, skip 0.25)
    S        = sum_i s_i, so the data term is a weighted *mean*
    lam      the L2 strength; the bias b is deliberately not penalised

Why the bias is not penalised: penalising it would pull the predicted like rate
towards 50% regardless of the data. Alen may genuinely like 20% of the corpus;
that is a fact about him, not overfitting.

The gradient
------------
The derivative of the cross-entropy through the sigmoid collapses to the
prediction error, which is the one fact that makes logistic regression pleasant:

    dL/dw = (1/S) sum_i s_i (p_i - y_i) x_i + lam w
    dL/db = (1/S) sum_i s_i (p_i - y_i)

An example pushes `w` along its own feature vector, by how wrong it was,
scaled by its weight. A confident correct prediction (p near y) barely moves
anything; a confident wrong one moves it a lot.

Worked example (M5, asserted in `tests/test_logreg.py`)
-------------------------------------------------------
One example, no penalty: w = (2, -1), b = 0, x = (0.8, 0.6), y = 0.

    z = 2(0.8) + (-1)(0.6) + 0 = 1.0
    p = sigma(1.0)             = 0.731
    loss = -ln(1 - 0.731)      = 1.313       he disliked it; we said 73% like

    p - y = 0.731
    dL/dw = 0.731 (0.8, 0.6)   = (0.585, 0.439)
    dL/db = 0.731

One step with learning rate eta = 0.5:

    w <- (2, -1) - 0.5 (0.585, 0.439) = (1.708, -1.219)
    b <- 0 - 0.5 (0.731)              = -0.366
    new p = sigma(1.708(0.8) - 1.219(0.6) - 0.366) = sigma(0.269) = 0.567

The prediction moved from 0.731 towards the truth (0).

Optimiser: full-batch gradient descent with backtracking
--------------------------------------------------------
Each iteration uses every example (a few hundred to a few thousand, so this is
cheap). The step size is found by *backtracking*: try eta, and halve it until
the loss actually falls by a sensible amount (the Armijo condition):

    L(theta - eta g) <= L(theta) - c eta ||g||^2,   c = 1e-4

Then the next iteration starts from a slightly larger eta, so the step grows
back when the surface allows. Training stops when ||g|| < `tol`: at the
minimum the gradient is exactly zero, so its size measures how far off we are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: Gradient-norm tolerance for convergence. Tight, because the agreement test
#: compares coefficients with scikit-learn's to within 1e-3.
DEFAULT_TOL = 1e-8
DEFAULT_MAX_ITER = 200_000

#: Armijo sufficient-decrease constant, and the backtracking factor.
ARMIJO_C = 1e-4
BACKTRACK = 0.5
#: How much the step is allowed to grow after a successful iteration.
STEP_GROWTH = 1.5


def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-z), computed without overflow for large |z|.

    The naive formula overflows `exp(-z)` for z below about -709. Splitting on
    the sign keeps every exponent non-positive.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def objective(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    lam: float,
) -> float:
    """L(w, b) from the module docstring.

    Uses log(1 + e^z) - y z, an algebraically identical form of the
    cross-entropy that stays finite when p rounds to exactly 0 or 1.
    """
    z = X @ w + b
    # CE = log(1 + e^z) - y z, with logaddexp(0, z) = log(1 + e^z) computed stably.
    cross_entropy = np.logaddexp(0.0, z) - y * z
    S = s.sum()
    return float((s * cross_entropy).sum() / S + 0.5 * lam * (w @ w))


def gradient(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float]:
    """(dL/dw, dL/db), exactly as in the module docstring."""
    p = sigmoid(X @ w + b)
    S = s.sum()
    error = s * (p - y) / S  # each example's weighted prediction error, / S
    grad_w = X.T @ error + lam * w
    grad_b = float(error.sum())
    return grad_w, grad_b


def gd_step(
    w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, s: np.ndarray, lam: float, eta: float
) -> tuple[np.ndarray, float]:
    """One plain gradient-descent step with a fixed learning rate (for M5)."""
    grad_w, grad_b = gradient(w, b, X, y, s, lam)
    return w - eta * grad_w, b - eta * grad_b


@dataclass
class FitResult:
    """Weights, plus the evidence of how training went."""

    w: np.ndarray
    b: float
    iterations: int
    converged: bool
    gradient_norm: float
    loss_history: list[float] = field(default_factory=list)


def fit(
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray | None = None,
    *,
    lam: float = 1e-2,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    record_every: int = 1,
) -> FitResult:
    """Minimise L(w, b) by full-batch gradient descent with backtracking.

    `X` is (n, d), `y` holds 0/1 targets, `s` the example weights (all 1 when
    omitted). Starts from w = 0, b = 0: every prediction 0.5, the most honest
    guess before seeing anything.

    `loss_history` records the objective every `record_every` iterations, which
    is the loss curve notebook 03 plots.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = np.ones(len(y)) if s is None else np.asarray(s, dtype=np.float64)

    w = np.zeros(X.shape[1])
    b = 0.0
    eta = 1.0
    loss = objective(w, b, X, y, s, lam)
    history = [loss]

    grad_norm = np.inf
    iteration = 0
    for iteration in range(1, max_iter + 1):
        grad_w, grad_b = gradient(w, b, X, y, s, lam)
        grad_sq = float(grad_w @ grad_w + grad_b * grad_b)
        grad_norm = grad_sq**0.5
        if grad_norm < tol:
            break

        # Backtrack until the step decreases the loss enough (Armijo).
        while True:
            w_new = w - eta * grad_w
            b_new = b - eta * grad_b
            loss_new = objective(w_new, b_new, X, y, s, lam)
            if loss_new <= loss - ARMIJO_C * eta * grad_sq or eta < 1e-16:
                break
            eta *= BACKTRACK

        w, b, loss = w_new, b_new, loss_new
        eta *= STEP_GROWTH
        if iteration % record_every == 0:
            history.append(loss)

    converged = grad_norm < tol
    if not converged:
        logger.warning(
            "gradient descent stopped after %d iterations with |g| = %.2e (tol %.0e)",
            iteration,
            grad_norm,
            tol,
        )
    return FitResult(
        w=w,
        b=b,
        iterations=iteration,
        converged=converged,
        gradient_norm=float(grad_norm),
        loss_history=history,
    )
