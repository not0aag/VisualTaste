"""The feature matrices the learned rungs train on, and why they are standardised.

Three blocks (Part 2 §5.1)
--------------------------
    sig    SigLIP 2, projected to PCA-128      what the image *depicts*
    dino   DINOv2,   projected to PCA-128      how it *looks*: texture, layout
    col    colour64, the Hellinger histogram   its palette

A model asks for blocks by name -- `("sig",)`, `("sig", "dino", "col")` -- and gets
one matrix with the blocks side by side. That is how the Phase 9 ablation swaps
feature sets without touching the model.

Why standardise
---------------
Logistic regression's L2 penalty is `(lam/2) * ||w||^2`: it charges every weight
the same price. But PCA components are not the same size. On the 50,000-image
pool the first SigLIP component has standard deviation 0.184 and the 128th has
0.029 -- a factor of six. To have the same effect on the score, a weight on the
128th component must be six times larger than one on the first, and so pays
*thirty-six* times the penalty. Unstandardised, the penalty is not "prefer
simple models"; it is "ignore the small components", which is a different and
unintended prior.

So every column is rescaled to mean 0 and standard deviation 1 over the pool:

    z_ij = (x_ij - mean_j) / std_j

after which the penalty treats every dimension equally. Colour needs it even
more: its 64 bins range from almost-always-empty to almost-always-full.

Worked example (asserted in `tests/test_features.py`): a column holding
(1, 2, 3, 6) has mean 3 and population std sqrt(3.5) = 1.871, so it becomes
(-1.069, -0.535, 0.0, 1.604).

Pool statistics, not training statistics
----------------------------------------
The mean and std come from the whole 50,000-image pool, not from the few hundred
labelled images. Two reasons. The labelled images are a biased sample -- they are
what the feed chose to show -- so their statistics drift as the model changes,
and every retrain would see a slightly different feature space. And the pool is
fixed, so its statistics can be computed once, saved next to the PCA
(`<stem>_standardise.npz`, written by `jobs.select`), and reused by every model
version. This is not leakage: the pool contains no labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from visualtaste.rec.base import FeatureBundle

logger = logging.getLogger(__name__)

#: Block name -> where its raw values live in a `FeatureBundle`, in order of
#: preference. `sig` falls back to the raw SigLIP vectors and `dino` to the raw
#: DINOv2 ones only for small synthetic test bundles that have no PCA.
BLOCK_SOURCES: dict[str, tuple[str, ...]] = {
    "sig": ("pca", "siglip"),
    "dino": ("dinov2_pca", "dinov2"),
    "col": ("colour",),
}

#: The feature files `jobs.select` writes, per block, for standardisation stats.
STATS_STEMS: dict[str, str] = {"sig": "pca128", "dino": "dinov2_pca128", "col": "colour64"}

#: R3's blocks: `sig+dino` by the §5.6 rule applied to phase9_ablation
#: (DECISIONS #28). `col` helped on S3 but cost 0.014 log loss on S1.
DEFAULT_BLOCKS: tuple[str, ...] = ("sig", "dino")
#: Blocks a model drops, with a warning, when the bundle does not have them.
OPTIONAL_BLOCKS: frozenset[str] = frozenset({"dino", "col"})

#: A column whose std is below this is constant over the pool (an always-empty
#: colour bin). It is centred but not scaled, so it stays exactly zero rather
#: than being divided by almost nothing and exploding.
MIN_STD = 1e-6


def standardise(
    matrix: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(z, mean, std)` with `z = (matrix - mean) / std`, column-wise.

    Statistics are computed from `matrix` when not given. Population std
    (ddof=0), because the pool is the whole population the model will ever
    score, not a sample of one.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if mean is None:
        mean = matrix.mean(axis=0)
    if std is None:
        std = matrix.std(axis=0)
    safe_std = np.where(std < MIN_STD, 1.0, std).astype(np.float32)
    z = (matrix - mean.astype(np.float32)) / safe_std
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def save_stats(features_dir: Path, stem: str, matrix: np.ndarray) -> Path:
    """Write the pool mean and std of `matrix` next to its feature file."""
    _, mean, std = standardise(matrix)
    path = features_dir / f"{stem}_standardise.npz"
    tmp = path.with_name(f".{path.name}.part")
    with tmp.open("wb") as handle:
        np.savez(handle, mean=mean, std=std)
    tmp.replace(path)
    return path


def load_stats(features_dir: Path | None, block: str) -> tuple[np.ndarray, np.ndarray] | None:
    if features_dir is None:
        return None
    path = features_dir / f"{STATS_STEMS[block]}_standardise.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["mean"], data["std"]


def _raw_block(features: FeatureBundle, block: str) -> np.ndarray:
    if block not in BLOCK_SOURCES:
        raise KeyError(f"unknown feature block {block!r}; expected one of {sorted(BLOCK_SOURCES)}")
    for source in BLOCK_SOURCES[block]:
        array = getattr(features, source, None)
        if array is None:
            array = features.extra.get(source)
        if array is not None:
            return np.asarray(array)
    raise KeyError(
        f"feature block {block!r} is unavailable: none of {BLOCK_SOURCES[block]} is loaded"
    )


def standardised_block(
    features: FeatureBundle, block: str, *, features_dir: Path | None = None
) -> np.ndarray:
    """One standardised block over the whole pool, cached in `features.extra`.

    Cached because the feed calls this on every request and the block never
    changes for the life of the process; recomputing 50,000 x 128 values per
    request would spend a quarter of the latency budget on arithmetic whose
    answer is already known.
    """
    key = f"std:{block}"
    cached = features.extra.get(key)
    if cached is not None:
        return cached

    raw = _raw_block(features, block)
    stats = load_stats(features_dir, block)
    if stats is not None and stats[0].shape[0] == raw.shape[1]:
        z, _, _ = standardise(raw, *stats)
    else:
        z, _, _ = standardise(raw)
    features.extra[key] = z
    return z


def design_matrix(
    features: FeatureBundle,
    blocks: Sequence[str] = DEFAULT_BLOCKS,
    *,
    features_dir: Path | None = None,
) -> np.ndarray:
    """The blocks side by side: one row per pool image, standardised columns.

    Cached per combination, for the same reason as `standardised_block`: the
    concatenation is a 50,000-row copy, and the feed must not pay for it on
    every request.
    """
    blocks = tuple(blocks)
    key = "design:" + "+".join(blocks)
    cached = features.extra.get(key)
    if cached is not None:
        return cached

    parts = [standardised_block(features, block, features_dir=features_dir) for block in blocks]
    matrix = parts[0] if len(parts) == 1 else np.hstack(parts).astype(np.float32)
    features.extra[key] = matrix
    logger.info("design matrix %s: %d x %d", "+".join(blocks), *matrix.shape)
    return matrix


def block_available(features: FeatureBundle, block: str) -> bool:
    try:
        _raw_block(features, block)
    except KeyError:
        return False
    return True


# ---------------------------------------------------------------------------
# Time-respecting history features (Part 2 §4.6)
# ---------------------------------------------------------------------------
#
# Some features describe the *person*, not the image: how close an image is to
# their interests (R3+R5), how near it is to their likes (R6). Those features
# are computed from the history, and that is where leakage hides.
#
# Train a model on label i with a feature computed from all labels, including
# i itself and everything after, and the feature has seen the answer. The
# model learns "images near the interests are liked", and it is right -- the
# interests were built from those very likes. Replay metrics then look superb
# and serving, where the future is genuinely unknown, does not.
#
# So for a training example at position t (time order), history features are
# computed from labels strictly before t. Rebuilding the state for every
# example would be quadratic, so the state is built at `boundaries` (every 25
# labels by default) and each example uses the most recent boundary at or
# before it -- a state built from labels [0, b) with b <= t, never from t or
# later. The cost is that a feature can be up to 24 labels stale, which is the
# same staleness the live system has between retrains.


HISTORY_BOUNDARY_EVERY = 25


def history_boundaries(n_examples: int, every: int = HISTORY_BOUNDARY_EVERY) -> list[int]:
    """0, every, 2*every, ... below `n_examples`."""
    return list(range(0, max(n_examples, 1), every))


def time_respecting_features(
    n_examples: int,
    state_for_prefix,
    featurise,
    *,
    every: int = HISTORY_BOUNDARY_EVERY,
) -> np.ndarray:
    """History features for examples 0..n-1, each from strictly earlier labels.

    `state_for_prefix(b)` builds whatever summary of the history is needed from
    the first `b` labels (it is called with increasing b, so it may be
    incremental). `featurise(state, indices)` turns that state into one feature
    row per example index. `tests/test_features.py` plants a future label and
    asserts no earlier example's features change.
    """
    boundaries = history_boundaries(n_examples, every)
    blocks = []
    for position, start in enumerate(boundaries):
        end = boundaries[position + 1] if position + 1 < len(boundaries) else n_examples
        state = state_for_prefix(start)
        blocks.append(np.asarray(featurise(state, np.arange(start, end)), dtype=np.float32))
    return np.vstack(blocks) if blocks else np.zeros((0, 0), dtype=np.float32)
