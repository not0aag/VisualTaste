"""R6: a small neural ranker over every feature block (brief §7.5, Part 2 §8).

Intuition
---------
R3 draws one flat boundary through feature space. Some tastes are not flat:
"I like warm palettes, but only in architecture" is an AND of a colour
condition and a content condition, and no weighted sum of features can say
AND (simulator variant S4 is built to need exactly this). A neural network
stacks two learned non-linear layers between the features and the score, so
it can carve the space into regions a line cannot. The price: thousands of
parameters instead of 128, so it needs far more labels before it beats the
simpler model -- if it ever does. H5 measures when.

The network
-----------
    input   [SigLIP 768 | DINOv2 768 | colour 64 | profile 8]     = 1608
            each block LayerNorm-ed on its own, so no block dominates by scale
    h1      GELU(W1 x + b1)          256 units, then dropout 0.3
    h2      GELU(W2 h1 + b2)          64 units, then dropout 0.3
    logit   w3 . h2 + b3              p(like) = sigma(logit)

Per-block LayerNorm: z = (x - mean(x)) / std(x) * gamma + beta, over the
features of one image's block. SigLIP and DINOv2 vectors are unit-length while
the profile features are on their own scales; normalising each block lets the
first layer see them on equal footing.

The profile features (the 8 that describe the *person*)
--------------------------------------------------------
    1 cos to the Rocchio vector       4 max similarity to dislikes
    2 m(x), the R5 soft-max interest  5 mean of the top-5 like similarities
    3 max similarity to likes         6 distance to the nearest seen image
    7 how often x's cluster has been shown (log scale)
    8 the image's source collection (scaled to 0..1)

For a training example at time t they are computed from labels strictly before
t (`rec/features.time_respecting_features`); computing them from all labels
would let the network read the answer off feature 3.

The loss
--------
Pointwise: weighted binary cross-entropy, as R3, without the L2 term (AdamW's
weight decay plays that part). Optionally plus 0.5 x BPR, a *pairwise* loss over
(liked, disliked) pairs from the same session:

    BPR = -log sigma(s_pos - s_neg)

which only cares that a liked image outscores a disliked one, not by how much.
Worked example (M9, asserted in `tests/test_neural.py`): s_pos = 1.2, s_neg =
0.4 gives -log sigma(0.8) = 0.371. Sessions are blocks of 100 consecutive
labels (the log has no reliable session boundaries; see DECISIONS #25).

Training
--------
AdamW (lr 1e-3, weight decay 1e-2), batch 256, at most 200 epochs, stopping
when the log loss on the latest 15% of labels has not improved for 10 epochs,
and keeping the best epoch's weights. Five seeds give five networks; the score
is their mean probability and their spread is the uncertainty. Exploration is
bootstrap Thompson: each batch is ranked by one member chosen at random, then
the same probe quota and MMR as R4.

Serving cost
------------
Five networks over 50,000 rows is too slow for every request on CPU, so the
pool's scores are computed once per fit (and once after loading) and the feed
indexes them. The profile features therefore describe the history as of the
last training run, which is the same staleness R3 has between retrains.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from visualtaste.rec.base import (
    Explanation,
    FeatureBundle,
    InsufficientSignal,
    Recommender,
    require_fitted,
)
from visualtaste.rec.features import time_respecting_features
from visualtaste.rec.rerank import (
    EXPLORE_EVERY,
    MMR_LAMBDA,
    cluster_ratings_from,
    explore_batch,
    explore_slots,
)

if TYPE_CHECKING:
    from visualtaste.rec.labels import LabelSet

logger = logging.getLogger(__name__)

# Brief §7.5.
HIDDEN = (256, 64)
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 10
VALIDATION_SHARE = 0.15
ENSEMBLE = 5
BPR_WEIGHT = 0.5
BPR_PAIRS_PER_POSITIVE = 20
SESSION_LABELS = 100
PROFILE_FEATURES = 8
EXPOSURE_SCALE = math.log1p(100)
POSITIVE = ("like", "save")


def bpr_loss(s_pos, s_neg):
    """-log sigma(s_pos - s_neg), averaged; written with softplus to stay finite."""
    import torch.nn.functional as F

    return F.softplus(-(s_pos - s_neg)).mean()


# ---------------------------------------------------------------------------
# Profile features
# ---------------------------------------------------------------------------


@dataclass
class ProfileState:
    """What the history says, at one moment, for the 8 profile features."""

    liked: np.ndarray
    disliked: np.ndarray
    seen: np.ndarray
    rocchio: np.ndarray | None
    interests: Any  # r5_interests.InterestState
    exposure: dict[int, int] = field(default_factory=dict)


def build_profile_state(
    features: FeatureBundle, rows: np.ndarray, actions: list[str], tracker
) -> ProfileState:
    """The state from labels at `rows` (pool row indices) with these `actions`.

    `tracker` is an `InterestTracker` updated in place, so successive calls
    over a growing prefix recluster incrementally rather than from scratch.
    """
    from visualtaste.rec.r1_rocchio import rocchio_vector

    rows = np.asarray(rows, dtype=np.int64)
    liked_rows = [int(r) for r, a in zip(rows, actions, strict=True) if a in POSITIVE]
    disliked_rows = [int(r) for r, a in zip(rows, actions, strict=True) if a == "dislike"]
    liked = features.siglip[liked_rows]
    disliked = features.siglip[disliked_rows]
    rocchio = rocchio_vector(liked, disliked).astype(np.float32) if len(liked) else None
    if len(liked):
        tracker.update(liked)
    exposure: dict[int, int] = {}
    if features.cluster64 is not None and len(rows):
        clusters, counts = np.unique(np.asarray(features.cluster64)[rows], return_counts=True)
        exposure = {int(c): int(n) for c, n in zip(clusters, counts, strict=True)}
    return ProfileState(liked, disliked, features.siglip[rows], rocchio, tracker.state, exposure)


def profile_features(state: ProfileState, features: FeatureBundle, rows: np.ndarray) -> np.ndarray:
    """The 8 profile columns for these pool rows, from one history state."""
    from visualtaste.rec.r5_interests import logsumexp_score

    rows = np.asarray(rows, dtype=np.int64)
    x = features.siglip[rows].astype(np.float32)
    out = np.zeros((len(rows), PROFILE_FEATURES), dtype=np.float32)
    out[:, 5] = 1.0  # nothing seen yet: maximally novel
    if state.rocchio is not None:
        out[:, 0] = x @ state.rocchio
    if state.interests.k:
        out[:, 1] = logsumexp_score(state.interests.similarities(x))
    if len(state.liked):
        like_sims = x @ state.liked.T
        out[:, 2] = like_sims.max(axis=1)
        top = min(5, like_sims.shape[1])
        out[:, 4] = -np.sort(-like_sims, axis=1)[:, :top].mean(axis=1)
    if len(state.disliked):
        out[:, 3] = (x @ state.disliked.T).max(axis=1)
    if len(state.seen):
        out[:, 5] = 1.0 - (x @ state.seen.T).max(axis=1)
    if features.cluster64 is not None:
        clusters = np.asarray(features.cluster64)[rows]
        out[:, 6] = np.log1p([state.exposure.get(int(c), 0) for c in clusters]) / EXPOSURE_SCALE
    source = features.extra.get("source_index")
    if source is not None:
        source = np.asarray(source, dtype=np.float32)
        out[:, 7] = source[rows] / max(float(source.max()), 1.0)
    return out


def _blocks_for(features: FeatureBundle, rows: np.ndarray, profile: np.ndarray) -> list[np.ndarray]:
    blocks = [features.siglip[rows]]
    if features.dinov2 is not None:
        blocks.append(features.dinov2[rows])
    if features.colour is not None:
        blocks.append(features.colour[rows])
    blocks.append(profile)
    return [np.asarray(b, dtype=np.float32) for b in blocks]


# ---------------------------------------------------------------------------
# The network and its training
# ---------------------------------------------------------------------------


def make_network(sizes: list[int]):
    import torch
    from torch import nn

    class RankerNet(nn.Module):
        """Per-block LayerNorm, then 1608 -> 256 -> 64 -> 1 with GELU and dropout."""

        def __init__(self, block_sizes: list[int]) -> None:
            super().__init__()
            self.block_sizes = list(block_sizes)
            self.norms = nn.ModuleList(nn.LayerNorm(size) for size in block_sizes)
            self.mlp = nn.Sequential(
                nn.Linear(sum(block_sizes), HIDDEN[0]),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN[0], HIDDEN[1]),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN[1], 1),
            )

        def forward(self, x):
            parts = torch.split(x, self.block_sizes, dim=1)
            normed = [norm(part) for norm, part in zip(self.norms, parts, strict=True)]
            return self.mlp(torch.cat(normed, dim=1)).squeeze(1)

    return RankerNet(sizes)


def session_pairs(targets: np.ndarray, actions: list[str], rng: np.random.Generator) -> np.ndarray:
    """(positive, negative) index pairs within each 100-label block, <= 20 per positive."""
    pairs = []
    for start in range(0, len(targets), SESSION_LABELS):
        block = np.arange(start, min(start + SESSION_LABELS, len(targets)))
        positives = [i for i in block if targets[i] == 1]
        negatives = np.array([i for i in block if actions[i] == "dislike"])
        if not len(negatives):
            continue
        for p in positives:
            chosen = rng.choice(
                negatives, size=min(BPR_PAIRS_PER_POSITIVE, len(negatives)), replace=False
            )
            pairs.extend((p, int(n)) for n in chosen)
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


@dataclass
class MemberReport:
    seed: int
    best_epoch: int
    epochs_run: int
    best_validation_log_loss: float
    train_curve: list[float]
    validation_curve: list[float]


def train_member(
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    actions: list[str],
    sizes: list[int],
    *,
    seed: int,
    loss: str = "pointwise",
    device: str = "cpu",
    max_epochs: int = MAX_EPOCHS,
):
    """Train one network; early stopping on the time-ordered last 15%."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(VALIDATION_SHARE * len(y)))
    train, val = slice(0, len(y) - n_val), slice(len(y) - n_val, len(y))

    net = make_network(sizes).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    X_train, y_train, s_train = tensor(X[train]), tensor(y[train]), tensor(s[train])
    X_val, y_val, s_val = tensor(X[val]), tensor(y[val]), tensor(s[val])
    pairs = session_pairs(y[train], actions[train], rng) if loss == "bpr" else np.zeros((0, 2), int)

    best_state, best_loss, best_epoch, stale = None, math.inf, 0, 0
    train_curve, val_curve = [], []
    for epoch in range(max_epochs):
        net.train()
        order = torch.as_tensor(rng.permutation(len(y_train)), device=device)
        running = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            logits = net(X_train[batch])
            weights = s_train[batch]
            bce = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y_train[batch], weight=weights, reduction="sum"
                )
                / weights.sum()
            )
            objective = bce
            if len(pairs):
                chosen = pairs[rng.integers(0, len(pairs), size=min(BATCH_SIZE, len(pairs)))]
                both = net(X_train[torch.as_tensor(chosen.ravel(), device=device)]).view(-1, 2)
                objective = objective + BPR_WEIGHT * bpr_loss(both[:, 0], both[:, 1])
            optimiser.zero_grad()
            objective.backward()
            optimiser.step()
            running += float(bce.detach()) * len(batch)
        train_curve.append(running / len(order))

        net.eval()
        with torch.no_grad():
            val_loss = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    net(X_val), y_val, weight=s_val, reduction="sum"
                )
                / s_val.sum()
            )
        val_curve.append(val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    report = MemberReport(seed, best_epoch, len(val_curve), best_loss, train_curve, val_curve)
    return net, report


# ---------------------------------------------------------------------------
# The rung
# ---------------------------------------------------------------------------


class NeuralRecommender(Recommender):
    """R6: an ensemble of five small networks, exploring by bootstrap Thompson."""

    rung = "r6"
    stochastic = True

    def __init__(
        self,
        *,
        min_labels: int = 1500,
        loss: str = "pointwise",
        members: int = ENSEMBLE,
        seed: int = 1729,
        device: str | None = None,
        explore_every: int | None = EXPLORE_EVERY,
        mmr_lambda: float = MMR_LAMBDA,
        max_epochs: int = MAX_EPOCHS,
    ) -> None:
        super().__init__()
        self.min_labels = min_labels
        self.loss = loss
        self.members = members
        self.seed = seed
        self.device = device
        self.explore_every = explore_every
        self.mmr_lambda = mmr_lambda
        self.max_epochs = max_epochs

        self.nets: list[Any] = []
        self.sizes: list[int] = []
        self.reports: list[MemberReport] = []
        self.history_ids: list[str] = []
        self.history_actions: list[str] = []
        self.cluster_ratings: dict[int, tuple[int, int]] = {}
        self.labels_at_fit = 0
        self.cards_served = 0
        self.batches_served = 0
        self.last_batch: tuple[int, int] = (0, 0)
        self.train_seconds = 0.0
        self.peak_vram_mb = 0.0
        self._pool_scores: np.ndarray | None = None  # (members, pool)
        self._explore_rows: dict[int, dict[str, Any]] = {}

    def _device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    # --- Fitting -------------------------------------------------------------

    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        import torch

        from visualtaste.rec.r5_interests import InterestTracker

        keep = [features.position(i) is not None for i in labels.image_ids]
        ids = [i for i, k in zip(labels.image_ids, keep, strict=True) if k]
        actions = [a for a, k in zip(labels.actions, keep, strict=True) if k]
        y = labels.targets[keep].astype(np.float32)
        s = labels.weights[keep].astype(np.float32)
        if (y == 1).sum() < 5 or (y == 0).sum() < 5:
            raise InsufficientSignal("r6 needs at least 5 labels of each class")
        rows = features.positions(ids)

        tracker = InterestTracker(self.seed)

        def state_for_prefix(b: int) -> ProfileState:
            return build_profile_state(features, rows[:b], actions[:b], tracker)

        def featurise(state: ProfileState, indices: np.ndarray) -> np.ndarray:
            return profile_features(state, features, rows[indices])

        profile = time_respecting_features(len(rows), state_for_prefix, featurise)
        blocks = _blocks_for(features, rows, profile)
        self.sizes = [b.shape[1] for b in blocks]
        X = np.hstack(blocks)

        device = self._device()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.nets, self.reports = [], []
        for member in range(self.members):
            net, report = train_member(
                X,
                y,
                s,
                np.array(actions, dtype=object),
                self.sizes,
                seed=self.seed + member,
                loss=self.loss,
                device=device,
                max_epochs=self.max_epochs,
            )
            self.nets.append(net)
            self.reports.append(report)
        self.train_seconds = time.perf_counter() - started
        self.peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0

        self.history_ids, self.history_actions = ids, actions
        self.cluster_ratings = cluster_ratings_from(labels, features)
        self.labels_at_fit = len(labels)
        self._pool_scores = None
        logger.info(
            "r6 (%s): %d members on %d labels in %.1fs; best validation log loss %s",
            self.loss,
            self.members,
            len(y),
            self.train_seconds,
            [round(r.best_validation_log_loss, 4) for r in self.reports],
        )

    # --- Scoring -------------------------------------------------------------

    def member_scores(self, features: FeatureBundle, rows: np.ndarray) -> np.ndarray:
        """p(like) from each member for these rows: (members, rows)."""
        import torch

        from visualtaste.rec.r5_interests import InterestTracker

        positions = features.positions(self.history_ids)
        state = build_profile_state(
            features, positions, self.history_actions, InterestTracker(self.seed)
        )
        device = next(self.nets[0].parameters()).device
        out = np.zeros((len(self.nets), len(rows)), dtype=np.float32)
        for start in range(0, len(rows), 8_192):
            chunk = np.asarray(rows[start : start + 8_192])
            X = np.hstack(_blocks_for(features, chunk, profile_features(state, features, chunk)))
            with torch.no_grad():
                tensor = torch.as_tensor(X, device=device)
                for m, net in enumerate(self.nets):
                    out[m, start : start + len(chunk)] = torch.sigmoid(net(tensor)).cpu().numpy()
        return out

    def pool_scores(self, features: FeatureBundle) -> np.ndarray:
        """Every member's score for the whole pool, computed once per fit or load."""
        require_fitted(self)
        if self._pool_scores is None or self._pool_scores.shape[1] != len(features):
            self._pool_scores = self.member_scores(features, np.arange(len(features)))
        return self._pool_scores

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """The ensemble mean, from the per-fit cache when scoring the pool."""
        require_fitted(self)
        if self._pool_scores is None and len(candidates) < len(features) // 2:
            # A few rows (replay's test block): score them directly.
            return self.member_scores(features, np.asarray(candidates)).mean(axis=0)
        return self.pool_scores(features).mean(axis=0)[candidates]

    def uncertainty(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """The ensemble's standard deviation: how much the five networks disagree."""
        return self.pool_scores(features).std(axis=0)[candidates]

    def rerank(self, features: FeatureBundle, candidates: np.ndarray, scores: np.ndarray, n: int):
        """Bootstrap Thompson: rank by one randomly chosen member, then probe + MMR."""
        rng = np.random.default_rng((self.seed, self.batches_served))
        member = int(rng.integers(len(self.nets)))
        sampled = self.pool_scores(features)[member][candidates]
        n_explore = explore_slots(
            self.cards_served, n, self.labels_at_fit, every=self.explore_every
        )
        mmr, explore = explore_batch(
            features,
            candidates,
            sampled,
            n,
            n_explore,
            rng,
            cluster_ratings=self.cluster_ratings,
            mmr_lambda=self.mmr_lambda,
        )
        slots = mmr + explore
        chosen = np.array([slots[i] for i in rng.permutation(len(slots))], dtype=np.int64)
        self._explore_rows = {
            int(candidates[o]): {
                "cluster64": int(np.asarray(features.cluster64)[int(candidates[o])]),
                "ratings": list(
                    self.cluster_ratings.get(
                        int(np.asarray(features.cluster64)[int(candidates[o])]), (0, 0)
                    )
                ),
                "slot_before_shuffle": slots.index(o),
            }
            for o in explore
        }
        self.batches_served += 1
        self.cards_served += len(chosen)
        self.last_batch = (n, n_explore)
        return chosen, sampled.astype(np.float32), None

    def inclusion_probabilities(
        self, features, candidates, shown, *, n, n_explore, draws=200, seed=0
    ) -> np.ndarray:
        """Monte-Carlo P(shown): replay member choice, probe and MMR `draws` times."""
        rng = np.random.default_rng((self.seed, seed, 1))
        scores = self.pool_scores(features)[:, candidates]
        shown = [int(s) for s in shown]
        counts = np.zeros(len(shown))
        for _ in range(draws):
            member = int(rng.integers(len(self.nets)))
            mmr, explore = explore_batch(
                features,
                candidates,
                scores[member],
                n,
                n_explore,
                rng,
                cluster_ratings=self.cluster_ratings,
                mmr_lambda=self.mmr_lambda,
            )
            batch = set(mmr) | set(explore)
            counts += [offset in batch for offset in shown]
        return counts / draws

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        require_fitted(self)
        probe = self._explore_rows.get(int(image_index))
        if probe is not None:
            return Explanation(reason="explore", detail=dict(probe))
        scores = self.pool_scores(features)[:, image_index]
        return Explanation(
            reason="model_confident",
            detail={
                "probability": round(float(scores.mean()), 4),
                "ensemble_sd": round(float(scores.std()), 4),
            },
        )

    # --- Reporting and persistence -------------------------------------------

    def fit_report(self) -> dict[str, Any]:
        return {
            "trained_on_events": self.trained_on_events,
            "loss": self.loss,
            "members": [
                {
                    "seed": r.seed,
                    "best_epoch": r.best_epoch,
                    "epochs_run": r.epochs_run,
                    "best_validation_log_loss": r.best_validation_log_loss,
                }
                for r in self.reports
            ],
            "validation_log_loss": float(
                np.mean([r.best_validation_log_loss for r in self.reports])
            )
            if self.reports
            else None,
            "train_seconds": round(self.train_seconds, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
        }

    @property
    def params(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "min_labels": self.min_labels,
            "loss": self.loss,
            "members": self.members,
            "hidden": list(HIDDEN),
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": self.max_epochs,
            "patience": PATIENCE,
            "seed": self.seed,
        }

    def _save(self, path: Path) -> None:
        import torch

        if not self.nets:
            return
        for m, net in enumerate(self.nets):
            torch.save(net.state_dict(), path / f"member_{m}.pt")
        (path / "r6.json").write_text(
            json.dumps(
                {
                    "sizes": self.sizes,
                    "history_ids": self.history_ids,
                    "history_actions": self.history_actions,
                    "cluster_ratings": {str(k): v for k, v in self.cluster_ratings.items()},
                    "labels_at_fit": self.labels_at_fit,
                    "reports": [r.__dict__ for r in self.reports],
                    "members": len(self.nets),
                }
            ),
            encoding="utf-8",
        )

    def _load(self, path: Path) -> None:
        import torch

        if not self.is_fitted:
            return
        meta = json.loads((path / "r6.json").read_text(encoding="utf-8"))
        self.sizes = meta["sizes"]
        self.history_ids = meta["history_ids"]
        self.history_actions = meta["history_actions"]
        self.cluster_ratings = {int(k): tuple(v) for k, v in meta["cluster_ratings"].items()}
        self.labels_at_fit = meta["labels_at_fit"]
        self.reports = [MemberReport(**r) for r in meta["reports"]]
        device = self._device()
        self.nets = []
        for m in range(meta["members"]):
            net = make_network(self.sizes).to(device)
            net.load_state_dict(torch.load(path / f"member_{m}.pt", map_location=device))
            net.eval()
            self.nets.append(net)
        self._pool_scores = None
