"""The interface every rung of the ladder implements, and why it is this shape.

The ladder (brief §7.5) runs from a rule with no parameters to a neural network:

    R0  cold start      round-robin over clusters; learns nothing
    R1  Rocchio         one vector: liked mean minus disliked mean
    R2  kNN             similarity to the nearest liked and disliked images
    R3  logistic        a trained linear classifier on PCA-128 features
    R4  Thompson        R3 plus a posterior, sampled per batch for exploration
    R5  multi-interest  several centroids instead of one preference vector
    R6  neural          a small MLP over all feature blocks

All seven answer the same three questions -- fit on this history, score these
candidates, explain this score -- so the feed, the evaluation harness and the
Model Lab can treat them interchangeably. That interchangeability is the point:
it is what makes "does R3 actually beat R2?" a question you can answer by
swapping one object, rather than an argument.

Two deliberate choices in the signature
---------------------------------------
**`score` returns an array, not a single value.** The feed scores every unseen
candidate on every request -- about 50,000 of them -- and then applies MMR and
the exploration quota to the whole ranking. One matrix operation does that;
50,000 function calls would not fit in the 50 ms budget.

**`explain` is part of the interface, not an add-on.** A recommender that
cannot say why it chose something is not inspectable, and this one exists partly
to be understood. Requiring every rung to implement `explain` means no rung can
be added without being explainable.

What `score` returns
--------------------
A number where higher means "show this sooner". Some rungs return a probability
(R3, R6), others a similarity in [-1, 1] (R1, R2). They are *not* comparable
across rungs, and nothing in the system compares them across rungs -- the feed
ranks within one model, and evaluation compares models by their outcomes rather
than by their raw scores.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from visualtaste.rec.labels import LabelSet

logger = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    """Everything a model may look at, assembled once per request or per fit.

    Passed as one object rather than as separate arguments so that adding a
    feature block (R5's interest similarities, R6's profile features) does not
    change the signature of every rung.

    `image_ids` is the index: row `i` of every array belongs to
    `image_ids[i]`. Every rung relies on that alignment, so it is asserted in
    `__post_init__` rather than trusted.
    """

    image_ids: list[str]
    siglip: np.ndarray
    dinov2: np.ndarray | None = None
    colour: np.ndarray | None = None
    pca: np.ndarray | None = None
    cluster64: np.ndarray | None = None
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = len(self.image_ids)
        for name in ("siglip", "dinov2", "colour", "pca", "cluster64"):
            block = getattr(self, name)
            if block is not None and len(block) != expected:
                raise ValueError(
                    f"feature block {name!r} has {len(block)} rows but there are "
                    f"{expected} image ids; every block must align with the index"
                )
        self._position_of = {image_id: i for i, image_id in enumerate(self.image_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def position(self, image_id: str) -> int | None:
        return self._position_of.get(image_id)

    def positions(self, image_ids: list[str]) -> np.ndarray:
        """Row numbers for these ids, silently skipping unknown ones.

        Unknown ids are skipped because a labelled image can be quarantined or
        folded into a duplicate afterwards; a profile referencing it must not
        break training.
        """
        found = [self._position_of[i] for i in image_ids if i in self._position_of]
        return np.array(found, dtype=np.int64)

    def rows(self, block: str, image_ids: list[str]) -> np.ndarray:
        """The named feature block, restricted to these ids."""
        array = getattr(self, block, None)
        if array is None:
            array = self.extra.get(block)
        if array is None:
            raise KeyError(f"no feature block named {block!r}")
        index = self.positions(image_ids)
        if len(index) == 0:
            return np.zeros((0, array.shape[1]), dtype=np.float32)
        return array[index]


@dataclass
class Explanation:
    """Why one image was scored or shown, in a form the UI can render.

    `reason` is one of the fixed codes in §7.6 (`cold_start`, `similar_likes`,
    `interest:<k>`, `explore`, `model_confident`), so the frontend can switch on
    it. `detail` carries whatever that code needs -- the three nearest liked
    ids, a cluster label, a probability. `sentence` is the one line shown on the
    card.

    The sentence is *derived* from the reason and the detail by
    `rec.explain.sentence_for` unless one is passed explicitly. Rungs therefore
    produce a code and its evidence and never phrasing, which is what keeps the
    wording consistent across seven models: when "close to three images you
    liked" needs rewording, there is one place to reword it, and a new rung
    cannot invent its own voice by accident.
    """

    reason: str
    sentence: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sentence:
            from visualtaste.rec.explain import sentence_for

            self.sentence = sentence_for(self.reason, self.detail)

    def to_json(self) -> str:
        return json.dumps({"reason": self.reason, "sentence": self.sentence, **self.detail})


class Recommender(ABC):
    """The common interface. Every rung R0-R6 is one of these.

    Subclasses implement `_fit`, `score` and `explain`; `fit` handles the
    bookkeeping every rung shares (recording how many labels it saw, and
    whether it is usable at all).
    """

    #: Short identifier used in `rec_models.rung` and the Model Lab, e.g. "r3".
    rung: str = "base"

    #: Minimum labels before this rung can produce anything meaningful. The
    #: feed falls back to the rung below when it is not met.
    min_labels: int = 0

    def __init__(self) -> None:
        self.trained_on_events: int = 0
        self.is_fitted: bool = False

    # --- Fitting -------------------------------------------------------------

    def fit(self, labels: LabelSet, features: FeatureBundle) -> Recommender:
        """Learn from this history. Returns `self` so calls can be chained.

        A rung is `is_fitted` only if it had enough labels to be meaningful,
        which is what lets the feed fall back cleanly rather than serving
        confident nonsense from four examples.
        """
        self.trained_on_events = len(labels)

        if len(labels) < self.min_labels:
            logger.info(
                "%s: %d labels is below the %d it needs; not fitted",
                self.rung,
                len(labels),
                self.min_labels,
            )
            self.is_fitted = False
            return self

        self._fit(labels, features)
        self.is_fitted = True
        logger.info("%s: fitted on %d labels", self.rung, len(labels))
        return self

    @abstractmethod
    def _fit(self, labels: LabelSet, features: FeatureBundle) -> None:
        """Rung-specific fitting. Called only when `min_labels` is satisfied."""

    # --- Scoring -------------------------------------------------------------

    @abstractmethod
    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        """Score candidate rows. `candidates` holds row indices into `features`.

        Returns one float per candidate, higher meaning "show sooner". Row
        indices rather than ids because the caller already has them and the
        conversion would be pure overhead on 50,000 candidates.
        """

    @abstractmethod
    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        """Why this image got the score it did."""

    # --- Persistence ---------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        """The hyperparameters that define this model, for `rec_models`.

        Recorded so a version in the Model Lab can be reproduced, and so a
        comparison between two versions can say what actually differed.
        """
        return {"rung": self.rung, "min_labels": self.min_labels}

    def save(self, path: Path) -> None:
        """Persist to a directory. Subclasses add their own artefacts."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "params.json").write_text(
            json.dumps(
                {
                    "rung": self.rung,
                    "params": self.params,
                    "trained_on_events": self.trained_on_events,
                    "is_fitted": self.is_fitted,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._save(path)

    def _save(self, path: Path) -> None:  # noqa: B027 - an optional hook, not abstract
        """Rung-specific artefacts.

        Deliberately concrete and empty: R0 has no parameters, so forcing every
        rung to implement this would mean writing empty methods to satisfy the
        interface rather than because anything needed saving.
        """

    def load(self, path: Path) -> Recommender:
        """Restore from a directory written by `save`."""
        meta = json.loads((path / "params.json").read_text(encoding="utf-8"))
        self.trained_on_events = int(meta.get("trained_on_events", 0))
        self.is_fitted = bool(meta.get("is_fitted", False))
        self._load(path)
        return self

    def _load(self, path: Path) -> None:  # noqa: B027 - an optional hook, not abstract
        """Rung-specific restoration. Empty by default, as `_save` is."""


class UnfittedModelError(RuntimeError):
    """Scoring was attempted on a model that has not been fitted."""


def require_fitted(model: Recommender) -> None:
    """Guard for `score`: a clear error beats a confusing array of zeros."""
    if not model.is_fitted:
        raise UnfittedModelError(
            f"{model.rung} has not been fitted (it saw "
            f"{model.trained_on_events} labels and needs {model.min_labels}). "
            f"The feed should have fallen back to a lower rung."
        )
