"""Turn a history of swipes into training labels, which is a design decision.

This file looks like bookkeeping and is not. Every model in the ladder learns
from whatever this module says the labels are, so the choices here propagate
into every recommendation the system ever makes.

The mapping (brief §7.5)
------------------------
    action     y    weight    meaning
    -------    -    ------    -----------------------------------------------
    like       1    1.0       a positive example
    save       1    2.0       a positive example worth twice as much
    dislike    0    1.0       a negative example
    skip       0    0.25      a weak negative example

A weight multiplies that example's contribution to the loss. With weight 2.0 a
save counts like two likes; at 0.25 a skip counts like a quarter of a dislike.

Why a save is worth two likes
-----------------------------
Liking costs a flick. Saving means "I want to find this again", which is a
stronger and rarer signal. Weighting it higher lets a few hundred labels carry
more information than their count suggests.

Why a skip is a weak negative and not nothing
---------------------------------------------
A skip is genuinely ambiguous. It can mean "not for me", or "I am going too
fast to care", or "I like this but not enough to stop". Treating it as a full
dislike would be wrong; dropping it entirely throws away the most common action
in the log and, worse, biases the training set towards images that provoked a
reaction at all.

0.25 is a guess, and the brief says so: Phase 9 compares 0 / 0.25 / exclude on
the simulator and records which actually works. Until that measurement exists,
this number is a default, not a finding.

Dwell time is logged and never used as a label
----------------------------------------------
`events.dwell_ms` is recorded on every event and is deliberately not an input
here. It correlates with interest but also with distraction, with a slow
network, and with having walked away from the laptop. Phase 12 analyses whether
dwell predicts likes at all (as an analysis, reported honestly); it does not
become a label on the strength of a plausible story.

Re-rated images: the latest verdict wins
----------------------------------------
If Alen changes his mind from the History page, the earlier event is marked
`undone`/`superseded_by` and a new one is inserted. Both rows survive, because
the log is a factual record and time-ordered replay must be able to reconstruct
what was known at any past moment. But for *training*, only the latest
non-undone event for each image counts -- otherwise one image contributes two
contradictory examples.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Brief §7.5. Configurable because Phase 9 compares alternatives.
ACTION_TO_TARGET: dict[str, int] = {
    "like": 1,
    "save": 1,
    "dislike": 0,
    "skip": 0,
}

DEFAULT_ACTION_WEIGHTS: dict[str, float] = {
    "like": 1.0,
    "save": 2.0,
    "dislike": 1.0,
    "skip": 0.25,
}

POSITIVE_ACTIONS = frozenset({"like", "save"})
NEGATIVE_ACTIONS = frozenset({"dislike", "skip"})


@dataclass
class LabelSet:
    """The training set derived from one profile's history, in time order.

    Time order is not cosmetic. The validation split in R3 and R6 is the last
    15% *chronologically*, and `eval/replay.py` cuts the history at increasing
    points to measure what the model knew at each one. Shuffling here would
    quietly turn an honest time-ordered evaluation into a leaky random split.
    """

    image_ids: list[str] = field(default_factory=list)
    targets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    actions: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def positives(self) -> list[str]:
        """Liked and saved image ids, in time order."""
        return [
            image_id
            for image_id, action in zip(self.image_ids, self.actions, strict=True)
            if action in POSITIVE_ACTIONS
        ]

    @property
    def negatives(self) -> list[str]:
        return [
            image_id
            for image_id, action in zip(self.image_ids, self.actions, strict=True)
            if action in NEGATIVE_ACTIONS
        ]

    @property
    def strong_negatives(self) -> list[str]:
        """Disliked only -- skips excluded.

        Rocchio (R1) and kNN (R2) use this rather than all negatives: both
        subtract a *mean* of negative vectors, and a mean has no notion of
        weight, so including ambiguous skips at full strength would distort the
        preference vector.
        """
        return [
            image_id
            for image_id, action in zip(self.image_ids, self.actions, strict=True)
            if action == "dislike"
        ]

    @property
    def saves(self) -> list[str]:
        return [
            image_id
            for image_id, action in zip(self.image_ids, self.actions, strict=True)
            if action == "save"
        ]

    def counts(self) -> dict[str, int]:
        return {
            action: sum(1 for a in self.actions if a == action)
            for action in sorted(set(self.actions))
        }

    def class_balance(self) -> tuple[int, int]:
        """`(positives, negatives)` by target, for logging and sanity checks."""
        positive = int((self.targets == 1).sum())
        return positive, len(self.targets) - positive


def label_weights(
    skip_weight: float | None = None, *, save_weight: float | None = None
) -> dict[str, float]:
    """The weight table, with the two values Phase 9 varies overridable.

    `skip_weight=0.0` is the "skips count for nothing" arm of that comparison;
    excluding skips entirely is a separate option in `build_label_set`.
    """
    weights = dict(DEFAULT_ACTION_WEIGHTS)
    if skip_weight is not None:
        weights["skip"] = skip_weight
    if save_weight is not None:
        weights["save"] = save_weight
    return weights


def latest_events(connection: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    """One row per image: the most recent non-undone event, in time order.

    The SQL is worth reading closely, because "the latest verdict wins" is easy
    to get subtly wrong:

    - `undone = 0` drops events the user explicitly took back.
    - `superseded_by IS NULL` drops events replaced by a later verdict.
    - the `GROUP BY image_id` with `MAX(event_id)` then keeps the newest
      surviving row per image, so one image contributes exactly one example.

    Ordered by `event_id` rather than `created_at`: `created_at` has
    second-resolution, and twenty swipes can land in the same second, which
    would make the "time order" arbitrary within it. `event_id` is monotonic.
    """
    return connection.execute(
        """
        SELECT e.event_id, e.image_id, e.action, e.dwell_ms, e.created_at,
               e.impression_id
        FROM events e
        JOIN (
            SELECT image_id, MAX(event_id) AS newest
            FROM events
            WHERE profile_id = ? AND undone = 0 AND superseded_by IS NULL
            GROUP BY image_id
        ) latest ON e.event_id = latest.newest
        ORDER BY e.event_id
        """,
        (profile_id,),
    ).fetchall()


def build_label_set(
    connection: sqlite3.Connection,
    profile_id: int,
    *,
    weights: dict[str, float] | None = None,
    exclude_skips: bool = False,
) -> LabelSet:
    """Build the training set for one profile.

    `exclude_skips=True` is the third arm of the Phase 9 comparison: drop skips
    altogether rather than down-weighting them.
    """
    table = weights or DEFAULT_ACTION_WEIGHTS
    rows = latest_events(connection, profile_id)

    image_ids: list[str] = []
    targets: list[float] = []
    weight_values: list[float] = []
    actions: list[str] = []
    timestamps: list[str] = []

    for row in rows:
        action = row["action"]
        if action not in ACTION_TO_TARGET:
            # The schema has a CHECK constraint, so this means the constraint
            # was changed without updating this table.
            logger.warning("ignoring event with unknown action %r", action)
            continue
        if exclude_skips and action == "skip":
            continue

        weight = table.get(action, 0.0)
        if weight <= 0.0:
            # A zero-weight example contributes nothing to any loss, so it is
            # dropped rather than carried as a row of zeros.
            continue

        image_ids.append(row["image_id"])
        targets.append(float(ACTION_TO_TARGET[action]))
        weight_values.append(weight)
        actions.append(action)
        timestamps.append(row["created_at"])

    return LabelSet(
        image_ids=image_ids,
        targets=np.array(targets, dtype=np.float32),
        weights=np.array(weight_values, dtype=np.float32),
        actions=actions,
        timestamps=timestamps,
    )


def seen_image_ids(connection: sqlite3.Connection, profile_id: int) -> set[str]:
    """Every image this profile has acted on, including undone events.

    Deliberately broader than the training set: an image Alen rated and then
    un-rated has still been seen, and showing it again as if it were new would
    look like a bug. The feed excludes everything here.
    """
    return {
        row["image_id"]
        for row in connection.execute(
            "SELECT DISTINCT image_id FROM events WHERE profile_id = ?", (profile_id,)
        )
    }


def label_count(connection: sqlite3.Connection, profile_id: int) -> int:
    """How many usable labels this profile has. Drives the rung ladder.

    R0 hands over at 30, R3 becomes worth training at about 150, R6 refuses
    below 1,500 -- all measured against this number.
    """
    row = connection.execute(
        """
        SELECT COUNT(DISTINCT image_id) AS n
        FROM events
        WHERE profile_id = ? AND undone = 0 AND superseded_by IS NULL
        """,
        (profile_id,),
    ).fetchone()
    return int(row["n"]) if row else 0
