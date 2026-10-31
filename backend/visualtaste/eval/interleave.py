"""Team-draft interleaving: comparing two models on the same swipes.

Why this is the strongest comparison available here
---------------------------------------------------
Every other method in this project compares two models on *different* data, or
on estimated counterfactuals:

- replay compares them on the same logged history, but neither one chose it;
- IPS reweights one policy's log to guess at another's, with weights so noisy
  the estimate is usually unquotable.

Interleaving avoids both problems. Both models rank the candidates, their
choices are merged into one batch, and Alen swipes without knowing which model
contributed which card. Each like is credited to whichever model put that image
in the batch. Since both models are judged by the same person in the same
session on the same five-card batches, almost every confounder cancels.

The mechanism (brief §7.7)
--------------------------
Team-draft interleaving, by analogy with picking teams in a playground:

1. flip a coin for who picks first;
2. that model takes its highest-ranked candidate not already taken;
3. the other model takes its highest-ranked candidate not already taken;
4. alternate until the batch is full.

The coin flip per batch is what removes position bias. Without it, whichever
model picked first would always own slot 1, and slot 1 gets more attention than
slot 5 -- so the measured difference would partly be a measurement of position.

Reading the result
------------------
Each batch yields a winner: the model whose cards attracted more likes. Ties
(including the common case of no likes at all) are discarded, and the win
counts go to a sign test. With twenty batches the test can detect only a large
difference; `InterleavingResult.summarise` says so rather than reporting a
direction from noise.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from visualtaste.eval.metrics import sign_test

logger = logging.getLogger(__name__)

POSITIVE_ACTIONS = frozenset({"like", "save"})


@dataclass
class InterleavedBatch:
    """One merged batch: which model contributed each slot."""

    batch_id: str
    #: Candidate index per slot, in the order they will be shown.
    chosen: list[int]
    #: "a" or "b" per slot, naming the model that picked it.
    arms: list[str]
    #: Which arm won the coin flip for the first pick.
    first_pick: str

    def arm_of(self, slot: int) -> str:
        return self.arms[slot]


def team_draft(
    ranking_a: list[int],
    ranking_b: list[int],
    *,
    slots: int,
    rng: np.random.Generator,
    batch_id: str = "",
) -> InterleavedBatch:
    """Merge two rankings into one batch, alternating picks after a coin flip.

    `ranking_a` and `ranking_b` are candidate indices, best first. The same
    candidate may appear in both; whoever picks it first owns it, and the other
    model moves to its next unused choice -- which is what makes the credit
    assignment unambiguous.

    Worked example: A ranks [1, 2, 3], B ranks [2, 1, 4], the coin gives A the
    first pick, and there are four slots.

        A picks 1          -> chosen [1],       arms [a]
        B picks 2          -> chosen [1, 2],    arms [a, b]
        A picks 3  (2 gone) -> chosen [1, 2, 3], arms [a, b, a]
        B picks 4  (1 gone) -> chosen [1,2,3,4], arms [a, b, a, b]
    """
    chosen: list[int] = []
    arms: list[str] = []
    taken: set[int] = set()

    cursor = {"a": 0, "b": 0}
    rankings = {"a": ranking_a, "b": ranking_b}

    first = "a" if rng.random() < 0.5 else "b"
    turn = first

    while len(chosen) < slots:
        ranking = rankings[turn]

        # Advance past anything the other model already took.
        while cursor[turn] < len(ranking) and ranking[cursor[turn]] in taken:
            cursor[turn] += 1

        if cursor[turn] < len(ranking):
            candidate = ranking[cursor[turn]]
            cursor[turn] += 1
            chosen.append(candidate)
            arms.append(turn)
            taken.add(candidate)
        elif all(
            cursor[arm] >= len(rankings[arm])
            or all(c in taken for c in rankings[arm][cursor[arm] :])
            for arm in ("a", "b")
        ):
            # Both rankings exhausted: a short batch is the honest outcome
            # rather than padding it with repeats.
            break

        turn = "b" if turn == "a" else "a"

    return InterleavedBatch(batch_id=batch_id, chosen=chosen, arms=arms, first_pick=first)


@dataclass
class InterleavingResult:
    """Accumulated outcome of an interleaving experiment."""

    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    likes_a: int = 0
    likes_b: int = 0
    impressions_a: int = 0
    impressions_b: int = 0
    per_batch: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def comparisons(self) -> int:
        """Batches that produced a winner. Ties carry no information."""
        return self.wins_a + self.wins_b

    @property
    def p_value(self) -> float:
        return sign_test(self.wins_a, self.wins_b)

    @property
    def like_rate_a(self) -> float:
        return self.likes_a / self.impressions_a if self.impressions_a else float("nan")

    @property
    def like_rate_b(self) -> float:
        return self.likes_b / self.impressions_b if self.impressions_b else float("nan")

    def record_batch(self, batch: InterleavedBatch, actions: list[str]) -> None:
        """Credit one swiped batch to the two arms."""
        likes = {"a": 0, "b": 0}
        shown = {"a": 0, "b": 0}

        for slot, action in enumerate(actions[: len(batch.arms)]):
            arm = batch.arm_of(slot)
            shown[arm] += 1
            if action in POSITIVE_ACTIONS:
                likes[arm] += 1

        self.likes_a += likes["a"]
        self.likes_b += likes["b"]
        self.impressions_a += shown["a"]
        self.impressions_b += shown["b"]
        self.per_batch.append((batch.batch_id, likes["a"], likes["b"]))

        if likes["a"] > likes["b"]:
            self.wins_a += 1
        elif likes["b"] > likes["a"]:
            self.wins_b += 1
        else:
            self.ties += 1

    def summarise(self, label_a: str = "A", label_b: str = "B") -> str:
        """A sentence that reports the result, including when there is none."""
        if self.comparisons == 0:
            return (
                f"{label_a} vs {label_b}: {self.ties} batches, all tied. No "
                f"comparison yet -- a tie is usually a batch with no likes in it."
            )

        leader, trailing = (label_a, label_b) if self.wins_a > self.wins_b else (label_b, label_a)
        p = self.p_value
        verdict = (
            f"{leader} ahead, but p = {p:.3f} over {self.comparisons} comparisons, "
            f"so this is not yet distinguishable from chance"
            if p > 0.05
            else f"{leader} ahead of {trailing}, p = {p:.3f}"
        )
        return (
            f"{label_a} vs {label_b}: {self.wins_a} wins to {self.wins_b}, "
            f"{self.ties} tied. Like rates {self.like_rate_a:.3f} and "
            f"{self.like_rate_b:.3f}. {verdict}."
        )


def start_experiment(connection: sqlite3.Connection, *, version_a: str, version_b: str) -> int:
    """Record a new interleaving experiment and return its id."""
    cursor = connection.execute(
        """
        INSERT INTO experiments (version_a, version_b, method, started_at)
        VALUES (?, ?, 'team_draft', datetime('now'))
        """,
        (version_a, version_b),
    )
    connection.commit()
    return int(cursor.lastrowid or 0)


def finish_experiment(
    connection: sqlite3.Connection, experiment_id: int, result: InterleavingResult
) -> None:
    """Close an experiment, storing its outcome as JSON."""
    connection.execute(
        """
        UPDATE experiments
        SET ended_at = datetime('now'), result_json = ?
        WHERE experiment_id = ?
        """,
        (
            json.dumps(
                {
                    "wins_a": result.wins_a,
                    "wins_b": result.wins_b,
                    "ties": result.ties,
                    "likes_a": result.likes_a,
                    "likes_b": result.likes_b,
                    "impressions_a": result.impressions_a,
                    "impressions_b": result.impressions_b,
                    "like_rate_a": result.like_rate_a,
                    "like_rate_b": result.like_rate_b,
                    "p_value": result.p_value,
                }
            ),
            experiment_id,
        ),
    )
    connection.commit()


def result_from_impressions(
    connection: sqlite3.Connection, experiment_id: int
) -> InterleavingResult:
    """Rebuild an experiment's outcome from the logged impressions and events.

    Recomputed from the log rather than accumulated in memory, so the number in
    the Model Lab comes from the same rows as everything else and survives a
    restart mid-experiment.
    """
    rows = connection.execute(
        """
        SELECT i.batch_id, i.rank, i.arm, e.action
        FROM impressions i
        LEFT JOIN events e
          ON e.impression_id = i.impression_id AND e.undone = 0
        WHERE i.experiment_id = ? AND i.arm IS NOT NULL
        ORDER BY i.batch_id, i.rank
        """,
        (experiment_id,),
    ).fetchall()

    batches: dict[str, dict[str, list]] = {}
    for row in rows:
        entry = batches.setdefault(row["batch_id"], {"arms": [], "actions": []})
        entry["arms"].append(row["arm"])
        entry["actions"].append(row["action"] or "unseen")

    result = InterleavingResult()
    for batch_id, entry in batches.items():
        batch = InterleavedBatch(
            batch_id=batch_id,
            chosen=list(range(len(entry["arms"]))),
            arms=entry["arms"],
            first_pick=entry["arms"][0] if entry["arms"] else "a",
        )
        result.record_batch(batch, entry["actions"])
    return result
