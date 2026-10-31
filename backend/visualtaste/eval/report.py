"""Running a simulated session through the real feed code, and writing it up.

The rule this module exists to enforce
--------------------------------------
The simulator drives **the same code path the browser does** -- `rec/feed.py`,
the same `Recommender`, the same impression writes. It does not reimplement
scoring "equivalently".

That matters because the failures worth catching live in the glue. A simulator
with its own scoring loop would happily report AUC 0.82 for a model whose
feature blocks are misaligned in production, whose impressions are never
written, or whose candidate pool silently excludes everything. All three of
those bugs actually occurred during this build, and all three were caught by
running the real path.

The cost is that the simulator is slower -- it writes to SQLite and goes through
the full candidate scoring on every batch. Several thousand swipes still take
under a minute, which is a price worth paying for results that mean something.

What a session produces
-----------------------
A `SessionResult` holding the action sequence, the per-batch impressions, and
the simulated user who generated it (so the planted interests can be compared
against what the model found). From that, `summarise_session` computes the
metrics §7.7 asks for, and `render_markdown` writes the table that goes into
`docs/EVALUATION.md` -- with the simulator caveat attached, every time.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from visualtaste.eval.metrics import (
    coverage,
    intra_batch_diversity,
    like_rate,
    novelty,
    rolling_like_rate,
)
from visualtaste.eval.sim import SimulatedUser, create_simulated_profile
from visualtaste.rec.feed import build_batch

logger = logging.getLogger(__name__)

COLDSTART_CLUSTERS = 64


@dataclass
class SessionResult:
    """Everything one simulated session produced."""

    rung: str
    seed: int
    actions: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    batch_diversities: list[float] = field(default_factory=list)
    shown_clusters: list[int] = field(default_factory=list)
    novelties: list[float] = field(default_factory=list)
    user: SimulatedUser | None = None
    profile_id: int | None = None

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class SessionSummary:
    """The metrics §7.7 asks for, for one session."""

    rung: str
    seed: int
    swipes: int
    like_rate: float
    like_rate_last_100: float
    mean_batch_diversity: float
    coverage: float
    mean_novelty: float
    interests_recalled: float
    save_count: int
    skip_count: int


def run_session(
    connection: sqlite3.Connection,
    features,
    model,
    user: SimulatedUser,
    *,
    swipes: int,
    seed: int,
    batch_size: int = 5,
    refit_every: int = 20,
    rung_label: str | None = None,
) -> SessionResult:
    """Swipe `swipes` times through the real feed path.

    `refit_every` mirrors the online retraining the API does (every 20 labels by
    default, §7.5). Refitting on the simulator's own schedule rather than once
    at the start is deliberate: a model evaluated after training on all the data
    it will ever see is not the model that would have served those batches.
    """
    from visualtaste.rec.labels import build_label_set

    profile_id = create_simulated_profile(connection, note=f"simulator {model.rung} seed {seed}")
    connection.commit()

    result = SessionResult(
        rung=rung_label or model.rung, seed=seed, user=user, profile_id=profile_id
    )
    clusters = np.asarray(features.cluster64) if features.cluster64 is not None else None
    seen_positions: list[int] = []

    while len(result.actions) < swipes:
        batch = build_batch(
            connection,
            features,
            model,
            profile_id=profile_id,
            model_version=f"sim_{model.rung}_{seed}",
            n=batch_size,
        )
        if not batch.cards:
            logger.info("%s: the pool ran out after %d swipes", model.rung, len(result.actions))
            break

        positions = [features.position(card.image_id) for card in batch.cards]
        positions = [p for p in positions if p is not None]
        batch_vectors = features.siglip[positions]

        # Diversity and novelty are properties of the batch as offered, so they
        # are measured before the user reacts to it.
        result.batch_diversities.append(intra_batch_diversity(batch_vectors))
        if seen_positions:
            result.novelties.extend(
                novelty(batch_vectors, features.siglip[seen_positions]).tolist()
            )
        else:
            result.novelties.extend([1.0] * len(batch_vectors))

        actions = user.react(batch_vectors)

        with_events = zip(batch.cards, actions, strict=False)
        for card, action in with_events:
            connection.execute(
                """
                INSERT INTO events (impression_id, profile_id, image_id, action, dwell_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (card.impression_id, profile_id, card.image_id, action, 1_000),
            )
            result.actions.append(action)
            result.image_ids.append(card.image_id)
            if clusters is not None:
                position = features.position(card.image_id)
                if position is not None:
                    result.shown_clusters.append(int(clusters[position]))
        connection.commit()

        seen_positions.extend(positions)

        # Retrain on the schedule the live system uses.
        if refit_every and len(result.actions) % refit_every < batch_size:
            labels = build_label_set(connection, profile_id)
            model.fit(labels, features)

    logger.info(
        "%s seed %d: %d swipes, like rate %.3f",
        result.rung,
        seed,
        len(result.actions),
        like_rate(result.actions),
    )
    return result


def summarise_session(result: SessionResult) -> SessionSummary:
    """Reduce a session to the numbers §7.7 asks for."""
    recent = result.actions[-100:]
    return SessionSummary(
        rung=result.rung,
        seed=result.seed,
        swipes=len(result.actions),
        like_rate=like_rate(result.actions),
        like_rate_last_100=like_rate(recent) if recent else float("nan"),
        mean_batch_diversity=(
            float(np.mean(result.batch_diversities)) if result.batch_diversities else float("nan")
        ),
        coverage=coverage(result.shown_clusters, COLDSTART_CLUSTERS),
        mean_novelty=float(np.mean(result.novelties)) if result.novelties else float("nan"),
        interests_recalled=(
            result.user.recall_of_planted_interests() if result.user else float("nan")
        ),
        save_count=sum(1 for action in result.actions if action == "save"),
        skip_count=sum(1 for action in result.actions if action == "skip"),
    )


def aggregate(summaries: list[SessionSummary]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation per metric across seeds.

    Reported as mean ± sd across five seeds rather than one number, because a
    single simulator run differs from another by more than most of the rung
    differences being measured. A result quoted without its spread invites
    reading noise as a finding.
    """
    if not summaries:
        return {}

    fields = (
        "like_rate",
        "like_rate_last_100",
        "mean_batch_diversity",
        "coverage",
        "mean_novelty",
        "interests_recalled",
    )
    out: dict[str, tuple[float, float]] = {}
    for name in fields:
        values = np.array([getattr(s, name) for s in summaries], dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values):
            out[name] = (float(values.mean()), float(values.std(ddof=0)))
    return out


def _cell(stats: dict[str, tuple[float, float]], name: str, places: int = 3) -> str:
    """One `mean ± sd` table cell, or `--` when the metric was not computed.

    Module-level rather than a closure inside the loop: a nested function would
    capture the loop variable by reference, which works only because it is
    called immediately and breaks the moment anyone defers the call.
    """
    if name not in stats:
        return "--"
    mean, sd = stats[name]
    return f"{mean:.{places}f} ± {sd:.{places}f}"


def render_markdown(by_rung: dict[str, list[SessionSummary]], *, swipes: int, seeds: int) -> str:
    """The results table for `docs/EVALUATION.md`, caveat attached."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### Simulator study ({now})",
        "",
        f"**These are simulated results.** {seeds} seeds x {swipes} swipes per "
        f"rung, against a planted-taste user with three hidden interest "
        f"centroids. The simulator's truth is cosine distance to those "
        f"centroids, which is close to the inductive bias of Rocchio and of "
        f"logistic regression and nothing like a person -- so it flatters the "
        f"simpler rungs and is near-blind to what a neural ranker would be for. "
        f"Read `backend/visualtaste/eval/sim.py` before drawing conclusions.",
        "",
        "Every number is mean ± sd across seeds.",
        "",
        "| rung | like rate | last 100 | batch diversity | coverage | novelty | interests found |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for rung, summaries in by_rung.items():
        stats = aggregate(summaries)
        lines.append(
            f"| `{rung}` | {_cell(stats, 'like_rate')} "
            f"| {_cell(stats, 'like_rate_last_100')} "
            f"| {_cell(stats, 'mean_batch_diversity')} "
            f"| {_cell(stats, 'coverage')} "
            f"| {_cell(stats, 'mean_novelty')} "
            f"| {_cell(stats, 'interests_recalled', 2)} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_learning_curves(by_rung: dict[str, list[SessionResult]]) -> str:
    """A compact text sparkline of the rolling like rate per rung.

    Text rather than a PNG for this part: the curve's shape -- flat, rising,
    rising then plateauing -- is the whole message, and a sparkline carries it
    inside the document without a figure to open. The real plots go to
    `docs/figures/` in Phase 14.
    """
    blocks = "▁▂▃▄▅▆▇█"
    lines = ["**Rolling like rate** (window 50, left to right over the session):", ""]

    for rung, results in by_rung.items():
        if not results:
            continue
        curve = rolling_like_rate(results[0].actions, window=50)
        if len(curve) < 10:
            continue
        # Sample 40 evenly spaced points so every rung's line is the same width.
        indices = np.linspace(0, len(curve) - 1, 40).astype(int)
        sampled = [curve[i] for i in indices]
        spark = "".join(
            blocks[min(len(blocks) - 1, int(max(0.0, value) * len(blocks)))] for value in sampled
        )
        lines.append(f"- `{rung:<8}` {spark}  {sampled[0]:.2f} -> {sampled[-1]:.2f}")

    lines.append("")
    return "\n".join(lines)
