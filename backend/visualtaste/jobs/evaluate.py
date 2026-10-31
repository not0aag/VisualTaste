"""Run the simulator study: every rung, several seeds, into `docs/EVALUATION.md`.

What this job is for
--------------------
No real labels exist until Alen swipes, so every claim about whether a rung
learns anything has to come from the planted-taste simulator (brief §7.7). This
job is the only thing that produces those numbers. It runs the *real* feed path
-- `rec.feed.build_batch`, real impressions, real events in the real database on
`is_simulated = 1` profiles -- so a bug in serving shows up as a bad result
rather than being bypassed by a shortcut scoring loop.

    python -m visualtaste.jobs.evaluate --rungs r0,r1,r2 --seeds 5 --swipes 1000

What the numbers are and are not
--------------------------------
The simulator's ground truth is cosine distance to three hidden centroids. That
is almost exactly the inductive bias of Rocchio, and close to that of logistic
regression on these embeddings, so the study systematically flatters the simple
rungs. It says nothing about whether SigLIP 2's notion of similarity matches a
person's. Every table this job writes carries that caveat, because a reader who
sees "like rate 0.83" without it will believe something false.

Why it also measures scoring latency
------------------------------------
Scoring cost is not a side issue for the ladder: R1 scores with one 50,000-row
dot product regardless of label count, while R2 scores against *every* label, so
its cost grows as Alen swipes. Serving is budgeted at a 50 ms p95 (§7.5), so the
rung that wins on like rate may still be unservable at 2,000 labels. Measuring
it here, from the fitted models the study already built, is cheaper and more
honest than discovering it from a slow feed months later.

Repeated runs
-------------
Results are *appended* to `docs/EVALUATION.md` with their command and date, never
overwritten. A later run that disagrees with an earlier one is a finding, and
deleting the earlier one would hide it.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from visualtaste.api.deps import AppState, load_rung
from visualtaste.config import Settings, get_settings
from visualtaste.db import repo
from visualtaste.eval.report import (
    SessionResult,
    SessionSummary,
    render_learning_curves,
    render_markdown,
    run_session,
    summarise_session,
)
from visualtaste.eval.sim import LOGIT_OFFSET, make_user
from visualtaste.rec.base import FeatureBundle, Recommender
from visualtaste.rec.labels import build_label_set

logger = logging.getLogger(__name__)

DEFAULT_RUNGS = ("r0", "r1", "r2")
DEFAULT_SEEDS = 5
DEFAULT_SWIPES = 1_000

#: Repeats per latency measurement. Small, because the measurement is a matrix
#: multiply with no I/O and no cache effects worth averaging out.
LATENCY_REPEATS = 10


@dataclass
class StudyOutcome:
    """Everything one study run produced, so `main` has nothing to recompute."""

    summaries: dict[str, list[SessionSummary]]
    results: dict[str, list[SessionResult]]
    latencies: list[LatencyResult]
    offsets: list[float]


@dataclass
class LatencyResult:
    """How long one rung takes to score the whole candidate pool."""

    rung: str
    labels: int
    candidates: int
    mean_ms: float
    max_ms: float


def study(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    settings: Settings,
    *,
    rungs: list[str],
    seeds: int,
    swipes: int,
    batch_size: int = 5,
    refit_every: int = 20,
    calibrate: bool = True,
) -> StudyOutcome:
    """Run `seeds` sessions of `swipes` swipes for each rung.

    The same seed means the same planted user across rungs, which is what makes
    the comparison paired: R1 and R2 face identical hidden interests and
    identical reaction draws, so a difference between them is the models rather
    than the luck of the user.
    """
    summaries: dict[str, list[SessionSummary]] = {}
    results: dict[str, list[SessionResult]] = {}
    latencies: list[LatencyResult] = []
    offsets: list[float] = []

    for rung in rungs:
        summaries[rung] = []
        results[rung] = []

        for index in range(seeds):
            seed = settings.seed + index
            model = load_rung(rung, settings)
            user = make_user(
                features.siglip,
                seed=seed,
                cluster_labels=features.cluster64,
                calibrate=calibrate,
            )

            offsets.append(user.offset)
            started = time.perf_counter()
            result = run_session(
                connection,
                features,
                model,
                user,
                swipes=swipes,
                seed=seed,
                batch_size=batch_size,
                refit_every=refit_every,
            )
            elapsed = time.perf_counter() - started

            summary = summarise_session(result)
            summaries[rung].append(summary)
            results[rung].append(result)
            logger.info(
                "%s seed %d: %d swipes in %.1fs, like rate %.3f (last 100: %.3f)",
                rung,
                seed,
                summary.swipes,
                elapsed,
                summary.like_rate,
                summary.like_rate_last_100,
            )

            # Measure latency once per rung, on the last fitted model of the
            # first seed -- the state a real feed would be serving from after
            # this many labels.
            if index == 0 and result.profile_id is not None:
                latencies.append(measure_latency(connection, features, model, result.profile_id))

    logger.info(
        "simulated like-probability offsets used: %s",
        ", ".join(f"{offset:.3f}" for offset in sorted({round(o, 3) for o in offsets})),
    )
    return StudyOutcome(summaries, results, latencies, offsets)


def measure_latency(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    model: Recommender,
    profile_id: int,
) -> LatencyResult:
    """Time `model.score` over the full candidate pool.

    This is the dominant term in `/feed/next` for every rung up to R5, so it is
    the number that decides whether a rung is servable. It excludes the database
    round-trip and the impression writes, which were measured separately in
    Phase 6 and are roughly constant across rungs.
    """
    candidates = np.arange(len(features), dtype=np.int64)
    labels = build_label_set(connection, profile_id)

    timings = []
    for _ in range(LATENCY_REPEATS):
        started = time.perf_counter()
        model.score(features, candidates)
        timings.append((time.perf_counter() - started) * 1_000)

    return LatencyResult(
        rung=model.rung,
        labels=len(labels),
        candidates=len(candidates),
        mean_ms=float(np.mean(timings)),
        max_ms=float(np.max(timings)),
    )


def render_latency(latencies: list[LatencyResult]) -> str:
    """The scoring-cost table, with the budget stated so it can be failed."""
    if not latencies:
        return ""

    lines = [
        "**Scoring cost** (`model.score` over the whole candidate pool, "
        f"mean of {LATENCY_REPEATS} runs, after the session's labels). The "
        "`/feed/next` budget is a 50 ms p95 for the entire request, so a rung "
        "over roughly 40 ms here cannot be served as it stands.",
        "",
        "| rung | labels | candidates | mean | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in latencies:
        lines.append(
            f"| `{item.rung}` | {item.labels} | {item.candidates:,} "
            f"| {item.mean_ms:.1f} ms | {item.max_ms:.1f} ms |"
        )
    lines.append("")
    return "\n".join(lines)


def render_report(
    summaries: dict[str, list[SessionSummary]],
    results: dict[str, list[SessionResult]],
    latencies: list[LatencyResult],
    *,
    command: str,
    swipes: int,
    seeds: int,
    pool_size: int,
    offsets: list[float],
    calibrated: bool,
) -> str:
    """Assemble one dated, self-describing section for `docs/EVALUATION.md`."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "",
        "---",
        "",
        f"## Study run {now}",
        "",
        f"```\n{command}\n```",
        "",
        f"Pool: {pool_size:,} selected images. "
        f"Rungs: {', '.join(f'`{r}`' for r in summaries)}. "
        f"Retrained every 20 labels, five cards per batch.",
        "",
        _calibration_note(offsets, calibrated=calibrated),
        "",
        render_markdown(summaries, swipes=swipes, seeds=seeds),
        render_learning_curves(results),
        render_latency(latencies),
    ]
    return "\n".join(part for part in parts if part is not None)


def _calibration_note(offsets: list[float], *, calibrated: bool) -> str:
    """State which simulated user this run faced, in one line.

    Load-bearing rather than decorative: the same rung scores a like rate of
    0.88 against the saturated user and something far more informative against
    the calibrated one, so a table without this line is unreadable a month
    later.
    """
    shown = ", ".join(f"{offset:.3f}" for offset in sorted({round(o, 3) for o in offsets}))
    if not calibrated:
        return (
            f"Simulated user: the brief's literal `sigma(10 (cos - {LOGIT_OFFSET}))`. "
            f"On this corpus that user likes about 98% of everything (SigLIP 2 "
            f"cosines sit near 0.5), so the rungs are expected to be "
            f"indistinguishable here -- which is the point of running it."
        )
    return (
        f"Simulated user: `sigma(10 (cos - offset))` with the offset calibrated "
        f"per seed to a 50% baseline like rate (offsets used: {shown}; the "
        f"brief's constant {LOGIT_OFFSET} saturates on this corpus -- see "
        f"`backend/visualtaste/eval/sim.py`)."
    )


def append_report(path: Path, section: str) -> None:
    """Append to the document, creating it with a header if it is absent."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Evaluation\n", encoding="utf-8")

    with path.open("a", encoding="utf-8") as handle:
        handle.write(section if section.endswith("\n") else section + "\n")
    logger.info("appended a study section to %s", path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rungs",
        default=",".join(DEFAULT_RUNGS),
        help="comma-separated rungs to compare, e.g. r0,r1,r2",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--swipes", type=int, default=DEFAULT_SWIPES)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="markdown file to append to (default docs/EVALUATION.md)",
    )
    parser.add_argument(
        "--brief-offset",
        action="store_true",
        help=(
            "use the brief's literal like-probability offset of 0.25 instead of "
            "calibrating it. Produces a user who likes nearly everything on this "
            "corpus; kept so that claim stays reproducible."
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="write the report to stdout instead of the document",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    rungs = [rung.strip() for rung in args.rungs.split(",") if rung.strip()]

    state = AppState(settings)
    state.load()
    features = state.require_features()
    for note in state.notes:
        logger.warning("state note: %s", note)

    connection = repo.connect(settings.db_path)
    try:
        outcome = study(
            connection,
            features,
            settings,
            rungs=rungs,
            seeds=args.seeds,
            swipes=args.swipes,
            batch_size=args.batch_size,
            refit_every=args.refit_every,
            calibrate=not args.brief_offset,
        )
    finally:
        connection.close()

    command = (
        f"python -m visualtaste.jobs.evaluate --rungs {args.rungs} "
        f"--seeds {args.seeds} --swipes {args.swipes}"
        + (" --brief-offset" if args.brief_offset else "")
    )
    section = render_report(
        outcome.summaries,
        outcome.results,
        outcome.latencies,
        command=command,
        swipes=args.swipes,
        seeds=args.seeds,
        pool_size=len(features),
        offsets=outcome.offsets,
        calibrated=not args.brief_offset,
    )

    if args.print_only:
        # The report contains "+-" signs and sparkline block characters, and a
        # Windows console is cp1252 by default. Writing through a reconfigured
        # stream is the difference between a report and a UnicodeEncodeError.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(section)
    else:
        append_report(args.out or settings.project_root / "docs" / "EVALUATION.md", section)

    return 0


if __name__ == "__main__":
    sys.exit(main())
