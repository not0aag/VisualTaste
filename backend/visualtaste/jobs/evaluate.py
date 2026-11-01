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


def session_model(rung: str, settings: Settings, seed: int) -> Recommender:
    """The model for one simulated session.

    R0 is seeded with the *session's* seed. `load_rung("r0")` uses the global
    seed, which gave every simulated user the identical cold-start sequence --
    the regression found when re-running the Phase 8 study in Part 2 (docs/
    DECISIONS.md #19). The recorded Phase 8 numbers were made with R0 seeded
    per session, and this restores them.
    """
    if rung == "r0":
        from visualtaste.rec.r0_coldstart import ColdStartRecommender

        return ColdStartRecommender(seed=seed)
    return load_rung(rung, settings)


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
            model = session_model(rung, settings, seed)
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


@dataclass(frozen=True)
class Preset:
    """A named Part 2 study: what to run, and which hypotheses to test on it.

    `comparisons` are `(variant, first, second, metric, expect)`. Named presets
    rather than free-form flags so that every study in docs/EVALUATION.md is one
    command that anyone can re-run and get the same cached seeds back.
    """

    variants: tuple[str, ...]
    specs: tuple[str, ...]
    comparisons: tuple[tuple[str, str, str, str, str], ...]
    title: str
    #: Per-seed replay AUC/log loss. Off for log-generation runs, whose
    #: histories are replayed by the study that uses them instead.
    with_replay: bool = True


_S1, _S2, _S3, _S5 = "S1_semantic", "S2_appearance", "S3_colour", "S5_drift"

PRESETS: dict[str, Preset] = {
    "phase9_rungs": Preset(
        title="R0-R3 on S1, S2, S3 and S5",
        variants=(_S1, _S2, _S3, _S5),
        specs=("r0", "r1", "r2", "r3"),
        comparisons=tuple(
            (variant, "r3", other, metric, "higher")
            for variant in (_S1, _S2, _S3, _S5)
            for other in ("r1", "r2")
            for metric in ("like_rate", "interests_found", "replay_auc")
        ),
    ),
    "phase9_skip": Preset(
        title="Skip weight: 0.25 vs 1.0 vs excluded (S1)",
        variants=(_S1,),
        specs=("r3", "r3:skip=1.0", "r3:skip=exclude"),
        comparisons=(
            (_S1, "r3:skip=exclude", "r3", "like_rate", "higher"),
            (_S1, "r3:skip=exclude", "r3", "replay_log_loss", "lower"),
            (_S1, "r3:skip=exclude", "r3:skip=1.0", "like_rate", "higher"),
            (_S1, "r3:skip=exclude", "r3:skip=1.0", "replay_log_loss", "lower"),
        ),
    ),
    "phase9_ablation": Preset(
        title="Feature blocks: sig / sig+dino / sig+dino+col (S1, S2, S3)",
        variants=(_S1, _S2, _S3),
        specs=("r3", "r3:sig+dino", "r3:sig+dino+col"),
        comparisons=tuple(
            (variant, first, second, metric, expect)
            for variant in (_S1, _S2, _S3)
            for first, second in (("r3:sig+dino", "r3"), ("r3:sig+dino+col", "r3:sig+dino"))
            for metric, expect in (("replay_log_loss", "lower"), ("like_rate", "higher"))
        ),
    ),
}


_S6 = "S6_six_interests"

PRESETS["phase10_h3"] = Preset(
    title="H3: R4 (Thompson + MMR + quota) vs R3 on S1, S5 and S6",
    variants=(_S1, _S5, _S6),
    specs=("r0", "r3", "r4"),
    comparisons=tuple(
        (variant, "r4", "r3", metric, expect)
        for variant in (_S1, _S5, _S6)
        for metric, expect in (
            ("all_found", "higher"),
            ("interests_found", "higher"),
            ("time_to_all_found", "lower"),
            ("coverage_last_500", "higher"),
            ("diversity", "higher"),
            ("like_rate", "higher"),
        )
    ),
)
PRESETS["phase10_sensitivity"] = Preset(
    title="R4 sensitivity: tau_post, lambda_mmr and quota (S1)",
    variants=(_S1,),
    # tau 0.1 and 0.25 extend Part 2's grid: at tau 1 the posterior's sampled
    # logit deviation is 2-4x the MAP signal and 0.5 does not change that
    # (DECISIONS #25), so the grid as given contains no usable default.
    # r0 and r3 are run alongside for H3's gain-retention criterion.
    specs=(
        "r4",
        "r4:tau=0.1",
        "r4:tau=0.25",
        "r4:tau=0.5",
        "r4:tau=2",
        "r4:mmr=0.5",
        "r4:mmr=0.9",
        "r4:quota=0",
        "r4:quota=10",
        "r3",
        "r0",
    ),
    comparisons=tuple(
        (_S1, other, "r4", metric, expect)
        for other in (
            "r4:tau=0.1",
            "r4:tau=0.25",
            "r4:tau=0.5",
            "r4:tau=2",
            "r4:mmr=0.5",
            "r4:mmr=0.9",
            "r4:quota=0",
            "r4:quota=10",
        )
        for metric, expect in (("all_found", "higher"), ("like_rate", "higher"))
    ),
)

PRESETS["phase11_h4"] = Preset(
    title="H4: R4+R5 (interests on top of exploration) vs R4 on S1, S5 and S6",
    variants=(_S1, _S5, _S6),
    specs=("r4", "r4+r5", "r5"),
    comparisons=(
        *(
            (variant, "r4+r5", "r4", metric, expect)
            for variant in (_S1, _S6)
            for metric, expect in (
                ("replay_log_loss", "lower"),
                ("interests_found", "higher"),
                ("all_found", "higher"),
                ("like_rate", "higher"),
            )
        ),
        (_S5, "r4+r5", "r4", "drift_recovery", "lower"),
        (_S5, "r4+r5", "r4", "like_rate", "higher"),
    ),
)

_S4 = "S4_conjunctive"

PRESETS["phase12_logs"] = Preset(
    title="Phase 12 training logs: R4 as the logging policy, S1 and S4, 5,000 swipes",
    variants=(_S1, _S4),
    specs=("r4",),
    comparisons=(),
    with_replay=False,
)

FINAL_STUDY = "phase14_final"
#: The variants DEFAULT_RUNG is chosen on (Part 2 §10.2). S4 is left out: it
#: exists for R6, which cannot serve automatically.
CHOICE_VARIANTS = (_S1, _S2, _S3, _S5, _S6)
#: Rungs eligible to be the default: within the latency budget, automatic.
#: R2 misses the budget (F3) and R6 is manual-only, so neither can be chosen.
ELIGIBLE_DEFAULTS = ("r1", "r3", "r3+r5", "r4", "r4+r5")
CHOICE_METRICS = ("like_rate", "interests_found", "coverage_last_500")

PRESETS[FINAL_STUDY] = Preset(
    title="Final study: every serving rung on S1-S6",
    variants=(_S1, _S2, _S3, _S4, _S5, _S6),
    specs=("r0", "r1", "r2", "r3", "r3+r5", "r4", "r4+r5"),
    comparisons=tuple(
        (variant, first, "r3", "like_rate", "higher")
        for variant in (_S1, _S2, _S3, _S4, _S5, _S6)
        for first in ("r4", "r4+r5", "r3+r5")
    ),
)


#: ROADMAP D1: can R4 keep more of R3's like-rate gain without losing its
#: reach? The sensitivity study (phase10_sensitivity) found the Thompson noise,
#: not the probe quota, to be the main cost; these variants fade that noise as
#: labels accumulate, narrow MMR's shortlist, and lower the temperature again.
EXPLORE_STUDY = "phase15_explore"
_EXPLORE_VARIANTS = (
    "r4:tau=0.05",
    "r4:tau_halflife=200",
    "r4:mmr_top=50",
    "r4:mmr=0.9",
    "r4:tau_halflife=200,mmr_top=50,mmr=0.9",
)
PRESETS[EXPLORE_STUDY] = Preset(
    title="Phase 15: closing R4's like-rate gap (S1, S6)",
    variants=(_S1, _S6),
    specs=("r0", "r3", "r4", *_EXPLORE_VARIANTS),
    comparisons=tuple(
        (variant, other, "r4", metric, expect)
        for variant in (_S1, _S6)
        for other in _EXPLORE_VARIANTS
        for metric, expect in (("like_rate", "higher"), ("all_found", "higher"))
    ),
)


#: ROADMAP D2: H2 found R3 the better-calibrated model and kNN the better
#: ranker. Does ranking by a blend of the two beat either? S2 is where
#: appearance matters and kNN's neighbourhoods should help most.
HYBRID_STUDY = "phase15_hybrid"
_HYBRID_VARIANTS = ("r3:knn=0.3", "r3:knn=0.5", "r3:knn=0.7")
PRESETS[HYBRID_STUDY] = Preset(
    title="Phase 15: ranking by R3 and kNN together (S1, S2)",
    variants=(_S1, _S2),
    specs=("r0", "r2", "r3", *_HYBRID_VARIANTS),
    comparisons=tuple(
        (variant, other, "r3", metric, "higher")
        for variant in (_S1, _S2)
        for other in (*_HYBRID_VARIANTS, "r2")
        for metric in ("like_rate", "interests_found", "replay_auc")
    ),
)


#: ROADMAP D6: in S5 the planted taste changes mid-session, and every label
#: still counts forever. Does letting old labels fade speed up recovery?
DRIFT_STUDY = "phase15_drift"
_DRIFT_VARIANTS = ("r3:decay=150", "r3:decay=400", "r4:decay=150", "r4:decay=400")
PRESETS[DRIFT_STUDY] = Preset(
    title="Phase 15: letting old labels fade after a change of taste (S5)",
    variants=(_S5,),
    specs=("r0", "r3", "r4", *_DRIFT_VARIANTS),
    comparisons=tuple(
        (_S5, other, base, metric, expect)
        for other, base in (
            ("r3:decay=150", "r3"),
            ("r3:decay=400", "r3"),
            ("r4:decay=150", "r4"),
            ("r4:decay=400", "r4"),
        )
        for metric, expect in (("drift_recovery", "lower"), ("like_rate", "higher"))
    ),
)


def default_rung_choice(
    results: dict[tuple[str, str], list],
    *,
    variants=CHOICE_VARIANTS,
    eligible=ELIGIBLE_DEFAULTS,
) -> tuple[str, dict[str, dict[str, float]], dict[str, float]]:
    """Part 2 §10.2: the best mean rank on a combined score, among eligible rungs.

    Per variant: each rung's mean over seeds of like rate, interests found and
    coverage; each metric z-scored across the eligible rungs; the three z-scores
    added with equal weight; rungs ranked (1 = best). The default is the rung
    with the lowest mean rank over `variants`. Returns (choice, combined score
    per variant per rung, mean rank per rung).
    """
    combined: dict[str, dict[str, float]] = {}
    ranks: dict[str, list[int]] = {rung: [] for rung in eligible}
    for variant in variants:
        means = {
            rung: [
                float(np.mean([getattr(r, m) for r in results[(variant, rung)]]))
                for m in CHOICE_METRICS
            ]
            for rung in eligible
            if (variant, rung) in results
        }
        matrix = np.array(list(means.values()))
        spread = matrix.std(axis=0)
        z = (matrix - matrix.mean(axis=0)) / np.where(spread == 0, 1.0, spread)
        scores = dict(zip(means, z.sum(axis=1).tolist(), strict=True))
        combined[variant] = scores
        for position, rung in enumerate(sorted(scores, key=lambda r: -scores[r]), start=1):
            ranks[rung].append(position)
    mean_rank = {rung: float(np.mean(r)) for rung, r in ranks.items() if r}
    choice = min(mean_rank, key=lambda rung: (mean_rank[rung], rung))
    return choice, combined, mean_rank


def render_default_choice(results) -> list[str]:
    choice, combined, mean_rank = default_rung_choice(results)
    header = "| rung | " + " | ".join(CHOICE_VARIANTS) + " | mean rank |"
    lines = [
        "",
        "### Choosing DEFAULT_RUNG (Part 2 §10.2)",
        "",
        "Combined score per variant = z(like rate) + z(interests found) + z(coverage), each "
        "z-scored across the eligible rungs (R1, R3, R3+R5, R4, R4+R5; R2 misses the latency "
        "budget and R6 is manual-only). Rank 1 is best.",
        "",
        header,
        "|" + " --- |" * (len(CHOICE_VARIANTS) + 2),
    ]
    for rung in sorted(mean_rank, key=lambda r: mean_rank[r]):
        cells = " | ".join(f"{combined[v].get(rung, float('nan')):+.2f}" for v in CHOICE_VARIANTS)
        lines.append(f"| `{rung}` | {cells} | {mean_rank[rung]:.1f} |")
    runner_up = sorted(mean_rank, key=lambda r: mean_rank[r])[1]
    margin = {v: combined[v][choice] - combined[v].get("r3", float("nan")) for v in CHOICE_VARIANTS}
    lines += [
        "",
        f"**Chosen: `{choice}`** (mean rank {mean_rank[choice]:.1f}; runner-up `{runner_up}` at "
        f"{mean_rank[runner_up]:.1f}). Its combined-score margin over R3 per variant: "
        + ", ".join(f"{v} {m:+.2f}" for v, m in margin.items())
        + f"; mean {np.nanmean(list(margin.values())):+.2f}.",
        "",
    ]
    return lines


#: H5's label counts. 4,900 rather than 5,000 leaves a 100-event test block.
H5_CUTS = (500, 1_500, 3_000, 4_900)
R6_SPECS = ("r6:minlabels=0", "r6:minlabels=0,loss=bpr")


def run_phase12_h5(
    features: FeatureBundle,
    settings: Settings,
    *,
    seeds: int,
    out: Path | None,
    print_only: bool,
) -> int:
    """H5's learning curve and the (SIMULATED) dwell study, on phase12_logs.

    R6's 1,500-label minimum is a rule for real data; here it is lowered to 0
    so the curve can show what R6 does at 500 labels, which is the point.
    """
    from visualtaste.eval.study import learning_curve, render_curve, run_study

    connection = repo.connect(settings.db_path)
    try:
        logs = run_study(
            connection,
            features,
            settings,
            study="phase12_logs",
            variants=(_S1, _S4),
            specs=("r4",),
            seeds=seeds,
            swipes=5_000,
            cache_root=settings.data / "eval",
            with_replay=False,
        )
    finally:
        connection.close()

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "",
        "---",
        "",
        f"## H5: R6 vs R3+R5 by label count, and the dwell study (SIMULATED, {now})",
        "",
        f"```\npython -m visualtaste.jobs.evaluate --study phase12_h5 --seeds {seeds}\n```",
        "",
        f"Histories: {seeds} seeds x 5,000 swipes per variant, logged by R4 (the default "
        "chain), so the training data carries the selection bias real data will. At each "
        "label count both models are fitted on the first N events and scored on the next 100; "
        "paired by seed. R6 is the 5-member ensemble, trained on the GPU.",
        "",
    ]
    for variant in (_S1, _S4):
        histories = logs[(variant, "r4")]
        for r6 in R6_SPECS:
            for metric, expect in (("log_loss", "lower"), ("auc", "higher")):
                points = learning_curve(
                    histories,
                    features,
                    settings,
                    first=r6,
                    second="r3+r5",
                    metric=metric,
                    expect=expect,
                    cuts=H5_CUTS,
                )
                parts += [
                    f"### {variant}: `{r6}` vs `r3+r5`, replay {metric}",
                    "",
                    render_curve(points, first=r6, second="r3+r5", metric=metric),
                    "",
                ]
    parts += dwell_study(logs, features, settings)
    section = "\n".join(parts) + "\n"
    if print_only:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(section)
    else:
        append_report(out or settings.project_root / "docs" / "EVALUATION.md", section)
    return 0


def _dwell_fitter(X_pool, rows, log_dwell, y_all, s_all, *, use_dwell: bool):
    """A replay callback for R3 on SigLIP, optionally with a standardised log-dwell column."""
    from scipy.special import expit

    from visualtaste.rec.r3_logreg import fit_sklearn, tune_lambda

    def fit_and_score(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        X_train, X_test = X_pool[rows[train]], X_pool[rows[test]]
        if use_dwell:
            mu, sd = log_dwell[train].mean(), log_dwell[train].std() or 1.0
            X_train = np.hstack([X_train, ((log_dwell[train] - mu) / sd)[:, None]])
            X_test = np.hstack([X_test, ((log_dwell[test] - mu) / sd)[:, None]])
        lam = tune_lambda(X_train, y_all[train], s_all[train]).lam
        w, b = fit_sklearn(X_train, y_all[train], s_all[train], lam)
        return expit(X_test @ w + b)

    return fit_and_score


def dwell_study(logs, features: FeatureBundle, settings: Settings) -> list[str]:
    """Part 2 §8.3: does (simulated) dwell predict likes, and does it help R3?

    Two measurements per seed: the AUC of dwell_ms alone for like-vs-dislike,
    and R3's replay log loss with and without a log-dwell column. Dwell is
    observed *with* the verdict, after the image is shown, so a dwell feature
    can sharpen a model of past verdicts but can never rank an unseen image.
    It is reported, and the promotion rule is applied, with that limit stated.
    """

    from visualtaste.eval.metrics import paired_bootstrap, roc_auc, sign_test
    from visualtaste.eval.replay import replay
    from visualtaste.rec.features import design_matrix
    from visualtaste.rec.labels import ACTION_TO_TARGET, label_weights

    X_pool = design_matrix(features, ("sig",))
    weights = label_weights(settings.skip_weight)
    lines = [
        "### Dwell (SIMULATED)",
        "",
        "Simulated dwell is `exp(N(ln 2500 + 0.3 (p - 0.5) + 0.4 novelty, 0.5))` "
        "(`eval/sim.simulated_dwell`): weakly tied to liking, more to novelty.",
        "",
        "| variant | AUC of dwell alone | R3 replay log loss without | with log-dwell "
        "| diff | 90% CI | W-L | promote? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant in (_S1, _S4):
        aucs, without, with_dwell = [], [], []
        for history in logs[(variant, "r4")]:
            verdicts = [i for i, a in enumerate(history.actions) if a != "skip"]
            aucs.append(
                roc_auc(
                    np.array([history.actions[i] in ("like", "save") for i in verdicts], dtype=int),
                    np.array([history.dwell_ms[i] for i in verdicts], dtype=float),
                )
            )
            rows = np.array([features.position(i) for i in history.image_ids])
            log_dwell = np.log(np.maximum(np.array(history.dwell_ms, dtype=float), 1.0))
            y_all = np.array([ACTION_TO_TARGET[a] for a in history.actions], dtype=float)
            s_all = np.array([weights[a] for a in history.actions])

            without.append(
                replay(
                    history.actions,
                    _dwell_fitter(X_pool, rows, log_dwell, y_all, s_all, use_dwell=False),
                    label="r3",
                ).mean("log_loss")
            )
            with_dwell.append(
                replay(
                    history.actions,
                    _dwell_fitter(X_pool, rows, log_dwell, y_all, s_all, use_dwell=True),
                    label="r3+dwell",
                ).mean("log_loss")
            )
        diff, low, high = paired_bootstrap(np.array(with_dwell), np.array(without), samples=10_000)
        deltas = np.array(without) - np.array(with_dwell)  # positive = dwell helped
        wins, losses = int((deltas > 0).sum()), int((deltas < 0).sum())
        promote = "yes (but off by default for real data, #26)" if high < 0 else "no"
        lines.append(
            f"| {variant} | {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | {np.mean(without):.4f} "
            f"| {np.mean(with_dwell):.4f} | {diff:+.4f} | [{low:+.4f}, {high:+.4f}] "
            f"| {wins}-{losses} (p = {sign_test(wins, losses):.3g}) | {promote} |"
        )
    lines += [
        "",
        "Rule (DECISIONS #26): dwell becomes a feature only if it lowers log loss with a CI "
        "excluding 0, and even then stays off for real data until there are 1,000 real labels.",
        "",
    ]
    return lines


#: Seeds and swipes for the simulated interleaving check (Part 2 §6.4).
INTERLEAVE_SEEDS, INTERLEAVE_SWIPES = 20, 300


def run_phase10_checks(
    features: FeatureBundle,
    settings: Settings,
    *,
    seeds: int,
    swipes: int,
    out: Path | None,
    print_only: bool,
) -> int:
    """The interleaving and IPS checks (Part 2 §6.4), on S1.

    Interleaving: R3 vs R4, team-drafted, 20 seeds x 300 swipes; does its winner
    match the separate-run H3 study? IPS: from R4 sessions logged with Monte-
    Carlo propensities and a shadow R3, estimate R3's like rate with SNIPS and
    compare with R3's actual simulated like rate.
    """
    from visualtaste.eval.study import ips_check, run_interleaved_seed, run_study

    cache = settings.data / "eval"
    connection = repo.connect(settings.db_path)
    try:
        interleaved = [
            run_interleaved_seed(
                connection,
                features,
                settings,
                variant=_S1,
                first="r3",
                second="r4",
                seed=settings.seed + i,
                swipes=INTERLEAVE_SWIPES,
                cluster_labels=features.cluster64,
                cache_root=cache,
            )
            for i in range(INTERLEAVE_SEEDS)
        ]
        separate = run_study(
            connection,
            features,
            settings,
            study="phase10_h3",
            variants=(_S1,),
            specs=("r3", "r4"),
            seeds=seeds,
            swipes=swipes,
            cache_root=cache,
        )
        logged = run_study(
            connection,
            features,
            settings,
            study="phase10_ips",
            variants=(_S1,),
            specs=("r4",),
            seeds=seeds,
            swipes=swipes,
            cache_root=cache,
            with_replay=False,
            with_propensities=True,
            shadow="r3",
        )[(_S1, "r4")]
    finally:
        connection.close()

    from visualtaste.eval.metrics import sign_test

    wins_a = sum(s.wins_a for s in interleaved)
    wins_b = sum(s.wins_b for s in interleaved)
    seed_winner = [
        "r3" if s.wins_a > s.wins_b else "r4" if s.wins_b > s.wins_a else "tie" for s in interleaved
    ]
    r3_rate = float(np.mean([r.like_rate for r in separate[(_S1, "r3")]]))
    r4_rate = float(np.mean([r.like_rate for r in separate[(_S1, "r4")]]))
    separate_winner = "r3" if r3_rate > r4_rate else "r4"
    interleave_winner = "r3" if wins_a > wins_b else "r4"
    ips = ips_check(logged, separate[(_S1, "r3")])

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    section = "\n".join(
        [
            "",
            "---",
            "",
            f"## Phase 10 checks: interleaving and IPS (S1, {now})",
            "",
            f"```\npython -m visualtaste.jobs.evaluate --study phase10_checks --seeds {seeds}\n```",
            "",
            f"**SIMULATED.** Interleaving: R3 (arm a) vs R4 (arm b), team-draft, "
            f"{INTERLEAVE_SEEDS} seeds x {INTERLEAVE_SWIPES} swipes; a batch is won by the arm "
            f"whose cards got more likes.",
            "",
            "| batches won by R3 | won by R4 | sign test p | seeds won R3 / R4 / tied "
            "| interleaving winner | separate-run winner (like rate) | agree? |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| {wins_a} | {wins_b} | {sign_test(wins_a, wins_b):.3g} "
            f"| {seed_winner.count('r3')} / {seed_winner.count('r4')} / {seed_winner.count('tie')} "
            f"| {interleave_winner} | {separate_winner} (R3 {r3_rate:.3f}, R4 {r4_rate:.3f}) "
            f"| {'yes' if interleave_winner == separate_winner else '**no**'} |",
            "",
            "**IPS (educational).** R3's like rate estimated by self-normalised IPS from "
            f"{ips.seeds} R4 sessions logged with 200-draw Monte-Carlo propensities and a shadow "
            "R3 fitted on the same labels (R3's propensity for a card is 1 if it would have "
            "put it in its own top five, else 0). Interval: bootstrap over seeds.",
            "",
            "| IPS estimate | 90% CI | R3's actual like rate | bias | effective sample size "
            "| logged R4 cards |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {ips.estimate:.3f} | [{ips.low:.3f}, {ips.high:.3f}] | {ips.actual:.3f} "
            f"| {ips.bias:+.3f} | {ips.effective_sample_size:.0f} | {ips.logged_cards:,} |",
            "",
        ]
    )
    if print_only:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(section)
    else:
        append_report(out or settings.project_root / "docs" / "EVALUATION.md", section)
    return 0


#: H2 is not a grid of sessions but a replay over existing ones: R0's histories
#: from `phase9_rungs` on S1, replayed with R2 and R3 at every 50 labels.
H2_STUDY = "phase9_h2"


def run_h2(
    features: FeatureBundle,
    settings: Settings,
    *,
    seeds: int,
    swipes: int,
    out: Path | None,
    print_only: bool,
) -> int:
    from visualtaste.eval.study import learning_curve, render_curve, run_study

    connection = repo.connect(settings.db_path)
    try:
        histories = run_study(
            connection,
            features,
            settings,
            study="phase9_rungs",
            variants=(_S1,),
            specs=("r0",),
            seeds=seeds,
            swipes=swipes,
            cache_root=settings.data / "eval",
        )[(_S1, "r0")]
    finally:
        connection.close()

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "",
        "---",
        "",
        f"## H2: R3 vs R2 by label count, replayed on cold-start histories ({now})",
        "",
        f"```\npython -m visualtaste.jobs.evaluate --study {H2_STUDY} --seeds {seeds}\n```",
        "",
        f"**SIMULATED**, S1, {len(histories)} seeds. The histories were logged by R0, so "
        "neither rung chose the images it is tested on. At each cut both rungs are "
        "fitted on the first N events and scored on the next 100. R2's similarity "
        "score is Platt-scaled out-of-sample before log loss (`eval/study.py`).",
        "",
    ]
    for metric, expect in (("log_loss", "lower"), ("auc", "higher")):
        points = learning_curve(
            histories, features, settings, first="r3", second="r2", metric=metric, expect=expect
        )
        parts += [
            f"### Replay {metric}",
            "",
            render_curve(points, first="r3", second="r2", metric=metric),
            "",
        ]
    section = "\n".join(parts) + "\n"
    if print_only:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(section)
    else:
        append_report(out or settings.project_root / "docs" / "EVALUATION.md", section)
    return 0


def run_preset(
    name: str,
    *,
    seeds: int,
    swipes: int,
    out: Path | None,
    print_only: bool,
    only_variants: list[str] | None = None,
) -> int:
    """Run a Part 2 preset, using and filling the cache, and write it up.

    `only_variants` fills the cache for those variants and writes no report,
    so one large study can be split across parallel processes (each on its own
    database, DECISIONS #22). A final run without it then reads every seed
    from the cache and writes the whole section.
    """
    from visualtaste.eval.study import (
        COMPARISON_HEADER,
        compare,
        discovery_table,
        run_study,
        summary_table,
    )

    settings = get_settings()
    state = AppState(settings)
    state.load()
    features = state.require_features()

    if name == H2_STUDY:
        return run_h2(
            features, settings, seeds=seeds, swipes=swipes, out=out, print_only=print_only
        )
    if name == "phase10_checks":
        return run_phase10_checks(
            features, settings, seeds=seeds, swipes=swipes, out=out, print_only=print_only
        )
    if name == "phase12_h5":
        return run_phase12_h5(features, settings, seeds=seeds, out=out, print_only=print_only)

    preset = PRESETS[name]
    if only_variants:
        unknown = set(only_variants) - set(preset.variants)
        if unknown:
            raise SystemExit(f"{sorted(unknown)} are not variants of {name}")
        from dataclasses import replace

        preset = replace(preset, variants=tuple(only_variants), comparisons=())

    connection = repo.connect(settings.db_path)
    try:
        results = run_study(
            connection,
            features,
            settings,
            study=name,
            variants=preset.variants,
            specs=preset.specs,
            seeds=seeds,
            swipes=swipes,
            cache_root=settings.data / "eval",
            with_replay=preset.with_replay,
        )
    finally:
        connection.close()

    if only_variants:
        print(f"{name}: cached {sum(len(r) for r in results.values())} seeds for {only_variants}")
        return 0

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "",
        "---",
        "",
        f"## {preset.title} ({now})",
        "",
        f"```\npython -m visualtaste.jobs.evaluate --study {name} --seeds {seeds} "
        f"--swipes {swipes}\n```",
        "",
        "**SIMULATED.** Variants and their calibration: `backend/visualtaste/eval/sim.py`. "
        "Every number is mean ± sd across seeds; comparisons are paired by seed "
        "(Part 2 §4.1). An interest counts as found at 5 likes.",
        "",
    ]
    for variant in preset.variants:
        parts += [
            f"### {variant}",
            "",
            summary_table(results, variant),
            "",
            discovery_table(results, variant),
            "",
        ]
    parts += ["### Comparisons", "", COMPARISON_HEADER]
    for variant, first, second, metric, expect in preset.comparisons:
        parts.append(
            compare(
                results, variant=variant, first=first, second=second, metric=metric, expect=expect
            ).row()
        )
    if "r0" in preset.specs and "r3" in preset.specs:
        from visualtaste.eval.study import gain_retention

        parts += [
            "",
            "**Like-rate gain kept** (H3 needs R4 to keep >= 70% of R3's gain over R0): "
            "(R4 - R0) / (R3 - R0), paired bootstrap 90% CI over seeds.",
            "",
            "| variant | spec | gain kept | 90% CI | meets 70%? |",
            "| --- | --- | --- | --- | --- |",
        ]
        for variant in preset.variants:
            for candidate in (s for s in preset.specs if s.startswith("r4")):
                ratio, low, high = gain_retention(
                    results, variant=variant, candidate=candidate, reference="r3", baseline="r0"
                )
                verdict = "yes" if low >= 0.7 else "no" if high < 0.7 else "inconclusive"
                parts.append(
                    f"| {variant} | `{candidate}` | {ratio:.2f} | [{low:.2f}, {high:.2f}] "
                    f"| {verdict} |"
                )
    if name == FINAL_STUDY:
        parts += render_default_choice(results)
    section = "\n".join(parts) + "\n"

    if print_only:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(section)
    else:
        append_report(out or settings.project_root / "docs" / "EVALUATION.md", section)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=[*sorted(PRESETS), H2_STUDY, "phase10_checks", "phase12_h5"],
        default=None,
        help="run a named Part 2 study (cached under data/eval/) instead of the Phase 8 one",
    )
    parser.add_argument(
        "--rungs",
        default=",".join(DEFAULT_RUNGS),
        help="comma-separated rungs to compare, e.g. r0,r1,r2",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="with --study: only fill the cache for these variants (comma-separated); no report",
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

    if args.study is not None:
        return run_preset(
            args.study,
            seeds=args.seeds,
            swipes=args.swipes,
            out=args.out,
            print_only=args.print_only,
            only_variants=[v.strip() for v in args.variants.split(",")] if args.variants else None,
        )

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
