"""Part 2 simulator studies: many seeds, several variants, honest statistics.

`jobs/evaluate.py` runs the Phase 8 study as it was (one simulated user, five
seeds), and that path is kept unchanged so its recorded numbers stay
reproducible (Part 2 §4.7). This module is the Part 2 machinery on top of the
same session runner:

- **Variants** (`eval/sim.py`, S1-S6): the taste is planted somewhere other
  than SigLIP space, so the study is not graded on its own geometry.
- **Rung specs**: a rung plus options, as text, so an ablation is one string --
  `r3`, `r3:sig+dino`, `r3:sig+dino+col`, `r3:skip=0`, `r3:skip=exclude`.
- **Outcomes** (Part 2 §4.3) per seed, not only like rate: interests found and
  when, coverage of the last 500 impressions, diversity, novelty, replay AUC and
  log loss, and calibration error.
- **A cache.** Every `(study, variant, spec, seed)` result is a JSON file under
  `data/eval/<study>/`; a re-run skips it. A study killed halfway resumes.
- **Statistics** (Part 2 §4.1): paired by seed, a bootstrap 90% interval on the
  mean difference (10,000 resamples), the sign test with the losing seeds named,
  and a verdict -- *supported*, *not supported* or *inconclusive*. An interval
  that crosses 0 is inconclusive, never supported.

Why paired by seed
------------------
The same seed is the same planted user, facing the same reaction draws, under
every spec. Two specs' like rates differ from seed to seed far more than from
each other, so the per-seed *difference* carries the signal and comparing two
separate means would bury it.
"""

from __future__ import annotations

import itertools
import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from visualtaste.eval.metrics import (
    coverage,
    expected_calibration_error,
    like_rate,
    paired_bootstrap,
    roc_auc,
    sign_test,
)
from visualtaste.eval.replay import replay
from visualtaste.eval.report import COLDSTART_CLUSTERS, run_session
from visualtaste.eval.sim import DISCOVERY_HITS, VARIANTS, make_variant_user, space_clusters
from visualtaste.rec.labels import ACTION_TO_TARGET, LabelSet, label_weights

if TYPE_CHECKING:
    from collections.abc import Sequence

    from visualtaste.config import Settings
    from visualtaste.rec.base import FeatureBundle, Recommender

logger = logging.getLogger(__name__)

BOOTSTRAP_SAMPLES = 10_000
CREDIBILITY = 0.9
COVERAGE_WINDOW = 500

#: Rungs whose score is a probability, so log loss and calibration mean
#: something. R1 and R2 score similarities; replay Platt-scales them first.
PROBABILITY_RUNGS = frozenset({"r3", "r4", "r3+r5", "r4+r5", "r6"})


# ---------------------------------------------------------------------------
# Rung specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RungSpec:
    """`r3:sig+dino` or `r3:skip=exclude`: a rung and the options it runs with."""

    text: str
    rung: str
    blocks: tuple[str, ...] | None = None
    skip: float | str | None = None  # a weight, "exclude", or None for the default
    #: Model attributes to override, from options like `tau=0.5`, `mmr=0.9`,
    #: `quota=10` (one card in ten) or `quota=0` (no exploration).
    overrides: tuple[tuple[str, Any], ...] = ()

    #: Option name -> (model attribute, converter).
    OPTION_ATTRIBUTES: ClassVar[dict[str, tuple[str, Callable[[str], Any]]]] = {
        "tau": ("tau_post", float),
        "mmr": ("mmr_lambda", float),
        "quota": ("explore_every", lambda v: None if int(v) == 0 else int(v)),
        "loss": ("loss", str),
        "minlabels": ("min_labels", int),
        "epochs": ("max_epochs", int),
    }

    @classmethod
    def parse(cls, text: str) -> RungSpec:
        rung, _, options = text.partition(":")
        blocks: tuple[str, ...] | None = None
        skip: float | str | None = None
        overrides: list[tuple[str, Any]] = []
        for option in filter(None, options.split(",")):
            key, _, value = option.partition("=")
            if key == "skip":
                skip = "exclude" if value == "exclude" else float(value)
            elif key in cls.OPTION_ATTRIBUTES:
                attribute, convert = cls.OPTION_ATTRIBUTES[key]
                overrides.append((attribute, convert(value)))
            else:
                blocks = tuple(option.split("+"))
        return cls(text=text, rung=rung, blocks=blocks, skip=skip, overrides=tuple(overrides))

    @property
    def slug(self) -> str:
        """A filename-safe form for the cache."""
        return self.text.replace(":", "__").replace("+", "-").replace("=", "-").replace(",", "_")

    def build(self, settings: Settings, seed: int | None = None) -> Recommender:
        """A fresh model. R0 takes the session's seed (see jobs/evaluate.session_model)."""
        from visualtaste.jobs.evaluate import session_model

        model = session_model(self.rung, settings, settings.seed if seed is None else seed)
        if self.blocks is not None:
            if not hasattr(model, "blocks"):
                raise ValueError(f"{self.rung} does not take feature blocks")
            model.blocks = self.blocks
        for attribute, value in self.overrides:
            if not hasattr(model, attribute):
                raise ValueError(f"{self.rung} has no setting {attribute!r}")
            setattr(model, attribute, value)
        return model

    def label_options(self, settings: Settings) -> tuple[dict[str, float], bool]:
        """(weights, exclude_skips) for this spec's labels."""
        if self.skip == "exclude":
            return label_weights(settings.skip_weight), True
        if self.skip is not None:
            return label_weights(float(self.skip)), False
        return label_weights(settings.skip_weight), False


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Everything one `(variant, spec, seed)` session produced, as cached JSON."""

    study: str
    variant: str
    spec: str
    seed: int
    swipes: int
    seconds: float
    like_rate: float
    like_rate_last_100: float
    interests_planted: int
    interests_found: int
    all_found: bool
    #: interest id -> swipe at which it reached 5 likes, or None (censored).
    discovery: dict[str, int | None]
    coverage_last_500: float
    diversity: float
    novelty: float
    replay_auc: float
    replay_log_loss: float
    ece: float
    offset: float
    slope: float
    switched_at: int | None = None
    #: AUC of the user's *true* p(like) on this session's verdicts: the ceiling
    #: any model's replay AUC could reach on the same images. With 5% label
    #: noise and a soft sigmoid it is about 0.7, not 1.0.
    oracle_auc: float = float("nan")
    #: Swipe at which the last planted interest was found; swipes + 1 when one
    #: never was (censored, and counted as worse than any finite time).
    time_to_all_found: float = float("nan")
    propensities: list[float] = field(default_factory=list)
    shadow_shown: list[float] = field(default_factory=list)
    #: S5: swipes from the switch until the new interest was found
    #: (swipes - switch + 1 when it never was: censored).
    drift_recovery: float = float("nan")
    #: Interest rungs: mean share of interest ids kept across each recluster.
    id_stability: float = float("nan")
    interest_history: list[dict[str, Any]] = field(default_factory=list)
    #: The history itself, so later analyses (H2, dwell) need no re-run.
    image_ids: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    served_rungs: list[str] = field(default_factory=list)
    dwell_ms: list[int] = field(default_factory=list)
    true_probabilities: list[float] = field(default_factory=list)

    def metric(self, name: str) -> float:
        value = getattr(self, name)
        return float(value)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> SeedResult:
        return cls(**json.loads(text))


def interest_id_stability(history: Sequence[dict[str, Any]]) -> float:
    """Mean share of the previous interest ids still present after each recluster.

    1.0 means every recluster kept every id (only nudges and births); NaN when
    there was never a second clustering to compare with.
    """
    shares = [
        len(set(previous["ids"]) & set(current["ids"])) / len(previous["ids"])
        for previous, current in itertools.pairwise(history)
        if previous["ids"]
    ]
    return float(np.mean(shares)) if shares else float("nan")


def cache_path(root: Path, study: str, variant: str, spec: RungSpec, seed: int) -> Path:
    return root / study / variant / spec.slug / f"seed_{seed}.json"


def labels_from_history(
    image_ids: Sequence[str], actions: Sequence[str], weights: dict[str, float], exclude: bool
) -> LabelSet:
    """A LabelSet from a session's history, applying the spec's label options."""
    keep = [
        i
        for i, action in enumerate(actions)
        if not (exclude and action == "skip") and weights.get(action, 0.0) > 0
    ]
    return LabelSet(
        image_ids=[image_ids[i] for i in keep],
        targets=np.array([ACTION_TO_TARGET[actions[i]] for i in keep], dtype=np.float32),
        weights=np.array([weights[actions[i]] for i in keep], dtype=np.float32),
        actions=[actions[i] for i in keep],
        timestamps=[""] * len(keep),
    )


def replay_fit_and_score(
    spec: RungSpec,
    settings: Settings,
    features: FeatureBundle,
    image_ids: Sequence[str],
    actions: Sequence[str],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A `replay` callback: fit a fresh model on the prefix, score the next block.

    For rungs whose score is a similarity (R1, R2), the score is turned into a
    probability by Platt scaling -- a one-feature logistic fit -- learned
    out-of-sample: the model is fitted on the first 80% of the prefix and scored
    on the remaining 20%, the Platt fit is made on that, and then the model is
    refitted on the whole prefix. Without the out-of-sample step, kNN would be
    calibrated on images that are their own nearest neighbour.

    A model that cannot be fitted on the prefix predicts 0.5 for everything,
    which is exactly what "knows nothing yet" should cost in log loss.
    """
    weights, exclude = spec.label_options(settings)
    ids = list(image_ids)

    def fitted(train: np.ndarray):
        model = spec.build(settings)
        model.fit(
            labels_from_history(
                [ids[i] for i in train], [actions[i] for i in train], weights, exclude
            ),
            features,
        )
        return model if model.is_fitted else None

    def score(model, rows: np.ndarray) -> np.ndarray:
        positions = features.positions([ids[i] for i in rows])
        return np.asarray(model.score(features, positions), dtype=np.float64)

    def fit_and_score(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        model = fitted(train)
        if model is None:
            return np.full(len(test), 0.5)
        if spec.rung in PROBABILITY_RUNGS:
            return score(model, test)

        split = int(0.8 * len(train))
        early = fitted(train[:split])
        if early is None:
            return np.full(len(test), 0.5)
        late = train[split:]
        targets = np.array(
            [ACTION_TO_TARGET[actions[i]] for i in late if actions[i] != "skip"], dtype=float
        )
        raw = score(early, np.array([i for i in late if actions[i] != "skip"]))
        if len(np.unique(targets)) < 2:
            return np.full(len(test), float(targets.mean()) if len(targets) else 0.5)
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression().fit(raw.reshape(-1, 1), targets)
        return platt.predict_proba(score(model, test).reshape(-1, 1))[:, 1]

    return fit_and_score


def run_seed(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    settings: Settings,
    *,
    study: str,
    variant: str,
    spec: RungSpec,
    seed: int,
    swipes: int,
    cluster_labels: np.ndarray | None,
    cache_root: Path,
    with_replay: bool = True,
    with_propensities: bool = False,
    shadow: str | None = None,
) -> SeedResult:
    """One session, its §4.3 outcomes, cached. A cached result is returned as is.

    `with_propensities` estimates R4's inclusion probabilities inline, and
    `shadow` (a spec) rides along a second model fitted on the same labels,
    recording whether it would have shown each card. Both serve the IPS check.
    """
    path = cache_path(cache_root, study, variant, spec, seed)
    if path.exists():
        return SeedResult.from_json(path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    user = make_variant_user(variant, features, seed=seed, cluster_labels=cluster_labels)
    model = spec.build(settings, seed)
    weights, exclude = spec.label_options(settings)
    result = run_session(
        connection,
        features,
        model,
        user,
        swipes=swipes,
        seed=seed,
        rung_label=spec.text,
        label_weights=weights,
        exclude_skips=exclude,
        with_propensities=with_propensities,
        shadow=RungSpec.parse(shadow).build(settings, seed) if shadow else None,
    )

    replay_auc = replay_log_loss = float("nan")
    if with_replay and spec.rung != "r0":
        replayed = replay(
            result.actions,
            replay_fit_and_score(spec, settings, features, result.image_ids, result.actions),
            label=spec.text,
        )
        replay_auc, replay_log_loss = replayed.mean("auc"), replayed.mean("log_loss")

    ece = float("nan")
    if spec.rung in PROBABILITY_RUNGS:
        served = [
            (1.0 if action in ("like", "save") else 0.0, score)
            for action, score, rung in zip(
                result.actions, result.scores, result.served_rungs, strict=True
            )
            if rung == spec.rung and action != "skip"
        ]
        if served:
            targets, scores = map(np.array, zip(*served, strict=True))
            ece = expected_calibration_error(targets, scores)

    verdicts = [i for i, action in enumerate(result.actions) if action != "skip"]
    oracle_auc = roc_auc(
        np.array([result.actions[i] in ("like", "save") for i in verdicts], dtype=int),
        np.array([result.true_probabilities[i] for i in verdicts]),
    )

    times = list(user.discovery_times().values())
    time_to_all = (
        float(max(times)) if all(t is not None for t in times) else float(len(result.actions) + 1)
    )

    drift_recovery = float("nan")
    if user.switched_at is not None:
        new_id = max(user.interest_ids)
        found = user.discovered_at.get(new_id)
        drift_recovery = float(
            found - user.switched_at
            if found is not None
            else len(result.actions) - user.switched_at + 1
        )

    from visualtaste.rec.r5_interests import tracker_of

    tracker = tracker_of(model)
    history = list(tracker.state.history) if tracker is not None else []
    id_stability = interest_id_stability(history)

    recent = result.actions[-100:]
    outcome = SeedResult(
        oracle_auc=oracle_auc,
        time_to_all_found=time_to_all,
        drift_recovery=drift_recovery,
        id_stability=id_stability,
        interest_history=history,
        propensities=result.propensities,
        shadow_shown=result.shadow_shown,
        study=study,
        variant=variant,
        spec=spec.text,
        seed=seed,
        swipes=len(result.actions),
        seconds=round(time.perf_counter() - started, 2),
        like_rate=like_rate(result.actions),
        like_rate_last_100=like_rate(recent) if recent else float("nan"),
        interests_planted=len(user.interest_ids),
        interests_found=user.interests_found(DISCOVERY_HITS),
        all_found=user.interests_found(DISCOVERY_HITS) == len(user.interest_ids),
        discovery={str(k): v for k, v in user.discovery_times().items()},
        coverage_last_500=coverage(result.shown_clusters[-COVERAGE_WINDOW:], COLDSTART_CLUSTERS),
        diversity=float(np.mean(result.batch_diversities)),
        novelty=float(np.mean(result.novelties)),
        replay_auc=replay_auc,
        replay_log_loss=replay_log_loss,
        ece=ece,
        offset=user.offset,
        slope=user.slope,
        switched_at=user.switched_at,
        image_ids=result.image_ids,
        actions=result.actions,
        served_rungs=result.served_rungs,
        dwell_ms=result.dwell_ms,
        true_probabilities=result.true_probabilities,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(outcome.to_json(), encoding="utf-8")
    tmp.replace(path)
    logger.info(
        "%s %s %s seed %d: like %.3f, found %d/%d, coverage %.2f, replay AUC %.3f (%.1fs)",
        study,
        variant,
        spec.text,
        seed,
        outcome.like_rate,
        outcome.interests_found,
        outcome.interests_planted,
        outcome.coverage_last_500,
        outcome.replay_auc,
        outcome.seconds,
    )
    return outcome


def clusters_for_variant(features: FeatureBundle, variant: str, seed: int) -> np.ndarray | None:
    """Where a variant plants its interests from: cluster64, or its own k-means."""
    space = VARIANTS[variant].space
    if space == "siglip":
        return features.cluster64
    return space_clusters(np.asarray(getattr(features, space)), seed=seed)


def run_study(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    settings: Settings,
    *,
    study: str,
    variants: Sequence[str],
    specs: Sequence[str],
    seeds: int,
    swipes: int,
    cache_root: Path,
    with_replay: bool = True,
    with_propensities: bool = False,
    shadow: str | None = None,
) -> dict[tuple[str, str], list[SeedResult]]:
    """Every (variant, spec) over `seeds` paired seeds. Cached and resumable."""
    out: dict[tuple[str, str], list[SeedResult]] = {}
    for variant in variants:
        clusters = clusters_for_variant(features, variant, settings.seed)
        for text in specs:
            spec = RungSpec.parse(text)
            out[(variant, text)] = [
                run_seed(
                    connection,
                    features,
                    settings,
                    study=study,
                    variant=variant,
                    spec=spec,
                    seed=settings.seed + index,
                    swipes=swipes,
                    cluster_labels=clusters,
                    cache_root=cache_root,
                    with_replay=with_replay,
                    with_propensities=with_propensities,
                    shadow=shadow,
                )
                for index in range(seeds)
            ]
    return out


# ---------------------------------------------------------------------------
# Statistics (Part 2 §4.1)
# ---------------------------------------------------------------------------


@dataclass
class PairedComparison:
    metric: str
    first: str
    second: str
    variant: str
    difference: float
    low: float
    high: float
    wins: int
    losses: int
    p_value: float
    losing_seeds: list[int]
    seeds: int
    expect: str  # "higher" or "lower": which direction the hypothesis predicts

    @property
    def verdict(self) -> str:
        """supported / not supported / inconclusive, per Part 2 §4.1.

        Supported needs the whole interval on the predicted side of zero. An
        interval wholly on the other side is "not supported" -- a detectable
        effect in the wrong direction. Anything crossing zero is inconclusive.
        """
        if np.isnan(self.low) or self.low <= 0.0 <= self.high:
            return "inconclusive"
        positive = self.low > 0.0
        return "supported" if positive == (self.expect == "higher") else "not supported"

    def row(self) -> str:
        losers = ", ".join(str(s) for s in self.losing_seeds) or "none"
        return (
            f"| {self.variant} | `{self.first}` vs `{self.second}` | {self.metric} "
            f"| {self.difference:+.4f} | [{self.low:+.4f}, {self.high:+.4f}] "
            f"| {self.wins}-{self.losses} (p = {self.p_value:.3g}) | {losers} "
            f"| {self.seeds} | **{self.verdict}** |"
        )


COMPARISON_HEADER = (
    "| variant | comparison | metric | mean diff | 90% CI | sign test (W-L) "
    "| losing seeds | seeds | verdict |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def compare(
    results: dict[tuple[str, str], list[SeedResult]],
    *,
    variant: str,
    first: str,
    second: str,
    metric: str,
    expect: str = "higher",
) -> PairedComparison:
    """Paired comparison of two specs on one metric, seed by seed.

    For a metric where lower is better (log loss, ECE), pass `expect="lower"`:
    the difference is still `first - second`, and "supported" then means the
    interval is wholly below zero. A "loss" is a seed that went the wrong way.
    """
    a = {r.seed: r.metric(metric) for r in results[(variant, first)]}
    b = {r.seed: r.metric(metric) for r in results[(variant, second)]}
    seeds = sorted(s for s in a if s in b and not np.isnan(a[s]) and not np.isnan(b[s]))
    first_values = np.array([a[s] for s in seeds])
    second_values = np.array([b[s] for s in seeds])
    difference, low, high = paired_bootstrap(
        first_values, second_values, samples=BOOTSTRAP_SAMPLES, credibility=CREDIBILITY
    )
    sign = 1.0 if expect == "higher" else -1.0
    deltas = sign * (first_values - second_values)
    wins, losses = int((deltas > 0).sum()), int((deltas < 0).sum())
    return PairedComparison(
        metric=metric,
        first=first,
        second=second,
        variant=variant,
        difference=difference,
        low=low,
        high=high,
        wins=wins,
        losses=losses,
        p_value=sign_test(wins, losses),
        losing_seeds=[s for s, d in zip(seeds, deltas, strict=True) if d < 0],
        seeds=len(seeds),
        expect=expect,
    )


SUMMARY_METRICS = (
    ("like_rate", "like rate", 3),
    ("interests_found", "found", 2),
    ("all_found", "all found", 2),
    ("time_to_all_found", "swipes to find all", 0),
    ("drift_recovery", "S5 recovery (swipes)", 0),
    ("id_stability", "id stability", 2),
    ("coverage_last_500", "coverage", 2),
    ("diversity", "diversity", 3),
    ("novelty", "novelty", 3),
    ("replay_auc", "replay AUC", 3),
    ("oracle_auc", "oracle AUC", 3),
    ("replay_log_loss", "replay log loss", 3),
    ("ece", "ECE", 3),
)


def _cell(values: list[float], places: int) -> str:
    array = np.array(values, dtype=float)
    array = array[~np.isnan(array)]
    if not len(array):
        return "--"
    return f"{array.mean():.{places}f} ± {array.std():.{places}f}"


def summary_table(results: dict[tuple[str, str], list[SeedResult]], variant: str) -> str:
    """One row per spec for one variant: mean ± sd across seeds per §4.3 metric."""
    header = "| spec | seeds | " + " | ".join(label for _, label, _ in SUMMARY_METRICS) + " |"
    lines = [header, "|" + " --- |" * (len(SUMMARY_METRICS) + 2)]
    for (v, spec), seeds in results.items():
        if v != variant:
            continue
        cells = [
            _cell([float(getattr(r, name)) for r in seeds], places)
            for name, _, places in SUMMARY_METRICS
        ]
        lines.append(f"| `{spec}` | {len(seeds)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def discovery_table(results: dict[tuple[str, str], list[SeedResult]], variant: str) -> str:
    """Median swipes to find each interest, with how many seeds were censored."""
    lines = [
        "| spec | median swipes to find an interest | censored (never found) |",
        "| --- | --- | --- |",
    ]
    for (v, spec), seeds in results.items():
        if v != variant:
            continue
        times = [t for r in seeds for t in r.discovery.values()]
        found = [t for t in times if t is not None]
        median = f"{np.median(found):.0f}" if found else "--"
        lines.append(f"| `{spec}` | {median} | {len(times) - len(found)} of {len(times)} |")
    return "\n".join(lines)


def results_to_rows(results: dict[tuple[str, str], list[SeedResult]]) -> list[dict[str, Any]]:
    """Flat per-seed rows without the histories, for notebooks and figures."""
    heavy = {
        "image_ids",
        "actions",
        "served_rungs",
        "dwell_ms",
        "true_probabilities",
        "propensities",
        "shadow_shown",
        "interest_history",
    }
    return [
        {k: v for k, v in asdict(r).items() if k not in heavy}
        for seeds in results.values()
        for r in seeds
    ]


# ---------------------------------------------------------------------------
# H2: the learning curve, replayed on neutral histories
# ---------------------------------------------------------------------------

#: Label counts at which H2 compares the rungs: from 50, so "R3 overtakes R2
#: at about 150" is visible, to 900, leaving a 100-event test block.
H2_CUTS: tuple[int, ...] = tuple(range(50, 901, 50))


@dataclass
class CurvePoint:
    cut: int
    first_mean: float
    second_mean: float
    comparison: PairedComparison


def learning_curve(
    histories: Sequence[SeedResult],
    features: FeatureBundle,
    settings: Settings,
    *,
    first: str,
    second: str,
    metric: str,
    expect: str,
    cuts: Sequence[int] = H2_CUTS,
) -> list[CurvePoint]:
    """Replay two specs over the same histories; compare them cut by cut.

    The histories should come from a policy that favours neither model -- R0's
    cold-start round-robin -- because a history logged by R3 is a set of
    images R3 chose, and replaying R2 on it would test R2 on R3's exam.
    Paired by seed at each cut.
    """
    by_cut: dict[int, dict[str, dict[int, float]]] = {cut: {first: {}, second: {}} for cut in cuts}
    for history in histories:
        for text in (first, second):
            spec = RungSpec.parse(text)
            replayed = replay(
                history.actions,
                replay_fit_and_score(spec, settings, features, history.image_ids, history.actions),
                label=text,
                cuts=cuts,
            )
            for point in replayed.points:
                if point.usable:
                    by_cut[point.cut][text][history.seed] = getattr(point, metric)

    out: list[CurvePoint] = []
    for cut in cuts:
        a, b = by_cut[cut][first], by_cut[cut][second]
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        fake = {
            ("curve", first): [_point_result(first, s, a[s], metric) for s in seeds],
            ("curve", second): [_point_result(second, s, b[s], metric) for s in seeds],
        }
        comparison = compare(
            fake, variant="curve", first=first, second=second, metric=metric, expect=expect
        )
        out.append(
            CurvePoint(
                cut=cut,
                first_mean=float(np.mean([a[s] for s in seeds])),
                second_mean=float(np.mean([b[s] for s in seeds])),
                comparison=comparison,
            )
        )
    return out


def _point_result(spec: str, seed: int, value: float, metric: str) -> SeedResult:
    """A minimal SeedResult carrying one value, so `compare` can be reused."""
    blank = dict.fromkeys(f.name for f in SeedResult.__dataclass_fields__.values())
    blank.update(
        study="curve",
        variant="curve",
        spec=spec,
        seed=seed,
        discovery={},
        image_ids=[],
        actions=[],
        served_rungs=[],
        dwell_ms=[],
        true_probabilities=[],
    )
    blank[metric] = value
    return SeedResult(**blank)


def render_curve(points: Sequence[CurvePoint], *, first: str, second: str, metric: str) -> str:
    lines = [
        f"| labels | `{first}` {metric} | `{second}` {metric} | diff | 90% CI | W-L | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in points:
        c = point.comparison
        lines.append(
            f"| {point.cut} | {point.first_mean:.4f} | {point.second_mean:.4f} "
            f"| {c.difference:+.4f} | [{c.low:+.4f}, {c.high:+.4f}] "
            f"| {c.wins}-{c.losses} | {c.verdict} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 10: interleaving, IPS, and H3's "keeps 70% of the gain"
# ---------------------------------------------------------------------------


@dataclass
class InterleavedSeed:
    seed: int
    wins_a: int
    wins_b: int
    ties: int
    like_rate_a: float
    like_rate_b: float


def run_interleaved_seed(
    connection: sqlite3.Connection,
    features: FeatureBundle,
    settings: Settings,
    *,
    variant: str,
    first: str,
    second: str,
    seed: int,
    swipes: int,
    cluster_labels: np.ndarray | None,
    cache_root: Path,
    refit_every: int = 20,
) -> InterleavedSeed:
    """A simulated user swiping a team-draft feed of two specs (Part 2 §6.4).

    Both models are refitted on the shared labels every `refit_every` swipes,
    as the live system would. Until both can serve, R0 serves ordinary batches
    (the same bootstrap as every session), and only interleaved batches count.
    """
    from visualtaste.eval.interleave import InterleavedBatch, InterleavingResult
    from visualtaste.eval.sim import create_simulated_profile
    from visualtaste.rec.feed import build_batch, build_interleaved_batch
    from visualtaste.rec.labels import build_label_set
    from visualtaste.rec.r0_coldstart import ColdStartRecommender

    slug = f"{RungSpec.parse(first).slug}_vs_{RungSpec.parse(second).slug}"
    path = cache_root / "interleave" / variant / slug / f"seed_{seed}.json"
    if path.exists():
        return InterleavedSeed(**json.loads(path.read_text(encoding="utf-8")))

    user = make_variant_user(variant, features, seed=seed, cluster_labels=cluster_labels)
    specs = {"a": RungSpec.parse(first), "b": RungSpec.parse(second)}
    models = {arm: spec.build(settings, seed) for arm, spec in specs.items()}
    bootstrap = ColdStartRecommender(seed=seed)
    profile_id = create_simulated_profile(connection, note=f"interleave {first} vs {second} {seed}")
    connection.commit()

    result = InterleavingResult()
    done = 0
    while done < swipes:
        if all(m.is_fitted for m in models.values()):
            batch = build_interleaved_batch(
                connection,
                features,
                {arm: (m, specs[arm].text) for arm, m in models.items()},
                profile_id=profile_id,
                experiment_id=0,
            )
        else:
            batch = build_batch(
                connection, features, bootstrap, profile_id=profile_id, model_version="r0_bootstrap"
            )
        if not batch.cards:
            break
        positions = [features.position(card.image_id) for card in batch.cards]
        actions = user.react_rows(features, positions)
        for card, action in zip(batch.cards, actions, strict=False):
            connection.execute(
                "INSERT INTO events (impression_id, profile_id, image_id, action) "
                "VALUES (?, ?, ?, ?)",
                (card.impression_id, profile_id, card.image_id, action),
            )
        connection.commit()
        if batch.experiment_id is not None:
            arms = [card.arm or "a" for card in batch.cards]
            result.record_batch(
                InterleavedBatch(batch.batch_id, list(range(len(arms))), arms, arms[0]), actions
            )
        done += len(actions)
        if done % refit_every < len(actions):
            labels = build_label_set(connection, profile_id)
            for model in models.values():
                model.fit(labels, features)

    outcome = InterleavedSeed(
        seed=seed,
        wins_a=result.wins_a,
        wins_b=result.wins_b,
        ties=result.ties,
        like_rate_a=result.like_rate_a,
        like_rate_b=result.like_rate_b,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(outcome)), encoding="utf-8")
    logger.info(
        "interleave %s vs %s seed %d: %d-%d (%d tied)",
        first,
        second,
        seed,
        outcome.wins_a,
        outcome.wins_b,
        outcome.ties,
    )
    return outcome


@dataclass
class IpsCheck:
    """SNIPS estimate of a deterministic policy's like rate from R4's logs."""

    estimate: float
    low: float
    high: float
    actual: float
    bias: float
    effective_sample_size: float
    logged_cards: int
    seeds: int


def ips_check(logged: Sequence[SeedResult], actual: Sequence[SeedResult]) -> IpsCheck:
    """Estimate the shadow policy's like rate from logged R4 sessions, and compare.

    Only cards served by R4 itself count: those are the ones whose propensity
    was estimated and whose shadow indicator was computed against a fitted
    shadow. The interval is a bootstrap over seeds (each seed's log is
    resampled as a unit, because cards within a session are not independent).
    """
    from visualtaste.eval.propensity import snips

    per_seed = []
    for result in logged:
        keep = [
            i
            for i, rung in enumerate(result.served_rungs)
            if rung == "r4" and not np.isnan(result.propensities[i])
        ]
        rewards = np.array([result.actions[i] in ("like", "save") for i in keep], dtype=float)
        logged_p = np.array([result.propensities[i] for i in keep])
        new_p = np.array([result.shadow_shown[i] for i in keep])
        per_seed.append((rewards, logged_p, new_p))

    def pooled(indices: Sequence[int]):
        rewards = np.concatenate([per_seed[i][0] for i in indices])
        logged_p = np.concatenate([per_seed[i][1] for i in indices])
        new_p = np.concatenate([per_seed[i][2] for i in indices])
        return snips(rewards, logged_p, new_p)

    point = pooled(range(len(per_seed)))
    rng = np.random.default_rng(1729)
    boots = []
    for _ in range(2_000):
        estimate = pooled(rng.integers(0, len(per_seed), len(per_seed))).value
        if not np.isnan(estimate):
            boots.append(estimate)
    low, high = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)
    actual_rate = float(np.mean([r.like_rate for r in actual]))
    return IpsCheck(
        estimate=point.value,
        low=float(low),
        high=float(high),
        actual=actual_rate,
        bias=point.value - actual_rate,
        effective_sample_size=point.effective_sample_size,
        logged_cards=sum(len(s[0]) for s in per_seed),
        seeds=len(per_seed),
    )


def gain_retention(
    results: dict[tuple[str, str], list[SeedResult]],
    *,
    variant: str,
    candidate: str,
    reference: str,
    baseline: str,
) -> tuple[float, float, float]:
    """(ratio, low, high): candidate's like-rate gain over baseline / reference's.

    H3 asks R4 to keep at least 70% of R3's gain over R0. A ratio of two noisy
    differences is itself noisy, so the interval is a paired bootstrap over
    seeds of the ratio of mean gains.
    """

    def by_seed(spec: str) -> dict[int, float]:
        return {r.seed: r.like_rate for r in results[(variant, spec)]}

    c, r, b = by_seed(candidate), by_seed(reference), by_seed(baseline)
    seeds = sorted(set(c) & set(r) & set(b))
    gc = np.array([c[s] - b[s] for s in seeds])
    gr = np.array([r[s] - b[s] for s in seeds])
    ratio = float(gc.mean() / gr.mean()) if gr.mean() else float("nan")
    rng = np.random.default_rng(1729)
    draws = rng.integers(0, len(seeds), size=(BOOTSTRAP_SAMPLES, len(seeds)))
    denominators = gr[draws].mean(axis=1)
    ratios = gc[draws].mean(axis=1) / np.where(denominators == 0, np.nan, denominators)
    low, high = np.nanpercentile(ratios, [5, 95])
    return ratio, float(low), float(high)
