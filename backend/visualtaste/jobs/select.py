"""Curate the pool: safety screen, clusters, PCA, and the 50,000 selection.

Run:  python -m visualtaste.jobs.select
      python -m visualtaste.jobs.select --skip-safety
      python -m visualtaste.jobs.select --target 10000   (a smaller pool)

Everything here decides *what Alen can be shown*, which makes it the most
consequential job in the build after acquisition. In order:

1. **Safety screen** (§7.2 step 7). SigLIP 2 zero-shot, calibrated on 200
   random images. Flagged images become `quarantined` -- never deleted -- and
   appear on the Review page. If more than 0.5% of candidates flag, the brief's
   §3 Q5 says to ask the owner rather than act.
2. **cluster64** for the cold-start rung and the exploration quota.
3. **cluster400 + capped sampling** to pick 50,000 varied images (§7.2 step 9).
4. **PCA-128** fitted on the SigLIP 2 vectors of the selected pool, which is
   the feature space R3 learns in.

Order matters: the safety screen runs first so quarantined images cannot be
selected, and PCA is fitted on the *selected* pool so its components describe
the space the recommender will actually operate in.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from visualtaste.config import Settings, get_settings
from visualtaste.data.safety import (
    CALIBRATION_FALSE_FLAG_RATE,
    NEUTRAL_PROMPTS,
    PROMPT_TEMPLATE,
    UNSAFE_PROMPTS,
    build_screen,
    sample_reference_vectors,
)
from visualtaste.data.select import (
    CLUSTER_COUNT,
    COLDSTART_CLUSTER_COUNT,
    SELECTION_TARGET,
    SelectionResult,
    fit_clusters,
    fit_pca,
    select_diverse,
)
from visualtaste.db import repo
from visualtaste.vision.encoders import DINOV2_KEY, SIGLIP2_KEY, load_encoder, pick_device
from visualtaste.vision.store import EmbeddingStore

logger = logging.getLogger(__name__)

# Brief §3 Q5: above this share of candidates flagged, a human decides.
Q5_FLAG_RATE = 0.005


@dataclass
class SafetyOutcome:
    """What the safety screen did, and whether it needs a human."""

    screened: int
    flagged: int
    threshold: float
    calibration_size: int

    @property
    def flag_rate(self) -> float:
        return self.flagged / self.screened if self.screened else 0.0

    @property
    def needs_human_review(self) -> bool:
        """True when §3 Q5 applies: more than 0.5% of candidates flagged."""
        return self.flag_rate > Q5_FLAG_RATE


def run_safety_screen(settings: Settings, connection: sqlite3.Connection) -> SafetyOutcome:
    """Screen every usable image and quarantine what the screen flags."""
    store = EmbeddingStore(settings.embedding_dir(SIGLIP2_KEY), SIGLIP2_KEY)
    vectors, ids = store.load()

    usable = set(repo.image_ids_with_status(connection, "ok"))
    positions = [i for i, image_id in enumerate(ids.tolist()) if image_id in usable]
    if not positions:
        logger.warning("no usable images with SigLIP 2 embeddings to screen")
        return SafetyOutcome(0, 0, 0.0, 0)

    candidate_ids = [ids[i] for i in positions]
    candidate_vectors = vectors[positions]

    settings.apply_hf_env()
    encoder = load_encoder(SIGLIP2_KEY, device=pick_device())
    reference = sample_reference_vectors(candidate_vectors, seed=settings.seed)
    screen = build_screen(encoder, reference)

    flagged_mask, margins = screen.flag(candidate_vectors)
    flagged_ids = [candidate_ids[i] for i in np.flatnonzero(flagged_mask)]

    with repo.transaction(connection):
        for image_id in flagged_ids:
            repo.set_image_status(connection, image_id, "quarantined")
            repo.record_removal(connection, image_id, "safety_screen: flagged for review")

    outcome = SafetyOutcome(
        screened=len(candidate_ids),
        flagged=len(flagged_ids),
        threshold=screen.threshold,
        calibration_size=screen.calibration_size,
    )
    logger.info(
        "safety screen: %d of %d flagged (%.3f%%), threshold %.4f, margin range %.3f to %.3f",
        outcome.flagged,
        outcome.screened,
        100 * outcome.flag_rate,
        screen.threshold,
        float(margins.min()),
        float(margins.max()),
    )
    if outcome.needs_human_review:
        logger.warning(
            "%.2f%% of candidates were flagged, above the %.1f%% in §3 Q5 -- "
            "the owner should be asked whether to auto-remove them or review "
            "them in the app",
            100 * outcome.flag_rate,
            100 * Q5_FLAG_RATE,
        )
    return outcome


def assign_coldstart_clusters(settings: Settings, connection: sqlite3.Connection) -> int:
    """Compute `cluster64` for every usable image and store it.

    These are the clusters R0 walks in round-robin until it has 30 labels, and
    the ones R4's exploration quota draws from. Computed over all usable images
    rather than only the selected pool, so an image promoted into the pool later
    already has its cluster.
    """
    store = EmbeddingStore(settings.embedding_dir(SIGLIP2_KEY), SIGLIP2_KEY)
    vectors, ids = store.load()

    usable = set(repo.image_ids_with_status(connection, "ok"))
    positions = [i for i, image_id in enumerate(ids.tolist()) if image_id in usable]
    if not positions:
        return 0

    candidate_ids = [ids[i] for i in positions]
    labels, _ = fit_clusters(vectors[positions], k=COLDSTART_CLUSTER_COUNT, seed=settings.seed)

    with repo.transaction(connection):
        for image_id, label in zip(candidate_ids, labels, strict=True):
            connection.execute(
                "UPDATE images SET cluster64 = ? WHERE image_id = ?",
                (int(label), image_id),
            )

    logger.info("assigned cluster64 to %d images", len(candidate_ids))
    return len(candidate_ids)


def run_selection(
    settings: Settings, connection: sqlite3.Connection, *, target: int
) -> SelectionResult:
    """Cluster into 400, sample with caps, and mark the result `selected = 1`."""
    store = EmbeddingStore(settings.embedding_dir(SIGLIP2_KEY), SIGLIP2_KEY)
    vectors, ids = store.load()

    rows = repo.query(connection, "SELECT image_id, source FROM images WHERE status = 'ok'")
    sources = {row["image_id"]: row["source"] for row in rows}
    usable = set(sources)

    positions = [i for i, image_id in enumerate(ids.tolist()) if image_id in usable]
    candidate_ids = [ids[i] for i in positions]
    candidate_vectors = vectors[positions]
    logger.info("selecting %d from %d usable candidates", target, len(candidate_ids))

    labels, _ = fit_clusters(candidate_vectors, k=CLUSTER_COUNT, seed=settings.seed)
    result = select_diverse(candidate_ids, labels, sources, target=target, seed=settings.seed)

    # Written in one transaction: a half-applied selection would leave the feed
    # drawing from a pool that no longer matches the recorded counts.
    with repo.transaction(connection):
        connection.execute("UPDATE images SET selected = 0")
        for image_id, label in result.cluster_of.items():
            connection.execute(
                "UPDATE images SET cluster400 = ? WHERE image_id = ?", (label, image_id)
            )
        for image_id in result.selected:
            connection.execute("UPDATE images SET selected = 1 WHERE image_id = ?", (image_id,))

    logger.info(
        "selected %d images across %d clusters; largest source share %.1f%%",
        len(result.selected),
        len(result.per_cluster),
        100 * result.dominant_source_share,
    )
    return result


def run_pca(
    settings: Settings,
    connection: sqlite3.Connection,
    *,
    components: int = 128,
    model_key: str = SIGLIP2_KEY,
    stem: str = "pca128",
) -> np.ndarray:
    """Fit PCA-128 on the selected pool's vectors for one model and save it.

    Saved as both the fitted transform (`<stem>.joblib`, so a single new image
    can be projected at request time) and the projected matrix
    (`<stem>.npy` with its id index, so training does not re-project 50,000
    rows on every retrain), plus the pool mean and std of the projection
    (`<stem>_standardise.npz`) that `rec/features.py` standardises with.

    SigLIP 2 -> `pca128` is the original Phase 5 artefact; DINOv2 ->
    `dinov2_pca128` was added in Phase 9 for the feature-block ablation.
    """
    import joblib

    from visualtaste.rec.features import save_stats

    store = EmbeddingStore(settings.embedding_dir(model_key), model_key)
    vectors, ids = store.load()

    selected = {
        row["image_id"]
        for row in repo.query(connection, "SELECT image_id FROM images WHERE selected = 1")
    }
    positions = [i for i, image_id in enumerate(ids.tolist()) if image_id in selected]
    if not positions:
        logger.warning("nothing selected yet; PCA skipped")
        return np.zeros(0)

    pool_ids = [ids[i] for i in positions]
    reduced, model, cumulative = fit_pca(
        vectors[positions], components=components, seed=settings.seed
    )

    features = settings.features_dir
    features.mkdir(parents=True, exist_ok=True)

    for path, array in (
        (features / f"{stem}.npy", reduced),
        (features / f"{stem}_ids.npy", np.array(pool_ids, dtype="<U16")),
    ):
        tmp = path.with_name(f".{path.name}.part")
        with tmp.open("wb") as handle:
            np.save(handle, array)
        tmp.replace(path)

    model_path = features / f"{stem}.joblib"
    tmp_model = model_path.with_name(f".{model_path.name}.part")
    joblib.dump(model, tmp_model)
    tmp_model.replace(model_path)
    save_stats(features, stem, reduced)

    logger.info(
        "%s PCA-%d saved as %s: %d x %d, cumulative explained variance %.1f%%",
        model_key,
        components,
        stem,
        *reduced.shape,
        100 * cumulative[-1],
    )
    return cumulative


def save_colour_stats(settings: Settings, connection: sqlite3.Connection) -> None:
    """Pool mean and std of colour64, for standardising the `col` block.

    Over the selected pool only, for the same reason the PCA is: those are the
    rows every model scores.
    """
    from visualtaste.rec.features import save_stats

    features = settings.features_dir
    vectors = np.load(features / "colour64.npy")
    ids = np.load(features / "colour64_ids.npy", allow_pickle=False).tolist()
    selected = {
        row["image_id"]
        for row in repo.query(connection, "SELECT image_id FROM images WHERE selected = 1")
    }
    keep = [i for i, image_id in enumerate(ids) if image_id in selected]
    save_stats(features, "colour64", vectors[keep])
    logger.info("colour64 standardisation stats saved over %d pool images", len(keep))


def render_summary(
    safety: SafetyOutcome | None,
    selection: SelectionResult | None,
    cumulative_variance: np.ndarray,
    connection: sqlite3.Connection,
) -> str:
    """The per-source and per-cluster report §7.2 step 9 asks to be recorded."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"### Selection summary ({now})", ""]

    if safety is not None:
        lines += [
            "**Safety screen** (SigLIP 2 zero-shot, see `data/safety.py` for what "
            "this can and cannot do):",
            "",
            f"- Prompt template: `{PROMPT_TEMPLATE}`",
            f"- {len(UNSAFE_PROMPTS)} unsafe prompts vs {len(NEUTRAL_PROMPTS)} neutral ones",
            f"- Calibrated on {safety.calibration_size} random images at a "
            f"{100 * CALIBRATION_FALSE_FLAG_RATE:.1f}% false-flag target",
            f"- Threshold: {safety.threshold:.4f}",
            f"- Flagged: **{safety.flagged:,} of {safety.screened:,}** "
            f"({100 * safety.flag_rate:.3f}%)",
            f"- §3 Q5 triggered: **{'yes' if safety.needs_human_review else 'no'}** "
            f"(threshold {100 * Q5_FLAG_RATE:.1f}%)",
            "",
        ]

    if selection is not None:
        lines += [
            f"**Selection:** {len(selection.selected):,} images from "
            f"{len(selection.cluster_of):,} candidates.",
            "",
            f"- Clusters used: {len(selection.per_cluster)}",
            f"- Clusters hitting the per-cluster cap: {selection.capped_clusters}",
            f"- Filled at random after capping: {selection.filled_at_random:,}",
            "",
            "| source | selected | share |",
            "| --- | --- | --- |",
        ]
        total = sum(selection.per_source.values()) or 1
        for source, count in selection.per_source.items():
            lines.append(f"| `{source}` | {count:,} | {100 * count / total:.1f}% |")

        sizes = np.array(list(selection.per_cluster.values()))
        lines += [
            "",
            f"- Per-cluster sizes: min {sizes.min()}, median "
            f"{int(np.median(sizes))}, max {sizes.max()}, mean {sizes.mean():.1f}",
            "",
        ]

    if len(cumulative_variance):
        marks = [(32, 31), (64, 63), (128, 127)]
        lines += ["**PCA on the selected pool's SigLIP 2 vectors:**", ""]
        for label, index in marks:
            if index < len(cumulative_variance):
                lines.append(
                    f"- {label} components explain "
                    f"{100 * cumulative_variance[index]:.1f}% of the variance"
                )
        lines.append("")

    lines += [
        "**Status counts:** "
        + ", ".join(
            f"`{k}` = {v:,}" for k, v in sorted(repo.count_images_by_status(connection).items())
        ),
        "",
    ]
    return "\n".join(lines)


def _configure_logging(settings: Settings) -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / f"select_{datetime.now(UTC):%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=SELECTION_TARGET)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--skip-safety", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help=(
            "only (re)fit the Phase 9 feature files over the existing selection: "
            "DINOv2 PCA-128 and the standardisation stats for every block"
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    log_path = _configure_logging(settings)
    logger.info("log file: %s", log_path)

    connection = repo.init_db(settings.db_path)

    if args.features_only:
        from visualtaste.rec.features import save_stats

        save_stats(settings.features_dir, "pca128", np.load(settings.features_dir / "pca128.npy"))
        run_pca(
            settings,
            connection,
            components=args.pca_components,
            model_key=DINOV2_KEY,
            stem="dinov2_pca128",
        )
        save_colour_stats(settings, connection)
        connection.close()
        return 0

    safety = None if args.skip_safety else run_safety_screen(settings, connection)

    assign_coldstart_clusters(settings, connection)

    selection = None
    cumulative = np.zeros(0)
    if not args.skip_selection:
        selection = run_selection(settings, connection, target=args.target)
        cumulative = run_pca(settings, connection, components=args.pca_components)
        run_pca(
            settings,
            connection,
            components=args.pca_components,
            model_key=DINOV2_KEY,
            stem="dinov2_pca128",
        )
        save_colour_stats(settings, connection)

    summary = render_summary(safety, selection, cumulative, connection)
    print("\n" + summary)
    (settings.logs_dir / "select_summary.md").write_text(summary, encoding="utf-8")

    connection.close()
    # A non-zero exit when §3 Q5 applies, so the owner is asked rather than the
    # flagged images being silently kept or silently dropped.
    return 1 if safety is not None and safety.needs_human_review else 0


if __name__ == "__main__":
    raise SystemExit(main())
