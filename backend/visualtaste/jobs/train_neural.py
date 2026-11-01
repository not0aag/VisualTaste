"""Train R6, the neural ranker, outside the API (brief §7.5).

    python -m visualtaste.jobs.train_neural                     # the active real profile
    python -m visualtaste.jobs.train_neural --profile 12        # any profile by id
    python -m visualtaste.jobs.train_neural --history <a cached session .json under data/eval/>
    python -m visualtaste.jobs.train_neural --loss bpr --record

Why a job and not the API: five networks, up to 200 epochs each, take tens of
seconds even on the GPU, and the API must answer a swipe in 50 ms. Training
here never blocks the feed. The result is registered as an *inactive* version;
it serves only after someone activates it from the Model Lab, which is the only
way R6 is ever switched on.

It refuses to train below `VT_NEURAL_MIN_LABELS` (1,500) and says why: a
network with half a million parameters fitted to a few hundred labels learns
the labels, not the taste, and an honest refusal is better than a confident
overfit sitting in the version table.

`--record` appends the wall time and peak GPU memory to docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _labels_from_history(path: Path):
    from visualtaste.eval.study import SeedResult, labels_from_history
    from visualtaste.rec.labels import label_weights

    result = SeedResult.from_json(path.read_text(encoding="utf-8"))
    return labels_from_history(result.image_ids, result.actions, label_weights(0.25), False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--profile", type=int, default=None)
    source.add_argument("--history", type=Path, default=None, help="a cached simulator session")
    parser.add_argument("--loss", choices=("pointwise", "bpr"), default="pointwise")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--record", action="store_true", help="append timings to BENCHMARKS.md")
    args = parser.parse_args(argv)

    from visualtaste.api.deps import AppState
    from visualtaste.config import get_settings
    from visualtaste.db import repo
    from visualtaste.rec import registry
    from visualtaste.rec.labels import build_label_set, label_weights
    from visualtaste.rec.r6_neural import NeuralRecommender
    from visualtaste.vision.encoders import SIGLIP2_KEY

    settings = get_settings()
    state = AppState(settings)
    state.load()
    features = state.require_features()

    connection = repo.connect(settings.db_path)
    try:
        if args.history is not None:
            labels, origin = _labels_from_history(args.history), f"history {args.history}"
        else:
            profile_id = args.profile if args.profile is not None else state.profile_id
            if profile_id is None:
                print("No profile to train on.", file=sys.stderr)
                return 2
            labels = build_label_set(
                connection, profile_id, weights=label_weights(settings.skip_weight)
            )
            origin = f"profile {profile_id}"

        if len(labels) < settings.neural_min_labels:
            print(
                f"Refusing to train R6 on {len(labels)} labels from {origin}: it needs at least "
                f"{settings.neural_min_labels} (VT_NEURAL_MIN_LABELS). Below that a network this "
                f"size memorises the labels rather than learning the taste; R3 and R4 are the "
                f"right rungs until then.",
                file=sys.stderr,
            )
            return 2

        model = NeuralRecommender(
            min_labels=settings.neural_min_labels,
            loss=args.loss,
            members=args.members,
            seed=settings.seed,
        )
        model.fit(labels, features)
        if not model.is_fitted:
            print(f"R6 could not be fitted: {model.unfit_reason}", file=sys.stderr)
            return 2

        report = model.fit_report()
        version = registry.register(
            connection,
            model,
            models_root=settings.models_dir,
            embedding_key=SIGLIP2_KEY,
            metrics={**report, "trained_from": origin},
            activate=False,
        )
    finally:
        connection.close()

    print(
        f"Registered {version.version} (inactive): {len(labels)} labels, {args.loss} loss, "
        f"mean validation log loss {report['validation_log_loss']:.4f}, "
        f"{report['train_seconds']}s, peak GPU memory {report['peak_vram_mb']} MB. "
        f"Activate it from the Model Lab."
    )
    if args.record:
        # One self-contained block per run, so a row can never land under some
        # other section that was appended to the file in between.
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        block = (
            f"\n### R6 training run ({now})\n\n"
            "```\npython -m visualtaste.jobs.train_neural "
            f"{' '.join(argv or sys.argv[1:])}\n```\n\n"
            "| version | labels | loss | members | wall time | peak VRAM | device |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            f"| `{version.version}` | {len(labels):,} | {args.loss} | {args.members} "
            f"| {report['train_seconds']} s | {report['peak_vram_mb']} MB "
            f"| {model._device()} |\n"
        )
        with (settings.project_root / "docs" / "BENCHMARKS.md").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
