"""Compute the Embedding Map: PCA-50 -> UMAP-2 for SigLIP 2 and DINOv2.

    python -m visualtaste.jobs.project                 # both models
    python -m visualtaste.jobs.project --models siglip2_b16_224

Over the 50,000-image selected pool, cached under `<DATA>/derived_features/`
as `umap2d_<model_key>.{npy,_ids.npy,joblib}`. Seeded, so re-running draws
the same map. Run time is appended to docs/BENCHMARKS.md, because UMAP is the
slowest thing in the analytics stack and the README should say how long to wait.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="siglip2_b16_224,dinov2_b14")
    args = parser.parse_args(argv)

    from visualtaste.api.deps import AppState
    from visualtaste.config import get_settings
    from visualtaste.viz.projection import fit_projection, save

    settings = get_settings()
    state = AppState(settings)
    state.load()
    features = state.require_features()

    rows = []
    for model_key in [m.strip() for m in args.models.split(",") if m.strip()]:
        vectors = features.siglip if model_key.startswith("siglip") else features.dinov2
        if vectors is None:
            logger.warning("no vectors for %s; skipped", model_key)
            continue
        projection = fit_projection(
            vectors, features.image_ids, model_key=model_key, seed=settings.seed
        )
        save(projection, settings.features_dir)
        rows.append(f"| `{model_key}` | {len(vectors):,} | {projection.seconds:.0f} s |")

    if rows:
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        with (settings.project_root / "docs" / "BENCHMARKS.md").open("a", encoding="utf-8") as fh:
            fh.write(
                f"\n## UMAP projection ({now})\n\n```\npython -m visualtaste.jobs.project "
                f"--models {args.models}\n```\n\nPCA-50 then UMAP (cosine, n_neighbors 30, "
                f"min_dist 0.1, seeded, so single-threaded). Machine: "
                f"{platform.processor() or platform.machine()}. Run alongside the simulator "
                "studies, so on a busy CPU.\n\n"
                "| model | points | wall time |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
