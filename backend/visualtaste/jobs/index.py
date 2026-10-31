"""Build the FAISS indexes and fold away embedding-level near-duplicates.

Run:  python -m visualtaste.jobs.index
      python -m visualtaste.jobs.index --skip-dedupe

Two jobs, both pure vector search, which is why they share a file:

**The indexes.** One `IndexFlatIP` per model, saved to `<DATA>/indexes/`. The
API loads these at startup; `rec/r2_knn.py` and `GET /images/{id}/similar` are
the callers.

**Embedding near-duplicates** (brief §7.2 step 8). The pHash pass in Phase 3
catches re-encodes and resizes. This pass catches what pHash misses: a crop, a
border, or a colour grade changes the coarse light-and-dark layout a pHash
fingerprints, while leaving the texture and layout DINOv2 encodes nearly
untouched. DINOv2 cosine >= 0.95 folds the smaller copy into the larger, same
rule as before.

The two methods fail in different ways, which is the point of running both.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from visualtaste.config import Settings, get_settings
from visualtaste.data.dedupe import (
    EMBEDDING_COSINE_THRESHOLD,
    find_embedding_duplicate_groups,
)
from visualtaste.db import repo
from visualtaste.search.index import VectorIndex
from visualtaste.vision.encoders import DINOV2_KEY, ENCODER_SPECS
from visualtaste.vision.store import EmbeddingStore

logger = logging.getLogger(__name__)


def build_indexes(settings: Settings) -> dict[str, int]:
    """Build and save a FAISS index for every model that has been embedded."""
    built: dict[str, int] = {}

    for model_key in sorted(ENCODER_SPECS):
        store = EmbeddingStore(settings.embedding_dir(model_key), model_key)
        if not store.vectors_path.exists():
            logger.warning("%s has no merged embeddings yet; run jobs.embed first", model_key)
            continue

        vectors, ids = store.load()
        index = VectorIndex(vectors, ids)
        index.save(settings.index_path(model_key))
        built[model_key] = len(index)

    return built


def mark_embedding_duplicates(
    settings: Settings,
    connection: sqlite3.Connection,
    *,
    threshold: float = EMBEDDING_COSINE_THRESHOLD,
) -> int:
    """Fold DINOv2 near-duplicates into their largest copy. Returns the count.

    Only images currently `ok` are considered: one already folded away by the
    pHash pass must not be folded again into a different group, which would make
    the `dup_of` chains ambiguous.
    """
    store = EmbeddingStore(settings.embedding_dir(DINOV2_KEY), DINOV2_KEY)
    if not store.vectors_path.exists():
        logger.warning("no DINOv2 embeddings; skipping embedding-level dedupe")
        return 0

    vectors, ids = store.load()
    usable = set(repo.image_ids_with_status(connection, "ok"))

    keep_positions = [i for i, image_id in enumerate(ids.tolist()) if image_id in usable]
    if not keep_positions:
        logger.warning("no usable images with DINOv2 embeddings")
        return 0

    candidate_ids = [ids[i] for i in keep_positions]
    candidate_vectors = vectors[keep_positions]

    # Pixel counts decide which copy survives, so they have to come from the
    # database rather than being inferred from the vectors.
    pixels = {
        row["image_id"]: (row["width"] or 0) * (row["height"] or 0)
        for row in repo.query(
            connection, "SELECT image_id, width, height FROM images WHERE status = 'ok'"
        )
    }

    groups = find_embedding_duplicate_groups(
        candidate_ids, candidate_vectors, threshold=threshold, pixels=pixels
    )

    folded = 0
    with repo.transaction(connection):
        for group in groups:
            for duplicate in group.duplicates:
                repo.set_image_status(connection, duplicate, "near_dup", dup_of=group.kept)
                folded += 1

    logger.info(
        "embedding near-duplicates: folded %d images across %d groups (cosine >= %.2f)",
        folded,
        len(groups),
        threshold,
    )
    return folded


def render_summary(built: dict[str, int], folded: int, connection: sqlite3.Connection) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### Index and embedding-dedupe summary ({now})",
        "",
        "| model | vectors indexed |",
        "| --- | --- |",
    ]
    for model_key, count in sorted(built.items()):
        lines.append(f"| `{model_key}` | {count:,} |")
    lines += [
        "",
        f"- Embedding-level near-duplicates folded away: **{folded:,}** "
        f"(DINOv2 cosine >= {EMBEDDING_COSINE_THRESHOLD}).",
        "- Status counts now: "
        + ", ".join(
            f"`{k}` = {v:,}" for k, v in sorted(repo.count_images_by_status(connection).items())
        ),
        "",
    ]
    return "\n".join(lines)


def _configure_logging(settings: Settings) -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / f"index_{datetime.now(UTC):%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dedupe", action="store_true", help="build indexes only")
    parser.add_argument(
        "--threshold",
        type=float,
        default=EMBEDDING_COSINE_THRESHOLD,
        help=f"DINOv2 cosine threshold (default {EMBEDDING_COSINE_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    _configure_logging(settings)

    connection = repo.init_db(settings.db_path)

    folded = 0
    if not args.skip_dedupe:
        folded = mark_embedding_duplicates(settings, connection, threshold=args.threshold)

    # The indexes hold a vector for *every* embedded image, including ones
    # folded away or quarantined. That is deliberate: keeping the row order
    # identical to `vectors.npy` means the id mapping never drifts, and the
    # Review page still wants neighbours for a quarantined image. Callers
    # filter by status instead -- `GET /images/{id}/similar` drops non-`ok`
    # neighbours, and the feed draws only from `selected = 1 AND status = 'ok'`.
    built = build_indexes(settings)

    summary = render_summary(built, folded, connection)
    print("\n" + summary)
    (settings.logs_dir / "index_summary.md").write_text(summary, encoding="utf-8")

    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
