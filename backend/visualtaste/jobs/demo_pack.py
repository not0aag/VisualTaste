"""Build the demo pack: a small, real collection that runs without a GPU.

The full build downloads ~80,000 images and embeds them on a GPU, which takes
hours. The demo pack is a stratified slice of the finished collection, small
enough to download in a minute and complete enough to run every page:

- N selected images, an even share from each of the 64 visual clusters, from
  the CC0, public-domain and CC BY sources only (Unsplash is left out: the
  Unsplash Lite dataset terms do not clearly allow redistributing the files);
- their display and thumbnail files (display re-encoded smaller);
- their SigLIP 2 and DINOv2 vectors, FAISS indexes, PCA-128 and colour
  features, and Embedding Map coordinates;
- the standardisation statistics of the full collection, unchanged, so the
  models see features on the same scale as in the full build;
- a fresh database holding just those images, and an attribution list.

Not included: the UMAP models (74 MB each), so the map's "taste over time"
slider needs the full build.

    python -m visualtaste.jobs.demo_pack --images 2000 --out demo-pack.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from visualtaste.config import Settings, get_settings
from visualtaste.db import repo
from visualtaste.vision.encoders import DINOV2_KEY, SIGLIP2_KEY

logger = logging.getLogger(__name__)

EXCLUDED_SOURCES = ("unsplash_lite",)
DISPLAY_QUALITY = 72
SEED = 1729
#: Feature files cut to the pack's rows: (array, ids) pairs, plus their stats.
FEATURE_BLOCKS = ("pca128", "colour64", "dinov2_pca128")


def choose_images(
    connection: sqlite3.Connection,
    count: int,
    seed: int = SEED,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """An even share of each cluster64, by a fixed seed, from redistributable sources."""
    placeholders = ",".join("?" for _ in EXCLUDED_SOURCES)
    rows = connection.execute(
        "SELECT image_id, cluster64 FROM images "
        "WHERE selected = 1 AND status = 'ok' AND cluster64 IS NOT NULL "
        f"AND source NOT IN ({placeholders}) ORDER BY image_id",
        EXCLUDED_SOURCES,
    ).fetchall()
    by_cluster: dict[int, list[str]] = {}
    for image_id, cluster in rows:
        if image_id not in exclude:
            by_cluster.setdefault(int(cluster), []).append(image_id)

    rng = np.random.default_rng(seed)
    per_cluster = -(-count // len(by_cluster))  # ceiling division
    chosen: list[str] = []
    for cluster in sorted(by_cluster):
        members = by_cluster[cluster]
        take = min(per_cluster, len(members))
        chosen += [members[i] for i in rng.choice(len(members), take, replace=False)]
    rng.shuffle(chosen)
    return sorted(chosen[:count])


def _subset(ids_path: Path, array_path: Path, keep: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rows of a feature file for `keep`, in `keep`'s order."""
    ids = np.load(ids_path)
    position = {str(image_id): i for i, image_id in enumerate(ids.tolist())}
    rows = [position[i] for i in keep]
    return np.load(array_path)[rows], np.array(keep, dtype=ids.dtype)


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array)


def build_pack(
    settings: Settings, count: int, out: Path, exclude: frozenset[str] = frozenset()
) -> dict:
    """Write the pack as a .tar.gz at `out`. Returns its manifest."""
    source_db = repo.connect(settings.db_path, read_only=True)
    keep = choose_images(source_db, count, exclude=exclude)
    source_db.close()
    logger.info("demo pack: %d images chosen", len(keep))

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"

        # 1. A fresh database with just these images.
        db_path = data / "db" / "visualtaste.db"
        db_path.parent.mkdir(parents=True)
        connection = repo.init_db(db_path)
        # Read from only; a plain path, since this connection is not in URI mode.
        connection.execute("ATTACH DATABASE ? AS src", (str(settings.db_path),))
        marks = ",".join("?" for _ in keep)
        with connection:
            for table in ("images", "image_sources"):
                connection.execute(
                    f"INSERT INTO {table} SELECT * FROM src.{table} WHERE image_id IN ({marks})",
                    keep,
                )
            connection.execute("INSERT INTO embedding_models SELECT * FROM src.embedding_models")
        attribution = connection.execute(
            f"SELECT image_id, source, author, license_url, source_url FROM images "
            f"WHERE image_id IN ({marks}) ORDER BY image_id",
            keep,
        ).fetchall()
        connection.execute("DETACH DATABASE src")
        connection.close()

        # 2. Image files: thumbnails as they are, display images re-encoded.
        for image_id in keep:
            thumb = data / "derived" / "thumb" / image_id[:2] / f"{image_id}.webp"
            thumb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(settings.thumb_path(image_id), thumb)
            display = data / "derived" / "display" / image_id[:2] / f"{image_id}.webp"
            display.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(settings.display_path(image_id)) as image:
                image.save(display, "WEBP", quality=DISPLAY_QUALITY, method=6)

        # 3. Vectors and indexes for both models.
        from visualtaste.search.index import VectorIndex

        for model_key in (SIGLIP2_KEY, DINOV2_KEY):
            source = settings.embedding_dir(model_key)
            vectors, ids = _subset(source / "ids.npy", source / "vectors.npy", keep)
            target = data / "embeddings" / model_key
            _save(target / "vectors.npy", vectors.astype(np.float32))
            _save(target / "ids.npy", ids)
            meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
            # Only the count changes; EmbeddingMeta rejects unknown fields.
            meta["count"] = len(keep)
            (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            VectorIndex(vectors, ids).save(data / "indexes" / f"{model_key}.faiss")

        # 4. Feature blocks cut to these rows; full-collection statistics kept.
        features = data / "derived_features"
        for block in FEATURE_BLOCKS:
            src = settings.features_dir
            if not (src / f"{block}.npy").exists():
                continue
            array, ids = _subset(src / f"{block}_ids.npy", src / f"{block}.npy", keep)
            _save(features / f"{block}.npy", array)
            _save(features / f"{block}_ids.npy", ids)
            stats = src / f"{block}_standardise.npz"
            if stats.exists():
                shutil.copyfile(stats, features / stats.name)
        for model_key in (SIGLIP2_KEY, DINOV2_KEY):
            ids_path = settings.features_dir / f"umap2d_{model_key}_ids.npy"
            if ids_path.exists():
                coords, ids = _subset(
                    ids_path, settings.features_dir / f"umap2d_{model_key}.npy", keep
                )
                _save(features / f"umap2d_{model_key}.npy", coords)
                _save(features / f"umap2d_{model_key}_ids.npy", ids)

        # 5. The concept vocabulary, for interest labels and taste in words.
        concepts = settings.data / "concepts"
        if concepts.exists():
            shutil.copytree(concepts, data / "concepts")

        # 6. Credit, and a manifest.
        lines = ["# Demo pack attribution", "", "| image | source | author | licence |"]
        lines += ["| --- | --- | --- | --- |"]
        lines += [
            f"| {image_id} | {source} | {author or ''} | {licence} |"
            for image_id, source, author, licence, _ in attribution
        ]
        (data / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        sources: dict[str, int] = {}
        for _, source, *_rest in attribution:
            sources[source] = sources.get(source, 0) + 1
        manifest = {
            "images": len(keep),
            "sources": sources,
            "clusters": 64,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "excluded_sources": list(EXCLUDED_SOURCES),
            "includes_umap_models": False,
        }
        (data / "PACK.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w:gz") as archive:
            archive.add(data, arcname="data")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    manifest.update(bytes=out.stat().st_size, sha256=digest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the demo pack.")
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("demo-pack.tar.gz"))
    parser.add_argument(
        "--exclude",
        type=Path,
        help="a JSON list of image ids to leave out (e.g. images whose licence "
        "could not be confirmed)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    exclude: frozenset[str] = frozenset()
    if args.exclude:
        exclude = frozenset(json.loads(args.exclude.read_text(encoding="utf-8")))
    manifest = build_pack(get_settings(), args.images, args.out.resolve(), exclude)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
