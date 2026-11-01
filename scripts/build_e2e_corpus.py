"""Build a small sandboxed corpus for the Playwright smoke test.

Run:  python scripts/build_e2e_corpus.py [--root ./.e2e] [--images 40]

The end-to-end test needs a working corpus: rows in the database, WebP
derivatives on disk, embeddings, and cluster assignments. It must *not* use the
real 80,000-image one -- an automated browser clicking through 20 swipes would
write 20 events into Alen's actual history.

So this writes a disposable one: generated gradient images rather than real
photographs, because the test is checking that a keystroke becomes a database
row, not that the images are interesting.

Idempotent: re-running rebuilds the corpus from scratch, so a test run never
inherits events from the previous one.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# The script runs before the package is necessarily on the path (it is invoked
# by Playwright's webServer from the repo root), so make sure it is.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from visualtaste.config import Settings
from visualtaste.db import repo
from visualtaste.vision.store import EmbeddingStore

DEFAULT_IMAGES = 40
EMBEDDING_DIM = 32


def build(root: Path, image_count: int) -> Settings:
    if root.exists():
        shutil.rmtree(root)

    settings = Settings(
        project_root=root,
        data_root=root / "data",
        archive_root=root / "archive",
        min_free_gb=0,
        seed=1729,
    )
    settings.ensure_dirs()

    connection = repo.init_db(settings.db_path)
    rng = np.random.default_rng(settings.seed)
    ids = [f"{n:016x}" for n in range(image_count)]

    from PIL import Image

    for index, image_id in enumerate(ids):
        repo.upsert_image(
            connection,
            image_id=image_id,
            source=["megalith", "pd12m", "open_images", "unsplash_lite"][index % 4],
            source_id=f"e2e{index}",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            orig_path=f"e2e/{image_id}.jpg",
            source_url=f"https://example.invalid/photo/{index}",
            author=f"Test Photographer {index}",
            author_url="https://example.invalid/author",
            title=f"Test image {index}",
            width=800,
            height=600,
            phash=f"{index:016x}",
            dominant_colours=["#334455", "#99aabb"],
        )
        connection.execute(
            "UPDATE images SET selected = 1, cluster64 = ?, cluster400 = ? WHERE image_id = ?",
            (index % 8, index, image_id),
        )

        # A distinguishable gradient per image, so a human watching the trace
        # can see the card actually change.
        gradient = np.zeros((300, 400, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.linspace(0, 255, 400, dtype=np.uint8)[None, :]
        gradient[:, :, 1] = (index * 37) % 255
        gradient[:, :, 2] = np.linspace(255, 0, 300, dtype=np.uint8)[:, None]
        picture = Image.fromarray(gradient, mode="RGB")
        for path in (settings.display_path(image_id), settings.thumb_path(image_id)):
            path.parent.mkdir(parents=True, exist_ok=True)
            picture.save(path, format="WEBP", quality=80)

    connection.commit()
    connection.close()

    for model_key in ("siglip2_b16_224", "dinov2_b14"):
        vectors = rng.normal(size=(image_count, EMBEDDING_DIM)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store = EmbeddingStore(settings.embedding_dir(model_key), model_key)
        store.write_shard(0, ids, vectors)
        store.merge()
        store.write_meta(
            checkpoint=f"{model_key}/e2e",
            dim=EMBEDDING_DIM,
            count=image_count,
            preprocess="generated",
            device="cpu",
        )

    print(f"built an end-to-end corpus of {image_count} images at {root}")
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".e2e"))
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)

    build(args.root.resolve(), args.images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
