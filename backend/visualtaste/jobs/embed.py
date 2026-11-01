"""Embed the corpus with one of the frozen encoders.

Run:  python -m visualtaste.jobs.embed --model siglip2_b16_224
      python -m visualtaste.jobs.embed --model dinov2_b14
      python -m visualtaste.jobs.embed --model all --batch-size 128

What this job is doing, in one sentence: turning 80,000 WebP files into an
80,000 x 768 matrix of unit vectors, twice, without ever having to start over.

Where the time goes
-------------------
The Phase 1 benchmark measured both encoders at 320-440 images/second on
synthetic tensors -- about four minutes of pure GPU time for 80,000 images. The
GPU is therefore *not* the bottleneck; reading and decoding 80,000 WebP files
is. That single fact decides the whole design:

- a `DataLoader` with worker processes does the decoding, so the GPU is fed
  rather than waiting on `Image.open`;
- the batch size comes from the measured benchmark, not from a guess;
- `pin_memory` overlaps the host-to-device copy with the next batch's decode.

The incremental contract
------------------------
The job embeds exactly the set difference between "images the database says are
usable" and "ids already in the store". So:

- killing it and re-running costs at most one shard of 2,048 rows;
- adding ten new images to the corpus embeds ten images, not 80,010;
- running it twice in a row does nothing the second time.

A single image that fails to decode is recorded in `embedding_errors` and the
batch continues. One corrupt file four hours into a run must not cost the run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from visualtaste.config import Settings, check_disk_headroom, get_settings
from visualtaste.db import repo
from visualtaste.vision.encoders import (
    ENCODER_SPECS,
    ImageEncoder,
    load_encoder,
    pick_device,
)
from visualtaste.vision.store import SHARD_ROWS, EmbeddingStore, verify_normalised

logger = logging.getLogger(__name__)

# Measured on this machine (docs/BENCHMARKS.md, 2026-10-30): throughput is flat
# from batch 32 upwards and peak VRAM at 256 is 2.8 GB of the 6 GB available,
# so these are comfortable rather than tight.
DEFAULT_BATCH_SIZES = {
    "siglip2_b16_224": 128,
    "dinov2_b14": 256,
}
#: On CPU a large batch only costs memory: there is no parallel hardware to fill.
CPU_BATCH_SIZE = 16

# DataLoader worker processes. These decode WebP files, which is the real
# bottleneck, so there are more of them than there are spare cores.
DEFAULT_LOADER_WORKERS = 6

LOG_EVERY_BATCHES = 20


@dataclass
class EmbedStats:
    """What one embedding run did."""

    model_key: str
    wanted: int = 0
    already_done: int = 0
    embedded: int = 0
    failed: int = 0
    seconds: float = 0.0

    @property
    def images_per_second(self) -> float:
        return self.embedded / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.model_key}: wanted={self.wanted} already_done={self.already_done} "
            f"embedded={self.embedded} failed={self.failed} "
            f"({self.images_per_second:.0f} img/s)"
        )


class DisplayImageDataset(Dataset):
    """Reads display WebP files and applies the encoder's own preprocessing.

    A failed decode returns a zero tensor flagged by `ok=False` rather than
    raising: an exception inside a DataLoader worker tears down the whole
    loader, which would mean one unreadable file ends a four-hour job.
    """

    def __init__(self, image_ids: list[str], settings: Settings, encoder: ImageEncoder):
        self.image_ids = image_ids
        self.settings = settings
        self.preprocess = encoder.preprocess
        self.image_size = encoder.spec.image_size

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, bool]:
        from PIL import Image

        image_id = self.image_ids[index]
        path = self.settings.display_path(image_id)
        try:
            with Image.open(path) as image:
                image.load()
                tensor = self.preprocess(image.convert("RGB"))
            return tensor, index, True
        except Exception:
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, index, False


def _worker_init(_worker_id: int) -> None:
    """One thread per DataLoader worker, for the same reason as in ingestion.

    Pillow and NumPy would each try to use every core, and with six workers the
    machine would spend its time context-switching instead of decoding.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def ids_to_embed(
    connection: sqlite3.Connection, store: EmbeddingStore, *, limit: int | None = None
) -> tuple[list[str], int]:
    """The usable images not yet embedded for this model. Returns `(ids, done)`.

    "Usable" is `status = 'ok'`: rejected, quarantined and near-duplicate images
    are not embedded, because nothing will ever score them. Sorted so a run is
    reproducible and a resumed run picks up in the same order.
    """
    wanted = repo.image_ids_with_status(connection, "ok")
    done = store.embedded_ids()
    todo = [image_id for image_id in wanted if image_id not in done]
    if limit is not None:
        todo = todo[:limit]
    return todo, len(done)


def embed_model(
    settings: Settings,
    connection: sqlite3.Connection,
    model_key: str,
    *,
    batch_size: int | None = None,
    loader_workers: int = DEFAULT_LOADER_WORKERS,
    limit: int | None = None,
    merge: bool = True,
) -> EmbedStats:
    """Embed every outstanding image for one model, writing shards as it goes."""
    import time

    spec = ENCODER_SPECS[model_key]
    store = EmbeddingStore(settings.embedding_dir(model_key), model_key)
    todo, already = ids_to_embed(connection, store, limit=limit)
    stats = EmbedStats(model_key=model_key, wanted=len(todo), already_done=already)

    if not todo:
        logger.info("%s: nothing to embed (%d already done)", model_key, already)
        if merge and store.shard_paths():
            store.merge()
        return stats

    settings.apply_hf_env()
    device = pick_device()
    default_batch = (
        DEFAULT_BATCH_SIZES.get(model_key, 64) if device.type == "cuda" else CPU_BATCH_SIZE
    )
    batch = batch_size or default_batch

    logger.info(
        "%s: embedding %d images (%d already done) on %s, batch %d",
        model_key,
        len(todo),
        already,
        device,
        batch,
    )
    encoder = load_encoder(model_key, device=device)
    repo.register_embedding_model(
        connection,
        model_key=model_key,
        family=spec.family,
        checkpoint=spec.checkpoint,
        dim=spec.dim,
        preprocess=f"{spec.family} native, {spec.image_size}px",
    )
    connection.commit()

    dataset = DisplayImageDataset(todo, settings, encoder)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init if loader_workers > 0 else None,
        # Workers are expensive to spawn on Windows; keeping them alive across
        # the whole run avoids paying for it once per batch.
        persistent_workers=loader_workers > 0,
        prefetch_factor=4 if loader_workers > 0 else None,
    )

    shard_index = store.next_shard_index()
    buffer_ids: list[str] = []
    buffer_vectors: list[np.ndarray] = []
    started = time.perf_counter()

    for batch_number, (tensors, indices, ok_flags) in enumerate(loader):
        usable = ok_flags.numpy()
        index_list = indices.numpy()

        # Record the failures, then drop them from the batch rather than
        # embedding a zero tensor and storing a meaningless vector.
        for position in np.flatnonzero(~usable):
            image_id = todo[int(index_list[position])]
            repo.record_embedding_error(
                connection, image_id, model_key, "could not decode display derivative"
            )
            stats.failed += 1

        if not usable.any():
            connection.commit()
            continue

        keep = torch.from_numpy(np.flatnonzero(usable))
        vectors = encoder.encode_images(tensors[keep])
        buffer_vectors.append(vectors)
        buffer_ids.extend(todo[int(i)] for i in index_list[usable])
        stats.embedded += len(vectors)

        # Flush whenever a full shard has accumulated. This is the unit of work
        # a crash can cost.
        while len(buffer_ids) >= SHARD_ROWS:
            stacked = np.concatenate(buffer_vectors)
            store.write_shard(shard_index, buffer_ids[:SHARD_ROWS], stacked[:SHARD_ROWS])
            shard_index += 1
            buffer_ids = buffer_ids[SHARD_ROWS:]
            remaining = stacked[SHARD_ROWS:]
            buffer_vectors = [remaining] if len(remaining) else []
            connection.commit()
            check_disk_headroom(settings.data, settings.min_free_gb)

        if batch_number % LOG_EVERY_BATCHES == 0:
            elapsed = time.perf_counter() - started
            rate = stats.embedded / elapsed if elapsed else 0
            done_now = already + stats.embedded
            logger.info(
                "%s: %d/%d this run (%d total), %.0f img/s",
                model_key,
                stats.embedded,
                len(todo),
                done_now,
                rate,
            )

    # Whatever is left over is a short final shard.
    if buffer_ids:
        store.write_shard(shard_index, buffer_ids, np.concatenate(buffer_vectors))

    stats.seconds = time.perf_counter() - started
    connection.commit()
    logger.info("%s", stats.summary())

    if merge:
        vectors, ids = store.merge()
        verify_normalised(vectors)
        store.write_meta(
            checkpoint=spec.checkpoint,
            dim=spec.dim,
            count=len(ids),
            preprocess=f"{spec.family} native, {spec.image_size}px",
            device=str(device),
        )
        logger.info("%s: %d vectors merged and verified unit-length", model_key, len(ids))

    return stats


def render_summary(all_stats: list[EmbedStats], settings: Settings) -> str:
    """A Markdown block for docs/BENCHMARKS.md: real-data embedding throughput."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## Real-data embedding throughput ({now})",
        "",
        "- Command: `python -m visualtaste.jobs.embed --model all`",
        "- Includes WebP decode, preprocessing, host-to-device copy and the "
        "forward pass -- unlike the synthetic benchmark above, which times the "
        "forward pass alone.",
        "",
        "| model | embedded | failed | images/s | batch | total on disk |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for stats in all_stats:
        store = EmbeddingStore(settings.embedding_dir(stats.model_key), stats.model_key)
        meta = store.read_meta()
        total = meta.count if meta else 0
        batch = DEFAULT_BATCH_SIZES.get(stats.model_key, 64)
        lines.append(
            f"| `{stats.model_key}` | {stats.embedded} | {stats.failed} "
            f"| {stats.images_per_second:.0f} | {batch} | {total} |"
        )
    lines.append("")

    synthetic = {"siglip2_b16_224": 438, "dinov2_b14": 322}
    for stats in all_stats:
        if stats.images_per_second > 0 and stats.model_key in synthetic:
            ratio = stats.images_per_second / synthetic[stats.model_key]
            lines.append(
                f"- `{stats.model_key}`: {ratio:.0%} of the synthetic-tensor rate "
                f"({synthetic[stats.model_key]} img/s), so the remaining "
                f"{1 - ratio:.0%} is decoding and IO."
            )
    lines.append("")
    return "\n".join(lines)


def _configure_logging(settings: Settings, verbose: bool) -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / f"embed_{datetime.now(UTC):%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
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
    parser.add_argument(
        "--model",
        default="all",
        choices=[*sorted(ENCODER_SPECS), "all"],
        help="model_key to embed, or 'all' (default)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--loader-workers", type=int, default=DEFAULT_LOADER_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="embed at most N images")
    parser.add_argument(
        "--no-merge", action="store_true", help="leave shards unmerged (for a partial run)"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    log_path = _configure_logging(settings, args.verbose)

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; this will run on the CPU and be roughly 30x slower.")

    connection = repo.init_db(settings.db_path)
    models = sorted(ENCODER_SPECS) if args.model == "all" else [args.model]

    logger.info("log file: %s", log_path)
    all_stats = [
        embed_model(
            settings,
            connection,
            model_key,
            batch_size=args.batch_size,
            loader_workers=args.loader_workers,
            limit=args.limit,
            merge=not args.no_merge,
        )
        for model_key in models
    ]

    if not args.no_merge:
        summary = render_summary(all_stats, settings)
        with (settings.docs_dir / "BENCHMARKS.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + summary)
        print("\n" + summary)

    connection.close()
    return 0 if all(s.failed == 0 for s in all_stats) else 1


if __name__ == "__main__":
    # DataLoader workers are spawned on Windows, so the entry point must be
    # guarded or each worker would start its own embedding run.
    raise SystemExit(main())
