"""PD12M: 25,000 public-domain images from museums, archives and iNaturalist.

Public Domain 12M is a 12.4-million-image dataset assembled by Spawning. The
metadata lives in 125 parquet files on Hugging Face (`Spawning/PD12M`, metadata
licensed CDLA-Permissive-2.0); the images themselves sit in a public S3 bucket
and are fetched over plain unsigned HTTPS.

This is the richest of the four sources, metadata-wise, and the only one that
publishes a genuine per-image licence URL (CC0 or the Public Domain Mark), an
MD5, dimensions and the originating institution. We use all four:

- the **licence URL** satisfies the §7.1 gate directly;
- the **MD5** is verified on arrival, so a corrupted transfer never enters a
  write-once archive;
- the **dimensions** let us reject a 6000x200 panorama before downloading it;
- the **institution** drives the stratified sample described below.

Why stratify
------------
The raw distribution is lopsided: in the first metadata shard, Wikimedia
Commons alone is 57% of rows and iNaturalist another 20%. Sampling uniformly
would hand the recommender a corpus that is mostly bird photographs and
Wikimedia miscellany, and "Alen's taste" would partly be an artefact of that.

So instead of a uniform sample we round-robin across institutions: take one
image from each institution in turn, in a seeded random order within each, until
the target is met. Small museums are therefore fully represented and the giants
only fill what is left over. The §7.1 instruction is "so no single institution
dominates", and round-robin achieves it without a hand-tuned cap.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from visualtaste.data.manifest import Candidate
from visualtaste.data.sources import (
    dimensions_acceptable,
    download_http_candidates,
    needs_planning,
)

if TYPE_CHECKING:
    from visualtaste.config import Settings
    from visualtaste.data.http import FetchStats, PoliteClient
    from visualtaste.data.manifest import Manifest

logger = logging.getLogger(__name__)

SOURCE = "pd12m"
DEFAULT_TARGET = 25_000
HF_REPO = "Spawning/PD12M"

# Each metadata shard holds 100,000 rows. Roughly 5% of PD12M images exceed the
# 8 MB cap (measured: 60 HEAD requests, median 1.06 MB, see DECISIONS.md #2),
# and a few percent more fail or fall outside the shape rules, so we plan from a
# pool comfortably larger than the target and stop as soon as it is met.
METADATA_SHARDS = 4

# Containers Pillow handles well and that are worth having. TIFF and GIF are
# excluded: a handful of rows, and both are usually scans or animations that the
# ingestion rules would drop anyway.
WANTED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

# A small number of PD12M rows name no institution. They are public domain and
# perfectly usable, so they get their own stratification bucket rather than
# being dropped or silently labelled "nan".
UNKNOWN_INSTITUTION = "(institution not recorded)"


def _text(value: object) -> str | None:
    """Normalise a pandas cell to `str | None`.

    Needed because pandas 3 keeps missing values as NaN even through
    `.astype(str)`, and `bool(float("nan"))` is `True`. A plain
    `str(row["hash"]) if row.get("hash")` therefore yields the string "nan",
    which as an expected MD5 would make every single image fail verification.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _number(value: object) -> int | None:
    """Normalise a pandas numeric cell to `int | None`, NaN included."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def plan(settings: Settings, target: int = DEFAULT_TARGET) -> list[Candidate]:
    """Choose which PD12M images to fetch. Deterministic for a given seed.

    Returns more candidates than `target` on purpose: the download loop stops at
    the target, and the surplus absorbs rejections without a second planning pass.
    """
    frame = _load_metadata(settings)
    frame = _apply_gates(frame)
    logger.info("pd12m: %d candidates survive the metadata gates", len(frame))

    chosen = _round_robin_by_institution(frame, target=target, seed=settings.seed)
    logger.info(
        "pd12m: planned %d candidates across %d institutions (largest share %.1f%%)",
        len(chosen),
        chosen["source"].nunique(),
        100 * chosen["source"].value_counts(normalize=True, dropna=False).iloc[0],
    )

    return [
        Candidate(
            source=SOURCE,
            source_id=str(row["id"]),
            url=str(row["url"]),
            license_url=str(row["license"]),
            # PD12M has no per-image landing page; the institution is the
            # provenance, so it is recorded as the author for attribution.
            author=_text(row.get("source")),
            title=_short_caption(row.get("caption")),
            expected_md5=_text(row.get("hash")),
            declared_width=_number(row.get("width")),
            declared_height=_number(row.get("height")),
        )
        for row in chosen.to_dict("records")
    ]


def _load_metadata(settings: Settings) -> pd.DataFrame:
    """Download and concatenate the first `METADATA_SHARDS` metadata parquets.

    `hf_hub_download` caches inside `<DATA>/cache/hf` and resumes partial
    transfers, so an interrupted plan costs nothing to redo.
    """
    from huggingface_hub import hf_hub_download

    settings.apply_hf_env()
    frames = []
    for shard in range(METADATA_SHARDS):
        path = hf_hub_download(
            HF_REPO,
            f"metadata/pd12m.{shard:03d}.parquet",
            repo_type="dataset",
        )
        frames.append(
            pd.read_parquet(
                path,
                columns=["id", "url", "caption", "width", "height", "mime_type", "hash",
                         "license", "source"],
            )
        )
    return pd.concat(frames, ignore_index=True)


def _apply_gates(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows we would not keep anyway, cheaply, before any download."""
    before = len(frame)

    # The §7.1 licence gate. A few dozen rows per shard have a null licence.
    frame = frame[frame["license"].notna() & (frame["license"].astype(str) != "")]
    no_licence = before - len(frame)

    frame = frame[frame["mime_type"].isin(WANTED_MIME_TYPES)]
    frame = frame[
        [
            dimensions_acceptable(w, h)
            for w, h in zip(frame["width"], frame["height"], strict=True)
        ]
    ]

    # The same picture can appear under two institutions; keep one row per URL.
    frame = frame.drop_duplicates(subset=["url"])

    logger.info(
        "pd12m: %d rows -> %d after gates (%d had no licence URL)",
        before,
        len(frame),
        no_licence,
    )
    return frame.reset_index(drop=True)


def _round_robin_by_institution(
    frame: pd.DataFrame, *, target: int, seed: int
) -> pd.DataFrame:
    """Take images one institution at a time, so no single archive dominates.

    The surplus factor is deliberate: we plan about 1.3x the target so that
    rejected oversized scans and dead URLs do not leave us short, and the
    download loop simply stops early when the target is met.
    """
    rng = np.random.default_rng(seed)
    wanted = int(target * 1.3)

    # A seeded shuffle within each institution, so the choice is random but
    # reproducible. Institutions are visited in a fixed (sorted) order.
    by_institution: dict[str, list[int]] = defaultdict(list)
    for position, institution in enumerate(frame["source"]):
        by_institution[_text(institution) or UNKNOWN_INSTITUTION].append(position)
    for positions in by_institution.values():
        rng.shuffle(positions)

    queues = [by_institution[name] for name in sorted(by_institution)]
    picked: list[int] = []
    cursor = 0
    while len(picked) < wanted and any(queues):
        progressed = False
        for queue in queues:
            if cursor < len(queue):
                picked.append(queue[cursor])
                progressed = True
                if len(picked) >= wanted:
                    break
        if not progressed:
            break  # Every institution is exhausted.
        cursor += 1

    return frame.iloc[picked].reset_index(drop=True)


def _short_caption(caption: object) -> str | None:
    """PD12M captions are machine-generated paragraphs; keep a usable title.

    They are descriptive rather than authored ("The image shows a silver
    coin..."), so they are stored as a title for the UI but never used as a
    label or a feature.
    """
    if caption is None or not isinstance(caption, str) or not caption.strip():
        return None
    text = caption.strip()
    return text if len(text) <= 200 else text[:197] + "..."


async def fetch(
    settings: Settings,
    manifest: Manifest,
    target: int = DEFAULT_TARGET,
    *,
    client: PoliteClient,
) -> FetchStats:
    """Plan (if needed) and download PD12M images into the archive."""
    if needs_planning(manifest, target):
        added = manifest.add_all(plan(settings, target))
        logger.info("pd12m: added %d new candidates to the manifest", added)
        manifest.flush(force=True)

    return await download_http_candidates(
        settings, client, manifest, source=SOURCE, target=target
    )
