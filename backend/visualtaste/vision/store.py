"""Where embeddings live on disk, and how a four-hour job survives a reboot.

The layout (brief §6.2)
-----------------------
    <DATA>/embeddings/<model_key>/
        shards/shard_00000.npz     float16 vectors + their ids, 2,048 rows each
        vectors.npy                float32, every row, merged
        ids.npy                    the image_id for each row of vectors.npy
        meta.json                  what produced these numbers

Why shards, and why float16 in them
-----------------------------------
Embedding 80,000 images twice is the longest single job in this project. Writing
one big array at the end would mean a crash at 95% costs the whole run. So
vectors are appended in blocks of 2,048: each block is written atomically to its
own file, and a restart reads which ids are already covered and embeds only the
rest.

The shards are float16 because that is the precision the GPU produced them at,
so storing float32 there would double the disk for digits that are not real.
The *merged* file is float32, because that is what FAISS, scikit-learn and
PyTorch all want, and converting 50,000 x 768 on every load is wasted time.

How much space: 80,000 x 768 x 4 bytes = 245 MB per model merged, 123 MB in
shards. Small enough that keeping both is not worth optimising away.

Why `meta.json` matters more than it looks
------------------------------------------
Six months from now the only way to answer "were these vectors made by the same
model as those ones?" is a file that says so. It records the checkpoint, the
dimension, the preprocessing, the torch and CUDA versions, and the date. A
mismatch there explains an otherwise baffling drop in recommendation quality.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Rows per shard, per brief §7.3. At 768 dimensions in float16 that is a 3 MB
# file: small enough to write often, large enough that the file count stays in
# the dozens rather than the thousands.
SHARD_ROWS = 2_048

SHARD_PREFIX = "shard_"

# Width of the fixed-length string dtype used for id arrays. Matches the
# 16-hex-character `image_id` defined in §6.2.
IMAGE_ID_WIDTH = 16


@dataclass
class EmbeddingMeta:
    """Provenance for one `(model_key, corpus)` pair of embedding files."""

    model_key: str
    checkpoint: str
    dim: int
    count: int
    preprocess: str
    torch_version: str
    cuda_version: str | None
    device: str
    created_at: str
    normalised: bool = True
    shard_dtype: str = "float16"
    merged_dtype: str = "float32"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read(cls, path: Path) -> EmbeddingMeta:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class EmbeddingStore:
    """Read and write the shard set and merged files for one `model_key`."""

    def __init__(self, root: Path, model_key: str) -> None:
        self.model_key = model_key
        self.dir = root
        self.shards_dir = root / "shards"
        self.vectors_path = root / "vectors.npy"
        self.ids_path = root / "ids.npy"
        self.meta_path = root / "meta.json"

    # --- Writing -------------------------------------------------------------

    def write_shard(self, index: int, ids: list[str], vectors: np.ndarray) -> Path:
        """Write one shard atomically. `vectors` is (N, D) and already normalised.

        Stored as a single `.npz` rather than two `.npy` files so that the ids
        and the vectors cannot be separated: a shard whose id list and vector
        block disagreed would attach Alen's labels to the wrong pictures, which
        is the worst failure this system has available to it.
        """
        if len(ids) != len(vectors):
            raise ValueError(f"shard {index}: {len(ids)} ids but {len(vectors)} vectors")

        # The id arrays are `<U16`, matching the 16-hex-character image_id from
        # §6.2. A longer string would be truncated to fit *silently*, and two
        # ids sharing a 16-character prefix would then collapse into one row --
        # a data-corrupting failure with no error message.
        too_long = [image_id for image_id in ids if len(image_id) > IMAGE_ID_WIDTH]
        if too_long:
            raise ValueError(
                f"shard {index}: {len(too_long)} id(s) longer than "
                f"{IMAGE_ID_WIDTH} characters would be truncated, e.g. {too_long[0]!r}"
            )

        self.shards_dir.mkdir(parents=True, exist_ok=True)
        path = self.shards_dir / f"{SHARD_PREFIX}{index:05d}.npz"
        tmp = path.with_name(f".{path.name}.part")

        with tmp.open("wb") as handle:
            np.savez(
                handle,
                vectors=vectors.astype(np.float16),
                ids=np.array(ids, dtype=f"<U{IMAGE_ID_WIDTH}"),
            )
        tmp.replace(path)
        return path

    def next_shard_index(self) -> int:
        """One past the highest shard already written."""
        existing = self.shard_paths()
        if not existing:
            return 0
        return max(int(p.stem.removeprefix(SHARD_PREFIX)) for p in existing) + 1

    # --- Reading -------------------------------------------------------------

    def shard_paths(self) -> list[Path]:
        if not self.shards_dir.is_dir():
            return []
        return sorted(self.shards_dir.glob(f"{SHARD_PREFIX}*.npz"))

    def embedded_ids(self) -> set[str]:
        """Every image_id already covered, across shards and the merged file.

        This is the incremental contract: the job embeds the set difference
        between what the database wants and what this returns. A partially
        written shard is ignored (it is a `.part` file until it is complete), so
        the answer is always consistent with what is on disk.
        """
        covered: set[str] = set()

        if self.ids_path.exists():
            covered.update(np.load(self.ids_path, allow_pickle=False).tolist())

        for path in self.shard_paths():
            try:
                with np.load(path, allow_pickle=False) as shard:
                    covered.update(shard["ids"].tolist())
            except (OSError, ValueError, KeyError) as exc:
                # A shard truncated by a power cut mid-write. Logged rather than
                # fatal: the ids it held are simply re-embedded.
                logger.warning("ignoring unreadable shard %s (%s)", path.name, exc)

        return covered

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """The merged `(vectors, ids)`. Raises if `merge` has not been run."""
        if not self.vectors_path.exists() or not self.ids_path.exists():
            raise FileNotFoundError(
                f"no merged embeddings for {self.model_key} at {self.dir}. "
                f"Run `python -m visualtaste.jobs.embed --model {self.model_key}`."
            )
        vectors = np.load(self.vectors_path)
        ids = np.load(self.ids_path, allow_pickle=False)
        if len(vectors) != len(ids):
            raise ValueError(
                f"{self.model_key}: {len(vectors)} vectors but {len(ids)} ids; "
                f"the merged files disagree and must be rebuilt"
            )
        return vectors, ids

    def load_as_dict(self) -> dict[str, np.ndarray]:
        vectors, ids = self.load()
        return dict(zip(ids.tolist(), vectors, strict=True))

    # --- Merging -------------------------------------------------------------

    def merge(self, *, keep_shards: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Combine every shard (and any previous merge) into the final files.

        Rows come out sorted by `image_id`, which matters for two reasons: the
        output is then deterministic for a given corpus, so a checksum means
        something; and FAISS row order becomes stable, so an index built today
        and a `cluster64` assignment computed tomorrow refer to the same rows.
        """
        collected: dict[str, np.ndarray] = {}

        if self.vectors_path.exists() and self.ids_path.exists():
            previous_vectors, previous_ids = self.load()
            collected.update(zip(previous_ids.tolist(), previous_vectors, strict=True))

        for path in self.shard_paths():
            try:
                with np.load(path, allow_pickle=False) as shard:
                    shard_ids = shard["ids"].tolist()
                    shard_vectors = shard["vectors"]
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("skipping unreadable shard %s (%s)", path.name, exc)
                continue
            collected.update(zip(shard_ids, shard_vectors, strict=True))

        if not collected:
            raise RuntimeError(f"nothing to merge for {self.model_key} in {self.dir}")

        ordered_ids = sorted(collected)
        matrix = np.stack([collected[image_id] for image_id in ordered_ids]).astype(np.float32)

        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_vectors = self.vectors_path.with_name(f".{self.vectors_path.name}.part")
        tmp_ids = self.ids_path.with_name(f".{self.ids_path.name}.part")
        # Through a handle, because `np.save` appends ".npy" to any path that
        # does not already end in it and would leave the replace pointing at a
        # file that was never written.
        with tmp_vectors.open("wb") as handle:
            np.save(handle, matrix)
        with tmp_ids.open("wb") as handle:
            np.save(handle, np.array(ordered_ids, dtype=f"<U{IMAGE_ID_WIDTH}"))
        tmp_vectors.replace(self.vectors_path)
        tmp_ids.replace(self.ids_path)

        if not keep_shards:
            for path in self.shard_paths():
                path.unlink(missing_ok=True)

        logger.info(
            "%s: merged %d vectors x %d dims -> %s",
            self.model_key,
            matrix.shape[0],
            matrix.shape[1],
            self.vectors_path,
        )
        return matrix, np.array(ordered_ids, dtype=f"<U{IMAGE_ID_WIDTH}")

    # --- Metadata ------------------------------------------------------------

    def write_meta(
        self,
        *,
        checkpoint: str,
        dim: int,
        count: int,
        preprocess: str,
        device: str,
    ) -> EmbeddingMeta:
        import torch

        meta = EmbeddingMeta(
            model_key=self.model_key,
            checkpoint=checkpoint,
            dim=dim,
            count=count,
            preprocess=preprocess,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            device=device,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        meta.write(self.meta_path)
        return meta

    def read_meta(self) -> EmbeddingMeta | None:
        if not self.meta_path.exists():
            return None
        return EmbeddingMeta.read(self.meta_path)


def verify_normalised(vectors: np.ndarray, *, tolerance: float = 1e-3) -> None:
    """Raise unless every row has unit length, within `tolerance`.

    The Phase 4 Definition of Done asks for this check, and it is worth having
    as a guard rather than only a test: every similarity in the system is a dot
    product that silently assumes unit length, so a batch that escaped
    normalisation would not crash -- it would just quietly rank badly.
    """
    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.max(np.abs(norms - 1.0))) if len(norms) else 0.0
    if worst > tolerance:
        raise ValueError(
            f"embeddings are not unit length: worst deviation {worst:.2e} exceeds {tolerance:.0e}"
        )
