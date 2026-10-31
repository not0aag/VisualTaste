"""Nearest-neighbour search over the embeddings, via FAISS.

Intuition
---------
Three different features of this project are the same question underneath:
"which images are most similar to this one?" The feed's kNN rung (R2) asks it of
Alen's likes, `GET /images/{id}/similar` asks it of one image, and
near-duplicate detection asks it of everything at once.

The formula
-----------
Every vector was L2-normalised when it was created (`vision/encoders.py`), so

    cos(a, b) = a . b

and finding the nearest neighbours by cosine similarity is finding the largest
inner products. That is exactly what `faiss.IndexFlatIP` computes -- IP for
inner product.

Why "Flat" and not something cleverer
-------------------------------------
FAISS offers approximate indexes (IVF, HNSW, PQ) that trade a little accuracy
for a lot of speed on large corpora. This corpus is 50,000 x 768 float32 = 150
MB, and a brute-force search over it is one matrix multiply: measured at well
under 10 ms, inside the feed's 50 ms budget with room to spare.

So an exact index is the right choice here, and the approximate ones would be
premature optimisation that also makes results non-reproducible. If this corpus
ever reached millions, `IndexIVFFlat` would be the next step -- and the
interface in this module would not change.

A note on row order
-------------------
FAISS returns *row positions*, not image ids. The mapping from one to the other
is the `ids.npy` written beside the vectors, and it only holds because
`EmbeddingStore.merge` sorts rows by image_id. That is why that sort exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """A FAISS inner-product index plus the id mapping that makes it meaningful.

    Wraps FAISS rather than exposing it, for one reason: FAISS speaks in row
    positions and every caller here speaks in image ids. Keeping the translation
    in one place means a mismatch between the two cannot be introduced by a
    caller.
    """

    def __init__(self, vectors: np.ndarray, ids: np.ndarray | list[str]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = list(ids) if not isinstance(ids, np.ndarray) else ids.tolist()

        if len(vectors) != len(ids):
            raise ValueError(f"{len(vectors)} vectors but {len(ids)} ids; they must align")

        self.ids: list[str] = ids
        self.dim = int(vectors.shape[1]) if len(vectors) else 0
        self.vectors = vectors
        self._position_of = {image_id: i for i, image_id in enumerate(ids)}

        self.index = faiss.IndexFlatIP(self.dim)
        if len(vectors):
            self.index.add(vectors)

    def __len__(self) -> int:
        return len(self.ids)

    # --- Lookups -------------------------------------------------------------

    def position(self, image_id: str) -> int | None:
        """The row number for an image id, or None if it is not in this index."""
        return self._position_of.get(image_id)

    def vector_for(self, image_id: str) -> np.ndarray | None:
        position = self.position(image_id)
        return None if position is None else self.vectors[position]

    def vectors_for(self, image_ids: list[str]) -> np.ndarray:
        """The rows for these ids, skipping any that are absent.

        Absent ids are skipped rather than raising, because an image can be
        labelled and then quarantined by the safety screen; a profile that
        referenced it would otherwise break the feed.
        """
        positions = [
            self._position_of[image_id] for image_id in image_ids if image_id in self._position_of
        ]
        if not positions:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.vectors[positions]

    # --- Search --------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int, *, exclude: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """The `k` most similar images to one query vector, nearest first.

        `exclude` drops ids after the search rather than before it, so the
        search itself stays a single matrix multiply. To keep `k` results after
        exclusions, the search over-fetches -- which costs nothing measurable at
        this corpus size.
        """
        if len(self) == 0 or k <= 0:
            return []

        exclude = exclude or set()
        fetch = min(len(self), k + len(exclude))

        vector = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        similarities, positions = self.index.search(vector, fetch)

        results: list[tuple[str, float]] = []
        for position, similarity in zip(positions[0], similarities[0], strict=True):
            if position < 0:  # FAISS pads with -1 when it runs out of vectors
                continue
            image_id = self.ids[int(position)]
            if image_id in exclude:
                continue
            results.append((image_id, float(similarity)))
            if len(results) >= k:
                break
        return results

    def search_many(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Batched search. Returns `(similarities, positions)`, both (N, k).

        Positions rather than ids, because the callers that need batching
        (near-duplicate detection, kNN scoring) work in array space and would
        only convert the ids straight back.
        """
        if len(self) == 0:
            return np.zeros((len(queries), 0), np.float32), np.zeros((len(queries), 0), np.int64)
        matrix = np.ascontiguousarray(queries, dtype=np.float32)
        return self.index.search(matrix, min(k, len(self)))

    def similarity_to_all(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query against every row, in index order.

        For scoring a whole candidate pool, which is what the feed does on every
        request, this is more useful than a top-k search: one matrix multiply
        gives every score, and the ranking happens afterwards alongside MMR and
        the exploration quota.
        """
        vector = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        return self.vectors @ vector

    # --- Persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the FAISS index. The ids live in `ids.npy` beside the vectors.

        Saving the index at all is a convenience: rebuilding it from
        `vectors.npy` takes under a second for a flat index. It is kept because
        the API loads it at startup and a second of startup time is worth
        avoiding, not because it would be expensive to lose.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        faiss.write_index(self.index, str(tmp))
        tmp.replace(path)
        logger.info("wrote FAISS index with %d vectors to %s", len(self), path)

    @classmethod
    def load(cls, index_path: Path, vectors: np.ndarray, ids: np.ndarray) -> VectorIndex:
        """Load a saved index, falling back to rebuilding it if it is unusable.

        A flat index takes well under a second to rebuild, so a corrupt or
        stale file is an inconvenience rather than a problem -- and silently
        rebuilding is much better than refusing to start the API.
        """
        built = cls(vectors, ids)
        if not index_path.exists():
            return built

        try:
            loaded = faiss.read_index(str(index_path))
        except (RuntimeError, OSError) as exc:
            logger.warning("rebuilding unreadable index %s (%s)", index_path.name, exc)
            return built

        if loaded.ntotal != len(ids) or loaded.d != built.dim:
            logger.warning(
                "index %s holds %d x %d vectors but the store has %d x %d; rebuilding",
                index_path.name,
                loaded.ntotal,
                loaded.d,
                len(ids),
                built.dim,
            )
            return built

        built.index = loaded
        return built


def build_from_store(store, index_path: Path) -> VectorIndex:
    """Build and save an index from a merged `EmbeddingStore`."""
    vectors, ids = store.load()
    index = VectorIndex(vectors, ids)
    index.save(index_path)
    return index
