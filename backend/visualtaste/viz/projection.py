"""Squashing 768 dimensions into 2, for the Embedding Map (maths unit M4).

Intuition
---------
Nobody can look at a 768-dimensional space. The Embedding Map needs every
image as a point on a screen, placed so that images the model considers
similar land near each other. Two steps do that:

1. **PCA to 50 dimensions.** PCA finds the directions along which the 50,000
   vectors vary most and keeps the top 50. That discards mostly noise, and it
   makes step 2 about 15 times faster because UMAP's nearest-neighbour search
   now runs in 50 dimensions instead of 768.
2. **UMAP to 2.** UMAP builds a graph linking each image to its 30 nearest
   neighbours (by cosine), then lays the graph out in the plane so that
   neighbours stay close. `min_dist` = 0.1 sets how tightly points may pack:
   small values give crisp clumps, large values a more even spread.

What the picture can and cannot say
-----------------------------------
UMAP keeps *local* structure: points close on the map are close in the
embedding. It does not keep global structure. Two clumps far apart on the map
are different, but their distance means little, and the map's axes mean
nothing. "The taste moved left" is not a finding; "the preference vector now
sits inside the cluster of botanical prints" is.

Seeded (random_state), so the same corpus always draws the same map. That
costs UMAP's multithreading, and the run time is recorded in BENCHMARKS.md.

Taste over time
---------------
`Projection.transform` places *new* points -- the preference vector after N
labels, the interest centroids -- onto the existing map with
`umap.transform`, without moving the 50,000. That is the time slider.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PCA_COMPONENTS = 50
N_NEIGHBORS = 30
MIN_DIST = 0.1
METRIC = "cosine"


@dataclass
class Projection:
    """A fitted PCA-50 -> UMAP-2 projection and the map it drew."""

    model_key: str
    image_ids: list[str]
    coords: np.ndarray  # (n, 2) float32
    pca: Any
    umap: Any
    seconds: float = 0.0

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Place new vectors on the existing map (the taste-over-time slider)."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return np.asarray(self.umap.transform(self.pca.transform(vectors)), dtype=np.float32)


def fit_projection(
    vectors: np.ndarray, image_ids: list[str], *, model_key: str, seed: int = 1729
) -> Projection:
    """PCA-50, then UMAP (cosine, 30 neighbours, min_dist 0.1), seeded."""
    import umap
    from sklearn.decomposition import PCA

    started = time.perf_counter()
    vectors = np.asarray(vectors, dtype=np.float32)
    pca = PCA(n_components=min(PCA_COMPONENTS, vectors.shape[1], len(vectors)), random_state=seed)
    reduced = pca.fit_transform(vectors)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(N_NEIGHBORS, len(vectors) - 1),
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=seed,
    )
    coords = reducer.fit_transform(reduced).astype(np.float32)
    seconds = time.perf_counter() - started
    logger.info(
        "%s: projected %d points in %.1fs (PCA-%d keeps %.1f%% of the variance)",
        model_key,
        len(vectors),
        seconds,
        pca.n_components_,
        100 * pca.explained_variance_ratio_.sum(),
    )
    return Projection(model_key, list(image_ids), coords, pca, reducer, seconds)


def paths(features_dir: Path, model_key: str) -> tuple[Path, Path, Path]:
    return (
        features_dir / f"umap2d_{model_key}.npy",
        features_dir / f"umap2d_{model_key}_ids.npy",
        features_dir / f"umap2d_{model_key}.joblib",
    )


def save(projection: Projection, features_dir: Path) -> None:
    import joblib

    coords_path, ids_path, model_path = paths(features_dir, projection.model_key)
    for path, array in (
        (coords_path, projection.coords),
        (ids_path, np.array(projection.image_ids, dtype="<U16")),
    ):
        tmp = path.with_name(f".{path.name}.part")
        with tmp.open("wb") as handle:
            np.save(handle, array)
        tmp.replace(path)
    tmp = model_path.with_name(f".{model_path.name}.part")
    joblib.dump({"pca": projection.pca, "umap": projection.umap}, tmp)
    tmp.replace(model_path)


def load_coords(features_dir: Path, model_key: str) -> tuple[list[str], np.ndarray] | None:
    """Just the map (ids and 2-D coordinates), which is all the API serves."""
    coords_path, ids_path, _ = paths(features_dir, model_key)
    if not (coords_path.exists() and ids_path.exists()):
        return None
    return np.load(ids_path).tolist(), np.load(coords_path)


def load(features_dir: Path, model_key: str) -> Projection | None:
    """The full projection, including the models needed for `transform`."""
    import joblib

    coords = load_coords(features_dir, model_key)
    _, _, model_path = paths(features_dir, model_key)
    if coords is None or not model_path.exists():
        return None
    models = joblib.load(model_path)
    return Projection(model_key, coords[0], coords[1], models["pca"], models["umap"])
