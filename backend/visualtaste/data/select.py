"""Choose 50,000 images from 80,000, so that the corpus is varied by construction.

The problem being solved
------------------------
A corpus is not just a pile of images; it is the space of things Alen can
possibly be shown, and therefore the space in which "his taste" can possibly be
expressed. If 40% of the corpus were bird photographs -- which is roughly what
iNaturalist plus Wikimedia would deliver unchecked -- then a recommender could
look like it had learned he likes birds when all it had learned is what the
corpus contains.

Downloading is already stratified by source and institution (`data/sources/`).
This is the second pass, done in embedding space rather than by metadata,
because two photographs can come from different institutions and still be the
same kind of picture.

The method (brief §7.2 step 9)
------------------------------
1. **k-means with k=400** over the SigLIP 2 embeddings of every usable
   candidate. Each cluster is a region of visual meaning -- "snowy mountains",
   "illuminated manuscripts", "studio portraits on white".
2. **Sample uniformly within each cluster**, capped at `1.5 x (50,000 / 400)` =
   187 images per cluster. The cap is what does the real work: without it, a
   cluster holding 8,000 near-identical bird photographs would contribute 8,000
   of the 50,000.
3. **Fill the remainder at random** from whatever is left, because the caps will
   usually leave the target short.

Why k=400 and why that cap
--------------------------
400 clusters over 80,000 images is an average of 200 images each, which is
roughly the granularity at which a cluster is still describable in words. The
1.5x multiplier allows genuinely large clusters to stay somewhat larger than
small ones -- the corpus really does contain more landscapes than illuminated
manuscripts, and flattening that completely would be its own distortion -- while
bounding how far the imbalance can go.

See maths unit M8 for k-means, and notebook 06.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

SELECTION_TARGET = 50_000
CLUSTER_COUNT = 400

# Brief §7.2 step 9: the per-cluster cap is this multiple of an equal share.
CLUSTER_CAP_MULTIPLIER = 1.5

# Clusters for the cold-start rung (R0), which walks them in round-robin until
# it has 30 labels. Far coarser than the 400 used for selection: R0 needs
# clusters a person would notice the difference between.
COLDSTART_CLUSTER_COUNT = 64

# Brief §7.2 step 9: if one source exceeds this share of the selection, say so.
SOURCE_DOMINANCE_WARNING = 0.50


@dataclass
class SelectionResult:
    """Which images were selected, and the counts needed to explain why."""

    selected: list[str]
    cluster_of: dict[str, int]
    per_cluster: dict[int, int] = field(default_factory=dict)
    per_source: dict[str, int] = field(default_factory=dict)
    capped_clusters: int = 0
    filled_at_random: int = 0

    @property
    def dominant_source_share(self) -> float:
        total = sum(self.per_source.values())
        if not total:
            return 0.0
        return max(self.per_source.values()) / total


def fit_clusters(
    vectors: np.ndarray, *, k: int, seed: int, sample_for_fit: int = 20_000
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster `vectors` into `k` groups. Returns `(labels, centroids)`.

    MiniBatchKMeans rather than the full algorithm: fitting 400 clusters over
    80,000 x 768 exactly would take many minutes and repeat it on every re-run,
    while the mini-batch version converges to a practically identical partition
    in seconds. The assignment step afterwards is exact for every image -- only
    the centroid *fitting* is approximate.

    Seeded, so the same corpus always produces the same clusters. That matters
    beyond reproducibility: `cluster64` is stored per image and read by the
    cold-start rung and the exploration quota, so a re-run that reshuffled the
    cluster numbering would silently invalidate the exploration statistics.
    """
    from sklearn.cluster import MiniBatchKMeans

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    k = min(k, len(vectors))
    if k < 1:
        raise ValueError("cannot cluster an empty set of vectors")

    rng = np.random.default_rng(seed)
    if len(vectors) > sample_for_fit:
        fit_rows = vectors[rng.choice(len(vectors), sample_for_fit, replace=False)]
    else:
        fit_rows = vectors

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=3,
        batch_size=4_096,
        max_iter=100,
    ).fit(fit_rows)

    labels = model.predict(vectors)
    logger.info(
        "k-means k=%d over %d vectors (fitted on %d): %d non-empty clusters",
        k,
        len(vectors),
        len(fit_rows),
        len(np.unique(labels)),
    )
    return labels.astype(np.int32), model.cluster_centers_.astype(np.float32)


def select_diverse(
    image_ids: list[str],
    labels: np.ndarray,
    sources: dict[str, str],
    *,
    target: int = SELECTION_TARGET,
    seed: int,
) -> SelectionResult:
    """Pick `target` images, capped per cluster, filling the rest at random.

    Deterministic for a given seed, so the selected corpus is reproducible.
    """
    if len(image_ids) != len(labels):
        raise ValueError(f"{len(image_ids)} ids but {len(labels)} labels")

    rng = np.random.default_rng(seed)
    cluster_of = {image_id: int(label) for image_id, label in zip(image_ids, labels, strict=True)}

    members: dict[int, list[str]] = {}
    for image_id, label in cluster_of.items():
        members.setdefault(label, []).append(image_id)

    cluster_count = len(members)
    cap = max(1, int(CLUSTER_CAP_MULTIPLIER * target / max(1, cluster_count)))

    selected: list[str] = []
    per_cluster: dict[int, int] = {}
    capped = 0

    # Clusters in numeric order, with a seeded shuffle inside each: the choice
    # is random but the run is reproducible.
    for label in sorted(members):
        pool = members[label]
        rng.shuffle(pool)
        take = pool[:cap]
        if len(pool) > cap:
            capped += 1
        selected.extend(take)
        per_cluster[label] = len(take)

    logger.info(
        "cluster caps at %d per cluster: %d clusters capped, %d images selected so far",
        cap,
        capped,
        len(selected),
    )

    # Step 3: the caps usually leave us short of the target, so fill from
    # whatever was left over, at random across all clusters.
    filled = 0
    if len(selected) < target:
        chosen = set(selected)
        remainder = [image_id for image_id in image_ids if image_id not in chosen]
        rng.shuffle(remainder)
        shortfall = target - len(selected)
        extra = remainder[:shortfall]
        for image_id in extra:
            per_cluster[cluster_of[image_id]] = per_cluster.get(cluster_of[image_id], 0) + 1
        selected.extend(extra)
        filled = len(extra)
        logger.info("filled %d of a %d shortfall at random", filled, shortfall)
    elif len(selected) > target:
        # More than enough even after capping: trim at random rather than by
        # cluster order, which would bias towards low-numbered clusters.
        rng.shuffle(selected)
        dropped = selected[target:]
        selected = selected[:target]
        for image_id in dropped:
            per_cluster[cluster_of[image_id]] -= 1

    per_source: dict[str, int] = {}
    for image_id in selected:
        source = sources.get(image_id, "unknown")
        per_source[source] = per_source.get(source, 0) + 1

    result = SelectionResult(
        selected=sorted(selected),
        cluster_of=cluster_of,
        per_cluster={label: n for label, n in sorted(per_cluster.items()) if n > 0},
        per_source=dict(sorted(per_source.items())),
        capped_clusters=capped,
        filled_at_random=filled,
    )

    if result.dominant_source_share > SOURCE_DOMINANCE_WARNING:
        dominant = max(result.per_source, key=lambda s: result.per_source[s])
        logger.warning(
            "source %r is %.1f%% of the selection, above the %.0f%% threshold "
            "in §7.2 step 9 -- record this in DECISIONS.md and consider "
            "tightening its per-cluster share",
            dominant,
            100 * result.dominant_source_share,
            100 * SOURCE_DOMINANCE_WARNING,
        )

    return result


def fit_pca(
    vectors: np.ndarray, *, components: int = 128, seed: int
) -> tuple[np.ndarray, object, np.ndarray]:
    """Fit PCA to `components` dimensions. Returns `(reduced, model, variance)`.

    Why reduce at all: the logistic regression in R3 learns one weight per
    input dimension. At 768 dimensions and 150 labels that is five times more
    parameters than examples, and the model would fit the noise. 128 components
    keeps most of the structure while bringing the ratio to something a few
    hundred labels can actually constrain.

    `variance` is the cumulative explained variance ratio, which the brief asks
    to be logged -- it is the honest answer to "how much did we throw away".
    """
    from sklearn.decomposition import PCA

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    components = min(components, *vectors.shape)

    model = PCA(n_components=components, random_state=seed)
    reduced = model.fit_transform(vectors).astype(np.float32)
    cumulative = np.cumsum(model.explained_variance_ratio_)

    logger.info(
        "PCA to %d components explains %.1f%% of the variance (32 components: %.1f%%, 64: %.1f%%)",
        components,
        100 * cumulative[-1],
        100 * cumulative[min(31, len(cumulative) - 1)],
        100 * cumulative[min(63, len(cumulative) - 1)],
    )
    return reduced, model, cumulative
