"""Find near-duplicate images, so the feed never shows the same picture twice.

Why this matters more than it sounds
------------------------------------
Four sources, two of which draw on Flickr and one on Wikimedia Commons, will
deliver the same photograph more than once. Exact byte-for-byte duplicates are
already free: `image_id` is the SHA-256 of the bytes, so they collapse into one
row with no comparison at all (`db/repo.upsert_image`).

Near-duplicates are the real problem. The same photograph re-encoded at a
different quality, resized, or watermarked has completely different bytes and a
completely different hash, but is the same picture. Left in, it damages the
system in three ways: the feed shows Alen the same image twice and looks broken;
the classifier sees the same labelled example several times and over-weights it;
and the diversity metrics report variety that is not there.

How a perceptual hash works
---------------------------
`imagehash.phash` is a 64-bit fingerprint built to survive re-encoding:

1. shrink the image to 32x32 greyscale -- throwing away all fine detail;
2. take a 2-D discrete cosine transform, which expresses the image as a sum of
   increasingly fine checkerboard patterns;
3. keep the top-left 8x8 block of coefficients, which is the coarse structure
   (where it is light, where it is dark) and discards the fine structure where
   compression artefacts live;
4. emit one bit per coefficient: 1 if above the median, 0 if below.

Two images are near-duplicates when their hashes differ in few bit positions.
That count is the **Hamming distance**, and the brief's threshold (§7.2 step 5)
is 6 bits out of 64.

Why 6: a re-encode or a modest resize typically moves 0-4 bits. Genuinely
different photographs of a similar scene usually differ by 20 or more. The gap
between those two populations is wide, and 6 sits in it -- strict enough that
two different sunsets are not merged, loose enough to catch a JPEG re-save.

Which copy survives
-------------------
The highest-resolution one. It makes the better 1024 px derivative, and its
embedding is computed from more real detail. The others get
`status='near_dup'` and `dup_of=<the kept id>`, so the group remains walkable
-- nothing is deleted, and the Review page can show what was folded together.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import imagehash
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

# Brief §7.2 step 5. Expressed in bits of a 64-bit hash.
PHASH_HAMMING_THRESHOLD = 6

# Brief §7.2 step 8: the same idea applied to DINOv2 embeddings, which catch
# near-duplicates that survive a crop or a colour shift and so fool a pHash.
EMBEDDING_COSINE_THRESHOLD = 0.95

# Length of the prefix used to bucket hashes before comparing them. See
# `find_near_duplicate_groups`.
_BUCKET_BITS = 16


def phash_for(thumb: Image.Image) -> str:
    """The 64-bit perceptual hash of a thumbnail, as 16 hex characters.

    Computed from the 256 px thumbnail rather than the original: pHash shrinks
    its input to 32x32 immediately, so feeding it a 5504 px scan would cost a
    full decode for an identical answer.
    """
    return str(imagehash.phash(thumb))


def hamming_distance(left: str, right: str) -> int:
    """How many of the 64 bits differ between two hex-encoded pHashes.

    `int(..., 16)` parses the hex, XOR marks the differing bits as 1s, and
    `bit_count` counts them. This is the whole of the comparison.
    """
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass
class DuplicateGroup:
    """One kept image and the near-duplicates folded into it."""

    kept: str
    duplicates: list[str] = field(default_factory=list)


class _DisjointSet:
    """Union-find over image ids, for grouping duplicates transitively.

    This exists because of a bug worth understanding. The obvious algorithm --
    walk the images, and for each one claim every unclaimed image similar to it
    -- resolves only one *layer* of overlap per pass. Given a chain where A
    resembles B and B resembles C but A does not resemble C, A claims B, and
    then C is skipped because its only partner is already claimed. C survives,
    and a second run folds it.

    That showed up as a dedupe pass that was not idempotent: successive runs
    folded 48 images, then 12, then more, slowly converging. Nothing was
    *wrong* in the database, but "run the job twice and the second run is a
    no-op" is a property the whole resumability contract depends on.

    Connected components fix it properly: similarity is treated as an
    undirected graph, every component collapses to one kept image in a single
    pass, and a second pass finds nothing.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression, so repeated lookups stay near-constant.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def components(self) -> dict[str, list[str]]:
        """Root -> every member of that component, including the root."""
        groups: dict[str, list[str]] = {}
        for item in self._parent:
            groups.setdefault(self.find(item), []).append(item)
        return groups


def _groups_from_edges(
    edges: list[tuple[str, str]], pixels: dict[str, int]
) -> list[DuplicateGroup]:
    """Collapse each connected component to its highest-resolution member.

    The kept image is the one with the most pixels (ties broken by id, so the
    result is deterministic): it makes the better 1024 px derivative and its
    embedding was computed from more real detail.
    """
    forest = _DisjointSet()
    for left, right in edges:
        forest.union(left, right)

    groups: list[DuplicateGroup] = []
    for members in forest.components().values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda image_id: (-pixels.get(image_id, 0), image_id))
        groups.append(DuplicateGroup(kept=ordered[0], duplicates=ordered[1:]))
    return groups


@dataclass(frozen=True)
class HashedImage:
    """The minimum needed to decide duplicate groups: identity, hash, size."""

    image_id: str
    phash: str
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return max(0, self.width) * max(0, self.height)


def find_near_duplicate_groups(
    images: Sequence[HashedImage], *, threshold: int = PHASH_HAMMING_THRESHOLD
) -> list[DuplicateGroup]:
    """Group near-duplicates, keeping the highest-resolution member of each.

    The naive approach compares every pair, which at 80,000 images is 3.2
    billion comparisons. Instead, hashes are bucketed by their leading bits
    first: two hashes within 6 bits of each other usually share a prefix, so
    only images inside a bucket (and its immediate neighbours by prefix) need
    comparing.

    This is a heuristic, not an exact nearest-neighbour search. It can miss a
    pair whose difference happens to fall entirely in the leading bits, which
    is why §7.2 step 8 runs a second, embedding-based pass afterwards: the two
    methods fail in different ways, and between them very little gets through.
    """
    ordered = sorted(images, key=lambda item: (-item.pixels, item.image_id))

    buckets: dict[str, list[HashedImage]] = defaultdict(list)
    for item in ordered:
        buckets[_bucket_key(item.phash)].append(item)

    # Collect every near-duplicate *pair*, then let connected components decide
    # the groups. See `_DisjointSet` for why one-group-per-image-per-pass is
    # not enough.
    edges: list[tuple[str, str]] = []
    for item in ordered:
        for other in _bucket_neighbours(buckets, item.phash):
            if other.image_id <= item.image_id:
                continue  # Each unordered pair considered once.
            if hamming_distance(item.phash, other.phash) <= threshold:
                edges.append((item.image_id, other.image_id))

    pixels = {item.image_id: item.pixels for item in images}
    groups = _groups_from_edges(edges, pixels)

    total = sum(len(g.duplicates) for g in groups)
    logger.info(
        "pHash near-duplicates: %d groups covering %d images to fold away",
        len(groups),
        total,
    )
    return groups


def _bucket_key(phash: str) -> str:
    """The leading hex characters of a hash, used as a coarse bucket."""
    return phash[: _BUCKET_BITS // 4]


def _bucket_neighbours(
    buckets: dict[str, list[HashedImage]], phash: str
) -> Iterable[HashedImage]:
    """Every image in this hash's bucket, plus buckets one bit away from it.

    Checking single-bit variations of the prefix recovers most of the pairs a
    plain prefix match would miss, at 16x the lookups rather than 80,000x the
    comparisons.
    """
    prefix_length = _BUCKET_BITS // 4
    seen_keys: set[str] = set()

    key = _bucket_key(phash)
    seen_keys.add(key)
    yield from buckets.get(key, ())

    value = int(key, 16)
    for bit in range(_BUCKET_BITS):
        neighbour = format(value ^ (1 << bit), f"0{prefix_length}x")
        if neighbour in seen_keys:
            continue
        seen_keys.add(neighbour)
        yield from buckets.get(neighbour, ())


def find_embedding_duplicate_groups(
    image_ids: Sequence[str],
    vectors,
    *,
    threshold: float = EMBEDDING_COSINE_THRESHOLD,
    pixels: dict[str, int] | None = None,
) -> list[DuplicateGroup]:
    """Group near-duplicates by DINOv2 cosine similarity (brief §7.2 step 8).

    `vectors` is an (N, D) array of L2-normalised DINOv2 embeddings aligned with
    `image_ids`, so cosine similarity is a plain dot product and the whole
    similarity matrix is one matrix multiply.

    DINOv2 catches what pHash misses. A crop, a border, or a colour grade
    changes the coarse light-and-dark structure that pHash fingerprints, but
    leaves the appearance -- texture, shape, layout -- that DINOv2 encodes
    almost untouched.

    Done in row blocks: the full 80,000 x 80,000 similarity matrix would be
    25 GB, while 512 rows at a time is 160 MB and just as correct.
    """
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float32)
    if len(image_ids) != len(vectors):
        raise ValueError(
            f"{len(image_ids)} ids but {len(vectors)} vectors; they must align"
        )

    sizes = pixels or {}

    # Every pair above the threshold, collected in row blocks so the full
    # 80,000 x 80,000 similarity matrix (25 GB) is never materialised. At a
    # 0.95 threshold the number of surviving pairs is small -- a few hundred
    # across this corpus -- so holding the edge list is trivial.
    edges: list[tuple[str, str]] = []
    block = 512

    for start in range(0, len(vectors), block):
        chunk = vectors[start : start + block]
        similarity = chunk @ vectors.T  # (block, N), cosine because unit-length

        for offset, row in enumerate(similarity):
            index = start + offset
            for j in np.flatnonzero(row >= threshold):
                partner = int(j)
                # Each unordered pair once, and never an image with itself.
                if partner > index:
                    edges.append((image_ids[index], image_ids[partner]))

    groups = _groups_from_edges(edges, sizes)

    total = sum(len(g.duplicates) for g in groups)
    logger.info(
        "embedding near-duplicates: %d groups covering %d images (cosine >= %.2f)",
        len(groups),
        total,
        threshold,
    )
    return groups
