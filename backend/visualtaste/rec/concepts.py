"""Putting a taste into words, and the correction that makes it possible.

What this does
--------------
SigLIP 2 shares a space between images and text, so a concept like "brutalist"
and a photograph of a concrete building become comparable vectors. That lets
the system describe a preference vector in English: score every concept against
it and report the ones that stand out.

The problem: raw cosines are not comparable
-------------------------------------------
Image-text cosines in SigLIP 2 sit in a narrow band -- roughly 0.0 to 0.2 -- and
each concept has its own baseline within it. "Photograph" scores highly against
*every* image because almost everything in the corpus is one; "cyanotype"
scores low against everything because the word is rare in the training
distribution. Ranking by raw cosine therefore produces the same answer for
every user: photograph, portrait, landscape, building. Technically true,
completely uninformative.

That offset is the image-text modality gap. The fix the brief specifies (§7.6)
is to report **z-scores against the mean concept similarity**:

    z_c = (cos(u, t_c) - mean_c' cos(u, t_c')) / sd_c' cos(u, t_c')

Now the question is not "how similar is this taste to the word brutalist" but
"how much more brutalist than average is this taste", and the answer differs
between people, which is the entire point.

A worked example makes the difference concrete. Suppose a taste scores
"photograph" 0.18, "brutalist" 0.12, "cyanotype" 0.04, and the mean concept
similarity is 0.11 with a standard deviation of 0.03:

    photograph  z = (0.18 - 0.11) / 0.03 = +2.3
    brutalist   z = (0.12 - 0.11) / 0.03 = +0.3
    cyanotype   z = (0.04 - 0.11) / 0.03 = -2.3

and against a *different* taste whose raw scores are 0.17, 0.19, 0.05 with the
same mean, brutalist becomes +2.7 and photograph +2.0. The raw ordering barely
moved; the z-scores separated the two people.

What is still wrong with it
---------------------------
Z-scoring removes the per-concept offset only to the extent that the offset is
shared across concepts. It cannot fix a concept whose baseline is unusual in a
way the standard deviation does not capture, and it says nothing about whether
SigLIP 2's idea of "brutalist" matches Alen's. So these words are a readable
summary, not a measurement -- which is why the Profile page shows them as
descriptions rather than as scores.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CONCEPTS_FILE = Path(__file__).parent / "concepts.txt"

# SigLIP 2's own prompt template. Recorded here and written into the cache
# metadata, because a different template shifts every similarity -- so a cache
# built under one template must not be read under another.
PROMPT_TEMPLATE = "a photo of {}."

# Concepts reported per direction by default.
TOP_N = 8

# A z-score below this is not worth printing: it says the taste is about as
# related to that word as to the average word, which is not a description.
MIN_INTERESTING_Z = 0.75


@dataclass
class ConceptScore:
    """One concept and how much it stands out for a taste."""

    concept: str
    z_score: float
    raw_similarity: float

    def __str__(self) -> str:
        return f"{self.concept} (z={self.z_score:+.2f})"


def load_concepts(path: Path | None = None) -> list[str]:
    """Read the vocabulary, dropping comments and blanks.

    The file holds 637 terms rather than the brief's "~400". More vocabulary
    costs 637 text embeddings once and buys a richer description, so the
    overshoot is deliberate -- the brief's figure is a scale, and this is the
    same order.
    """
    source = path or CONCEPTS_FILE
    lines = source.read_text(encoding="utf-8").splitlines()
    concepts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not concepts:
        raise ValueError(f"no concepts found in {source}")
    return concepts


def embed_concepts(encoder, concepts: list[str]) -> np.ndarray:
    """Embed each concept through the prompt template. Returns unit vectors."""
    prompts = [PROMPT_TEMPLATE.format(concept) for concept in concepts]
    return encoder.encode_texts(prompts)


def score_concepts(
    direction: np.ndarray, concept_vectors: np.ndarray, concepts: list[str]
) -> list[ConceptScore]:
    """Z-score every concept against one direction in embedding space.

    `direction` is a preference vector, an interest centroid, or anything else
    that lives in the image space. It need not be unit length -- it is
    normalised here, because a z-score of a cosine requires a cosine.
    """
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("cannot describe a zero-length direction")
    direction = direction / norm

    similarity = (np.asarray(concept_vectors, dtype=np.float32) @ direction).astype(np.float64)
    if len(similarity) != len(concepts):
        raise ValueError(f"{len(concept_vectors)} concept vectors but {len(concepts)} concepts")

    mean = float(similarity.mean())
    spread = float(similarity.std())
    if spread < 1e-12:
        # Every concept equally similar: no description is possible, and
        # dividing by zero would produce infinities that look like findings.
        logger.warning("concept similarities have no spread; cannot z-score")
        return [
            ConceptScore(concept, 0.0, float(value))
            for concept, value in zip(concepts, similarity, strict=True)
        ]

    return [
        ConceptScore(
            concept=concept,
            z_score=float((value - mean) / spread),
            raw_similarity=float(value),
        )
        for concept, value in zip(concepts, similarity, strict=True)
    ]


def describe(
    direction: np.ndarray,
    concept_vectors: np.ndarray,
    concepts: list[str],
    *,
    top_n: int = TOP_N,
    minimum_z: float = MIN_INTERESTING_Z,
) -> list[ConceptScore]:
    """The `top_n` concepts that most describe a direction.

    Concepts below `minimum_z` are dropped rather than padded out to `top_n`:
    a taste the vocabulary cannot describe should return three words, not three
    words and five shrugs.
    """
    scored = score_concepts(direction, concept_vectors, concepts)
    scored.sort(key=lambda item: -item.z_score)
    return [item for item in scored[:top_n] if item.z_score >= minimum_z]


def describe_negative(
    direction: np.ndarray,
    concept_vectors: np.ndarray,
    concepts: list[str],
    *,
    top_n: int = TOP_N,
    minimum_z: float = MIN_INTERESTING_Z,
) -> list[ConceptScore]:
    """The concepts a direction is *least* like -- "taste in words" for dislikes.

    Applied to the preference vector this answers "what is this taste furthest
    from", which is related to but not the same as describing the dislike
    vector. The Profile page shows both, because they differ: the things he
    actively rejected and the things his taste simply has nothing to do with
    are different kinds of absence.
    """
    scored = score_concepts(direction, concept_vectors, concepts)
    scored.sort(key=lambda item: item.z_score)
    return [item for item in scored[:top_n] if item.z_score <= -minimum_z]


def label_for(
    direction: np.ndarray,
    concept_vectors: np.ndarray,
    concepts: list[str],
    *,
    words: int = 3,
) -> str:
    """A short label for an interest centroid: its top concepts, comma-joined.

    Used by R5 to name each interest on the Profile page (§7.5). Three words
    because a label is read at a glance; the full z-score list is available
    underneath it.
    """
    top = describe(direction, concept_vectors, concepts, top_n=words, minimum_z=0.0)
    if not top:
        return "unnamed interest"
    return ", ".join(item.concept for item in top)


class ConceptCache:
    """Concept vectors on disk, so the text encoder is loaded once ever.

    Embedding 637 short strings takes a couple of seconds on the GPU and is
    completely deterministic, so it is done once and cached. The cache records
    the prompt template and the concept list alongside the vectors: a cache
    built under a different template or vocabulary would silently produce
    wrong descriptions, so it is invalidated rather than reused.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.vectors_path = directory / "concept_vectors.npy"
        self.meta_path = directory / "concept_meta.json"

    def load(self) -> tuple[list[str], np.ndarray] | None:
        if not (self.vectors_path.exists() and self.meta_path.exists()):
            return None

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("prompt_template") != PROMPT_TEMPLATE:
            logger.info("concept cache was built with a different prompt template")
            return None

        concepts = meta.get("concepts", [])
        vectors = np.load(self.vectors_path)
        if len(concepts) != len(vectors):
            logger.warning("concept cache is inconsistent; rebuilding")
            return None
        return concepts, vectors

    def save(self, concepts: list[str], vectors: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

        tmp_vectors = self.vectors_path.with_name(f".{self.vectors_path.name}.part")
        with tmp_vectors.open("wb") as handle:
            np.save(handle, vectors.astype(np.float32))
        tmp_vectors.replace(self.vectors_path)

        tmp_meta = self.meta_path.with_name(f".{self.meta_path.name}.part")
        tmp_meta.write_text(
            json.dumps(
                {
                    "prompt_template": PROMPT_TEMPLATE,
                    "count": len(concepts),
                    "dim": int(vectors.shape[1]),
                    "concepts": concepts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_meta.replace(self.meta_path)
        logger.info("cached %d concept vectors in %s", len(concepts), self.directory)

    def ensure(self, encoder_factory) -> tuple[list[str], np.ndarray]:
        """Load the cache, building it with `encoder_factory()` if necessary.

        The encoder is constructed lazily so that a warm cache never loads the
        SigLIP 2 weights at all -- which is what keeps API startup fast.
        """
        cached = self.load()
        if cached is not None:
            return cached

        concepts = load_concepts()
        logger.info("embedding %d concepts (no usable cache)", len(concepts))
        vectors = embed_concepts(encoder_factory(), concepts)
        self.save(concepts, vectors)
        return concepts, vectors


def cache_for(settings) -> ConceptCache:
    """The project's concept cache: `<DATA>/concepts/`."""
    return ConceptCache(settings.data / "concepts")


def main() -> int:
    """`python -m visualtaste.rec.concepts`: build the concept cache once.

    Loads the SigLIP 2 text encoder (GPU if available), embeds the vocabulary
    through the prompt template, and saves it. The API only ever *reads* the
    cache, so it never has to load a text model at startup.
    """
    from visualtaste.config import get_settings
    from visualtaste.vision.encoders import SIGLIP2_KEY, load_encoder

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()
    settings.apply_hf_env()
    concepts, vectors = cache_for(settings).ensure(lambda: load_encoder(SIGLIP2_KEY))
    logger.info("concept cache ready: %d concepts x %d dims", len(concepts), vectors.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
