"""A zero-shot safety screen, and an honest account of what it can and cannot do.

Why this exists
---------------
The corpus is 80,000 images pulled from Flickr archives, museum collections and
a stock photography site. Almost all of it is unremarkable. But a public-domain
archive contains medical photographs, war documentation and anatomical studies,
and an app whose whole purpose is to show one image at a time, full screen,
unprompted, should not hand someone a corpse at breakfast.

How it works
------------
SigLIP 2 shares a space between images and text (`vision/encoders.py`), so a
sentence and a picture become comparable vectors. That allows a classifier with
no training data at all:

1. embed a set of unsafe descriptions and a set of neutral ones;
2. embed the image;
3. compare. If the image sits closer to the unsafe sentences than to the
   neutral ones, it is a candidate for quarantine.

The score used is a **margin**, not a raw similarity:

    unsafe_score(x) = max_u cos(x, u) - max_n cos(x, n)

where `u` ranges over the unsafe prompts and `n` over the neutral ones.

Why a margin rather than `max_u cos(x, u)` alone: raw image-text cosines in
SigLIP 2 sit in a narrow band (roughly 0.0-0.2) and that band shifts with
image style -- a black-and-white photograph scores lower against every prompt
than a bright colour one. Subtracting the best neutral match cancels most of
that offset, which is the same modality-gap correction that `rec/concepts.py`
makes with z-scores.

Calibration
-----------
The threshold is not a guess. It is set from 200 images sampled from the corpus
and treated as the "obviously safe" reference set, at the percentile where no
more than 0.5% of them would be flagged (brief §7.2 step 7). The top 1% of all
candidates by margin are flagged regardless, as the brief also specifies.

What this is not
----------------
This is a coarse filter with no ground truth behind it, and it should be
described that way:

- it will miss things, because six sentences cannot describe everything
  unpleasant;
- it will flag innocent images -- anatomical drawings, red paint, butchery,
  medical illustration -- because they genuinely are close to the unsafe
  prompts in meaning;
- the "obviously safe" calibration set is *assumed* safe, not verified. If the
  corpus contained 2% unsafe images, 2% of that sample is unsafe too, and the
  threshold is correspondingly too permissive.

Nothing is deleted. Flagged images get `status='quarantined'`, which removes
them from selection and from the feed, and the Review page shows them so a
human can decide. If more than 0.5% of all candidates flag, the brief's §3 Q5
asks the owner what to do rather than acting unilaterally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Brief §7.2 step 7 names these six.
UNSAFE_PROMPTS: tuple[str, ...] = (
    "explicit nudity",
    "pornography",
    "graphic violence",
    "gore",
    "a corpse",
    "a severe medical injury",
)

# The neutral set is the other half of the comparison, and it has to span the
# corpus rather than describe one kind of picture: these are the categories the
# four sources actually contain. A neutral set that was all landscapes would
# make every portrait look relatively unsafe.
NEUTRAL_PROMPTS: tuple[str, ...] = (
    "a landscape photograph",
    "a portrait of a person",
    "a photograph of an animal",
    "a building",
    "a city street",
    "a still life of objects",
    "a painting",
    "a drawing or illustration",
    "a historical photograph",
    "a flower",
    "food on a plate",
    "an abstract pattern",
    "a piece of furniture",
    "a vehicle",
    "a museum artefact",
    "a document or page of text",
)

# SigLIP 2's own prompt template. Logged in the metadata because a different
# template shifts every similarity, and a screen calibrated under one template
# is not calibrated under another.
PROMPT_TEMPLATE = "a photo of {}."

# Brief §7.2 step 7: flag the top 1% by margin, plus anything above the
# calibrated threshold.
TOP_FRACTION_FLAGGED = 0.01

# The calibration target: no more than this share of the reference set may be
# flagged by the threshold.
CALIBRATION_FALSE_FLAG_RATE = 0.005

CALIBRATION_SAMPLE_SIZE = 200


@dataclass
class SafetyScreen:
    """A calibrated screen: prompt embeddings plus the threshold they imply."""

    unsafe_vectors: np.ndarray
    neutral_vectors: np.ndarray
    threshold: float
    calibration_size: int
    calibration_flagged: int

    def margins(self, image_vectors: np.ndarray) -> np.ndarray:
        """The unsafe margin for each image. Higher means more suspicious."""
        return unsafe_margin(image_vectors, self.unsafe_vectors, self.neutral_vectors)

    def flag(self, image_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decide which images to quarantine. Returns `(flagged_mask, margins)`.

        Two rules, unioned, per §7.2 step 7: above the calibrated threshold, or
        in the top 1% of this corpus by margin. The second rule means the screen
        always surfaces *something* for review even if the threshold turns out
        to be too permissive for this particular corpus.
        """
        margins = self.margins(image_vectors)
        if len(margins) == 0:
            return np.zeros(0, dtype=bool), margins

        above_threshold = margins > self.threshold

        cutoff_count = max(1, int(len(margins) * TOP_FRACTION_FLAGGED))
        # `argpartition` finds the top k without sorting all 80,000.
        top_positions = np.argpartition(-margins, cutoff_count - 1)[:cutoff_count]
        in_top_fraction = np.zeros(len(margins), dtype=bool)
        in_top_fraction[top_positions] = True

        return above_threshold | in_top_fraction, margins


def unsafe_margin(
    image_vectors: np.ndarray, unsafe_vectors: np.ndarray, neutral_vectors: np.ndarray
) -> np.ndarray:
    """`max_u cos(x, u) - max_n cos(x, n)` for every image.

    All inputs are L2-normalised, so each cosine is a dot product and the whole
    calculation is two matrix multiplies.

    Worked example (asserted in `tests/test_safety.py`): an image whose best
    unsafe cosine is 0.18 and best neutral cosine is 0.11 has a margin of 0.07.
    An image at 0.05 unsafe and 0.20 neutral has a margin of -0.15 -- clearly
    on the safe side, even though its raw unsafe similarity is not zero.
    """
    images = np.asarray(image_vectors, dtype=np.float32)
    if images.ndim == 1:
        images = images.reshape(1, -1)
    if len(images) == 0:
        return np.zeros(0, dtype=np.float32)

    best_unsafe = (images @ np.asarray(unsafe_vectors, dtype=np.float32).T).max(axis=1)
    best_neutral = (images @ np.asarray(neutral_vectors, dtype=np.float32).T).max(axis=1)
    return (best_unsafe - best_neutral).astype(np.float32)


def calibrate(
    unsafe_vectors: np.ndarray,
    neutral_vectors: np.ndarray,
    reference_vectors: np.ndarray,
) -> SafetyScreen:
    """Set the threshold from a reference set assumed to be safe.

    The threshold is the margin above which at most `CALIBRATION_FALSE_FLAG_RATE`
    of the reference set sits. With 200 reference images and a 0.5% target that
    is the single highest-scoring one, so the threshold lands just above the most
    unsafe-looking image in a sample of ordinary photographs.

    The assumption is stated rather than hidden: these images are *assumed*
    safe because they were drawn from a public-domain corpus at random, not
    because anyone checked them.
    """
    margins = unsafe_margin(reference_vectors, unsafe_vectors, neutral_vectors)
    if len(margins) == 0:
        raise ValueError("cannot calibrate a safety screen with no reference images")

    percentile = 100 * (1 - CALIBRATION_FALSE_FLAG_RATE)
    threshold = float(np.percentile(margins, percentile))
    flagged = int((margins > threshold).sum())

    logger.info(
        "safety screen calibrated on %d reference images: threshold %.4f flags %d of them (%.2f%%)",
        len(margins),
        threshold,
        flagged,
        100 * flagged / len(margins),
    )
    return SafetyScreen(
        unsafe_vectors=np.asarray(unsafe_vectors, dtype=np.float32),
        neutral_vectors=np.asarray(neutral_vectors, dtype=np.float32),
        threshold=threshold,
        calibration_size=len(margins),
        calibration_flagged=flagged,
    )


def embed_prompts(encoder, prompts: tuple[str, ...]) -> np.ndarray:
    """Embed prompt strings through the template, returning unit vectors."""
    filled = [PROMPT_TEMPLATE.format(prompt) for prompt in prompts]
    return encoder.encode_texts(filled)


def build_screen(encoder, reference_vectors: np.ndarray) -> SafetyScreen:
    """Embed both prompt sets and calibrate against `reference_vectors`."""
    unsafe = embed_prompts(encoder, UNSAFE_PROMPTS)
    neutral = embed_prompts(encoder, NEUTRAL_PROMPTS)
    return calibrate(unsafe, neutral, reference_vectors)


def sample_reference_vectors(
    vectors: np.ndarray, *, seed: int, size: int = CALIBRATION_SAMPLE_SIZE
) -> np.ndarray:
    """A seeded random sample of the corpus, to calibrate against.

    Random rather than hand-picked: a hand-picked "obviously safe" set would be
    chosen to look safe, and the threshold would then be calibrated against a
    cleaner distribution than the corpus actually has.
    """
    rng = np.random.default_rng(seed)
    if len(vectors) <= size:
        return vectors
    return vectors[rng.choice(len(vectors), size, replace=False)]
