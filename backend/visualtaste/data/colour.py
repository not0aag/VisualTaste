"""A 64-number colour descriptor, and the five colours a human would name.

Intuition
---------
The neural encoders are surprisingly bad at colour. SigLIP 2 and DINOv2 both
learn that a thing is a chair, in a room, shot from above -- but "warm ochre"
and "cold grey" land close together in their spaces, because for most of their
training objectives the colour was not the point. Visual taste is substantially
about colour, so this module supplies what they leave out: an explicit,
interpretable histogram that the ranker can weigh for itself.

The formula, in four steps
-------------------------
Start from the 256 px thumbnail, as a (H, W, 3) array of sRGB bytes.

1. **Convert to CIELAB.** Not RGB, because distance in RGB does not match
   perception: (0,0,0)->(10,10,10) and (200,200,200)->(210,210,210) are the
   same RGB distance and nowhere near the same perceived difference. CIELAB is
   built so that Euclidean distance approximates perceived difference.

       sRGB -> linear RGB     (undo the display gamma curve)
       linear RGB -> XYZ      (a 3x3 matrix, D65 white point)
       XYZ -> L*a*b*          (a cube-root curve per axis)

   L* is lightness, 0 (black) to 100 (white). a* runs green(-) to red(+),
   b* runs blue(-) to yellow(+), each roughly -128..127.

2. **Histogram into 4 x 4 x 4 = 64 bins.** Four bins per axis, so each bin is a
   broad region like "dark, slightly red, quite yellow". Coarse on purpose: with
   32 bins per axis almost no two photographs would share a bin, and the
   descriptor would memorise rather than generalise.

3. **L1-normalise:** divide by the pixel count, so every histogram sums to 1.
   This makes a 5504x5504 scan comparable to a 1024x768 photograph -- the
   descriptor describes proportions of colour, not amounts.

4. **Take the element-wise square root** (the Hellinger transform). This is the
   step that is easy to skip and worth understanding:

       Euclidean distance between sqrt(p) and sqrt(q)
           = sqrt( sum_i ( sqrt(p_i) - sqrt(q_i) )^2 )
           = sqrt(2) * Hellinger distance between p and q

   So after this transform, the plain dot products and Euclidean distances that
   every downstream model computes are *already* the statistically appropriate
   distance for probability distributions. Without it, one dominant bin at 0.9
   swamps the comparison; sqrt(0.9) = 0.95 against sqrt(0.01) = 0.1 compresses
   that dominance and lets the smaller bins matter.

Worked example (asserted in `tests/test_colour.py`)
---------------------------------------------------
An image of 4 pixels where 3 fall in bin 0 and 1 falls in bin 63:

    counts        = [3, 0, ..., 0, 1]
    L1-normalised = [0.75, 0, ..., 0, 0.25]
    sqrt          = [0.866..., 0, ..., 0, 0.5]
    sum of squares = 0.75 + 0.25 = 1.0

That last line is the useful identity: after the Hellinger transform every
descriptor is a unit vector, because the squares are the original probabilities
and those sum to 1. The descriptor needs no separate normalisation step.

See maths unit M1 for cosine similarity, and notebook 04.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BINS_PER_AXIS = 4
DESCRIPTOR_SIZE = BINS_PER_AXIS**3  # 64
DOMINANT_COLOUR_COUNT = 5

# Pixels sampled before clustering for the display swatches. Measured on this
# machine: k-means over 4,096 pixels with scikit-learn's default n_init=4 costs
# 1,350 ms per image, which across 80,000 images is over an hour spent on
# decoration. Over 1,024 pixels with a single seeded initialisation it is
# 139 ms and the swatches are indistinguishable -- five broad colour regions
# are not a subtle estimation problem. See DECISIONS.md #9.
DOMINANT_SAMPLE_PIXELS = 1_024
DOMINANT_MAX_ITER = 25

# CIELAB axis ranges used for binning. L* is exact; a* and b* are clipped to
# the range real sRGB colours can reach, which is well inside the theoretical
# -128..127.
L_RANGE = (0.0, 100.0)
AB_RANGE = (-110.0, 110.0)

# sRGB D65 -> XYZ. The standard matrix from IEC 61966-2-1.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

# The D65 reference white, which is what L*a*b* measures relative to.
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])

# Constants of the L*a*b* transfer function. `_EPSILON` is (6/29)^3 and
# `_KAPPA` is 841/108; below the epsilon the cube root is replaced by a linear
# segment, which keeps the function's slope finite near black.
_EPSILON = 216 / 24389
_KAPPA = 24389 / 27


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB gamma curve. Input and output are in [0, 1].

    Displays apply a roughly 2.2-power curve, so stored sRGB values are not
    proportional to light. Every colour-space conversion has to undo that first,
    or the arithmetic is being done on perceptual values as if they were
    physical ones.
    """
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb_bytes: np.ndarray) -> np.ndarray:
    """Convert an (..., 3) array of sRGB bytes to CIELAB.

    Returns the same shape, with L* in [0, 100] and a*, b* in roughly
    [-110, 110]. Written out rather than imported so that the one place colour
    space matters is readable; `skimage.color.rgb2lab` does the same thing.
    """
    srgb = np.asarray(rgb_bytes, dtype=np.float64) / 255.0
    linear = srgb_to_linear(srgb)

    # (..., 3) @ (3, 3).T applies the matrix to every pixel at once.
    xyz = linear @ _SRGB_TO_XYZ.T
    scaled = xyz / _WHITE_D65

    # The cube-root curve, with a linear segment near zero.
    f = np.where(
        scaled > _EPSILON,
        np.cbrt(scaled),
        (_KAPPA * scaled + 16) / 116,
    )

    f_x, f_y, f_z = f[..., 0], f[..., 1], f[..., 2]
    lightness = 116 * f_y - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)
    return np.stack([lightness, a, b], axis=-1)


def _bin_indices(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map values in [low, high] onto 0..BINS_PER_AXIS-1, clipping outliers."""
    scaled = (values - low) / (high - low) * BINS_PER_AXIS
    return np.clip(scaled.astype(np.int32), 0, BINS_PER_AXIS - 1)


def lab_histogram(lab: np.ndarray) -> np.ndarray:
    """A 64-bin Hellinger-transformed CIELAB histogram from an (N, 3) Lab array.

    Steps 2-4 of the module docstring. The result is a unit vector in float32.
    """
    flat = lab.reshape(-1, 3)
    if flat.size == 0:
        return np.zeros(DESCRIPTOR_SIZE, dtype=np.float32)

    l_bin = _bin_indices(flat[:, 0], *L_RANGE)
    a_bin = _bin_indices(flat[:, 1], *AB_RANGE)
    b_bin = _bin_indices(flat[:, 2], *AB_RANGE)

    # One flat bin index per pixel: a 3-D histogram addressed as a 1-D array.
    flat_bins = (l_bin * BINS_PER_AXIS + a_bin) * BINS_PER_AXIS + b_bin
    counts = np.bincount(flat_bins, minlength=DESCRIPTOR_SIZE).astype(np.float64)

    # Step 3: L1-normalise, so the descriptor is proportions, not pixel counts.
    total = counts.sum()
    if total == 0:  # pragma: no cover - bincount always sums to the pixel count
        return np.zeros(DESCRIPTOR_SIZE, dtype=np.float32)
    probabilities = counts / total

    # Step 4: Hellinger. The result is automatically unit-length, because the
    # squares are the probabilities and those sum to 1.
    return np.sqrt(probabilities).astype(np.float32)


def colour_descriptor(thumb: Image.Image) -> np.ndarray:
    """The 64-number descriptor for one thumbnail. Expects RGB."""
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    pixels = np.asarray(thumb, dtype=np.uint8)
    return lab_histogram(rgb_to_lab(pixels))


def dominant_colours(
    thumb: Image.Image, count: int = DOMINANT_COLOUR_COUNT, *, seed: int = 1729
) -> list[str]:
    """The `count` most representative colours, as hex strings for the UI.

    k-means in CIELAB rather than RGB, so the clusters group colours a person
    would call similar. Returned largest-cluster-first, which is what makes the
    list read like "this photograph is mostly dark blue, then grey, then...".

    This is for display only. The histogram above is the feature the models see;
    five hex strings would be a lossy and oddly-quantised thing to learn from.
    """
    from sklearn.cluster import KMeans

    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")

    pixels = np.asarray(thumb, dtype=np.uint8).reshape(-1, 3)
    if len(pixels) == 0:
        return []

    rng = np.random.default_rng(seed)
    if len(pixels) > DOMINANT_SAMPLE_PIXELS:
        pixels = pixels[rng.choice(len(pixels), DOMINANT_SAMPLE_PIXELS, replace=False)]

    lab = rgb_to_lab(pixels).astype(np.float32)

    # k cannot exceed the number of distinct colours present, or k-means emits
    # empty clusters. Each RGB triple is packed into one integer and counted,
    # rather than calling `np.unique(lab, axis=0)` -- that lexsorts a view of
    # float rows and costs more than the clustering it is there to protect.
    packed = (
        pixels[:, 0].astype(np.int32) << 16
        | pixels[:, 1].astype(np.int32) << 8
        | pixels[:, 2].astype(np.int32)
    )
    clusters = max(1, min(count, len(np.unique(packed))))

    kmeans = KMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=1,
        max_iter=DOMINANT_MAX_ITER,
    ).fit(lab)

    # Order by how many pixels each cluster claimed, biggest first.
    _, sizes = np.unique(kmeans.labels_, return_counts=True)
    order = np.argsort(-sizes)

    out: list[str] = []
    for index in order:
        members = pixels[kmeans.labels_ == index]
        # The mean of the member pixels in RGB, rather than converting the Lab
        # centroid back: the centroid can land outside the sRGB gamut, and a
        # colour swatch that cannot be displayed is not useful to show.
        mean_rgb = members.mean(axis=0).round().astype(int).clip(0, 255)
        out.append("#{:02x}{:02x}{:02x}".format(*mean_rgb))
    return out
