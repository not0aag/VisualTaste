"""Frozen image encoders: the two models that turn pixels into vectors.

Intuition
---------
Neither model is trained here, and neither ever learns anything about Alen.
They are fixed functions that map an image to a point in a 768-dimensional
space, chosen so that images a human would call similar land close together.
All the learning in this project happens *downstream* of these vectors, on a
few hundred of his labels -- which is only tractable because the hard part
(understanding what an image contains) is already done by these encoders.

Two encoders, because they disagree in useful ways:

    siglip2_b16_224   SigLIP 2 ViT-B/16, trained on image-text pairs. It knows
                      *what* is in a picture and shares a space with text, so
                      the same model can score the concept "brutalist" against
                      an image. Primary features, concepts, safety screen.
    dinov2_b14        DINOv2 ViT-B/14, trained on images alone (self-supervised).
                      It knows *how* things look -- texture, shape, layout --
                      with no language bias. Appearance similarity, near-dupes.

The formula
-----------
Every encoder returns L2-normalised vectors:

    x = f(image)              raw encoder output, 768 numbers
    x_hat = x / ||x||         ||x_hat|| = 1

Normalising up front means cosine similarity is just a dot product:

    cos(a_hat, b_hat) = a_hat . b_hat

That is one `numpy` matrix multiply for 50,000 images instead of a division per
pair, and it is why `faiss.IndexFlatIP` (inner product) can be used to answer
cosine-similarity queries. See maths unit M1 and notebook 01.

Worked example
--------------
A = (1, 2, 2), ||A|| = 3, so A_hat = (1/3, 2/3, 2/3).
B = (2, 1, 2), ||B|| = 3, so B_hat = (2/3, 1/3, 2/3).
A_hat . B_hat = (2 + 2 + 4) / 9 = 8/9 = 0.8889.
`test_encoders.py` asserts `l2_normalise` reproduces this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL.Image import Image

logger = logging.getLogger(__name__)

SIGLIP2_KEY = "siglip2_b16_224"
DINOV2_KEY = "dinov2_b14"


@dataclass(frozen=True)
class EncoderSpec:
    """What a `model_key` means, in enough detail to reproduce an embedding run.

    `checkpoint` is written into `meta.json` beside the vectors, so a future run
    can tell whether its vectors came from the same weights. The exact
    `open_clip` tag was read from a live `open_clip.list_pretrained()` on
    2026-10-30 rather than copied from the brief -- model registries move.
    """

    key: str
    family: str
    checkpoint: str
    dim: int
    image_size: int
    has_text_encoder: bool


ENCODER_SPECS: dict[str, EncoderSpec] = {
    SIGLIP2_KEY: EncoderSpec(
        key=SIGLIP2_KEY,
        family="open_clip",
        # Unsuffixed means 224 px in open_clip's SigLIP 2 naming; the -256/-384
        # /-512 variants are the higher-resolution checkpoints.
        checkpoint="ViT-B-16-SigLIP2/webli",
        dim=768,
        image_size=224,
        has_text_encoder=True,
    ),
    DINOV2_KEY: EncoderSpec(
        key=DINOV2_KEY,
        family="transformers",
        checkpoint="facebook/dinov2-base",
        dim=768,
        image_size=224,
        has_text_encoder=False,
    ),
}


def l2_normalise(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale each vector to unit length, so dot product == cosine similarity.

    `eps` guards the degenerate all-zero vector, which would otherwise divide by
    zero. A real encoder never produces one, but a test fixture might.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def pick_device() -> torch.device:
    """CUDA if it is really there, else CPU.

    The CPU path exists so that tests and small jobs run on any machine; it is
    roughly 30x slower, which is fine for a few hundred fixture images and
    hopeless for 80,000.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder:
    """A loaded, frozen encoder: preprocessing + forward pass + normalisation.

    Instances are stateless apart from the weights, and are safe to reuse across
    batches. They are deliberately *not* `torch.nn.Module` subclasses: nothing
    here is ever trained, and inheriting from Module would invite someone to
    call `.train()` or hand it to an optimiser.
    """

    def __init__(
        self,
        spec: EncoderSpec,
        model: torch.nn.Module,
        preprocess: Callable[[Image], torch.Tensor],
        device: torch.device,
        encode_image: Callable[[torch.Tensor], torch.Tensor],
        encode_text: Callable[[list[str]], torch.Tensor] | None = None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.preprocess = preprocess
        self._model = model
        self._encode_image = encode_image
        self._encode_text = encode_text

    @property
    def dim(self) -> int:
        return self.spec.dim

    def _autocast(self) -> torch.autocast | torch.amp.autocast:
        """Half precision on the GPU, full precision on the CPU.

        float16 roughly halves the memory and time of a forward pass with no
        measurable effect on cosine similarities (the vectors are stored as
        float16 on disk anyway). CPU autocast to float16 is slow and poorly
        supported, so the CPU path stays in float32.
        """
        enabled = self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=enabled)

    @torch.inference_mode()
    def encode_images(self, batch: torch.Tensor) -> np.ndarray:
        """Embed a preprocessed batch, shape (B, 3, H, W) -> (B, dim) float32.

        Returns L2-normalised float32 regardless of the autocast dtype, because
        everything downstream (FAISS, PCA, the classifiers) assumes unit-length
        float32 rows.
        """
        batch = batch.to(self.device, non_blocking=True)
        with self._autocast():
            features = self._encode_image(batch)
        return l2_normalise(features.float().cpu().numpy())

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text into the *same* space as the images (SigLIP 2 only).

        This is what makes "taste in words" and the zero-shot safety screen
        possible: a concept string and an image become comparable vectors. Note
        the modality gap -- text-image cosines are systematically lower than
        image-image ones, which is why `rec/concepts.py` reports z-scores
        instead of raw similarities.
        """
        if self._encode_text is None:
            raise NotImplementedError(f"{self.spec.key} is image-only; use {SIGLIP2_KEY} for text.")
        with self._autocast():
            features = self._encode_text(texts)
        return l2_normalise(features.float().cpu().numpy())


def _load_siglip2(spec: EncoderSpec, device: torch.device) -> ImageEncoder:
    import open_clip

    model_name, pretrained = spec.checkpoint.split("/", 1)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer(texts).to(device)
        return model.encode_text(tokens)

    return ImageEncoder(
        spec=spec,
        model=model,
        preprocess=preprocess,
        device=device,
        encode_image=model.encode_image,
        encode_text=encode_text,
    )


def _load_dinov2(spec: EncoderSpec, device: torch.device) -> ImageEncoder:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(spec.checkpoint)
    model = AutoModel.from_pretrained(spec.checkpoint).to(device).eval()

    def preprocess(image: Image) -> torch.Tensor:
        # The processor returns a batch of 1; drop that axis so this matches the
        # open_clip transform signature and the Dataset can stack samples.
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    def encode_image(batch: torch.Tensor) -> torch.Tensor:
        # DINOv2 emits one token per image patch plus a leading CLS token.
        # We take CLS (index 0): it is the token the self-supervised objective
        # trained to summarise the whole image, and it is what the DINOv2
        # paper's own retrieval benchmarks use. Mean-pooling the patch tokens
        # is the documented alternative and scores slightly worse on
        # instance-level retrieval, which is exactly our near-dupe use case.
        return model(pixel_values=batch).last_hidden_state[:, 0]

    return ImageEncoder(
        spec=spec,
        model=model,
        preprocess=preprocess,
        device=device,
        encode_image=encode_image,
    )


def load_encoder(model_key: str, device: torch.device | None = None) -> ImageEncoder:
    """Load one encoder by `model_key`. Weights come from the local HF cache.

    Call `Settings.apply_hf_env()` before this so the cache lands inside the
    project rather than in the user profile.
    """
    if model_key not in ENCODER_SPECS:
        known = ", ".join(sorted(ENCODER_SPECS))
        raise KeyError(f"unknown model_key {model_key!r}; known keys: {known}")

    spec = ENCODER_SPECS[model_key]
    device = device or pick_device()
    logger.info("loading %s (%s) onto %s", spec.key, spec.checkpoint, device)

    loader = _load_siglip2 if spec.family == "open_clip" else _load_dinov2
    return loader(spec, device)
