"""Words into the image space: text search, and a described taste to start from.

SigLIP 2 was trained to put a caption and its image close together, so a short
description embeds into the same 768-dimensional space as the photographs.
Two things use that:

- **Search** (`GET /api/search`): rank the pool by cosine to a query.
- **A described taste** (`/api/profile/seed-text`): a few words saved per
  profile, which `rec/r0_text.TextSeededColdStart` uses to aim part of the cold
  start while there is nothing learned yet.

The text tower is loaded on first use, not at startup: most sessions never
search, and the model costs about a gigabyte of memory and a while to load.
After that a query takes about 40 ms. Text-image cosines are systematically
lower than image-image ones (the modality gap), so scores here are only
compared with each other.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from visualtaste.rec.concepts import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

#: Longer descriptions are cut: SigLIP's text tower reads 64 tokens.
MAX_TEXT_CHARACTERS = 200


class TextEncoder:
    """SigLIP 2's text tower, loaded once on first use, thread-safe."""

    def __init__(self, factory: Callable[[], object] | None = None) -> None:
        self._factory = factory
        self._encoder = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._encoder is None:
                if self._factory is None:
                    from visualtaste.vision.encoders import SIGLIP2_KEY, load_encoder

                    self._factory = lambda: load_encoder(SIGLIP2_KEY)
                logger.info("loading the SigLIP 2 text encoder (first text query)")
                self._encoder = self._factory()
        return self._encoder

    def embed(self, text: str) -> np.ndarray:
        """A unit vector for `text`, through SigLIP's own prompt template."""
        return _cached_embed(self, clean(text))

    def embed_uncached(self, text: str) -> np.ndarray:
        encoder = self._load()
        vector = encoder.encode_texts([PROMPT_TEMPLATE.format(text)])[0]
        return np.asarray(vector, dtype=np.float32)


@lru_cache(maxsize=256)
def _cached_embed(encoder: TextEncoder, text: str) -> np.ndarray:
    vector = encoder.embed_uncached(text)
    vector.setflags(write=False)
    return vector


def clean(text: str) -> str:
    """Trimmed, single-spaced, and short enough for the text tower."""
    text = " ".join(text.split())[:MAX_TEXT_CHARACTERS]
    if not text:
        raise ValueError("the text is empty")
    return text


def search(
    pool: np.ndarray, query: np.ndarray, k: int, exclude: set[int] | None = None
) -> list[tuple[int, float]]:
    """The `k` pool rows closest to `query`, best first, as (row, cosine)."""
    scores = pool @ query
    if exclude:
        scores = scores.copy()
        scores[list(exclude)] = -np.inf
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(row), float(scores[row])) for row in top if np.isfinite(scores[row])]


# --- A described taste, saved per profile -----------------------------------


def seed_path(data_dir: Path, profile_id: int) -> Path:
    return data_dir / "profile_seeds" / f"{profile_id}.json"


def save_seed(data_dir: Path, profile_id: int, text: str, vector: np.ndarray) -> None:
    """Saved with its vector, so a restart does not need the text encoder."""
    path = seed_path(data_dir, profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(
        json.dumps({"text": text, "vector": [round(float(v), 6) for v in vector]}),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_seed(data_dir: Path, profile_id: int) -> tuple[str, np.ndarray] | None:
    path = seed_path(data_dir, profile_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["text"], np.asarray(data["vector"], dtype=np.float32)


def clear_seed(data_dir: Path, profile_id: int) -> bool:
    path = seed_path(data_dir, profile_id)
    if path.exists():
        path.unlink()
        return True
    return False
