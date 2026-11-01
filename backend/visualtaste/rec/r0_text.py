"""R0, aimed by a description: the cold start for someone who said what they like.

Plain R0 covers the collection evenly, which is right when nothing is known.
When the person has described their taste in a few words ("foggy forests, old
maps"), some of the first cards can start there instead. Each batch then has:

- `text_share` of its cards (1 of 5 by default) chosen from the images whose
  SigLIP 2 vectors are closest to the description, diversified with MMR so
  they are not five versions of the same picture;
- the rest from plain R0, so the collection is still covered and an interest
  the description missed can still be found.

It serves only while the learned rung cannot (as R0 does), so the description
shapes the first few dozen cards and then the person's own verdicts take over.
"""

from __future__ import annotations

import numpy as np

from visualtaste.rec.base import Explanation, FeatureBundle, Recommender
from visualtaste.rec.r0_coldstart import ColdStartRecommender
from visualtaste.rec.rerank import MMR_LAMBDA, mmr_select

#: Share of each cold-start batch aimed at the description: 20%, measured in
#: `phase15_text` (DECISIONS #30). 60% raised the like rate more but found
#: fewer of the user's other interests; 20% kept them all in S1.
TEXT_SHARE = 0.2
#: MMR picks from this many of the closest candidates.
TEXT_SHORTLIST = 200


class TextSeededColdStart(Recommender):
    rung = "r0+text"

    def __init__(
        self,
        base: ColdStartRecommender,
        text: str,
        vector: np.ndarray,
        *,
        text_share: float = TEXT_SHARE,
    ) -> None:
        super().__init__()
        self.base = base
        self.text = text
        self.vector = np.asarray(vector, dtype=np.float32)
        self.text_share = text_share
        self.is_fitted = True
        self._text_rows: dict[int, float] = {}

    def _fit(self, labels, features) -> None:
        """Nothing to learn, like R0."""

    def score(self, features: FeatureBundle, candidates: np.ndarray) -> np.ndarray:
        return self.base.score(features, candidates)

    def rerank(self, features: FeatureBundle, candidates: np.ndarray, scores: np.ndarray, n: int):
        """`text_share` of the batch near the description, the rest from R0."""
        n = min(n, len(candidates))
        n_text = min(n, round(n * self.text_share))
        # Score every row, then index: the gather costs more than the product.
        similarity = (features.siglip @ self.vector)[candidates]
        take = min(TEXT_SHORTLIST, len(candidates))
        shortlist = np.argpartition(-similarity, take - 1)[:take]
        picks = mmr_select(
            similarity[shortlist].astype(np.float64),
            features.siglip[candidates[shortlist]],
            n_text,
            lam=MMR_LAMBDA,
        )
        text_offsets = [int(shortlist[i]) for i in picks]

        rest = np.array(scores, dtype=np.float64)
        rest[text_offsets] = -np.inf
        base_offsets = [int(i) for i in np.argsort(-rest)[: n - len(text_offsets)]]

        self._text_rows = {int(candidates[o]): float(similarity[o]) for o in text_offsets}
        # Interleave, so the described images do not all arrive first.
        chosen: list[int] = []
        for k in range(max(len(text_offsets), len(base_offsets))):
            if k < len(text_offsets):
                chosen.append(text_offsets[k])
            if k < len(base_offsets):
                chosen.append(base_offsets[k])
        return np.array(chosen, dtype=np.int64), None, None

    def note_shown(self, cluster: int | None) -> None:
        self.base.note_shown(cluster)

    def explain(self, features: FeatureBundle, image_index: int) -> Explanation:
        similarity = self._text_rows.get(int(image_index))
        if similarity is None:
            return self.base.explain(features, image_index)
        return Explanation(
            reason="text_seed",
            detail={"text": self.text, "similarity": round(similarity, 4)},
        )

    @property
    def params(self) -> dict[str, object]:
        return {"rung": self.rung, "text": self.text, "text_share": self.text_share}
