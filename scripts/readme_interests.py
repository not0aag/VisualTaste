"""Render docs/images/interests-found.jpg: what one session discovered.

One recorded session from the final study (S6: six hidden interests, served
by the default R4+R5). The simulated user is rebuilt from its seed, and the
script checks it is the same user by recomputing the like probability of all
1,000 shown images against the cached ones. Then, for each hidden interest:
the swipe at which it counted as found (5 likes), its three nearest
vocabulary words, and the liked images closest to it.

    python scripts/readme_interests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readme_strips import (
    MONO,
    MUTED,
    OUT,
    SANS,
    SANS_B,
    SURFACE,
    TEXT,
    TEXT_2,
    rounded,
    spaced,
    thumb,
)

ROOT = Path(__file__).resolve().parents[1]
VARIANT, SPEC_DIR, SEED = "S6_six_interests", "r4-r5", 1748
SESSION = ROOT / "data" / "eval" / "phase14_final" / VARIANT / SPEC_DIR / f"seed_{SEED}.json"
PER_ROW = 9
#: One accent per interest row (the app's palette).
ACCENTS = [
    (63, 216, 240),
    (165, 148, 249),
    (61, 220, 151),
    (255, 193, 69),
    (255, 107, 129),
    (96, 165, 250),
]


def rebuild_user(features, session):
    from visualtaste.eval.sim import make_variant_user
    from visualtaste.eval.study import clusters_for_variant

    user = make_variant_user(
        VARIANT, features, seed=SEED, cluster_labels=clusters_for_variant(features, VARIANT, SEED)
    )
    user.offset = session["offset"]
    rows = np.array([features.position(i) for i in session["image_ids"]])
    probabilities = user.like_probability(features.siglip[rows])
    gap = float(np.max(np.abs(probabilities - np.array(session["true_probabilities"]))))
    if gap > 1e-5:
        raise SystemExit(f"rebuilt user does not match the session (max gap {gap:.2e})")
    return user, rows


def main() -> None:
    from visualtaste.api.deps import AppState
    from visualtaste.config import get_settings
    from visualtaste.rec.concepts import cache_for

    settings = get_settings()
    state = AppState(settings)
    state.load()
    features = state.require_features()
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    user, rows = rebuild_user(features, session)
    concepts, concept_vectors = cache_for(settings).load()

    liked = [
        (row, image_id)
        for row, image_id, action in zip(
            rows, session["image_ids"], session["actions"], strict=True
        )
        if action in ("like", "save")
    ]
    liked_rows = np.array([r for r, _ in liked])
    owner = user.best_interest(features.siglip[liked_rows])
    found = {int(k): int(v) for k, v in session["discovery"].items()}
    order = sorted(found, key=found.get)

    size, gap, pad, label_w = 112, 8, 44, 300
    width = pad * 2 + label_w + PER_ROW * size + (PER_ROW - 1) * gap
    row_h = size + 22
    height = pad + 84 + len(order) * row_h + 40
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)
    spaced(draw, (pad, pad - 6), "ONE SESSION, SIX HIDDEN INTERESTS", MONO(15), MUTED, 3)
    likes = len(liked)
    draw.text(
        (pad, pad + 16),
        f"All six found by swipe {max(found.values())}; {likes} likes in 1,000 swipes",
        font=SANS_B(30),
        fill=TEXT,
    )

    y = pad + 84
    for rank, k in enumerate(order):
        accent = ACCENTS[rank % len(ACCENTS)]
        draw.rounded_rectangle((pad, y + 6, pad + 5, y + size - 6), 3, fill=accent)
        spaced(draw, (pad + 20, y + 8), f"INTEREST {rank + 1}", MONO(13), accent, 2.5)
        words = [concepts[i] for i in np.argsort(-(concept_vectors @ user.centroids[k]))[:3]]
        line = ", ".join(words)
        fnt = SANS_B(21)
        while draw.textlength(line, font=fnt) > label_w - 36 and fnt.size > 14:
            fnt = SANS_B(fnt.size - 1)
        draw.text((pad + 20, y + 32), line, font=fnt, fill=TEXT)
        mine = np.flatnonzero(owner == k)
        draw.text(
            (pad + 20, y + 66),
            f"found at swipe {found[k]} · {len(mine)} likes",
            font=SANS(16),
            fill=TEXT_2,
        )
        closeness = features.siglip[liked_rows[mine]] @ user.centroids[k]
        x = pad + label_w
        for index in mine[np.argsort(-closeness)][:PER_ROW]:
            tile = rounded(thumb(liked[index][1], size), 10)
            canvas.paste(tile, (x, y), tile)
            x += size + gap
        y += row_h

    draw.text(
        (pad, height - 34),
        "SIMULATED user (phase14_final, S6, seed 1748), served by the default R4+R5. "
        "Words: the interest's three nearest vocabulary terms in SigLIP 2 space.",
        font=SANS(14),
        fill=MUTED,
    )
    canvas.save(OUT / "interests-found.jpg", quality=88, optimize=True)
    print("wrote", OUT / "interests-found.jpg", canvas.size)


if __name__ == "__main__":
    main()
