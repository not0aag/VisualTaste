"""Render the README's photo composites from real data.

- learning-strip.png: the first and last 12 cards of one recorded simulator
  session, each with the verdict the simulated user gave it.
- collection-mosaic.png: one selected image from each of the 64 visual
  clusters (cluster64), chosen with a fixed seed.

    python scripts/readme_strips.py
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
THUMBS = ROOT / "data" / "derived" / "thumb"
OUT = ROOT / "docs" / "images"

SURFACE = (5, 8, 13)
PANEL = (10, 15, 24)
TEXT = (226, 234, 243)
TEXT_2 = (163, 178, 198)
MUTED = (85, 101, 126)
ACCENT = (63, 216, 240)
VERDICT = {
    "like": ((61, 220, 151), "LIKE"),
    "save": ((255, 193, 69), "SAVE"),
    "dislike": ((255, 107, 129), "NOPE"),
    "skip": ((123, 140, 165), "SKIP"),
}

SESSION = ROOT / "data" / "eval" / "phase10_h3" / "S1_semantic" / "r3" / "seed_1748.json"


def font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


SANS_B = lambda s: font(["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"], s)  # noqa: E731
SANS = lambda s: font(["segoeui.ttf", "arial.ttf"], s)  # noqa: E731
MONO = lambda s: font(["consola.ttf", "cour.ttf"], s)  # noqa: E731


def thumb(image_id: str, size: int) -> Image.Image:
    image = Image.open(THUMBS / image_id[:2] / f"{image_id}.webp").convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius, fill=255)
    out = Image.new("RGBA", image.size)
    out.paste(image, (0, 0), mask)
    return out


def spaced(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill, spacing: float) -> None:
    """Letter-spaced text, for the uppercase mono labels."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + spacing


def liked_pct(actions: list[str]) -> int:
    """Share of these verdicts that were likes or saves, as a whole percent."""
    return round(100 * sum(a in ("like", "save") for a in actions) / max(1, len(actions)))


def learning_strip() -> None:
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    ids, actions, rungs = session["image_ids"], session["actions"], session["served_rungs"]
    n = len(ids)
    rows = [
        (range(0, 12), "SWIPES 1–12", "Cold start: a deliberate spread across the collection"),
        (
            range(n - 12, n),
            f"SWIPES {n - 11:,}–{n:,}",
            f"After learning: served by {rungs[-1].upper()}",
        ),
    ]

    size, gap, pad = 112, 10, 40
    width = pad * 2 + 12 * size + 11 * gap
    row_h = 30 + size + 34
    height = pad + 54 + len(rows) * (row_h + 26) + 44
    canvas = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, pad - 6), "Watch it learn", font=SANS_B(30), fill=TEXT)
    sub = (
        f"One recorded session of a simulated user. Liked: {liked_pct(actions[:100])}% "
        f"of the first 100 cards, {liked_pct(actions[-100:])}% of the last 100."
    )
    draw.text((pad, pad + 34), sub, font=SANS(17), fill=TEXT_2)

    y = pad + 80
    for indices, label, caption in rows:
        chosen = list(indices)
        liked = sum(actions[i] in ("like", "save") for i in chosen)
        spaced(draw, (pad, y), label, MONO(15), ACCENT, 2.4)
        lw = draw.textlength(label, font=MONO(15)) + 2.4 * len(label)
        draw.text((pad + lw + 16, y - 3), caption, font=SANS(17), fill=TEXT_2)
        tally = f"{liked} of 12 liked"
        draw.text(
            (width - pad - draw.textlength(tally, font=SANS_B(17)), y - 3),
            tally,
            font=SANS_B(17),
            fill=TEXT,
        )
        top = y + 30
        for k, i in enumerate(chosen):
            x = pad + k * (size + gap)
            colour, word = VERDICT[actions[i]]
            canvas.paste(rounded(thumb(ids[i], size), 6), (x, top), rounded(thumb(ids[i], size), 6))
            if actions[i] not in ("like", "save"):
                # Dim what was passed over, so the likes stand out.
                veil = Image.new("RGBA", (size, size), (5, 8, 13, 120))
                canvas.paste(veil, (x, top), rounded(veil, 6))
            draw.rounded_rectangle((x, top + size + 8, x + size, top + size + 11), 2, fill=colour)
            spaced(draw, (x + 2, top + size + 15), word, MONO(12), colour, 1.6)
        y += row_h + 26

    foot = f"SIMULATED · phase10_h3 · S1_semantic · r3 · seed {session['seed']}"
    spaced(draw, (pad, height - 26), foot, MONO(12), MUTED, 1.2)
    canvas.save(OUT / "learning-strip.png", optimize=True)


def collection_mosaic() -> None:
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'db' / 'visualtaste.db'}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT image_id, cluster64 FROM images "
        "WHERE selected = 1 AND status = 'ok' AND cluster64 IS NOT NULL"
    ).fetchall()
    db.close()
    by_cluster: dict[int, list[str]] = {}
    for image_id, cluster in rows:
        by_cluster.setdefault(int(cluster), []).append(image_id)
    rng = random.Random(1729)

    def readable(image_id: str) -> bool:
        # Skip near-black or flat thumbnails: at mosaic size they read as broken tiles.
        from PIL import ImageStat

        stat = ImageStat.Stat(thumb(image_id, 48).convert("L"))
        return stat.mean[0] > 40 and stat.stddev[0] > 25

    picks = []
    for cluster in sorted(by_cluster):
        candidates = sorted(by_cluster[cluster])
        rng.shuffle(candidates)
        picks.append(next((c for c in candidates[:40] if readable(c)), candidates[0]))
    picks = picks[:64]

    cols, size, gap = 16, 104, 6
    rows_n = (len(picks) + cols - 1) // cols
    width = cols * size + (cols - 1) * gap
    height = rows_n * size + (rows_n - 1) * gap
    canvas = Image.new("RGB", (width, height), SURFACE)
    for k, image_id in enumerate(picks):
        x, y = (k % cols) * (size + gap), (k // cols) * (size + gap)
        tile = rounded(thumb(image_id, size), 5)
        canvas.paste(tile, (x, y), tile)
    canvas.save(OUT / "collection-mosaic.jpg", quality=86, optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    learning_strip()
    collection_mosaic()
    for name in ("learning-strip.png", "collection-mosaic.jpg"):
        print(name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")
