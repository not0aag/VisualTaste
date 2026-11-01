"""Capture the GitHub Pages demo from a running sandbox API.

The demo (frontend `VITE_DEMO=1`, `src/demo/server.ts`) replays what a real
backend answered. This script asks a running API for every response the pages
need, records the model's real feed picks batch by batch, and copies the image
files those responses refer to.

Run it against a SANDBOX copy of the database, never the real one: recording
the feed posts events (skips) to mark each card as seen.

    python scripts/capture_demo.py --api http://127.0.0.1:8003 --out demo-assets

Unsplash images are left out, because the Unsplash Lite dataset terms do not
clearly allow redistributing the files; every other source is CC0, public
domain or CC BY (credited on every card).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from visualtaste.config import get_settings  # noqa: E402

EXCLUDED_SOURCES = {"unsplash_lite"}
MODELS = ("siglip2_b16_224", "dinov2_b14")
RETRAIN_EVERY = 20

#: Every GET the pages make with a fixed path, exactly as client.ts builds it.
STATIC_ROUTES = [
    "/api/health",
    "/api/profile",
    "/api/profile/taste",
    "/api/stats/summary",
    "/api/stats/learning-curve?window=50",
    "/api/stats/diversity",
    "/api/stats/calibration",
    "/api/stats/exploration",
    "/api/stats/scores",
    "/api/models",
    "/api/experiments",
    "/api/eval/studies",
    "/api/review",
    *(f"/api/viz/projection?model={m}" for m in MODELS),
    *(f"/api/viz/taste-path?model={m}&every=50" for m in MODELS),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8003")
    parser.add_argument("--db", required=True, help="the sandbox database the API serves")
    parser.add_argument("--out", type=Path, default=ROOT / "demo-assets")
    parser.add_argument("--cards", type=int, default=300)
    parser.add_argument("--history", type=int, default=200)
    parser.add_argument("--similar", type=int, default=48)
    args = parser.parse_args()

    settings = get_settings()
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    source_of = dict(db.execute("SELECT image_id, source FROM images"))
    allowed = lambda image_id: source_of.get(image_id) not in EXCLUDED_SOURCES  # noqa: E731

    client = httpx.Client(base_url=args.api, timeout=120)
    get = lambda path: client.get(path).raise_for_status().json()  # noqa: E731

    # Pages that only read: captured as they are, before any swipes.
    routes: dict[str, object] = {path: get(path) for path in STATIC_ROUTES}
    # The demo shows one profile; an empty list hides the simulated-profile picker.
    routes["/api/profiles?include_simulated=true"] = []

    history: list[dict] = []
    offset = 0
    while len(history) < args.history:  # the endpoint pages at most 200 at a time
        page = get(f"/api/history?action=all&limit=200&offset={offset}")
        history += [i for i in page["items"] if allowed(i["image"]["image_id"])]
        offset += len(page["items"])
        if not page["items"] or offset >= page["total"]:
            break
    history = history[: args.history]

    similar = {}
    for item in history[: args.similar]:
        image_id = item["image"]["image_id"]
        found = get(f"/api/images/{image_id}/similar?k=12")
        found["neighbours"] = [n for n in found["neighbours"] if allowed(n["image"]["image_id"])]
        similar[image_id] = found

    # The feed: the model's real picks, in order. Each card is marked seen with
    # a skip, the lightest verdict, so the next batch is new.
    feed, model_version, rung = [], "", ""
    while len(feed) < args.cards:
        batch = get("/api/feed/next?n=5")
        if not batch["cards"]:
            break
        model_version, rung = batch["model_version"], batch["rung"]
        for card in batch["cards"]:
            client.post(
                "/api/events",
                json={
                    "image_id": card["image"]["image_id"],
                    "action": "skip",
                    "impression_id": card["impression_id"],
                },
            ).raise_for_status()
            if allowed(card["image"]["image_id"]):
                feed.append(card)
    feed = feed[: args.cards]

    fixtures = {
        "routes": routes,
        "feed": feed,
        "model_version": model_version,
        "rung": rung,
        "history": history,
        "similar": similar,
        "retrain_every": RETRAIN_EVERY,
    }

    # The files every captured response points at.
    displays = {c["image"]["image_id"] for c in feed}
    thumbs = set(displays)
    thumbs |= {i["image"]["image_id"] for i in history}
    thumbs |= {n["image"]["image_id"] for s in similar.values() for n in s["neighbours"]}
    for card in routes["/api/profile/taste"]["interests"]:
        # Unbundled images would show as broken tiles, so the cards lose them.
        card["central_image_ids"] = [i for i in card["central_image_ids"] if allowed(i)]
        thumbs |= set(card["central_image_ids"])

    out = args.out
    shutil.rmtree(out, ignore_errors=True)
    (out / "display").mkdir(parents=True)
    (out / "thumb").mkdir(parents=True)
    for image_id in displays:
        shutil.copyfile(settings.display_path(image_id), out / "display" / f"{image_id}.webp")
    for image_id in thumbs:
        shutil.copyfile(settings.thumb_path(image_id), out / "thumb" / f"{image_id}.webp")
    (out / "fixtures.json").write_text(json.dumps(fixtures, separators=(",", ":")), "utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(
        f"{len(feed)} feed cards, {len(history)} history items, {len(similar)} similar sets, "
        f"{len(displays)} display + {len(thumbs)} thumb files, {size:.1f} MB -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
