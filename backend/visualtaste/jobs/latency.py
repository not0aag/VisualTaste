"""Measure `/feed/next` end to end: the 50 ms p95 budget, per rung and label count.

    python -m visualtaste.jobs.latency --rungs r0,r1,r3,r4 --labels 0,100,1000,3000

Part 2 §4.5: 500 requests through FastAPI's `TestClient` at 0, 100, 1,000 and
3,000 labels, for each rung that may serve automatically (R2 is measured and
reported but exempt, F3). The number that matters is the p95 of the whole
request -- candidate set, scoring, re-ranking, explanation, impression writes,
JSON -- because that is what the person swiping waits for.

How a label count is produced: a simulated profile (flagged `is_simulated`, so
it never mixes with real data) gets N events on random pool images, with
verdicts from a planted-taste user so both classes are present. The rung is
fitted on them and swapped in as the serving model; then the requests run.
Background work (R4's propensity estimate, about a second) runs after the
response in production. `TestClient` runs it before returning, so it is timed
on its own and subtracted, and reported in a separate column.

The benchmark runs on a temporary copy of the database, so its simulated
profiles never reach the real one.

Results append to `docs/BENCHMARKS.md` with the command, date and machine.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REQUESTS = 500
BUDGET_MS = 50.0


def _seed_profile(state, connection, labels: int, seed: int) -> int:
    """A simulated profile with `labels` events from a planted-taste user."""
    from visualtaste.eval.sim import create_simulated_profile, make_variant_user

    features = state.require_features()
    profile_id = create_simulated_profile(connection, note=f"latency benchmark {labels} labels")
    if labels:
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(features), labels, replace=False)
        user = make_variant_user(
            "S1_semantic", features, seed=seed, cluster_labels=features.cluster64
        )
        actions = user.react_rows(features, rows)
        connection.executemany(
            "INSERT INTO events (profile_id, image_id, action) VALUES (?, ?, ?)",
            [(profile_id, features.image_ids[r], a) for r, a in zip(rows, actions, strict=True)],
        )
    connection.commit()
    return profile_id


def measure(rung: str, labels: int, requests: int, seed: int, settings=None) -> dict[str, float]:
    from fastapi.testclient import TestClient

    from visualtaste.api import deps
    from visualtaste.api.deps import AppState, serving_rung
    from visualtaste.api.main import app
    from visualtaste.api.routes import feed as feed_route
    from visualtaste.config import get_settings
    from visualtaste.rec.labels import build_label_set, label_weights

    settings = settings or get_settings()
    background_ms: list[float] = []
    background: list[float] = []
    estimate = feed_route._estimate_propensities

    def timed_estimate(*args, **kwargs):
        started = time.perf_counter()
        try:
            return estimate(*args, **kwargs)
        finally:
            background_ms.append((time.perf_counter() - started) * 1_000)

    state = AppState(settings)
    deps.set_state(state)
    with TestClient(app) as client:
        connection = state.connect()
        try:
            profile_id = _seed_profile(state, connection, labels, seed)
            state.profile_id = profile_id
            model = serving_rung(rung, settings)
            model.fit(
                build_label_set(
                    connection, profile_id, weights=label_weights(settings.skip_weight)
                ),
                state.require_features(),
            )
        finally:
            connection.close()
        state.swap(model, f"latency_{rung}_{labels}")
        state.experiment = None

        client.get("/api/feed/next")  # warm-up: first-request allocations
        timings = []
        feed_route._estimate_propensities = timed_estimate
        try:
            for _ in range(requests):
                background_ms.clear()
                started = time.perf_counter()
                response = client.get("/api/feed/next")
                elapsed = (time.perf_counter() - started) * 1_000
                timings.append(elapsed - sum(background_ms))
                if response.status_code != 200:
                    raise RuntimeError(f"{rung} at {labels} labels: HTTP {response.status_code}")
                if background_ms:
                    background.append(sum(background_ms))
        finally:
            feed_route._estimate_propensities = estimate
        served = response.json()["rung"]

    array = np.array(timings)
    return {
        "background": float(np.median(background)) if background else 0.0,
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "served": served,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", default="r0,r1,r3,r4")
    parser.add_argument("--labels", default="0,100,1000,3000")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    import shutil
    import tempfile

    from visualtaste.config import get_settings

    settings = get_settings()
    scratch = Path(tempfile.mkdtemp(prefix="vt_latency_"))
    copy = scratch / "visualtaste.db"
    shutil.copy2(settings.db_path, copy)
    settings = settings.model_copy(update={"db_file": copy})

    rows = []
    for rung in [r.strip() for r in args.rungs.split(",") if r.strip()]:
        for labels in [int(n) for n in args.labels.split(",")]:
            result = measure(rung, labels, args.requests, seed=1729 + labels, settings=settings)
            exempt = rung == "r2"
            ok = result["p95"] < BUDGET_MS
            rows.append(
                f"| `{rung}` | {labels:,} | `{result['served']}` | {result['p50']:.1f} "
                f"| {result['p95']:.1f} | {result['max']:.1f} "
                f"| {result['background']:.0f} "
                f"| {'exempt (F3)' if exempt else 'yes' if ok else '**no**'} |"
            )
            print(rows[-1], flush=True)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    section = "\n".join(
        [
            "",
            f"## `/feed/next` latency ({now})",
            "",
            f"```\npython -m visualtaste.jobs.latency --rungs {args.rungs} --labels {args.labels} "
            f"--requests {args.requests}\n```",
            "",
            f"{args.requests} requests per row through `TestClient`, end to end, after one "
            f"warm-up request. Machine: {platform.processor() or platform.machine()}, "
            f'Python {platform.python_version()}. "served" is the rung that actually served '
            "the batches (below its label minimum a rung falls back to R0). Budget: p95 < 50 ms.",
            "",
            "| rung | labels | served | p50 ms | p95 ms | max ms | after response, p50 ms "
            "| within budget |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )
    shutil.rmtree(scratch, ignore_errors=True)
    if args.print_only:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(section)
    else:
        path = get_settings().project_root / "docs" / "BENCHMARKS.md"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
