"""Draw the evaluation figures from cached study results into docs/figures/.

    python -m visualtaste.jobs.figures --study phase14_final

Reads only the JSON cache under `data/eval/<study>/`, so it never re-runs a
session. Per variant, two figures:

- `<study>_<variant>_outcomes.png`: like rate, interests found and coverage per
  rung, mean with a 90% bootstrap interval over seeds.
- `<study>_<variant>_curves.png`: the rolling like rate (window 100) and the
  coverage of the last 500 cards over the session, averaged over seeds.

Every figure is titled SIMULATED.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

METRICS = (
    ("like_rate", "like rate"),
    ("interests_found", "interests found"),
    ("coverage_last_500", "coverage (last 500)"),
)


def load_study(cache: Path, study: str) -> dict[tuple[str, str], list]:
    from visualtaste.eval.study import SeedResult

    results: dict[tuple[str, str], list] = {}
    for path in sorted((cache / study).glob("*/*/seed_*.json")):
        result = SeedResult.from_json(path.read_text(encoding="utf-8"))
        results.setdefault((result.variant, result.spec), []).append(result)
    return results


def _interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    means = values[rng.integers(0, len(values), size=(5_000, len(values)))].mean(axis=1)
    return float(values.mean()), *np.percentile(means, [5, 95]).tolist()


def draw(results, study: str, out: Path, cluster64: np.ndarray | None, positions) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    written: list[Path] = []
    for variant in sorted({v for v, _ in results}):
        specs = [s for v, s in results if v == variant]

        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4))
        for ax, (metric, label) in zip(axes, METRICS, strict=True):
            stats = [
                _interval(np.array([getattr(r, metric) for r in results[(variant, s)]], float), rng)
                for s in specs
            ]
            mean = [m for m, _, _ in stats]
            err = [[m - lo for m, lo, _ in stats], [hi - m for m, _, hi in stats]]
            ax.bar(specs, mean, yerr=err, capsize=3, color="#6b8fb5")
            ax.set_title(label)
            ax.tick_params(axis="x", rotation=40)
        fig.suptitle(f"SIMULATED · {study} · {variant} · mean and 90% CI over seeds")
        fig.tight_layout()
        path = out / f"{study}_{variant}_outcomes.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        for spec in specs:
            likes, covers = [], []
            for r in results[(variant, spec)]:
                liked = np.isin(r.actions, ["like", "save"]).astype(float)
                likes.append(np.convolve(liked, np.ones(100) / 100, mode="valid"))
                if cluster64 is not None:
                    clusters = cluster64[positions(r.image_ids)]
                    covers.append(
                        [
                            len(set(clusters[max(0, i - 500) : i])) / 64
                            for i in range(100, len(clusters) + 1)
                        ]
                    )
            n = min(len(x) for x in likes)
            axes[0].plot(np.mean([x[:n] for x in likes], axis=0), label=spec)
            if covers:
                m = min(len(x) for x in covers)
                axes[1].plot(np.mean([x[:m] for x in covers], axis=0), label=spec)
        axes[0].set(title="like rate (window 100)", xlabel="swipe")
        axes[1].set(title="coverage of the last 500 cards", xlabel="swipe", ylim=(0, 1.05))
        axes[0].legend(fontsize=7)
        fig.suptitle(f"SIMULATED · {study} · {variant} · averaged over seeds")
        fig.tight_layout()
        path = out / f"{study}_{variant}_curves.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    args = parser.parse_args(argv)

    from visualtaste.api.deps import AppState
    from visualtaste.config import get_settings

    settings = get_settings()
    state = AppState(settings)
    state.load()
    features = state.require_features()
    results = load_study(settings.data / "eval", args.study)
    if not results:
        print(f"no cached results for {args.study}", file=sys.stderr)
        return 2
    cluster64 = np.asarray(features.cluster64) if features.cluster64 is not None else None
    for path in draw(
        results, args.study, settings.docs_dir / "figures", cluster64, features.positions
    ):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
