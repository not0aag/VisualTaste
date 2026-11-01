"""Render the README's result charts from the cached simulator sessions.

Every number comes from data/eval/<study>/<variant>/<rung>/seed_*.json, the
same cache docs/EVALUATION.md is computed from. Run after the studies:

    python scripts/readme_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "docs" / "images"

# Chart tokens. Series colours were checked with a CVD/contrast validator
# against the surface below (lightness band, chroma, deutan/tritan separation).
SURFACE = "#0a0f18"
TEXT = "#e2eaf3"
TEXT_2 = "#a3b2c6"
MUTED = "#55657e"
GRID = "#1b2433"
SERIES = {"r4": "#139bb4", "r3": "#8576e6", "r0": "#c98500"}
NAMES = {
    "r4": "R4 · Thompson sampling",
    "r3": "R3 · logistic regression",
    "r0": "R0 · cold start",
}
ORDER = ("r4", "r3", "r0")


def font(*names: str) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in available), "DejaVu Sans")


SANS = font("Segoe UI", "Helvetica Neue", "Arial")
MONO = font("Consolas", "Menlo", "DejaVu Sans Mono")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": SANS,
        "font.size": 10.5,
        "text.color": TEXT,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.linewidth": 1,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_2,
        "legend.fontsize": 9.5,
    }
)


def sessions(study: str, variant: str, rung: str) -> list[dict]:
    paths = sorted((EVAL / study / variant / rung).glob("seed_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def title(fig, ax, heading: str, sub: str) -> None:
    ax.set_title(heading, loc="left", fontsize=15, fontweight="bold", color=TEXT, pad=30)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=9.5, color=TEXT_2, va="bottom")


def footer(fig, text: str) -> None:
    fig.text(0.012, 0.02, text, fontsize=8, color=MUTED, family=MONO)


def end_label(ax, x, y, rung: str, dy: float = 0.0) -> None:
    """Direct label at a line's end: a coloured dot beside text in text ink."""
    ax.plot([x], [y], "o", ms=6, color=SERIES[rung], mec=SURFACE, mew=2, zorder=5, clip_on=False)
    ax.annotate(
        NAMES[rung].split(" · ")[0],
        (x, y),
        xytext=(9, dy),
        textcoords="offset points",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=TEXT,
    )


# --- 1. Interests found over a session (S6, six planted interests) ----------


def chart_discovery() -> None:
    swipes = np.arange(0, 1001)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    labels = []
    for rung in ORDER:
        runs = sessions("phase10_h3", "S6_six_interests", rung)
        found = np.array(
            [
                [
                    sum(1 for t in s["discovery"].values() if t is not None and t <= x)
                    for x in swipes
                ]
                for s in runs
            ]
        )
        mean = found.mean(axis=0)
        ax.plot(
            swipes,
            mean,
            color=SERIES[rung],
            lw=2,
            label=f"{NAMES[rung]}  (n = {len(runs)} seeds)",
            drawstyle="steps-post",
        )
        labels.append((rung, mean[-1]))
    for rung, y in labels:
        end_label(ax, 1000, y, rung, dy={"r4": 8, "r0": -8}.get(rung, 0))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 6.3)
    ax.set_yticks(range(7))
    ax.set_xlabel("swipes")
    ax.set_ylabel("planted interests found (of 6)")
    title(
        fig,
        ax,
        "Greedy learning stops finding interests. Exploration doesn't.",
        "Simulated users with six hidden interests. R0 shows everything evenly; "
        "R4 keeps that reach while learning.",
    )
    ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0.02))
    footer(fig, "SIMULATED · phase10_h3 · S6_six_interests · 20 seeds × 1,000 swipes")
    fig.subplots_adjust(left=0.07, right=0.9, top=0.84, bottom=0.17)
    fig.savefig(OUT / "chart-discovery.png")
    plt.close(fig)


# --- 2. The trade-off: like rate against coverage, every session -------------


def chart_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    centres = {}
    for rung in ORDER[::-1]:
        runs = [
            s
            for v in ("S1_semantic", "S5_drift", "S6_six_interests")
            for s in sessions("phase10_h3", v, rung)
        ]
        x = np.array([s["like_rate"] for s in runs])
        y = np.array([s["coverage_last_500"] for s in runs])
        ax.scatter(
            x,
            y,
            s=46,
            color=SERIES[rung],
            edgecolors=SURFACE,
            linewidths=1.6,
            alpha=0.95,
            label=f"{NAMES[rung]}  (n = {len(runs)})",
            zorder=3,
        )
        centres[rung] = (x.mean(), y.mean())
    offsets = {"r4": (-20, 26), "r3": (60, 95), "r0": (26, 24)}
    for rung, (cx, cy) in centres.items():
        ax.annotate(
            NAMES[rung].split(" · ")[0],
            (cx, cy),
            xytext=offsets[rung],
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            color=TEXT,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )
    ax.set_xlabel("like rate over the session")
    ax.set_ylabel("coverage of the last 500 cards")
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    title(
        fig,
        ax,
        "The exploration trade-off, session by session",
        "Each dot is one simulated session. R3 maximises likes by narrowing; "
        "R4 keeps the collection open.",
    )
    ax.legend(loc="lower left")
    footer(
        fig, "SIMULATED · phase10_h3 · S1_semantic + S5_drift + S6_six_interests · 20 seeds each"
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.17)
    fig.savefig(OUT / "chart-tradeoff.png")
    plt.close(fig)


# --- 3. Like rate as the model learns (S1) -----------------------------------


def chart_learning() -> None:
    window = 100
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    ends = []
    for rung in ORDER:
        runs = sessions("phase10_h3", "S1_semantic", rung)
        liked = np.array([[a in ("like", "save") for a in s["actions"]] for s in runs], dtype=float)
        kernel = np.ones(window) / window
        rolling = np.array([np.convolve(row, kernel, mode="valid") for row in liked])
        x = np.arange(window, liked.shape[1] + 1)
        mean = rolling.mean(axis=0)
        lo, hi = np.percentile(rolling, [25, 75], axis=0)
        ax.fill_between(x, lo, hi, color=SERIES[rung], alpha=0.12, lw=0)
        ax.plot(x, mean, color=SERIES[rung], lw=2, label=f"{NAMES[rung]}  (n = {len(runs)} seeds)")
        ends.append((rung, x[-1], mean[-1]))
    for rung, x, y in ends:
        end_label(ax, x, y, rung)
    ax.set_xlim(window, 1000)
    ax.set_ylim(0.3, 0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("swipes")
    ax.set_ylabel(f"like rate (rolling {window})")
    title(
        fig,
        ax,
        "Learning lifts the like rate; exploring spends some of it",
        "Like rate of simulated users as each model learns. "
        "Mean across seeds; band = middle 50% of seeds.",
    )
    ax.legend(loc="upper left", ncol=3)
    footer(fig, "SIMULATED · phase10_h3 · S1_semantic · 20 seeds × 1,000 swipes")
    fig.subplots_adjust(left=0.08, right=0.9, top=0.84, bottom=0.17)
    fig.savefig(OUT / "chart-learning.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chart_discovery()
    chart_tradeoff()
    chart_learning()
    for name in ("chart-discovery", "chart-tradeoff", "chart-learning"):
        print(name, f"{(OUT / f'{name}.png').stat().st_size / 1e3:.0f} kB")
