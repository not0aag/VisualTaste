"""Render the README's result charts from the cached simulator sessions.

Every number comes from data/eval/<study>/<variant>/<rung>/seed_*.json, the
same cache docs/EVALUATION.md is computed from. Run after the studies:

    python scripts/readme_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.patches

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
        runs = sessions("phase14_final", "S6_six_interests", rung)
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
    footer(fig, "SIMULATED · phase14_final · S6_six_interests · 20 seeds × 1,000 swipes")
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
            for s in sessions("phase14_final", v, rung)
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
        fig, "SIMULATED · phase14_final · S1_semantic + S5_drift + S6_six_interests · 20 seeds each"
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
        runs = sessions("phase14_final", "S1_semantic", rung)
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
    footer(fig, "SIMULATED · phase14_final · S1_semantic · 20 seeds × 1,000 swipes")
    fig.subplots_adjust(left=0.08, right=0.9, top=0.84, bottom=0.17)
    fig.savefig(OUT / "chart-learning.png")
    plt.close(fig)


# --- 4. Scoreboard: every rung on every scenario (phase14_final) ------------

SCENARIOS = (
    ("S1_semantic", "S1 subject"),
    ("S2_appearance", "S2 look"),
    ("S3_colour", "S3 colour"),
    ("S4_conjunctive", "S4 both"),
    ("S5_drift", "S5 drift"),
    ("S6_six_interests", "S6 six"),
)
BOARD_RUNGS = (
    ("r0", "r0", "R0 cold start"),
    ("r1", "r1", "R1 Rocchio"),
    ("r2", "r2", "R2 kNN"),
    ("r3", "r3", "R3 logistic regression"),
    ("r3-r5", "r3+r5", "R3+R5"),
    ("r4", "r4", "R4 Thompson sampling"),
    ("r4-r5", "r4+r5", "R4+R5 (default)"),
)


def _ramp(hex_low: str, hex_high: str, value: float) -> tuple[float, float, float]:
    low = np.array(matplotlib.colors.to_rgb(hex_low))
    high = np.array(matplotlib.colors.to_rgb(hex_high))
    return tuple(low + (high - low) * float(np.clip(value, 0, 1)))


def _ink(rgb) -> str:
    r, g, b = rgb
    return "#05080d" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else TEXT


def chart_scoreboard() -> None:
    like = np.full((len(BOARD_RUNGS), len(SCENARIOS)), np.nan)
    found = np.full_like(like, np.nan)
    for i, (folder, _, _) in enumerate(BOARD_RUNGS):
        for j, (variant, _) in enumerate(SCENARIOS):
            runs = sessions("phase14_final", variant, folder)
            if runs:
                like[i, j] = np.mean([r["like_rate"] for r in runs])
                found[i, j] = np.mean([bool(r["all_found"]) for r in runs])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), dpi=200, gridspec_kw={"wspace": 0.06})
    panels = (
        (axes[0], like, "Like rate", "#0d2a33", "#3fd8f0", 0.42, 0.72, "{:.2f}"),
        (axes[1], found, "Found every interest", "#1c1838", "#a594f9", 0.0, 1.0, "{:.0%}"),
    )
    for ax, grid, heading, low, high, vmin, vmax, fmt in panels:
        ax.set_xlim(0, len(SCENARIOS))
        ax.set_ylim(len(BOARD_RUNGS), 0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(BOARD_RUNGS)):
            for j in range(len(SCENARIOS)):
                value = grid[i, j]
                rgb = _ramp(low, high, (value - vmin) / (vmax - vmin))
                ax.add_patch(
                    matplotlib.patches.FancyBboxPatch(
                        (j + 0.04, i + 0.06),
                        0.92,
                        0.88,
                        boxstyle="round,pad=0,rounding_size=0.08",
                        fc=rgb,
                        ec=SURFACE,
                        lw=0,
                    )
                )
                ax.text(
                    j + 0.5,
                    i + 0.52,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color=_ink(rgb),
                )
        ax.set_xticks(
            np.arange(len(SCENARIOS)) + 0.5,
            [label for _, label in SCENARIOS],
            fontsize=8.5,
            color=TEXT_2,
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title(heading, loc="left", fontsize=12.5, fontweight="bold", color=TEXT, pad=40)
    axes[0].set_yticks(
        np.arange(len(BOARD_RUNGS)) + 0.5,
        [name for *_, name in BOARD_RUNGS],
        fontsize=10,
        color=TEXT,
    )
    axes[1].set_yticks([])
    fig.suptitle(
        "Every recommender on every planted taste",
        x=0.012,
        y=1.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.012,
        0.955,
        "Mean of 20 seeds × 1,000 swipes per cell. Brighter is better. "
        "R3 likes most; R4 and R4+R5 never miss an interest.",
        fontsize=9.5,
        color=TEXT_2,
    )
    footer(fig, "SIMULATED · phase14_final · docs/EVALUATION.md")
    fig.subplots_adjust(left=0.15, right=0.99, top=0.80, bottom=0.08)
    fig.savefig(OUT / "chart-scoreboard.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# --- 5. Closing R4's gap: like rate against coverage (phase15_explore) -------

EXPLORE_POINTS = (
    ("r0", "R0", "r0"),
    ("r3", "R3", "r3"),
    ("r4", "R4", "r4"),
    ("r4__tau-0.05", "cooler", None),
    ("r4__tau_halflife-200", "fading τ", None),
    ("r4__mmr_top-50", "narrow MMR", None),
    ("r4__mmr-0.9", "MMR 0.9", None),
    ("r4__tau_halflife-200_mmr_top-50_mmr-0.9", "all three", "best"),
)


def chart_explore() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.8), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for ax, (variant, label) in zip(
        axes,
        (("S1_semantic", "S1 · three interests"), ("S6_six_interests", "S6 · six interests")),
        strict=True,
    ):
        points = {}
        for folder, name, kind in EXPLORE_POINTS:
            runs = sessions("phase15_explore", variant, folder)
            x = np.mean([r["coverage_last_500"] for r in runs])
            y = np.mean([r["like_rate"] for r in runs])
            points[folder] = (x, y)
            if kind in ("r0", "r3", "r4"):
                colour, size, ink = SERIES[kind], 70, TEXT
            elif kind == "best":
                colour, size, ink = "#3ddc97", 120, TEXT
            else:
                colour, size, ink = "#3a5f7d", 46, TEXT_2
            ax.scatter(
                [x],
                [y],
                s=size,
                color=colour,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=4,
                label="one setting alone"
                if kind is None and not ax.get_legend_handles_labels()[1]
                else None,
            )
            if kind is not None:
                offset = (-62, 10) if kind == "best" else (8, 6)
                ax.annotate(
                    name,
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=ink,
                )
        r4, best = points["r4"], points[EXPLORE_POINTS[-1][0]]
        ax.annotate(
            "",
            xy=best,
            xytext=r4,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#3ddc97",
                "lw": 1.6,
                "shrinkA": 7,
                "shrinkB": 9,
            },
        )
        ax.set_title(label, loc="left", fontsize=11.5, fontweight="bold", color=TEXT, pad=8)
        ax.set_xlabel("coverage of the collection, last 500 swipes")
        ax.set_xlim(0.2, 1.08)
        ax.legend(loc="lower left", handletextpad=0.3)
    axes[0].set_ylabel("like rate")
    fig.suptitle(
        "Closing exploration's cost: higher is more likes, further right is more of the collection",
        x=0.012,
        y=1.03,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.012,
        0.955,
        "Mean of 20 seeds × 1,000 swipes. The three R4 settings together keep "
        "70% (S1) and 80% (S6) of R3's gain and still find every interest.",
        fontsize=9.5,
        color=TEXT_2,
    )
    footer(fig, "SIMULATED · phase15_explore · docs/EVALUATION.md")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.14)
    fig.savefig(OUT / "chart-explore.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chart_discovery()
    chart_tradeoff()
    chart_learning()
    chart_scoreboard()
    chart_explore()
    for name in (
        "chart-discovery",
        "chart-tradeoff",
        "chart-learning",
        "chart-scoreboard",
        "chart-explore",
    ):
        print(name, f"{(OUT / f'{name}.png').stat().st_size / 1e3:.0f} kB")
