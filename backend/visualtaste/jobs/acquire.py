"""Download the candidate pool.

Run:  python -m visualtaste.jobs.acquire                      (all four sources)
      python -m visualtaste.jobs.acquire --source pd12m       (just one)
      python -m visualtaste.jobs.acquire --source pd12m --target 500   (a taste)

Targets come from brief §7.1 and add up to 80,000:

    megalith       30,000   public-domain Flickr, via the DrawThings mirror
    pd12m          25,000   museum and archive scans, stratified by institution
    open_images    15,000   CC BY 2.0 photographs, from the validation split
    unsplash_lite  10,000   contemporary photography, for stylistic range

The job is resumable at every level. Stopping it (Ctrl-C, a reboot, the disk
guard tripping) costs the current batch at most; re-running it continues where
it left off and downloads nothing it already has. That is the point of
`data/manifest.py`, and it is why this is safe to leave running unattended.

Each source is independent: one source being blocked or failing does not stop
the others. The run ends with a per-source summary, and a non-zero exit status
if any source failed more than 5% of its attempts -- the brief's §9 gate.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from visualtaste.config import DiskGuardTripped, Settings, get_settings
from visualtaste.data.http import FetchStats, PoliteClient, SourceBlocked
from visualtaste.data.manifest import Manifest
from visualtaste.data.sources import megalith, openimages, pd12m, unsplash_lite

logger = logging.getLogger(__name__)

# The brief's §9 Phase 2 gate: investigate a source that fails more than this.
MAX_ACCEPTABLE_FAILURE_RATE = 0.05

SOURCES = {
    megalith.SOURCE: megalith,
    pd12m.SOURCE: pd12m,
    openimages.SOURCE: openimages,
    unsplash_lite.SOURCE: unsplash_lite,
}

DEFAULT_TARGETS = {
    megalith.SOURCE: megalith.DEFAULT_TARGET,
    pd12m.SOURCE: pd12m.DEFAULT_TARGET,
    openimages.SOURCE: openimages.DEFAULT_TARGET,
    unsplash_lite.SOURCE: unsplash_lite.DEFAULT_TARGET,
}


@dataclass
class SourceOutcome:
    """What happened to one source in one run."""

    source: str
    target: int
    stats: FetchStats | None
    total_stored: int
    blocked: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        if self.blocked or self.error or self.stats is None:
            return False
        return self.stats.failure_rate <= MAX_ACCEPTABLE_FAILURE_RATE


async def acquire_source(
    settings: Settings, source: str, target: int, *, client: PoliteClient
) -> SourceOutcome:
    """Run one source to its target, converting its failure modes into a report.

    Exceptions are caught here rather than allowed to propagate: a dead source
    must not cost us the other three, and the manifest has already been flushed
    by the time anything is raised.
    """
    module = SOURCES[source]
    manifest = Manifest.load(settings.manifest_path(source), source)

    try:
        stats = await module.fetch(settings, manifest, target, client=client)
    except SourceBlocked as exc:
        # Brief §0.6: we do not work around a block. This is reported, and §3 Q4
        # is the owner's decision to make.
        logger.error("%s: BLOCKED -- %s", source, exc)
        return SourceOutcome(source, target, None, manifest.succeeded, blocked=str(exc))
    except DiskGuardTripped as exc:
        logger.error("%s: stopped by the disk guard -- %s", source, exc)
        return SourceOutcome(source, target, None, manifest.succeeded, error=str(exc))
    except Exception as exc:  # one source must not sink the whole run
        logger.exception("%s: unexpected failure", source)
        return SourceOutcome(
            source, target, None, manifest.succeeded, error=f"{type(exc).__name__}: {exc}"
        )

    return SourceOutcome(source, target, stats, manifest.succeeded)


async def acquire_all(settings: Settings, targets: dict[str, int]) -> list[SourceOutcome]:
    """Run each requested source in turn, sharing one polite HTTP client.

    Sequential rather than concurrent, deliberately: four sources downloading at
    once would multiply the concurrency limit by four, which is exactly the
    politeness budget the brief sets. It also keeps the log readable.
    """
    outcomes: list[SourceOutcome] = []
    async with PoliteClient(settings) as client:
        for source, target in targets.items():
            logger.info("=== %s: target %d ===", source, target)
            outcomes.append(await acquire_source(settings, source, target, client=client))
    return outcomes


def render_summary(outcomes: list[SourceOutcome]) -> str:
    """The per-source summary table, for the log and for docs/PROGRESS.md."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### Acquisition summary ({now})",
        "",
        "| source | target | stored | new this run | rejected | failed | failure rate | state |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for outcome in outcomes:
        stats = outcome.stats
        if stats is None:
            state = "BLOCKED" if outcome.blocked else "ERROR"
            lines.append(
                f"| `{outcome.source}` | {outcome.target} | {outcome.total_stored} "
                f"| -- | -- | -- | -- | {state} |"
            )
            continue

        rejected = sum(stats.rejected.values())
        failed = sum(stats.failed.values())
        state = "ok" if outcome.ok else "HIGH FAILURE RATE"
        lines.append(
            f"| `{outcome.source}` | {outcome.target} | {outcome.total_stored} "
            f"| {stats.downloaded} | {rejected} | {failed} "
            f"| {stats.failure_rate:.2%} | {state} |"
        )

    total_stored = sum(o.total_stored for o in outcomes)
    total_bytes = sum(o.stats.bytes_written for o in outcomes if o.stats)
    lines += [
        "",
        f"- Total stored across these sources: **{total_stored:,}** images.",
        f"- Written this run: {total_bytes / 1024**3:.2f} GB.",
    ]

    for outcome in outcomes:
        if outcome.stats and outcome.stats.rejected:
            detail = ", ".join(
                f"{reason}={count}" for reason, count in sorted(outcome.stats.rejected.items())
            )
            lines.append(f"- `{outcome.source}` rejections: {detail}.")
        if outcome.stats and outcome.stats.failed:
            detail = ", ".join(
                f"{reason}={count}" for reason, count in sorted(outcome.stats.failed.items())
            )
            lines.append(f"- `{outcome.source}` failures: {detail}.")
        if outcome.blocked:
            lines.append(f"- `{outcome.source}` **BLOCKED**: {outcome.blocked}")
        if outcome.error:
            lines.append(f"- `{outcome.source}` **ERROR**: {outcome.error}")

    lines.append("")
    return "\n".join(lines)


def _configure_logging(settings: Settings, verbose: bool) -> Path:
    """Log to both the console and a timestamped file under `logs/`."""
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / f"acquire_{datetime.now(UTC):%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    # These two are chatty at INFO and would bury our own progress lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        help="source to acquire; repeatable. Default: all four.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="override the per-source target (useful for a small trial run)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug-level logging")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    log_path = _configure_logging(settings, args.verbose)

    chosen = args.source or list(SOURCES)
    targets = {
        source: args.target if args.target is not None else DEFAULT_TARGETS[source]
        for source in chosen
    }

    logger.info("archive root: %s", settings.archive)
    logger.info("log file: %s", log_path)
    logger.info("targets: %s", targets)

    outcomes = asyncio.run(acquire_all(settings, targets))

    summary = render_summary(outcomes)
    print("\n" + summary)
    (settings.logs_dir / "acquire_summary.md").write_text(summary, encoding="utf-8")

    problems = [o for o in outcomes if not o.ok]
    if problems:
        logger.warning(
            "%d of %d sources need attention: %s",
            len(problems),
            len(outcomes),
            ", ".join(o.source for o in problems),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
