"""Measure encoder throughput, so the embedding job's batch size is a fact.

Run:  python -m visualtaste.jobs.benchmark

This times the forward pass only, on synthetic random tensors, deliberately:
it separates "how fast can this GPU run the model" from "how fast can this disk
decode WebP files". Phase 4 adds the second measurement on real data, and the
gap between the two tells us whether the pipeline is GPU-bound or IO-bound.

Why it matters: this laptop's RTX 3060 has 6 GB of VRAM, which is small. The
largest batch that fits is an empirical question, and a batch that OOMs after
four hours of embedding is a wasted evening. An out-of-memory result here is a
recorded measurement, not a failure.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from visualtaste.config import get_settings
from visualtaste.vision.encoders import ENCODER_SPECS, load_encoder, pick_device

DEFAULT_BATCH_SIZES = (32, 64, 128, 256)

# Timed runs per batch size, after one untimed warm-up run. The first forward
# pass through a freshly loaded model pays for CUDA kernel autotuning and lazy
# weight transfers, which would otherwise dominate a small measurement.
WARMUP_RUNS = 1
TIMED_RUNS = 5

# A batch size counts as "as fast as the best" if it is within this fraction of
# the best measured throughput. Run-to-run variance on a thermally throttled
# laptop GPU is a few percent, so picking the strict argmax would hand us a
# batch size chosen by noise.
THROUGHPUT_TOLERANCE = 0.05


@dataclass
class BenchmarkRow:
    """One (model, batch size) measurement."""

    model_key: str
    batch_size: int
    images_per_second: float | None
    peak_vram_mb: float | None
    seconds_per_batch: float | None
    status: str  # "ok" | "out_of_memory" | "error: ..."


def _nvidia_smi_line() -> str:
    """GPU name and driver version, or a note if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return f"nvidia-smi unavailable ({exc})"


def _cpu_line() -> str:
    """CPU model and core count, via PowerShell's CIM (the brief's command)."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$c = Get-CimInstance Win32_Processor | Select-Object -First 1; "
                '"$($c.Name.Trim()) / $($c.NumberOfCores) cores / '
                '$($c.NumberOfLogicalProcessors) threads"',
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{platform.processor()} (CIM unavailable: {exc})"


def benchmark_model(
    model_key: str,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
) -> list[BenchmarkRow]:
    """Time one encoder across batch sizes on synthetic input."""
    settings = get_settings()
    settings.apply_hf_env()

    device = pick_device()
    spec = ENCODER_SPECS[model_key]
    encoder = load_encoder(model_key, device=device)
    rows: list[BenchmarkRow] = []

    for batch_size in batch_sizes:
        # Random pixels, not real images: we are timing matrix multiplies, and
        # the model's runtime does not depend on what the picture shows.
        batch = torch.randn(batch_size, 3, spec.image_size, spec.image_size, device=device)

        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            for _ in range(WARMUP_RUNS):
                encoder.encode_images(batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(TIMED_RUNS):
                encoder.encode_images(batch)
            if device.type == "cuda":
                # Without this, perf_counter would stop before the GPU has
                # actually finished: CUDA calls are queued asynchronously.
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            seconds_per_batch = elapsed / TIMED_RUNS
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
            rows.append(
                BenchmarkRow(
                    model_key=model_key,
                    batch_size=batch_size,
                    images_per_second=batch_size / seconds_per_batch,
                    peak_vram_mb=peak_mb,
                    seconds_per_batch=seconds_per_batch,
                    status="ok",
                )
            )
        except torch.cuda.OutOfMemoryError:
            # Expected at the top of the range on a 6 GB card. Record and move on.
            rows.append(BenchmarkRow(model_key, batch_size, None, None, None, "out_of_memory"))
        finally:
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

        row = rows[-1]
        if row.status == "ok":
            print(
                f"  {model_key:18} batch {batch_size:4}  "
                f"{row.images_per_second:7.1f} img/s  "
                f"peak {row.peak_vram_mb or 0:7.0f} MB"
            )
        else:
            print(f"  {model_key:18} batch {batch_size:4}  {row.status}")

    return rows


def recommended_batch_size(rows: list[BenchmarkRow]) -> int | None:
    """The largest batch size that is still within `THROUGHPUT_TOLERANCE` of the best.

    Throughput plateaus well before the memory limit, so beyond some point a
    bigger batch buys nothing. Among the batch sizes that all hit that plateau
    we take the largest, because each batch costs one Python round-trip and one
    synchronisation, and Phase 4 pays that cost tens of thousands of times.

    Taking the strict argmax instead would be choosing by measurement noise:
    on a laptop GPU that throttles, two batch sizes on the same plateau can
    swap places between runs.
    """
    ok = [r for r in rows if r.status == "ok" and r.images_per_second is not None]
    if not ok:
        return None
    best = max(r.images_per_second or 0.0 for r in ok)
    floor = best * (1 - THROUGHPUT_TOLERANCE)
    on_plateau = [r for r in ok if (r.images_per_second or 0.0) >= floor]
    return max(r.batch_size for r in on_plateau)


def render_markdown(rows: list[BenchmarkRow], *, command: str) -> str:
    """Format results as the Markdown block appended to docs/BENCHMARKS.md."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## Synthetic-tensor encoder throughput ({now})",
        "",
        f"- Command: `{command}`",
        f"- GPU: {_nvidia_smi_line()}",
        f"- CPU: {_cpu_line()}",
        f"- torch {torch.__version__}, CUDA build {torch.version.cuda}, "
        f"Python {platform.python_version()}",
        f"- Method: {WARMUP_RUNS} warm-up + {TIMED_RUNS} timed forward passes on "
        "random tensors, fp16 autocast, `torch.inference_mode()`. "
        "Forward pass only -- no image decoding, no disk IO.",
        "",
        "| model_key | batch | images/s | peak VRAM (MB) | s/batch | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        ips = f"{r.images_per_second:.1f}" if r.images_per_second else "--"
        vram = f"{r.peak_vram_mb:.0f}" if r.peak_vram_mb else "--"
        spb = f"{r.seconds_per_batch:.3f}" if r.seconds_per_batch else "--"
        lines.append(f"| `{r.model_key}` | {r.batch_size} | {ips} | {vram} | {spb} | {r.status} |")

    lines.append("")
    for model_key in dict.fromkeys(r.model_key for r in rows):
        model_rows = [r for r in rows if r.model_key == model_key]
        best = recommended_batch_size(model_rows)
        if best is None:
            lines.append(f"- `{model_key}`: no batch size completed.")
            continue
        chosen = next(r for r in model_rows if r.batch_size == best)
        rate = chosen.images_per_second or 0.0
        est_minutes = 80_000 / rate / 60 if rate else float("nan")
        lines.append(
            f"- `{model_key}`: recommended batch **{best}** "
            f"({rate:.0f} img/s, peak {chosen.peak_vram_mb or 0:.0f} MB) -> "
            f"~{est_minutes:.0f} min of pure GPU time for 80k images "
            f"(estimate: excludes decoding and IO)."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(ENCODER_SPECS),
        help="model_key to benchmark; repeatable. Default: all.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="batch sizes to try (default: 32 64 128 256)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Markdown file to append to (default: docs/BENCHMARKS.md)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="also write raw rows as JSON")
    args = parser.parse_args(argv)

    settings = get_settings()
    model_keys = args.model or sorted(ENCODER_SPECS)
    batch_sizes = tuple(args.batch_sizes)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; these numbers are CPU numbers.")

    print(f"benchmarking {', '.join(model_keys)} at batches {batch_sizes}")
    rows: list[BenchmarkRow] = []
    for model_key in model_keys:
        rows.extend(benchmark_model(model_key, batch_sizes))

    command = (
        "python -m visualtaste.jobs.benchmark"
        + ("" if not args.model else " " + " ".join(f"--model {m}" for m in args.model))
        + (
            ""
            if batch_sizes == DEFAULT_BATCH_SIZES
            else " --batch-sizes " + " ".join(str(b) for b in batch_sizes)
        )
    )
    markdown = render_markdown(rows, command=command)

    out_path = args.out or (settings.docs_dir / "BENCHMARKS.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + markdown)
    print(f"\nappended results to {out_path}")

    if args.json_out:
        args.json_out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
