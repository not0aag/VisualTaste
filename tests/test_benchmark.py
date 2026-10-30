"""Tests for the benchmark's reporting logic (not for the GPU measurement)."""

from __future__ import annotations

from visualtaste.jobs.benchmark import (
    BenchmarkRow,
    recommended_batch_size,
    render_markdown,
)


def row(batch: int, ips: float | None, status: str = "ok", vram: float | None = 100.0):
    return BenchmarkRow(
        model_key="siglip2_b16_224",
        batch_size=batch,
        images_per_second=ips,
        peak_vram_mb=vram,
        seconds_per_batch=(batch / ips) if ips else None,
        status=status,
    )


class TestRecommendedBatchSize:
    def test_prefers_the_largest_batch_on_the_throughput_plateau(self):
        # 437 and 438 are the same speed in practice. Picking the strict argmax
        # would choose 64 on one run and 128 on the next; we want the larger.
        rows = [row(32, 414.0), row(64, 436.6), row(128, 437.9), row(256, 406.2)]

        # 406 is 92.7% of 438, outside the 5% tolerance, so 256 is excluded.
        assert recommended_batch_size(rows) == 128

    def test_includes_a_larger_batch_that_is_within_tolerance(self):
        rows = [row(32, 414.0), row(64, 436.6), row(128, 437.9), row(256, 425.0)]

        # 425 is 97% of 438, on the plateau, so the larger batch wins.
        assert recommended_batch_size(rows) == 256

    def test_excludes_a_batch_that_ran_out_of_memory(self):
        rows = [row(32, 414.0), row(64, 436.0), row(128, None, "out_of_memory", None)]

        assert recommended_batch_size(rows) == 64

    def test_a_clearly_faster_small_batch_wins(self):
        rows = [row(32, 500.0), row(64, 300.0), row(128, 250.0)]

        assert recommended_batch_size(rows) == 32

    def test_returns_none_when_nothing_completed(self):
        rows = [
            row(32, None, "out_of_memory", None),
            row(64, None, "error: boom", None),
        ]

        assert recommended_batch_size(rows) is None


class TestRenderMarkdown:
    def test_reports_the_throughput_of_the_recommended_batch_not_the_best(self):
        # If these two disagree the summary must quote the one we will actually
        # use, otherwise the time estimate is for a batch size nobody runs.
        rows = [row(64, 436.6), row(128, 437.9), row(256, 425.0)]

        markdown = render_markdown(rows, command="pytest")

        assert "recommended batch **256**" in markdown
        assert "425 img/s" in markdown

    def test_every_row_appears_in_the_table(self):
        rows = [row(32, 400.0), row(64, None, "out_of_memory", None)]

        markdown = render_markdown(rows, command="pytest")

        assert "| 32 | 400.0 |" in markdown
        assert "out_of_memory" in markdown

    def test_records_the_command_so_the_number_is_reproducible(self):
        markdown = render_markdown([row(32, 400.0)], command="python -m visualtaste.jobs.benchmark")

        assert "Command: `python -m visualtaste.jobs.benchmark`" in markdown

    def test_says_which_numbers_are_estimates(self):
        # The brief forbids presenting arithmetic as measurement.
        markdown = render_markdown([row(32, 400.0)], command="pytest")

        assert "estimate" in markdown

    def test_handles_a_model_where_nothing_fitted(self):
        rows = [row(32, None, "out_of_memory", None)]

        markdown = render_markdown(rows, command="pytest")

        assert "no batch size completed" in markdown
