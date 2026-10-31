"""Tests for the embedding job's selection logic and failure handling.

The encoders themselves are not loaded: these cover which images the job
decides to embed, and what happens when one of them will not decode -- the two
behaviours that determine whether a four-hour run survives a bad file.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from visualtaste.config import Settings
from visualtaste.db import repo
from visualtaste.jobs.embed import (
    DEFAULT_BATCH_SIZES,
    DisplayImageDataset,
    EmbedStats,
    ids_to_embed,
    render_summary,
)
from visualtaste.vision.encoders import ENCODER_SPECS
from visualtaste.vision.store import EmbeddingStore


@pytest.fixture
def settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        min_free_gb=0,
    )


@pytest.fixture
def db(settings):
    settings.ensure_dirs()
    connection = repo.init_db(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def store(settings):
    return EmbeddingStore(settings.embedding_dir("siglip2_b16_224"), "siglip2_b16_224")


def add_image(db, image_id: str, status: str = "ok") -> None:
    repo.upsert_image(
        db,
        image_id=image_id,
        source="pd12m",
        source_id=image_id,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        orig_path="p",
        status=status,
    )
    db.commit()


def hex_id(n: int) -> str:
    return f"{n:016x}"


class TestWhichImagesGetEmbedded:
    def test_every_usable_image_when_nothing_is_done(self, db, store):
        for n in range(5):
            add_image(db, hex_id(n))

        todo, done = ids_to_embed(db, store)

        assert len(todo) == 5
        assert done == 0

    def test_only_ok_images_are_embedded(self, db, store):
        # Nothing will ever score a rejected, quarantined or near-duplicate
        # image, so embedding one would be wasted GPU time.
        add_image(db, hex_id(1), "ok")
        add_image(db, hex_id(2), "rejected")
        add_image(db, hex_id(3), "near_dup")
        add_image(db, hex_id(4), "quarantined")

        todo, _ = ids_to_embed(db, store)

        assert todo == [hex_id(1)]

    def test_already_embedded_images_are_skipped(self, db, store):
        for n in range(5):
            add_image(db, hex_id(n))
        store.write_shard(0, [hex_id(0), hex_id(1)], np.eye(2, 8, dtype=np.float32))

        todo, done = ids_to_embed(db, store)

        assert set(todo) == {hex_id(2), hex_id(3), hex_id(4)}
        assert done == 2

    def test_a_finished_corpus_has_nothing_to_do(self, db, store):
        # Running the job twice in a row must do nothing the second time.
        ids = [hex_id(n) for n in range(4)]
        for image_id in ids:
            add_image(db, image_id)
        store.write_shard(0, ids, np.eye(4, 8, dtype=np.float32))

        todo, done = ids_to_embed(db, store)

        assert todo == []
        assert done == 4

    def test_adding_ten_images_embeds_ten(self, db, store):
        # The incremental property, stated as the brief states it.
        original = [hex_id(n) for n in range(20)]
        for image_id in original:
            add_image(db, image_id)
        store.write_shard(0, original, np.zeros((20, 8), dtype=np.float32))

        for n in range(20, 30):
            add_image(db, hex_id(n))
        todo, done = ids_to_embed(db, store)

        assert len(todo) == 10
        assert done == 20

    def test_the_order_is_stable_so_a_resumed_run_matches(self, db, store):
        for n in (7, 2, 9, 4):
            add_image(db, hex_id(n))

        first, _ = ids_to_embed(db, store)
        second, _ = ids_to_embed(db, store)

        assert first == second == sorted(first)

    def test_a_limit_truncates_the_work(self, db, store):
        for n in range(10):
            add_image(db, hex_id(n))

        todo, _ = ids_to_embed(db, store, limit=3)

        assert len(todo) == 3

    def test_an_empty_database_is_not_an_error(self, db, store):
        assert ids_to_embed(db, store) == ([], 0)


class TestDatasetFailureHandling:
    def _dataset(self, settings, image_ids):
        class FakeEncoder:
            spec = ENCODER_SPECS["siglip2_b16_224"]

            @staticmethod
            def preprocess(image):
                # Stands in for the model's own transform.
                return torch.from_numpy(
                    np.asarray(image.resize((224, 224)), dtype=np.float32).transpose(2, 0, 1)
                )

        return DisplayImageDataset(image_ids, settings, FakeEncoder())

    def test_reads_a_display_derivative(self, settings):
        image_id = hex_id(1)
        path = settings.display_path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (300, 200), (50, 100, 150)).save(path, format="WEBP")

        tensor, index, ok = self._dataset(settings, [image_id])[0]

        assert ok is True
        assert index == 0
        assert tensor.shape == (3, 224, 224)

    def test_a_missing_file_yields_a_flag_not_an_exception(self, settings):
        # An exception inside a DataLoader worker tears down the whole loader,
        # which would mean one unreadable file ends a four-hour job.
        tensor, index, ok = self._dataset(settings, [hex_id(99)])[0]

        assert ok is False
        assert index == 0
        assert tensor.shape == (3, 224, 224)
        assert not tensor.any(), "a failed read should produce a blank, not junk"

    def test_a_corrupt_file_yields_a_flag_too(self, settings):
        image_id = hex_id(2)
        path = settings.display_path(image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a webp at all")

        _, _, ok = self._dataset(settings, [image_id])[0]

        assert ok is False

    def test_the_index_identifies_which_image_failed(self, settings):
        # The job uses this index to look up the id and record the failure
        # against the right image.
        good = hex_id(1)
        path = settings.display_path(good)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(path, format="WEBP")
        dataset = self._dataset(settings, [hex_id(99), good])

        _, bad_index, bad_ok = dataset[0]
        _, good_index, good_ok = dataset[1]

        assert (bad_index, bad_ok) == (0, False)
        assert (good_index, good_ok) == (1, True)

    def test_length_is_the_number_of_images(self, settings):
        assert len(self._dataset(settings, [hex_id(n) for n in range(7)])) == 7


class TestErrorRecording:
    def test_a_decode_failure_lands_in_embedding_errors(self, db):
        add_image(db, hex_id(1))

        repo.record_embedding_error(
            db, hex_id(1), "siglip2_b16_224", "could not decode display derivative"
        )
        db.commit()

        rows = repo.query(db, "SELECT * FROM embedding_errors")
        assert len(rows) == 1
        assert rows[0]["model_key"] == "siglip2_b16_224"

    def test_the_model_is_registered_with_its_checkpoint(self, db):
        spec = ENCODER_SPECS["siglip2_b16_224"]

        repo.register_embedding_model(
            db,
            model_key=spec.key,
            family=spec.family,
            checkpoint=spec.checkpoint,
            dim=spec.dim,
            preprocess="open_clip native, 224px",
        )
        db.commit()

        row = repo.query(db, "SELECT * FROM embedding_models")[0]
        assert row["checkpoint"] == "ViT-B-16-SigLIP2/webli"
        assert row["dim"] == 768


class TestBatchSizes:
    def test_both_models_have_a_measured_batch_size(self):
        # These come from docs/BENCHMARKS.md, not from a guess.
        assert set(DEFAULT_BATCH_SIZES) == set(ENCODER_SPECS)

    def test_the_batch_sizes_fit_the_measured_vram(self):
        # Peak measured at batch 256 was 2.8 GB of the 6 GB available, so both
        # of these are comfortable rather than tight.
        assert DEFAULT_BATCH_SIZES["siglip2_b16_224"] == 128
        assert DEFAULT_BATCH_SIZES["dinov2_b14"] == 256


class TestStatsAndSummary:
    def test_throughput_is_computed_from_the_measured_time(self):
        stats = EmbedStats("siglip2_b16_224", embedded=1000, seconds=4.0)

        assert stats.images_per_second == pytest.approx(250.0)

    def test_throughput_is_zero_rather_than_dividing_by_zero(self):
        assert EmbedStats("m").images_per_second == 0.0

    def test_the_summary_names_the_counters(self):
        stats = EmbedStats("dinov2_b14", wanted=10, already_done=2, embedded=8, failed=1)
        stats.seconds = 2.0

        text = stats.summary()

        assert "wanted=10" in text
        assert "already_done=2" in text
        assert "failed=1" in text

    def test_the_markdown_separates_io_cost_from_gpu_cost(self, settings, store):
        # The interesting number in Phase 4 is how much slower real data is than
        # synthetic tensors, because the gap is decode and IO.
        store.write_shard(0, [hex_id(1)], np.ones((1, 8), dtype=np.float32) / np.sqrt(8))
        store.merge()
        store.write_meta(
            checkpoint="ViT-B-16-SigLIP2/webli",
            dim=768,
            count=1,
            preprocess="open_clip native, 224px",
            device="cuda",
        )
        stats = EmbedStats("siglip2_b16_224", embedded=1000, seconds=10.0)

        text = render_summary([stats], settings)

        assert "`siglip2_b16_224`" in text
        assert "decoding and IO" in text
        assert "100 " in text  # 1000 images / 10 s
