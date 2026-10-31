"""Tests for the embedding store: sharding, resume, merge, provenance.

No model is loaded here. The store's job is to not lose or misalign vectors,
which is testable with arrays of known numbers -- and misalignment is the
failure that matters most, because it would attach Alen's labels to the wrong
pictures without anything appearing to break.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.vision.store import (
    IMAGE_ID_WIDTH,
    SHARD_ROWS,
    EmbeddingMeta,
    EmbeddingStore,
    verify_normalised,
)


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(tmp_path / "siglip2_b16_224", "siglip2_b16_224")


def unit_rows(count: int, dim: int = 8, *, seed: int = 0) -> np.ndarray:
    """`count` random unit vectors, as a real encoder would produce."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, dim)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def ids_for(count: int, prefix: str = "a") -> list[str]:
    """Realistic ids: exactly 16 characters, like a truncated SHA-256."""
    return [f"{prefix}{i:0{16 - len(prefix)}d}" for i in range(count)]


class TestShardWriting:
    def test_writes_a_shard_and_reports_its_ids(self, store):
        vectors = unit_rows(10)

        store.write_shard(0, ids_for(10), vectors)

        assert len(store.shard_paths()) == 1
        assert store.embedded_ids() == set(ids_for(10))

    def test_shards_hold_float16_to_halve_the_disk(self, store):
        # The GPU produced these at float16, so float32 on disk would be storing
        # digits that are not real.
        store.write_shard(0, ids_for(4), unit_rows(4))

        with np.load(store.shard_paths()[0], allow_pickle=False) as shard:
            assert shard["vectors"].dtype == np.float16

    def test_ids_and_vectors_cannot_be_separated(self, store):
        # They live in one .npz on purpose: a shard whose id list and vector
        # block disagreed would mislabel images silently.
        store.write_shard(0, ids_for(3), unit_rows(3))

        with np.load(store.shard_paths()[0], allow_pickle=False) as shard:
            assert set(shard.files) == {"vectors", "ids"}
            assert len(shard["ids"]) == len(shard["vectors"])

    def test_a_length_mismatch_is_rejected_loudly(self, store):
        with pytest.raises(ValueError, match="ids but"):
            store.write_shard(0, ids_for(3), unit_rows(5))

    def test_an_over_long_id_is_rejected_rather_than_truncated(self, store):
        # The id arrays are a fixed-width dtype, so a 20-character id would be
        # cut to 16 with no error -- and two ids sharing a 16-character prefix
        # would silently become one row.
        with pytest.raises(ValueError, match="longer than"):
            store.write_shard(0, ["x" * (IMAGE_ID_WIDTH + 4)], unit_rows(1))

    def test_ids_of_exactly_the_id_width_are_fine(self, store):
        store.write_shard(0, ["0123456789abcdef"], unit_rows(1))

        assert store.embedded_ids() == {"0123456789abcdef"}

    def test_no_part_file_survives_a_write(self, store):
        store.write_shard(0, ids_for(3), unit_rows(3))

        assert list(store.shards_dir.glob(".*.part")) == []

    def test_shard_indices_continue_after_a_restart(self, store):
        store.write_shard(0, ids_for(3), unit_rows(3))
        store.write_shard(1, ids_for(3, "b"), unit_rows(3))

        # A fresh process opening the same directory must not overwrite shard 0.
        reopened = EmbeddingStore(store.dir, store.model_key)

        assert reopened.next_shard_index() == 2

    def test_an_empty_store_starts_at_shard_zero(self, store):
        assert store.next_shard_index() == 0
        assert store.embedded_ids() == set()
        assert store.shard_paths() == []


class TestIncrementalContract:
    def test_embedded_ids_spans_shards_and_the_merged_file(self, store):
        store.write_shard(0, ids_for(5), unit_rows(5))
        store.merge()
        store.write_shard(store.next_shard_index(), ids_for(3, "later"), unit_rows(3))

        covered = store.embedded_ids()

        assert covered == set(ids_for(5)) | set(ids_for(3, "later"))

    def test_an_unreadable_shard_is_skipped_not_fatal(self, store, caplog):
        # A shard truncated by a power cut mid-write. Its ids are simply
        # re-embedded; losing 2,048 rows of work is the designed worst case.
        store.write_shard(0, ids_for(5), unit_rows(5))
        store.write_shard(1, ids_for(5, "b"), unit_rows(5))
        store.shard_paths()[1].write_bytes(b"truncated garbage")

        with caplog.at_level("WARNING"):
            covered = store.embedded_ids()

        assert covered == set(ids_for(5))
        assert "unreadable shard" in caplog.text

    def test_a_part_file_is_not_counted_as_done(self, store):
        # Writes go to a dotted .part file first, so a half-written shard is
        # invisible to the resume logic rather than half-believed.
        store.write_shard(0, ids_for(5), unit_rows(5))
        (store.shards_dir / ".shard_00001.npz.part").write_bytes(b"incomplete")

        assert store.embedded_ids() == set(ids_for(5))


class TestMerge:
    def test_merges_shards_into_float32_vectors_and_ids(self, store):
        store.write_shard(0, ids_for(4), unit_rows(4))
        store.write_shard(1, ids_for(3, "b"), unit_rows(3, seed=1))

        vectors, ids = store.merge()

        assert vectors.shape == (7, 8)
        assert vectors.dtype == np.float32
        assert len(ids) == 7

    def test_rows_come_out_sorted_by_image_id(self, store):
        # Determinism: the same corpus produces the same bytes, so a checksum
        # means something, and FAISS row order is stable between runs.
        store.write_shard(0, ["zzz", "aaa", "mmm"], unit_rows(3))

        _, ids = store.merge()

        assert ids.tolist() == ["aaa", "mmm", "zzz"]

    def test_each_vector_stays_with_its_own_id(self, store):
        # The alignment property. Distinctive vectors make a swap detectable.
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
        store.write_shard(0, ["ccc", "aaa", "bbb"], vectors)

        merged, ids = store.merge()
        lookup = dict(zip(ids.tolist(), merged, strict=True))

        np.testing.assert_allclose(lookup["ccc"], [1.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(lookup["aaa"], [0.0, 1.0], atol=1e-3)
        np.testing.assert_allclose(lookup["bbb"], [0.6, 0.8], atol=1e-3)

    def test_merging_twice_does_not_duplicate_rows(self, store):
        store.write_shard(0, ids_for(4), unit_rows(4))
        store.merge()

        vectors, ids = store.merge()

        assert len(vectors) == 4
        assert len(set(ids.tolist())) == 4

    def test_a_later_shard_supersedes_an_earlier_value_for_the_same_id(self, store):
        # Re-embedding an image after a model change should replace its vector,
        # not leave two rows claiming the same id.
        store.write_shard(0, ["same"], np.array([[1.0, 0.0]], dtype=np.float32))
        store.merge()
        store.write_shard(1, ["same"], np.array([[0.0, 1.0]], dtype=np.float32))

        vectors, ids = store.merge()

        assert len(ids) == 1
        np.testing.assert_allclose(vectors[0], [0.0, 1.0], atol=1e-3)

    def test_merge_preserves_the_previous_merge(self, store):
        store.write_shard(0, ids_for(3), unit_rows(3))
        store.merge()
        for path in store.shard_paths():
            path.unlink()
        store.write_shard(0, ids_for(2, "new"), unit_rows(2))

        _, ids = store.merge()

        assert set(ids.tolist()) == set(ids_for(3)) | set(ids_for(2, "new"))

    def test_merging_nothing_fails_loudly(self, store):
        with pytest.raises(RuntimeError, match="nothing to merge"):
            store.merge()

    def test_shards_can_be_discarded_after_merging(self, store):
        store.write_shard(0, ids_for(3), unit_rows(3))

        store.merge(keep_shards=False)

        assert store.shard_paths() == []
        assert store.vectors_path.exists()

    def test_no_part_files_survive_a_merge(self, store):
        store.write_shard(0, ids_for(3), unit_rows(3))

        store.merge()

        assert list(store.dir.glob(".*.part")) == []


class TestLoading:
    def test_round_trips_through_disk(self, store):
        vectors = unit_rows(6)
        store.write_shard(0, ids_for(6), vectors)
        store.merge()

        loaded, ids = EmbeddingStore(store.dir, store.model_key).load()

        assert loaded.shape == vectors.shape
        assert len(ids) == 6
        # float16 on the way in, so compare at float16 precision.
        assert np.abs(loaded).max() <= 1.01

    def test_load_as_dict_keys_by_image_id(self, store):
        store.write_shard(0, ["aaa", "bbb"], unit_rows(2))
        store.merge()

        mapping = store.load_as_dict()

        assert set(mapping) == {"aaa", "bbb"}
        assert mapping["aaa"].shape == (8,)

    def test_loading_before_a_merge_says_what_to_run(self, store):
        with pytest.raises(FileNotFoundError, match=r"jobs\.embed"):
            store.load()

    def test_disagreeing_merged_files_are_detected(self, store):
        store.write_shard(0, ids_for(5), unit_rows(5))
        store.merge()
        # Corrupt the pairing, as a partial write or a bad manual edit would.
        with store.ids_path.open("wb") as handle:
            np.save(handle, np.array(ids_for(3), dtype="<U16"))

        with pytest.raises(ValueError, match="disagree"):
            store.load()


class TestNormalisationGuard:
    def test_accepts_unit_vectors(self):
        verify_normalised(unit_rows(20))

    def test_rejects_vectors_that_escaped_normalisation(self):
        # Every similarity in the system is a dot product that assumes unit
        # length. Un-normalised vectors would not crash anything -- they would
        # just quietly rank badly, which is far worse.
        vectors = unit_rows(5) * 3.0

        with pytest.raises(ValueError, match="not unit length"):
            verify_normalised(vectors)

    def test_tolerates_float16_rounding(self):
        # Shards are float16, so round-tripped vectors are unit length only to
        # about 1e-3 -- which is exactly the tolerance the brief specifies.
        vectors = unit_rows(50).astype(np.float16).astype(np.float32)

        verify_normalised(vectors, tolerance=1e-3)

    def test_an_empty_array_is_accepted(self):
        verify_normalised(np.zeros((0, 8), dtype=np.float32))

    def test_the_error_quotes_the_worst_deviation(self):
        with pytest.raises(ValueError) as excinfo:
            verify_normalised(unit_rows(3) * 2.0)

        assert "worst deviation" in str(excinfo.value)


class TestMeta:
    def test_round_trips_every_field(self, tmp_path):
        meta = EmbeddingMeta(
            model_key="siglip2_b16_224",
            checkpoint="ViT-B-16-SigLIP2/webli",
            dim=768,
            count=50_000,
            preprocess="open_clip native, 224px",
            torch_version="2.11.0+cu128",
            cuda_version="12.8",
            device="cuda",
            created_at="2026-10-30T20:00:00+00:00",
        )
        path = tmp_path / "meta.json"

        meta.write(path)

        assert EmbeddingMeta.read(path) == meta

    def test_records_what_is_needed_to_spot_a_model_change(self, store):
        # Six months on, the only way to answer "were these made by the same
        # model as those?" is a file that says so.
        store.write_shard(0, ids_for(3), unit_rows(3))
        store.merge()

        meta = store.write_meta(
            checkpoint="ViT-B-16-SigLIP2/webli",
            dim=768,
            count=3,
            preprocess="open_clip native, 224px",
            device="cuda",
        )

        assert meta.checkpoint == "ViT-B-16-SigLIP2/webli"
        assert meta.torch_version
        assert meta.created_at
        assert meta.normalised is True
        assert store.read_meta() == meta

    def test_an_absent_meta_file_reads_as_none(self, store):
        assert store.read_meta() is None


class TestShardSize:
    def test_the_shard_size_matches_the_brief(self):
        assert SHARD_ROWS == 2_048

    def test_a_full_size_shard_round_trips(self, store):
        # The real shard size, so the npz path is exercised at production scale
        # rather than only with toy arrays.
        vectors = unit_rows(SHARD_ROWS, dim=768)
        shard_ids = ids_for(SHARD_ROWS)

        store.write_shard(0, shard_ids, vectors)
        merged, merged_ids = store.merge()

        assert merged.shape == (SHARD_ROWS, 768)
        assert sorted(shard_ids) == merged_ids.tolist()
        verify_normalised(merged, tolerance=1e-3)
