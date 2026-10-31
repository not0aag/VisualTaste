"""Tests for the FAISS index wrapper.

The property that matters most: FAISS speaks in row positions and everything
else speaks in image ids, so the translation between them must never slip.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.search.index import VectorIndex


def unit(rows) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


@pytest.fixture
def index() -> VectorIndex:
    """Four unit vectors in the plane, at known angles.

    `east` and `near_east` are 0.1 radians apart, so their cosine is about
    0.995; `north` is orthogonal to `east`; `west` is opposite.
    """
    vectors = unit(
        [
            [1.0, 0.0],
            [np.cos(0.1), np.sin(0.1)],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    return VectorIndex(vectors, ["east", "near_east", "north", "west"])


class TestConstruction:
    def test_reports_its_size_and_dimension(self, index):
        assert len(index) == 4
        assert index.dim == 2

    def test_misaligned_ids_and_vectors_fail_loudly(self):
        with pytest.raises(ValueError, match="must align"):
            VectorIndex(np.zeros((3, 4), dtype=np.float32), ["a", "b"])

    def test_an_empty_index_is_usable_rather_than_broken(self):
        empty = VectorIndex(np.zeros((0, 8), dtype=np.float32), [])

        assert len(empty) == 0
        assert empty.search(np.ones(8, dtype=np.float32), 5) == []

    def test_accepts_a_numpy_id_array(self):
        built = VectorIndex(unit([[1.0, 0.0]]), np.array(["abc"], dtype="<U16"))

        assert built.ids == ["abc"]


class TestLookups:
    def test_position_maps_an_id_to_its_row(self, index):
        assert index.position("east") == 0
        assert index.position("west") == 3

    def test_an_unknown_id_has_no_position(self, index):
        assert index.position("nowhere") is None
        assert index.vector_for("nowhere") is None

    def test_vector_for_returns_the_right_row(self, index):
        np.testing.assert_allclose(index.vector_for("north"), [0.0, 1.0], atol=1e-6)

    def test_vectors_for_skips_ids_it_does_not_hold(self, index):
        # An image can be labelled and then quarantined by the safety screen, so
        # a profile may reference ids the index no longer contains. Skipping is
        # correct; raising would break the feed.
        rows = index.vectors_for(["east", "nowhere", "north"])

        assert rows.shape == (2, 2)

    def test_vectors_for_an_empty_list_has_the_right_shape(self, index):
        rows = index.vectors_for([])

        assert rows.shape == (0, 2)


class TestSearch:
    def test_finds_the_nearest_neighbour_first(self, index):
        results = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)

        assert results[0][0] == "east"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)
        assert results[1][0] == "near_east"

    def test_similarity_is_cosine_because_vectors_are_unit_length(self, index):
        results = dict(index.search(np.array([1.0, 0.0], dtype=np.float32), k=4))

        assert results["near_east"] == pytest.approx(np.cos(0.1), abs=1e-5)
        assert results["north"] == pytest.approx(0.0, abs=1e-5)
        assert results["west"] == pytest.approx(-1.0, abs=1e-5)

    def test_results_are_ordered_most_similar_first(self, index):
        similarities = [s for _, s in index.search(np.array([1.0, 0.0], np.float32), k=4)]

        assert similarities == sorted(similarities, reverse=True)

    def test_k_is_respected(self, index):
        assert len(index.search(np.array([1.0, 0.0], np.float32), k=2)) == 2

    def test_asking_for_more_than_exists_returns_everything(self, index):
        assert len(index.search(np.array([1.0, 0.0], np.float32), k=99)) == 4

    def test_excluded_ids_are_dropped_and_k_is_still_filled(self, index):
        # Exclusions are how the feed avoids re-showing a seen image, so the
        # search has to over-fetch rather than return a short list.
        results = index.search(np.array([1.0, 0.0], np.float32), k=2, exclude={"east", "near_east"})

        assert [image_id for image_id, _ in results] == ["north", "west"]

    def test_excluding_everything_returns_nothing(self, index):
        results = index.search(
            np.array([1.0, 0.0], np.float32),
            k=2,
            exclude={"east", "near_east", "north", "west"},
        )

        assert results == []

    def test_k_of_zero_returns_nothing(self, index):
        assert index.search(np.array([1.0, 0.0], np.float32), k=0) == []

    def test_a_row_shaped_query_is_accepted(self, index):
        # Callers pass both (D,) and (1, D); both must work.
        flat = index.search(np.array([1.0, 0.0], np.float32), k=1)
        shaped = index.search(np.array([[1.0, 0.0]], np.float32), k=1)

        assert flat == shaped


class TestBatchSearch:
    def test_returns_similarities_and_positions_per_query(self, index):
        queries = unit([[1.0, 0.0], [0.0, 1.0]])

        similarities, positions = index.search_many(queries, k=2)

        assert similarities.shape == (2, 2)
        assert positions.shape == (2, 2)
        assert index.ids[positions[0][0]] == "east"
        assert index.ids[positions[1][0]] == "north"

    def test_an_empty_index_returns_empty_arrays(self):
        empty = VectorIndex(np.zeros((0, 4), np.float32), [])

        similarities, positions = empty.search_many(np.zeros((3, 4), np.float32), k=5)

        assert similarities.shape[0] == 3
        assert positions.shape[0] == 3


class TestSimilarityToAll:
    def test_scores_every_row_in_index_order(self, index):
        scores = index.similarity_to_all(np.array([1.0, 0.0], np.float32))

        assert scores.shape == (4,)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[3] == pytest.approx(-1.0, abs=1e-5)

    def test_agrees_with_the_top_k_search(self, index):
        # The feed scores the whole pool rather than top-k, because MMR and the
        # exploration quota need every score. The two paths must not disagree.
        query = unit([[0.8, 0.6]])[0]

        all_scores = index.similarity_to_all(query)
        best_id, best_score = index.search(query, k=1)[0]

        assert index.ids[int(np.argmax(all_scores))] == best_id
        assert float(all_scores.max()) == pytest.approx(best_score, abs=1e-5)


class TestPersistence:
    def test_saves_and_reloads(self, index, tmp_path):
        path = tmp_path / "siglip2_b16_224.faiss"
        index.save(path)

        reloaded = VectorIndex.load(path, index.vectors, np.array(index.ids))

        assert len(reloaded) == 4
        assert reloaded.search(np.array([1.0, 0.0], np.float32), k=1)[0][0] == "east"

    def test_no_part_file_survives_a_save(self, index, tmp_path):
        index.save(tmp_path / "x.faiss")

        assert list(tmp_path.glob(".*.part")) == []

    def test_a_missing_file_simply_rebuilds(self, index, tmp_path):
        # A flat index rebuilds in well under a second, so a missing file is an
        # inconvenience rather than a reason to refuse to start the API.
        reloaded = VectorIndex.load(tmp_path / "absent.faiss", index.vectors, np.array(index.ids))

        assert len(reloaded) == 4

    def test_an_unreadable_file_rebuilds_with_a_warning(self, index, tmp_path, caplog):
        path = tmp_path / "corrupt.faiss"
        path.write_bytes(b"this is not a faiss index")

        with caplog.at_level("WARNING"):
            reloaded = VectorIndex.load(path, index.vectors, np.array(index.ids))

        assert len(reloaded) == 4
        assert "rebuilding" in caplog.text

    def test_a_stale_index_is_detected_and_rebuilt(self, index, tmp_path, caplog):
        # The corpus grows: an index saved before the last 10,000 images were
        # embedded must not be used as if it were complete.
        path = tmp_path / "stale.faiss"
        index.save(path)
        grown = np.vstack([index.vectors, unit([[0.6, 0.8]])])
        grown_ids = np.array([*index.ids, "newcomer"])

        with caplog.at_level("WARNING"):
            reloaded = VectorIndex.load(path, grown, grown_ids)

        assert len(reloaded) == 5
        assert "rebuilding" in caplog.text
        assert reloaded.position("newcomer") == 4
