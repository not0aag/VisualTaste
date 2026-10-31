"""Tests for the common recommender interface and the R0 cold-start rung."""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.rec.base import (
    Explanation,
    FeatureBundle,
    Recommender,
    UnfittedModelError,
    require_fitted,
)
from visualtaste.rec.labels import LabelSet
from visualtaste.rec.r0_coldstart import (
    COLDSTART_LABEL_TARGET,
    ColdStartRecommender,
    coldstart_is_active,
)


def hex_id(n: int) -> str:
    return f"{n:016x}"


def bundle(count: int = 10, *, clusters: list[int] | None = None, dim: int = 8):
    rng = np.random.default_rng(1729)
    siglip = rng.normal(size=(count, dim)).astype(np.float32)
    siglip /= np.linalg.norm(siglip, axis=1, keepdims=True)
    return FeatureBundle(
        image_ids=[hex_id(n) for n in range(count)],
        siglip=siglip,
        cluster64=np.array(clusters, dtype=np.int32) if clusters is not None else None,
    )


def labelset(count: int) -> LabelSet:
    return LabelSet(
        image_ids=[hex_id(n) for n in range(count)],
        targets=np.ones(count, dtype=np.float32),
        weights=np.ones(count, dtype=np.float32),
        actions=["like"] * count,
        timestamps=["2026-10-31T00:00:00"] * count,
    )


class TestFeatureBundle:
    def test_reports_its_length(self):
        assert len(bundle(7)) == 7

    def test_maps_ids_to_rows(self):
        features = bundle(5)

        assert features.position(hex_id(3)) == 3
        assert features.position("nope") is None

    def test_misaligned_blocks_are_rejected_at_construction(self):
        # Every rung relies on row i of every block belonging to image_ids[i],
        # so a mismatch must fail loudly rather than mislabel silently.
        with pytest.raises(ValueError, match="must align"):
            FeatureBundle(
                image_ids=[hex_id(0), hex_id(1)],
                siglip=np.zeros((2, 4), dtype=np.float32),
                dinov2=np.zeros((3, 4), dtype=np.float32),
            )

    def test_optional_blocks_may_be_absent(self):
        features = bundle(3)

        assert features.dinov2 is None
        assert features.pca is None

    def test_positions_skips_ids_it_does_not_know(self):
        # A labelled image can be quarantined or folded into a duplicate later,
        # so training must tolerate a profile referencing one.
        features = bundle(5)

        index = features.positions([hex_id(1), "gone", hex_id(4)])

        assert index.tolist() == [1, 4]

    def test_positions_of_nothing_is_empty(self):
        assert bundle(5).positions([]).tolist() == []

    def test_rows_returns_the_requested_block(self):
        features = bundle(5)

        rows = features.rows("siglip", [hex_id(0), hex_id(2)])

        assert rows.shape == (2, 8)
        np.testing.assert_allclose(rows[0], features.siglip[0])

    def test_rows_of_an_unknown_block_fails_loudly(self):
        with pytest.raises(KeyError, match="no feature block"):
            bundle(3).rows("nonexistent", [hex_id(0)])

    def test_rows_reads_from_the_extra_blocks_too(self):
        # R5's interest similarities arrive this way, so adding a feature block
        # does not change every rung's signature.
        features = bundle(4)
        features.extra["interests"] = np.ones((4, 3), dtype=np.float32)

        assert features.rows("interests", [hex_id(1)]).shape == (1, 3)

    def test_rows_for_no_known_ids_has_the_right_width(self):
        rows = bundle(4).rows("siglip", ["unknown"])

        assert rows.shape == (0, 8)


class TestExplanation:
    def test_serialises_reason_sentence_and_detail_together(self):
        explanation = Explanation(
            reason="similar_likes",
            sentence="Close to three images you liked.",
            detail={"neighbours": [hex_id(1)]},
        )

        import json

        decoded = json.loads(explanation.to_json())

        assert decoded["reason"] == "similar_likes"
        assert decoded["sentence"].startswith("Close to")
        assert decoded["neighbours"] == [hex_id(1)]

    def test_detail_defaults_to_empty(self):
        assert Explanation(reason="explore", sentence="x").detail == {}


class StubRecommender(Recommender):
    """A minimal rung, to exercise the base class's own behaviour."""

    rung = "stub"
    min_labels = 5

    def __init__(self) -> None:
        super().__init__()
        self.fit_calls = 0

    def _fit(self, labels, features) -> None:
        self.fit_calls += 1

    def score(self, features, candidates):
        require_fitted(self)
        return np.zeros(len(candidates), dtype=np.float32)

    def explain(self, features, image_index):
        return Explanation(reason="stub", sentence="stub")


class TestFitBookkeeping:
    def test_fitting_with_enough_labels_marks_the_model_fitted(self):
        model = StubRecommender().fit(labelset(10), bundle())

        assert model.is_fitted
        assert model.trained_on_events == 10
        assert model.fit_calls == 1

    def test_too_few_labels_leaves_the_model_unfitted_without_calling_fit(self):
        # This is what lets the feed fall back cleanly rather than serving
        # confident nonsense from four examples.
        model = StubRecommender().fit(labelset(2), bundle())

        assert not model.is_fitted
        assert model.fit_calls == 0
        assert model.trained_on_events == 2

    def test_exactly_the_minimum_is_enough(self):
        assert StubRecommender().fit(labelset(5), bundle()).is_fitted

    def test_fit_returns_self_so_calls_can_chain(self):
        model = StubRecommender()

        assert model.fit(labelset(9), bundle()) is model

    def test_scoring_an_unfitted_model_says_what_went_wrong(self):
        model = StubRecommender().fit(labelset(1), bundle())

        with pytest.raises(UnfittedModelError, match="fallen back"):
            model.score(bundle(), np.arange(3))


class TestPersistence:
    def test_saving_records_the_rung_and_what_it_saw(self):
        model = StubRecommender().fit(labelset(8), bundle())

        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1"
            model.save(path)
            stored = json.loads((path / "params.json").read_text(encoding="utf-8"))

        assert stored["rung"] == "stub"
        assert stored["trained_on_events"] == 8
        assert stored["is_fitted"] is True

    def test_loading_restores_the_bookkeeping(self, tmp_path):
        original = StubRecommender().fit(labelset(8), bundle())
        original.save(tmp_path / "v1")

        restored = StubRecommender().load(tmp_path / "v1")

        assert restored.trained_on_events == 8
        assert restored.is_fitted is True


class TestColdStart:
    def test_it_is_always_fitted_because_it_has_no_parameters(self):
        model = ColdStartRecommender().fit(labelset(0), bundle())

        assert model.is_fitted

    def test_it_spreads_across_clusters_rather_than_repeating_one(self):
        # The property the whole rung exists for. Ten images in three clusters:
        # successive picks should move between clusters, not stay in one.
        clusters = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        features = bundle(10, clusters=clusters)
        model = ColdStartRecommender(seed=1729).fit(labelset(0), features)

        chosen_clusters = []
        for _ in range(6):
            scores = model.score(features, np.arange(10))
            best = int(np.argmax(scores))
            chosen_clusters.append(clusters[best])
            model.note_shown(clusters[best])

        # Six picks over three clusters should touch all three, twice each.
        assert set(chosen_clusters) == {0, 1, 2}
        assert max(chosen_clusters.count(c) for c in {0, 1, 2}) <= 3

    def test_a_cluster_already_drawn_from_scores_lower(self):
        clusters = [0, 1]
        features = bundle(2, clusters=clusters)
        model = ColdStartRecommender().fit(labelset(0), features)

        model.note_shown(0)
        scores = model.score(features, np.arange(2))

        assert scores[1] > scores[0], "the undrawn cluster should win"

    def test_the_random_tiebreak_never_reorders_clusters(self):
        # The tiebreak is scaled below one full draw, so it only ever chooses
        # *which* image within a cluster -- never which cluster.
        clusters = [0] * 50 + [1] * 50
        features = bundle(100, clusters=clusters)
        model = ColdStartRecommender().fit(labelset(0), features)
        model.note_shown(0)
        model.note_shown(0)

        scores = model.score(features, np.arange(100))

        assert scores[50:].min() > scores[:50].max()

    def test_note_shown_ignores_a_missing_cluster(self):
        model = ColdStartRecommender()

        model.note_shown(None)  # must not raise

        assert sum(model._draws_per_cluster.values()) == 0

    def test_without_clusters_it_falls_back_to_random_and_warns(self, caplog):
        # A feed quietly serving random images looks the same as one that works,
        # so this must be loud.
        features = bundle(10, clusters=None)
        model = ColdStartRecommender().fit(labelset(0), features)

        with caplog.at_level("WARNING"):
            scores = model.score(features, np.arange(10))

        assert len(scores) == 10
        assert "jobs.select" in caplog.text

    def test_the_same_seed_produces_the_same_order(self):
        clusters = list(range(10))
        features = bundle(10, clusters=clusters)

        first = ColdStartRecommender(seed=7).fit(labelset(0), features)
        second = ColdStartRecommender(seed=7).fit(labelset(0), features)

        np.testing.assert_allclose(
            first.score(features, np.arange(10)), second.score(features, np.arange(10))
        )

    def test_it_explains_itself_as_cold_start(self):
        features = bundle(5, clusters=[0, 1, 2, 3, 4])
        model = ColdStartRecommender().fit(labelset(0), features)

        explanation = model.explain(features, 2)

        assert explanation.reason == "cold_start"
        assert explanation.detail["cluster64"] == 2
        assert "learning" in explanation.sentence.lower()

    def test_its_params_record_the_seed_and_the_target(self):
        params = ColdStartRecommender(seed=99).params

        assert params["seed"] == 99
        assert params["label_target"] == COLDSTART_LABEL_TARGET

    def test_the_handover_point_matches_the_brief(self):
        assert COLDSTART_LABEL_TARGET == 30
        assert coldstart_is_active(0)
        assert coldstart_is_active(29)
        assert not coldstart_is_active(30)
        assert not coldstart_is_active(500)
