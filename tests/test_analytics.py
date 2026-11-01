"""Phase 13's API: stats, the Embedding Map, review, evaluation, per-version metrics."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_api import build_corpus
from tests.test_training import POOL, alternate, swipe_batches
from visualtaste.api import deps
from visualtaste.api.deps import AppState
from visualtaste.api.main import app


@pytest.fixture
def settings(tmp_path):
    return build_corpus(tmp_path, POOL, clusters=40)


@pytest.fixture
def client(settings):
    deps.set_state(AppState(settings=settings))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def trained(client):
    swipe_batches(client, 60, alternate)
    body = client.post("/api/models/train", json={"rung": "r4"}).json()
    assert body["status"] == "swapped"
    swipe_batches(client, 20, alternate)  # some R4-served, explored, calibrated cards
    return body["version"]


class TestStats:
    def test_summary(self, client, trained):
        body = client.get("/api/stats/summary").json()
        assert body["labels"] == 80
        assert body["serving_rung"] == "r4"
        assert 0 <= body["like_rate"] <= 1

    def test_learning_curve_carries_coverage_and_switches(self, client, trained):
        body = client.get("/api/stats/learning-curve?window=10").json()
        assert len(body["like_rate"]) == len(body["coverage"]) == len(body["interests"]) == 80
        versions = [s["model_version"] for s in body["switches"]]
        assert versions[0] == "r0_bootstrap" and trained in versions
        assert all(0 <= c <= 1 for c in body["coverage"])

    def test_clusters_have_intervals(self, client, trained):
        body = client.get("/api/stats/clusters").json()
        assert len(body) == 64
        rated = [c for c in body if c["rate"]["trials"]]
        assert rated and all(
            c["rate"]["low"] <= c["rate"]["mean"] <= c["rate"]["high"] for c in rated
        )

    def test_diversity_calibration_exploration_scores(self, client, trained):
        diversity = client.get("/api/stats/diversity").json()
        assert len(diversity["batch_diversity"]) == len(diversity["batch_novelty"]) >= 16

        calibration = client.get("/api/stats/calibration").json()
        assert calibration["samples"] > 0 and calibration["ece"] is not None

        exploration = client.get("/api/stats/exploration").json()
        assert exploration["explore"]["trials"] >= 1

        scores = client.get("/api/stats/scores").json()
        # "Unseen" means no event: a shown-but-unswiped impression does not count.
        assert sum(scores["counts"]) == POOL - 80
        # The 20 swipes after training crossed the next retrain boundary, so a
        # newer R4 version may be serving by now; the histogram is of whichever is.
        assert scores["model_version"] == client.get("/api/models").json()["serving_version"]
        assert scores["model_version"].startswith("r4_") and trained.startswith("r4_")

    def test_profiles_hide_simulated_by_default(self, client, trained):
        assert all(not p["is_simulated"] for p in client.get("/api/profiles").json())


class TestModelLab:
    def test_evaluate_fills_the_metrics_table(self, client, trained):
        body = client.post(f"/api/models/{trained}/evaluate").json()
        metrics = body["metrics"]
        assert metrics["score_p95_ms"] > 0
        assert "replay_auc" in metrics and "ece" in metrics

    def test_eval_lists_and_validates_studies(self, client):
        assert "phase9_rungs" in client.get("/api/eval/studies").json()
        assert client.post("/api/eval/run", json={"study": "nope"}).status_code == 400


class TestReview:
    def test_keep_and_remove(self, client, settings):
        connection = deps.get_state().connect()
        connection.execute(
            "UPDATE images SET status = 'quarantined' WHERE image_id IN (?, ?)",
            ("0" * 16, f"{1:016x}"),
        )
        connection.commit()
        connection.close()

        listed = client.get("/api/review").json()
        assert {item["image"]["image_id"] for item in listed} == {"0" * 16, f"{1:016x}"}

        removed = client.post(f"/api/review/{'0' * 16}", json={"decision": "remove"}).json()
        kept = client.post(f"/api/review/{1:016x}", json={"decision": "keep"}).json()
        assert removed["status"] == "removed" and kept["status"] == "ok"

        connection = deps.get_state().connect()
        reason = connection.execute(
            "SELECT reason FROM removals WHERE image_id = ?", ("0" * 16,)
        ).fetchone()
        connection.close()
        assert reason["reason"] == "removed after safety review"
        assert client.get("/api/review").json() == []


class TestEmbeddingMap:
    def test_missing_projection_says_how_to_make_one(self, client):
        response = client.get("/api/viz/projection")
        assert response.status_code == 404
        assert "jobs.project" in response.json()["detail"]

    def test_projection_and_taste_path(self, client, settings, trained):
        from visualtaste.viz.projection import fit_projection, save

        features = deps.get_state().features
        save(
            fit_projection(features.siglip, features.image_ids, model_key="siglip2_b16_224"),
            settings.features_dir,
        )

        body = client.get("/api/viz/projection").json()
        assert len(body["x"]) == len(body["y"]) == len(body["image_ids"]) == POOL
        assert len(body["score"]) == POOL
        assert set(body["liked"]) <= set(body["image_ids"])
        assert np.isfinite(body["x"]).all()

        path = client.get("/api/viz/taste-path?every=20").json()
        assert [p["labels"] for p in path["path"]] == [20, 40, 60, 80]
