"""Online retraining, the hot swap, and the handover chain through the real API.

Part 2 §5.5 and §4.4. Each guarantee in `api/training.py` has a test here:
requests during a retrain see the old model, a failed retrain keeps it, two
retrains never run at once -- and a fresh profile, a profile with 30 likes and
no dislikes, and a profile whose active artefact was deleted all get a batch.
"""

from __future__ import annotations

import json
import shutil
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_api import build_corpus
from visualtaste.api import deps, training
from visualtaste.api.deps import AppState
from visualtaste.api.main import app
from visualtaste.rec import registry

POOL = 120


@pytest.fixture
def settings(tmp_path):
    # 40 clusters, so after 40 cold-start labels most are still barely rated
    # and R4's exploration quota has somewhere to go.
    return build_corpus(tmp_path, POOL, clusters=40)


@pytest.fixture
def client(settings):
    deps.set_state(AppState(settings=settings))
    with TestClient(app) as test_client:
        yield test_client


def swipe_batches(client, count: int, decide) -> None:
    """Swipe `count` cards, choosing each action with `decide(index, card)`."""
    done = 0
    while done < count:
        cards = client.get("/api/feed/next").json()["cards"]
        assert cards, "the pool ran out"
        for card in cards:
            if done >= count:
                break
            response = client.post(
                "/api/events",
                json={
                    "image_id": card["image"]["image_id"],
                    "action": decide(done, card),
                    "impression_id": card["impression_id"],
                },
            )
            assert response.status_code == 201
            done += 1


def alternate(index, _card):
    return "like" if index % 2 else "dislike"


def default_rung() -> str:
    """The configured default, chosen by the final study (DECISIONS #29)."""
    return deps.get_state().settings.default_rung


#: Reasons a learned rung gives; cold start's is `cold_start`.
LEARNED_REASONS = ("model_confident", "explore", "similar_likes")


class TestHandoverChain:
    def test_a_fresh_profile_serves_cold_start(self, client):
        response = client.get("/api/feed/next")
        body = response.json()
        assert response.status_code == 200
        assert body["model_version"] == "r0_bootstrap"
        assert body["cards"][0]["explanation"]["fallback"]["from"] == default_rung()

    def test_the_default_rung_takes_over_after_enough_labels_of_both_classes(self, client):
        swipe_batches(client, 40, alternate)  # crosses the boundaries at 20 and 40

        body = client.get("/api/feed/next").json()
        rung = default_rung()
        assert body["rung"] == rung
        assert body["model_version"].startswith(f"{rung}_")
        assert all(
            card["reason"] in LEARNED_REASONS or card["reason"].startswith("interest:")
            for card in body["cards"]
        )

        models = client.get("/api/models").json()
        active = [v for v in models["versions"] if v["is_active"]]
        assert len(active) == 1 and active[0]["rung"] == rung
        assert "validation_curve" in active[0]["metrics"]

    def test_thirty_likes_and_no_dislikes_still_serve(self, client):
        swipe_batches(client, 40, lambda *_: "like")

        response = client.get("/api/feed/next")
        body = response.json()
        assert response.status_code == 200
        assert body["model_version"] == "r0_bootstrap"
        assert "negative" in body["cards"][0]["explanation"]["fallback"]["reason"]

    def test_a_deleted_active_artefact_falls_back_on_restart(self, settings, client):
        swipe_batches(client, 40, alternate)
        connection = deps.get_state().connect()
        active = registry.active_version(connection)
        connection.close()
        assert active is not None
        shutil.rmtree(active.artifact_path)

        # A restart over the broken artefact: the startup retrain rebuilds one.
        deps.set_state(AppState(settings=settings))
        with TestClient(app) as restarted:
            response = restarted.get("/api/feed/next")
            assert response.status_code == 200
            assert len(response.json()["cards"]) == 5

    def test_reset_returns_to_cold_start_and_deactivates(self, client):
        swipe_batches(client, 40, alternate)
        client.post("/api/profile/reset")

        body = client.get("/api/feed/next").json()
        assert body["model_version"] == "r0_bootstrap"
        assert not any(v["is_active"] for v in client.get("/api/models").json()["versions"])


class TestHotSwap:
    def _trained_state(self, client) -> AppState:
        swipe_batches(client, 40, alternate)
        return deps.get_state()

    def test_requests_during_a_retrain_get_the_old_model(self, client, monkeypatch):
        state = self._trained_state(client)
        old_model, old_version = state.serving()

        started, release = threading.Event(), threading.Event()
        real_fit = type(old_model).fit

        def slow_fit(self, labels, features):
            started.set()
            release.wait(timeout=10)
            return real_fit(self, labels, features)

        monkeypatch.setattr(type(old_model), "fit", slow_fit)
        worker = threading.Thread(target=training.retrain, args=(state,))
        worker.start()
        assert started.wait(timeout=10)

        during = client.get("/api/feed/next").json()
        release.set()
        worker.join(timeout=30)

        assert during["model_version"] == old_version
        # Same label count in the same minute can reuse the version name, so
        # the swap is checked by identity.
        assert state.serving()[0] is not old_model

    def test_a_failed_retrain_keeps_the_old_model(self, client, monkeypatch):
        state = self._trained_state(client)
        before = state.serving()

        def broken_fit(self, labels, features):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(type(before[0]), "fit", broken_fit)
        outcome = training.retrain(state)

        assert outcome.status == "failed"
        assert "disk on fire" in outcome.detail
        assert state.serving() == before
        assert client.get("/api/feed/next").json()["model_version"] == before[1]

    def test_two_retrains_are_never_in_flight(self, client):
        state = self._trained_state(client)
        state.retrain_lock.acquire()
        try:
            assert training.retrain(state).status == "busy"
            assert client.post("/api/models/train", json={}).status_code == 409
        finally:
            state.retrain_lock.release()
        assert training.retrain(state).status == "swapped"


class TestModelsRoutes:
    def test_train_and_activate(self, client):
        swipe_batches(client, 30, alternate)
        trained = client.post("/api/models/train", json={"rung": "r3", "activate": False}).json()
        assert trained["status"] == "registered"

        activated = client.post(f"/api/models/{trained['version']}/activate").json()
        assert activated["is_active"]
        assert client.get("/api/feed/next").json()["model_version"] == trained["version"]

    def test_r6_cannot_be_trained_from_the_api(self, client):
        response = client.post("/api/models/train", json={"rung": "r6"})
        assert response.status_code == 400
        assert "train_neural" in response.json()["detail"]

    def test_an_unknown_version_is_404(self, client):
        assert client.post("/api/models/nope/activate").status_code == 404

    def test_training_with_too_few_labels_says_so(self, client):
        swipe_batches(client, 10, alternate)
        body = client.post("/api/models/train", json={"rung": "r3"}).json()
        assert body["status"] == "not_fitted"
        assert "needs 30" in body["detail"]


def test_the_retrain_scores_are_probabilities(client):
    swipe_batches(client, 40, alternate)
    model, _ = deps.get_state().serving()
    scores = model.score(deps.get_state().features, np.arange(POOL))
    assert np.all((scores > 0) & (scores < 1))


class TestR4ThroughTheApi:
    """Phase 10: R4 serving, propensities written after the response, interleaving."""

    def _r4_version(self, client) -> str:
        swipe_batches(client, 40, alternate)
        body = client.post("/api/models/train", json={"rung": "r4"}).json()
        assert body["status"] == "swapped"
        return body["version"]

    def test_every_r4_impression_gets_a_propensity(self, client):
        self._r4_version(client)
        body = client.get("/api/feed/next").json()
        assert body["rung"] == "r4"

        # TestClient runs background tasks before returning, so they are written.
        connection = deps.get_state().connect()
        rows = connection.execute(
            "SELECT propensity, explanation_json FROM impressions WHERE batch_id = ?",
            (body["batch_id"],),
        ).fetchall()
        connection.close()
        assert len(rows) == 5
        assert all(row["propensity"] is not None and row["propensity"] >= 0.005 for row in rows)
        assert all(json.loads(row["explanation_json"])["propensity_draws"] == 200 for row in rows)

    def test_a_batch_carries_one_exploration_card(self, client):
        self._r4_version(client)
        reasons = [c["reason"] for c in client.get("/api/feed/next").json()["cards"]]
        assert reasons.count("explore") == 1

    def test_deterministic_rungs_record_propensity_one(self, client):
        swipe_batches(client, 40, alternate)
        # R3 explicitly: the default rung (R4+R5) is stochastic.
        assert client.post("/api/models/train", json={"rung": "r3"}).json()["status"] == "swapped"
        cards = client.get("/api/feed/next").json()["cards"]
        assert {c["propensity"] for c in cards} == {1.0}

    def test_an_interleaving_experiment_end_to_end(self, client):
        r4 = self._r4_version(client)
        r3 = client.post("/api/models/train", json={"rung": "r3", "activate": False}).json()[
            "version"
        ]

        started = client.post("/api/experiments", json={"version_a": r3, "version_b": r4})
        assert started.status_code == 201
        experiment_id = started.json()["experiment_id"]

        for _ in range(4):
            body = client.get("/api/feed/next").json()
            assert body["rung"] == "interleaved"
            assert {c["explanation"]["arm"] for c in body["cards"]} <= {"a", "b"}
            for card in body["cards"]:
                client.post(
                    "/api/events",
                    json={
                        "image_id": card["image"]["image_id"],
                        "action": "like" if card["explanation"]["arm"] == "a" else "dislike",
                        "impression_id": card["impression_id"],
                    },
                )

        listed = client.get("/api/experiments").json()[0]
        assert listed["active"] and listed["wins_a"] + listed["ties"] >= 1
        assert listed["wins_b"] == 0

        stopped = client.post(f"/api/experiments/{experiment_id}/stop").json()
        assert not stopped["active"]
        assert client.get("/api/feed/next").json()["rung"] != "interleaved"

    def test_an_experiment_needs_two_loadable_versions(self, client):
        r4 = self._r4_version(client)
        assert (
            client.post("/api/experiments", json={"version_a": r4, "version_b": r4}).status_code
            == 400
        )
        assert (
            client.post("/api/experiments", json={"version_a": r4, "version_b": "nope"}).status_code
            == 404
        )


class TestInterestsThroughTheApi:
    def test_taste_explains_why_it_is_empty_early(self, client):
        swipe_batches(client, 20, alternate)
        body = client.get("/api/profile/taste").json()
        assert body["interests"] == []
        assert "interests appear at 50" in body["note"]
        assert "concept vocabulary" in body["note"]  # no cache in the temporary corpus

    def test_r3_plus_r5_persists_interests_and_the_profile_shows_them(self, client):
        swipe_batches(client, 110, alternate)  # 55 likes -> 2 interests
        trained = client.post("/api/models/train", json={"rung": "r3+r5"}).json()
        assert trained["status"] == "swapped"

        connection = deps.get_state().connect()
        rows = connection.execute(
            "SELECT interest_id, n_likes FROM interests WHERE model_version = ?",
            (trained["version"],),
        ).fetchall()
        connection.close()
        assert len(rows) >= 2

        body = client.get("/api/profile/taste").json()
        assert body["source"] == trained["version"]
        assert {card["interest_id"] for card in body["interests"]} == {
            r["interest_id"] for r in rows
        }
        for card in body["interests"]:
            assert 1 <= len(card["central_image_ids"]) <= 6
            interval = card["like_rate"]
            assert interval["low"] < interval["mean"] < interval["high"]
        assert body["history"][0]["born"]
