"""API tests against a temporary corpus, through FastAPI's TestClient.

Each test gets its own database, derivative files and embedding store under
`tmp_path`, so nothing here touches the real 80,000-image corpus and the tests
can run while a background job is writing to it.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visualtaste.api import deps
from visualtaste.api.deps import AppState
from visualtaste.api.main import app
from visualtaste.config import Settings
from visualtaste.db import repo
from visualtaste.vision.store import EmbeddingStore

IMAGE_COUNT = 24


def hex_id(n: int) -> str:
    return f"{n:016x}"


@pytest.fixture
def corpus(tmp_path):
    """A small, complete corpus: rows, derivatives, embeddings and clusters."""
    return build_corpus(tmp_path, IMAGE_COUNT)


def build_corpus(tmp_path, image_count: int, clusters: int = 6) -> Settings:
    """The `corpus` fixture's body, reusable at other sizes (tests/test_training.py)."""
    IMAGE_COUNT = image_count
    settings = Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        min_free_gb=0,
        seed=1729,
    )
    settings.ensure_dirs()

    connection = repo.init_db(settings.db_path)
    rng = np.random.default_rng(1729)

    vectors = rng.normal(size=(IMAGE_COUNT, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [hex_id(n) for n in range(IMAGE_COUNT)]

    for index, image_id in enumerate(ids):
        repo.upsert_image(
            connection,
            image_id=image_id,
            source="pd12m" if index % 2 else "open_images",
            source_id=f"src{index}",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            orig_path="original.jpg",
            source_url=f"https://example.invalid/photo/{index}",
            author=f"Photographer {index}",
            author_url="https://example.invalid/author",
            title=f"Image {index}",
            width=800,
            height=600,
            phash=f"{index:016x}",
            dominant_colours=["#112233", "#445566"],
        )
        connection.execute(
            "UPDATE images SET selected = 1, cluster64 = ?, cluster400 = ? WHERE image_id = ?",
            (index % clusters, index, image_id),
        )
        for path in (settings.display_path(image_id), settings.thumb_path(image_id)):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (index * 7 % 255, 90, 120)).save(path, format="WEBP")

    connection.commit()
    connection.close()

    # A *complete* corpus: both encoders plus the colour and PCA feature
    # blocks, so `GET /health` can legitimately report "ok". A fixture missing
    # a block would make every test run against a degraded app and hide
    # whatever only happens when everything is present.
    for model_key, dim in (("siglip2_b16_224", 16), ("dinov2_b14", 16)):
        block = rng.normal(size=(IMAGE_COUNT, dim)).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        store = EmbeddingStore(settings.embedding_dir(model_key), model_key)
        store.write_shard(0, ids, vectors if model_key == "siglip2_b16_224" else block)
        store.merge()
        store.write_meta(
            checkpoint=f"{model_key}/test",
            dim=dim,
            count=IMAGE_COUNT,
            preprocess="test",
            device="cpu",
        )

    settings.features_dir.mkdir(parents=True, exist_ok=True)
    for stem, width in (("colour64", 64), ("pca128", 8)):
        block = rng.normal(size=(IMAGE_COUNT, width)).astype(np.float32)
        with (settings.features_dir / f"{stem}.npy").open("wb") as handle:
            np.save(handle, block)
        with (settings.features_dir / f"{stem}_ids.npy").open("wb") as handle:
            np.save(handle, np.array(ids, dtype="<U16"))

    return settings


@pytest.fixture
def client(corpus):
    state = AppState(settings=corpus)
    deps.set_state(state)
    with TestClient(app) as test_client:
        yield test_client


def swipe(client, card, action="like"):
    return client.post(
        "/api/events",
        json={
            "image_id": card["image"]["image_id"],
            "action": action,
            "impression_id": card["impression_id"],
            "dwell_ms": 1200,
        },
    )


class TestHealth:
    def test_reports_ok_on_a_complete_corpus(self, client):
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["images_ok"] == IMAGE_COUNT
        assert body["images_selected"] == IMAGE_COUNT
        assert body["embeddings"]["siglip2_b16_224"] == IMAGE_COUNT
        assert body["notes"] == []

    def test_reports_the_schema_version(self, client):
        assert client.get("/api/health").json()["schema_version"] == 1

    def test_reports_disk_space_against_the_guard(self, client, monkeypatch):
        # Below the guard, downloads refuse to run; the feed still serves, so
        # low disk shows in its own fields rather than as `degraded`.
        import visualtaste.config as config

        monkeypatch.setattr(config, "free_gb", lambda path: 3.0)
        body = client.get("/api/health").json()

        assert body["disk_free_gb"] == 3.0
        assert body["disk_min_free_gb"] >= 0
        assert body["disk_ok"] is (body["disk_min_free_gb"] <= 3.0)
        assert body["status"] == "ok"

    def test_reports_degraded_with_a_reason_when_embeddings_are_missing(self, tmp_path):
        # During the build this is the normal state, and "degraded plus a note"
        # is the difference between "not finished yet" and "broken".
        settings = Settings(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            archive_root=tmp_path / "archive",
            min_free_gb=0,
        )
        settings.ensure_dirs()
        repo.init_db(settings.db_path).close()
        deps.set_state(AppState(settings=settings))

        with TestClient(app) as bare:
            body = bare.get("/api/health").json()

        assert body["status"] == "degraded"
        assert any("jobs.embed" in note for note in body["notes"])


class TestFeed:
    def test_returns_the_requested_number_of_cards(self, client):
        body = client.get("/api/feed/next?n=5").json()

        assert len(body["cards"]) == 5
        assert body["rung"] == "r0"

    def test_every_card_carries_its_attribution(self, client):
        # Open Images is CC BY 2.0 and requires credit, so this is a licensing
        # requirement rather than a nicety.
        body = client.get("/api/feed/next").json()

        for card in body["cards"]:
            attribution = card["image"]["attribution"]
            assert attribution["license_url"].startswith("http")
            assert attribution["author"]
            assert attribution["source"]

    def test_every_card_carries_an_explanation(self, client):
        body = client.get("/api/feed/next").json()

        for card in body["cards"]:
            assert card["reason"] == "cold_start"
            assert card["sentence"]

    def test_cards_come_with_display_and_thumb_urls(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]

        assert card["image"]["display_url"].endswith("/display")
        assert card["image"]["thumb_url"].endswith("/thumb")

    def test_an_impression_is_recorded_for_every_card(self, client, corpus):
        body = client.get("/api/feed/next?n=4").json()

        connection = repo.connect(corpus.db_path, read_only=True)
        rows = repo.query(connection, "SELECT * FROM impressions ORDER BY rank")
        connection.close()

        assert len(rows) == 4
        assert [row["rank"] for row in rows] == [0, 1, 2, 3]
        assert all(row["batch_id"] == body["batch_id"] for row in rows)
        assert all(row["reason"] == "cold_start" for row in rows)

    def test_the_impression_id_is_returned_so_outcomes_can_be_attributed(self, client):
        cards = client.get("/api/feed/next").json()["cards"]

        assert all(card["impression_id"] > 0 for card in cards)
        assert len({card["impression_id"] for card in cards}) == len(cards)

    def test_a_rated_image_is_not_offered_again(self, client):
        first = client.get("/api/feed/next?n=3").json()["cards"]
        for card in first:
            swipe(client, card)

        second = client.get("/api/feed/next?n=3").json()["cards"]

        rated = {card["image"]["image_id"] for card in first}
        assert not rated & {card["image"]["image_id"] for card in second}

    def test_cold_start_spreads_across_clusters(self, client, corpus):
        # The property R0 exists for: the first few cards should not all come
        # from one cluster.
        cards = client.get("/api/feed/next?n=6").json()["cards"]

        connection = repo.connect(corpus.db_path, read_only=True)
        clusters = {
            repo.get_image(connection, card["image"]["image_id"])["cluster64"] for card in cards
        }
        connection.close()

        assert len(clusters) >= 4, f"only {len(clusters)} distinct clusters in 6 cards"

    def test_an_exhausted_corpus_returns_no_cards_rather_than_failing(self, client):
        for _ in range(IMAGE_COUNT // 4 + 2):
            for card in client.get("/api/feed/next?n=4").json()["cards"]:
                swipe(client, card)

        body = client.get("/api/feed/next").json()

        assert body["cards"] == []

    def test_a_missing_corpus_is_a_503_not_an_empty_feed(self, tmp_path):
        # An empty feed and an unbuilt corpus look identical to the UI, and the
        # difference matters enormously to whoever is debugging it.
        settings = Settings(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            archive_root=tmp_path / "archive",
            min_free_gb=0,
        )
        settings.ensure_dirs()
        repo.init_db(settings.db_path).close()
        deps.set_state(AppState(settings=settings))

        with TestClient(app) as bare:
            response = bare.get("/api/feed/next")

        assert response.status_code == 503
        assert "jobs.embed" in response.json()["detail"]

    def test_n_is_validated(self, client):
        assert client.get("/api/feed/next?n=0").status_code == 422
        assert client.get("/api/feed/next?n=99").status_code == 422


class TestEvents:
    def test_recording_a_like(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]

        response = swipe(client, card, "like")

        assert response.status_code == 201
        body = response.json()
        assert body["action"] == "like"
        assert body["labels_so_far"] == 1
        assert body["superseded_event_id"] is None

    @pytest.mark.parametrize("action", ["like", "dislike", "skip", "save"])
    def test_all_four_actions_are_accepted(self, client, action):
        card = client.get("/api/feed/next").json()["cards"][0]

        assert swipe(client, card, action).status_code == 201

    def test_an_invalid_action_is_rejected(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]

        response = client.post(
            "/api/events",
            json={"image_id": card["image"]["image_id"], "action": "shrug"},
        )

        assert response.status_code == 422

    def test_an_unknown_image_is_a_404(self, client):
        response = client.post(
            "/api/events", json={"image_id": "ffffffffffffffff", "action": "like"}
        )

        assert response.status_code == 404

    def test_dwell_is_stored_and_capped(self, client, corpus):
        # A client is not to be trusted with an invariant: a laptop asleep for
        # an hour would otherwise log 3,600,000 ms.
        card = client.get("/api/feed/next").json()["cards"][0]

        client.post(
            "/api/events",
            json={
                "image_id": card["image"]["image_id"],
                "action": "like",
                "dwell_ms": 999_999,
            },
        )

        connection = repo.connect(corpus.db_path, read_only=True)
        dwell = repo.query(connection, "SELECT dwell_ms FROM events")[0]["dwell_ms"]
        connection.close()

        assert dwell == 30_000

    def test_a_negative_dwell_is_rejected(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]

        response = client.post(
            "/api/events",
            json={
                "image_id": card["image"]["image_id"],
                "action": "like",
                "dwell_ms": -5,
            },
        )

        assert response.status_code == 422

    def test_the_impression_link_is_stored(self, client, corpus):
        card = client.get("/api/feed/next").json()["cards"][0]

        swipe(client, card)

        connection = repo.connect(corpus.db_path, read_only=True)
        row = repo.query(connection, "SELECT impression_id FROM events")[0]
        connection.close()

        assert row["impression_id"] == card["impression_id"]

    def test_the_retrain_flag_fires_on_the_boundary(self, client):
        # Every 20 labels by default, so the UI can say "the model is updating"
        # at the right moments.
        # Swiped as they arrive, not collected first: an unrated card is
        # offered again on the next request, so batching them up would rate the
        # same image twice and supersede rather than add a label.
        flags: list[bool] = []
        while len(flags) < 20:
            for card in client.get("/api/feed/next?n=5").json()["cards"]:
                flags.append(swipe(client, card).json()["retrain_queued"])

        assert flags[19] is True, "the 20th label should queue a retrain"
        assert not any(flags[:19])


class TestChangingAVerdict:
    def test_a_second_verdict_supersedes_the_first(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        first = swipe(client, card, "like").json()

        second = swipe(client, card, "dislike").json()

        assert second["superseded_event_id"] == first["event_id"]
        assert second["labels_so_far"] == 1, "one image is still one label"

    def test_both_rows_survive_in_the_log(self, client, corpus):
        card = client.get("/api/feed/next").json()["cards"][0]
        swipe(client, card, "like")
        swipe(client, card, "dislike")

        connection = repo.connect(corpus.db_path, read_only=True)
        rows = repo.query(connection, "SELECT * FROM events ORDER BY event_id")
        connection.close()

        assert len(rows) == 2
        assert rows[0]["undone"] == 1
        assert rows[0]["superseded_by"] == rows[1]["event_id"]

    def test_the_latest_verdict_is_the_one_that_counts(self, client, corpus):
        from visualtaste.rec.labels import build_label_set

        card = client.get("/api/feed/next").json()["cards"][0]
        swipe(client, card, "like")
        swipe(client, card, "dislike")

        connection = repo.connect(corpus.db_path, read_only=True)
        labels = build_label_set(connection, 1)
        connection.close()

        assert labels.actions == ["dislike"]


class TestUndo:
    def test_undoing_removes_the_label(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        event = swipe(client, card).json()

        response = client.post(f"/api/events/{event['event_id']}/undo")

        assert response.status_code == 200
        assert response.json()["labels_so_far"] == 0

    def test_undoing_a_changed_verdict_revives_the_earlier_one(self, client, corpus):
        # Without reviving, undoing would leave the image unlabelled even though
        # the user had expressed an opinion about it twice.
        from visualtaste.rec.labels import build_label_set

        card = client.get("/api/feed/next").json()["cards"][0]
        swipe(client, card, "like")
        second = swipe(client, card, "dislike").json()

        client.post(f"/api/events/{second['event_id']}/undo")

        connection = repo.connect(corpus.db_path, read_only=True)
        labels = build_label_set(connection, 1)
        connection.close()
        assert labels.actions == ["like"]

    def test_undoing_twice_is_a_conflict_not_a_silent_success(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        event = swipe(client, card).json()
        client.post(f"/api/events/{event['event_id']}/undo")

        assert client.post(f"/api/events/{event['event_id']}/undo").status_code == 409

    def test_undoing_an_unknown_event_is_a_404(self, client):
        assert client.post("/api/events/9999/undo").status_code == 404

    def test_an_undone_image_is_still_not_offered_again(self, client):
        # It has been seen; offering it again as if new would look like a bug.
        card = client.get("/api/feed/next").json()["cards"][0]
        event = swipe(client, card).json()
        client.post(f"/api/events/{event['event_id']}/undo")

        later = client.get("/api/feed/next?n=10").json()["cards"]

        assert card["image"]["image_id"] not in {c["image"]["image_id"] for c in later}


class TestImages:
    def test_image_detail_includes_the_internals(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        image_id = card["image"]["image_id"]

        body = client.get(f"/api/images/{image_id}").json()

        assert body["status"] == "ok"
        assert body["selected"] is True
        assert body["cluster64"] is not None
        assert body["attribution"]["license_url"]

    def test_an_unknown_image_is_a_404(self, client):
        assert client.get("/api/images/ffffffffffffffff").status_code == 404

    @pytest.mark.parametrize(
        "bad_id",
        [
            "..",  # parent directory
            "..%2f..%2fdb",  # encoded traversal
            "FFFFFFFFFFFFFFFF",  # upper case: ids are lower-case hex
            "ffffffffffffffff0",  # 17 characters
            "fffffffffffffff",  # 15 characters
            "gggggggggggggggg",  # not hex
            "ffffffff%00ffffff",  # embedded NUL
        ],
    )
    @pytest.mark.parametrize("suffix", ["", "/display", "/thumb", "/similar"])
    def test_a_malformed_id_is_rejected_before_any_path_is_built(self, client, bad_id, suffix):
        response = client.get(f"/api/images/{bad_id}{suffix}")
        assert response.status_code in (404, 422)
        assert response.headers.get("content-type", "").startswith("application/json")

    def test_a_malformed_id_cannot_be_reviewed(self, client):
        response = client.post("/api/review/..", json={"decision": "keep"})
        assert response.status_code in (404, 422)


class TestCors:
    """The browser may call the API only from the dev server's own origins."""

    def test_the_dev_origin_is_allowed(self, client):
        response = client.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"})
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

    def test_any_other_origin_is_not(self, client):
        response = client.get("/api/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers

    def test_a_preflight_from_another_origin_is_refused(self, client):
        response = client.options(
            "/api/events",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert response.headers.get("access-control-allow-origin") is None

    def test_the_display_file_is_served_as_webp(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]

        response = client.get(card["image"]["display_url"])

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"

    def test_derivatives_are_served_with_an_immutable_cache_header(self, client):
        # The content of one of these URLs can never change: image_id is the
        # hash of the bytes. So the browser should never re-request one.
        card = client.get("/api/feed/next").json()["cards"][0]

        response = client.get(card["image"]["thumb_url"])

        assert "immutable" in response.headers["cache-control"]
        assert "max-age=31536000" in response.headers["cache-control"]

    def test_a_missing_derivative_says_what_to_run(self, client, corpus):
        card = client.get("/api/feed/next").json()["cards"][0]
        image_id = card["image"]["image_id"]
        corpus.display_path(image_id).unlink()

        response = client.get(f"/api/images/{image_id}/display")

        assert response.status_code == 404
        assert "jobs.ingest" in response.json()["detail"]

    def test_similar_returns_neighbours_ordered_by_similarity(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        image_id = card["image"]["image_id"]

        body = client.get(f"/api/images/{image_id}/similar?k=5").json()

        assert body["image_id"] == image_id
        assert len(body["neighbours"]) == 5
        similarities = [n["similarity"] for n in body["neighbours"]]
        assert similarities == sorted(similarities, reverse=True)

    def test_similar_never_returns_the_query_image(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        image_id = card["image"]["image_id"]

        body = client.get(f"/api/images/{image_id}/similar?k=10").json()

        assert image_id not in {n["image"]["image_id"] for n in body["neighbours"]}

    def test_an_unknown_similarity_model_is_rejected(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        image_id = card["image"]["image_id"]

        response = client.get(f"/api/images/{image_id}/similar?model=clip")

        assert response.status_code == 422


class TestHistory:
    def test_lists_rated_images_newest_first(self, client):
        cards = client.get("/api/feed/next?n=3").json()["cards"]
        for card in cards:
            swipe(client, card)

        body = client.get("/api/history").json()

        assert body["total"] == 3
        assert len(body["items"]) == 3
        event_ids = [item["event_id"] for item in body["items"]]
        assert event_ids == sorted(event_ids, reverse=True)

    def test_filters_by_action(self, client):
        cards = client.get("/api/feed/next?n=4").json()["cards"]
        swipe(client, cards[0], "like")
        swipe(client, cards[1], "dislike")
        swipe(client, cards[2], "like")

        body = client.get("/api/history?action=like").json()

        assert body["total"] == 2
        assert all(item["action"] == "like" for item in body["items"])

    def test_undone_events_are_hidden_by_default_and_shown_on_request(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        event = swipe(client, card).json()
        client.post(f"/api/events/{event['event_id']}/undo")

        assert client.get("/api/history").json()["total"] == 0
        assert client.get("/api/history?include_undone=true").json()["total"] == 1

    def test_pagination_reports_the_full_total(self, client):
        cards = client.get("/api/feed/next?n=5").json()["cards"]
        for card in cards:
            swipe(client, card)

        body = client.get("/api/history?limit=2&offset=1").json()

        assert body["total"] == 5
        assert len(body["items"]) == 2
        assert body["offset"] == 1

    def test_history_items_carry_the_image_and_its_attribution(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        swipe(client, card)

        item = client.get("/api/history").json()["items"][0]

        assert item["image"]["attribution"]["license_url"]
        assert item["image"]["thumb_url"]

    def test_an_empty_history_is_not_an_error(self, client):
        body = client.get("/api/history").json()

        assert body == {"items": [], "total": 0, "offset": 0, "limit": 50}


class TestProfile:
    def test_reports_counts_and_the_active_rung(self, client):
        cards = client.get("/api/feed/next?n=3").json()["cards"]
        swipe(client, cards[0], "like")
        swipe(client, cards[1], "save")
        swipe(client, cards[2], "skip")

        body = client.get("/api/profile").json()

        assert body["labels"] == 3
        assert body["counts"] == {"like": 1, "save": 1, "skip": 1}
        assert body["active_rung"] == "r0"
        assert body["coldstart_active"] is True
        assert body["is_simulated"] is False

    def test_reset_archives_rather_than_deletes(self, client, corpus):
        cards = client.get("/api/feed/next?n=2").json()["cards"]
        for card in cards:
            swipe(client, card)

        body = client.post("/api/profile/reset").json()

        assert body["archived_labels"] == 2
        assert body["new_profile_id"] != body["previous_profile_id"]

        connection = repo.connect(corpus.db_path, read_only=True)
        surviving = repo.query(connection, "SELECT * FROM events")
        ended = repo.query(connection, "SELECT * FROM profiles WHERE ended_at IS NOT NULL")
        connection.close()

        assert len(surviving) == 2, "reset deleted events"
        assert len(ended) == 1

    def test_the_new_profile_starts_empty(self, client):
        card = client.get("/api/feed/next").json()["cards"][0]
        swipe(client, card)
        client.post("/api/profile/reset")

        assert client.get("/api/profile").json()["labels"] == 0

    def test_images_rated_by_the_old_profile_are_offered_again(self, client):
        # A new profile is a different person as far as the recommender is
        # concerned, so the corpus is fresh for them.
        card = client.get("/api/feed/next").json()["cards"][0]
        swipe(client, card)
        client.post("/api/profile/reset")

        offered = {
            c["image"]["image_id"] for c in client.get("/api/feed/next?n=20").json()["cards"]
        }

        assert card["image"]["image_id"] in offered


class TestLoopbackOnly:
    def test_cors_allows_only_loopback_origins(self):
        # Brief §0.7: nothing is ever exposed beyond this machine.
        from visualtaste.api.main import DEV_ORIGINS

        assert DEV_ORIGINS == [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
        assert all("127.0.0.1" in o or "localhost" in o for o in DEV_ORIGINS)

    def test_the_configured_host_is_loopback(self, corpus):
        assert corpus.api_host == "127.0.0.1"
