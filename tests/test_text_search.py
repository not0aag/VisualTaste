"""Search by words and a described taste (ROADMAP D4, F9).

The real text tower is a SigLIP 2 model; here a stand-in encoder maps every
text onto one known image's vector, so the expected ranking is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_api import client, corpus, swipe  # noqa: F401 - pytest fixtures
from visualtaste.api import deps
from visualtaste.rec.r0_coldstart import ColdStartRecommender
from visualtaste.rec.r0_text import TextSeededColdStart
from visualtaste.search.text import TextEncoder, clean, load_seed, search

TARGET = 7  # the pool row every text "means"


class FakeTextModel:
    """Encodes any text as the vector of pool row TARGET."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        return np.stack([self.vector for _ in texts])


@pytest.fixture
def fake_text(client):  # noqa: F811
    state = deps.get_state()
    model = FakeTextModel(np.asarray(state.features.siglip[TARGET], dtype=np.float32))
    state.text_encoder = TextEncoder(factory=lambda: model)
    return model


class TestSearch:
    def test_returns_the_closest_images_best_first(self, client, fake_text):  # noqa: F811
        body = client.get("/api/search", params={"q": "  foggy   forest ", "k": 5}).json()
        state = deps.get_state()

        assert body["query"] == "foggy forest"
        assert body["results"][0]["image"]["image_id"] == state.features.image_ids[TARGET]
        scores = [r["similarity"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True) and len(scores) == 5

    def test_the_text_model_loads_once_and_queries_are_cached(self, client, fake_text):  # noqa: F811
        for _ in range(3):
            client.get("/api/search", params={"q": "lighthouse"})
        assert fake_text.calls == 1

    def test_an_empty_query_is_rejected(self, client, fake_text):  # noqa: F811
        assert client.get("/api/search", params={"q": ""}).status_code == 422
        assert client.get("/api/search", params={"q": "   "}).status_code == 422


class TestSeedText:
    def test_round_trip(self, client, fake_text):  # noqa: F811
        assert client.get("/api/profile/seed-text").json() == {"text": None}
        saved = client.post("/api/profile/seed-text", json={"text": "old maps"}).json()
        assert saved == {"text": "old maps"}
        assert client.get("/api/profile/seed-text").json() == {"text": "old maps"}
        assert client.delete("/api/profile/seed-text").json() == {"text": None}
        assert client.get("/api/profile/seed-text").json() == {"text": None}

    def test_the_cold_start_leans_on_the_description(self, client, fake_text):  # noqa: F811
        client.post("/api/profile/seed-text", json={"text": "old maps"})
        body = client.get("/api/feed/next", params={"n": 5}).json()
        reasons = [c["reason"] for c in body["cards"]]
        target_id = deps.get_state().features.image_ids[TARGET]

        assert body["rung"] == "r0+text"
        assert reasons.count("text_seed") == 1  # 20% of five (DECISIONS #30)
        assert reasons.count("cold_start") == 4
        assert target_id in {c["image"]["image_id"] for c in body["cards"]}
        seeded = next(c for c in body["cards"] if c["reason"] == "text_seed")
        assert seeded["sentence"] == 'Close to what you described: "old maps".'

    def test_the_description_survives_a_restart(self, client, fake_text):  # noqa: F811
        client.post("/api/profile/seed-text", json={"text": "harbours at dusk"})
        state = deps.get_state()
        text, vector = load_seed(state.settings.data, state.profile_id)
        assert text == "harbours at dusk"
        assert np.allclose(vector, state.features.siglip[TARGET], atol=1e-5)

    def test_forgetting_the_description_returns_to_plain_cold_start(
        self,
        client,  # noqa: F811
        fake_text,
    ):
        client.post("/api/profile/seed-text", json={"text": "old maps"})
        client.delete("/api/profile/seed-text")
        body = client.get("/api/feed/next", params={"n": 5}).json()
        assert {c["reason"] for c in body["cards"]} == {"cold_start"}


class TestUnits:
    def test_clean_trims_and_rejects_empty(self):
        assert clean("  a   b ") == "a b"
        with pytest.raises(ValueError):
            clean("   ")

    def test_search_excludes_rows(self):
        pool = np.eye(4, dtype=np.float32)
        hits = search(pool, pool[2], k=2, exclude={2})
        assert 2 not in [row for row, _ in hits]

    def test_the_seeded_cold_start_shares_r0s_round_robin(self):
        base = ColdStartRecommender(seed=1)
        seeded = TextSeededColdStart(base, "x", np.ones(4, dtype=np.float32))
        seeded.note_shown(3)
        assert base._draws_per_cluster == {3: 1}


class TestDescribedInterest:
    def test_the_description_is_the_three_nearest_words(self):
        from visualtaste.eval.study import describe_interest

        vectors = np.eye(5, dtype=np.float32)
        centroid = np.array([0.1, 0.9, 0.0, 0.5, 0.3], dtype=np.float32)
        words, vector = describe_interest(centroid, ["a", "b", "c", "d", "e"], vectors)

        assert words == "b, d, e"
        assert np.isclose(np.linalg.norm(vector), 1.0)
        assert vector[1] > 0 and vector[0] == 0

    def test_r0_carries_the_option_without_using_it(self):
        assert ColdStartRecommender().describe_interests == 0
