"""The simulator driving the real feed code, end to end.

This is the test that makes the whole evaluation harness worth trusting. A
simulator with its own scoring loop would happily report a healthy AUC for a
model whose feature blocks are misaligned, whose impressions are never written,
or whose candidate pool is empty -- and all three of those bugs occurred during
this build. So the session must go through `rec/feed.build_batch`, write real
impressions and real events, and be checked against the database afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.config import Settings
from visualtaste.db import repo
from visualtaste.eval.report import (
    aggregate,
    render_learning_curves,
    render_markdown,
    run_session,
    summarise_session,
)
from visualtaste.eval.sim import make_user
from visualtaste.rec.base import FeatureBundle, UnfittedModelError
from visualtaste.rec.feed import build_batch
from visualtaste.rec.labels import LabelSet, build_label_set
from visualtaste.rec.r0_coldstart import ColdStartRecommender
from visualtaste.rec.r1_rocchio import RocchioRecommender

POOL_SIZE = 400
CLUSTERS = 8
DIM = 24


@pytest.fixture
def corpus(tmp_path):
    """A small selected pool with clusters, embeddings and database rows."""
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
    # Clustered vectors, so cluster64 means something and a planted taste can
    # be concentrated rather than spread uniformly.
    centres = rng.normal(size=(CLUSTERS, DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    ids: list[str] = []
    vectors = np.zeros((POOL_SIZE, DIM), dtype=np.float32)
    labels = np.zeros(POOL_SIZE, dtype=np.int32)

    for index in range(POOL_SIZE):
        cluster = index % CLUSTERS
        point = centres[cluster] + rng.normal(scale=0.25, size=DIM).astype(np.float32)
        vectors[index] = point / np.linalg.norm(point)
        labels[index] = cluster

        image_id = f"{index:016x}"
        ids.append(image_id)
        repo.upsert_image(
            connection,
            image_id=image_id,
            source="pd12m",
            source_id=image_id,
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            orig_path="p",
            width=800,
            height=600,
        )
        connection.execute(
            "UPDATE images SET selected = 1, cluster64 = ? WHERE image_id = ?",
            (int(cluster), image_id),
        )
    connection.commit()

    features = FeatureBundle(image_ids=ids, siglip=vectors, cluster64=labels)
    yield settings, connection, features
    connection.close()


class TestRunSession:
    def test_it_produces_the_requested_number_of_swipes(self, corpus):
        settings, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)
        model = ColdStartRecommender(seed=settings.seed)

        result = run_session(connection, features, model, user, swipes=60, seed=1729, refit_every=0)

        assert len(result) == 60

    def test_impressions_and_events_both_reach_the_database(self, corpus):
        # The failure this catches: a simulator that scores images without
        # going through the feed would report metrics while writing nothing.
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=40,
            seed=1729,
            refit_every=0,
        )

        events = repo.query(
            connection,
            "SELECT * FROM events WHERE profile_id = ?",
            (result.profile_id,),
        )
        impressions = repo.query(
            connection,
            "SELECT * FROM impressions WHERE profile_id = ?",
            (result.profile_id,),
        )
        assert len(events) == 40
        assert len(impressions) >= 40

    def test_every_event_links_to_the_impression_that_offered_it(self, corpus):
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=25,
            seed=1729,
            refit_every=0,
        )

        unlinked = repo.query(
            connection,
            "SELECT COUNT(*) AS n FROM events WHERE profile_id = ? AND impression_id IS NULL",
            (result.profile_id,),
        )[0]["n"]
        assert unlinked == 0

    def test_the_session_runs_on_a_simulated_profile(self, corpus):
        # So it can never become the profile the app serves, and its events
        # cannot reach Alen's training data (brief §7.4).
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)
        real_profile = repo.create_profile(connection, note="real")
        connection.commit()

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=20,
            seed=1729,
            refit_every=0,
        )

        row = repo.query(
            connection, "SELECT * FROM profiles WHERE profile_id = ?", (result.profile_id,)
        )[0]
        assert row["is_simulated"] == 1
        assert repo.active_profile_id(connection) == real_profile

    def test_no_image_is_shown_twice(self, corpus):
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=100,
            seed=1729,
            refit_every=0,
        )

        assert len(set(result.image_ids)) == len(result.image_ids)

    def test_the_labels_it_wrote_are_readable_as_a_training_set(self, corpus):
        # The loop the whole project turns on: swipes become labels a model
        # can be fitted to.
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=50,
            seed=1729,
            refit_every=0,
        )
        labels = build_label_set(connection, result.profile_id)

        assert len(labels) == 50
        assert set(labels.actions) <= {"like", "save", "dislike", "skip"}
        assert labels.targets.shape == (50,)

    def test_it_stops_cleanly_when_the_pool_runs_out(self, corpus):
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=POOL_SIZE + 200,
            seed=1729,
            refit_every=0,
        )

        assert len(result) <= POOL_SIZE

    def test_refitting_on_a_schedule_does_not_break_the_session(self, corpus):
        # Mirrors the live system's retrain-every-20-labels behaviour.
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=60,
            seed=1729,
            refit_every=20,
        )

        assert len(result) == 60

    def test_cold_start_spreads_across_clusters(self, corpus):
        # R0's whole purpose, measured through the real feed rather than by
        # inspecting its score function.
        _, connection, features = corpus
        user = make_user(features.siglip, seed=1729, cluster_labels=features.cluster64)

        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=1),
            user,
            swipes=CLUSTERS * 2,
            seed=1729,
            refit_every=0,
        )

        assert len(set(result.shown_clusters)) == CLUSTERS

    def test_two_runs_with_the_same_seed_agree(self, corpus):
        _, connection, features = corpus

        first = run_session(
            connection,
            features,
            ColdStartRecommender(seed=5),
            make_user(features.siglip, seed=1729, cluster_labels=features.cluster64),
            swipes=30,
            seed=1729,
            refit_every=0,
        )
        second = run_session(
            connection,
            features,
            ColdStartRecommender(seed=5),
            make_user(features.siglip, seed=1729, cluster_labels=features.cluster64),
            swipes=30,
            seed=1729,
            refit_every=0,
        )

        assert first.actions == second.actions
        assert first.image_ids == second.image_ids


class TestSummaries:
    def _summary(self, corpus, seed=1729, swipes=80):
        _, connection, features = corpus
        user = make_user(features.siglip, seed=seed, cluster_labels=features.cluster64)
        result = run_session(
            connection,
            features,
            ColdStartRecommender(seed=seed),
            user,
            swipes=swipes,
            seed=seed,
            refit_every=0,
        )
        return result, summarise_session(result)

    def test_the_summary_reports_every_metric(self, corpus):
        _, summary = self._summary(corpus)

        assert summary.swipes == 80
        assert 0.0 <= summary.like_rate <= 1.0
        assert 0.0 <= summary.coverage <= 1.0
        assert 0.0 <= summary.interests_recalled <= 1.0
        assert summary.mean_batch_diversity > 0
        assert summary.mean_novelty > 0

    def test_diversity_and_novelty_are_measured_as_offered(self, corpus):
        # Both are properties of the batch the model chose, so they are
        # recorded before the user reacts to it.
        result, _summary_unused = self._summary(corpus)

        assert len(result.batch_diversities) == pytest.approx(len(result) / 5, abs=2)
        assert len(result.novelties) == len(result)

    def test_aggregating_across_seeds_gives_a_mean_and_spread(self, corpus):
        summaries = [self._summary(corpus, seed=seed, swipes=40)[1] for seed in (1, 2, 3)]

        stats = aggregate(summaries)

        assert "like_rate" in stats
        mean, sd = stats["like_rate"]
        assert 0.0 <= mean <= 1.0
        assert sd >= 0.0

    def test_aggregating_nothing_is_empty_rather_than_an_error(self):
        assert aggregate([]) == {}

    def test_the_markdown_leads_with_the_simulator_caveat(self, corpus):
        # A metric detached from its caveat is how a project talks itself into
        # believing something.
        _, summary = self._summary(corpus, swipes=40)

        text = render_markdown({"r0": [summary]}, swipes=40, seeds=1)

        assert "simulated results" in text
        assert "flatters the" in text
        assert "sim.py" in text
        assert "`r0`" in text

    def test_the_markdown_reports_mean_and_spread(self, corpus):
        summaries = [self._summary(corpus, seed=s, swipes=40)[1] for s in (1, 2)]

        text = render_markdown({"r0": summaries}, swipes=40, seeds=2)

        assert "±" in text

    def test_a_missing_metric_shows_as_a_dash_not_a_zero(self):
        from visualtaste.eval.report import SessionSummary

        empty = SessionSummary(
            rung="r0",
            seed=1,
            swipes=0,
            like_rate=float("nan"),
            like_rate_last_100=float("nan"),
            mean_batch_diversity=float("nan"),
            coverage=float("nan"),
            mean_novelty=float("nan"),
            interests_recalled=float("nan"),
            save_count=0,
            skip_count=0,
        )

        text = render_markdown({"r0": [empty]}, swipes=0, seeds=1)

        assert "--" in text

    def test_the_sparkline_shows_the_shape_of_the_curve(self, corpus):
        result, _ = self._summary(corpus, swipes=120)

        text = render_learning_curves({"r0": [result]})

        assert "Rolling like rate" in text
        assert "`r0" in text


class TestBootstrapHandover:
    """R0 serving on behalf of a rung that has not got enough labels yet.

    This is the ladder of §7.5 as an actual mechanism. Before it existed,
    activating R1 on a fresh profile made the very first `/feed/next` raise --
    the state every new profile starts in.
    """

    def _profile(self, connection):
        profile_id = repo.create_profile(connection, note="bootstrap test")
        connection.commit()
        return profile_id

    def test_an_unfitted_model_is_covered_by_the_bootstrap(self, corpus):
        settings, connection, features = corpus
        profile_id = self._profile(connection)

        batch = build_batch(
            connection,
            features,
            RocchioRecommender(),
            profile_id=profile_id,
            model_version="r1_v1",
            bootstrap=ColdStartRecommender(seed=settings.seed),
        )

        assert len(batch.cards) == 5
        assert batch.rung == "r0"

    def test_the_impression_records_the_rung_that_actually_served(self, corpus):
        # Attribution: a cold-start card filed under r1 would corrupt every
        # later comparison between the two.
        settings, connection, features = corpus
        profile_id = self._profile(connection)

        batch = build_batch(
            connection,
            features,
            RocchioRecommender(),
            profile_id=profile_id,
            model_version="r1_v1",
            bootstrap=ColdStartRecommender(seed=settings.seed),
        )

        versions = {
            row["model_version"]
            for row in connection.execute(
                "SELECT model_version FROM impressions WHERE profile_id = ?", (profile_id,)
            )
        }
        reasons = {card.reason for card in batch.cards}

        assert versions == {"r0_bootstrap"}
        assert reasons == {"cold_start"}

    def test_without_a_bootstrap_an_unfitted_model_still_raises(self, corpus):
        # The strict behaviour is what the unit tests rely on; the fallback is
        # opt-in rather than a silent default.
        _, connection, features = corpus
        profile_id = self._profile(connection)

        with pytest.raises(UnfittedModelError):
            build_batch(
                connection,
                features,
                RocchioRecommender(),
                profile_id=profile_id,
                model_version="r1_v1",
            )

    def test_a_fitted_model_serves_itself(self, corpus):
        settings, connection, features = corpus
        profile_id = self._profile(connection)
        model = RocchioRecommender().fit(self._likes(features), features)

        batch = build_batch(
            connection,
            features,
            model,
            profile_id=profile_id,
            model_version="r1_v1",
            bootstrap=ColdStartRecommender(seed=settings.seed),
        )

        assert batch.rung == "r1"
        assert batch.model_version == "r1_v1"
        assert {card.reason for card in batch.cards} == {"similar_likes"}

    def _likes(self, features) -> LabelSet:
        """Enough likes for R1 to be fitted, straight from the corpus."""
        ids = features.image_ids[:8]
        return LabelSet(
            image_ids=list(ids),
            targets=np.ones(len(ids), dtype=np.float32),
            weights=np.ones(len(ids), dtype=np.float32),
            actions=["like"] * len(ids),
            timestamps=["2026-10-31T00:00:00"] * len(ids),
        )
