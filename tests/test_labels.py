"""Tests for the swipe-to-label mapping.

This module looks like bookkeeping and is not: every model learns from whatever
it says the labels are, so a mistake here propagates into every recommendation
the system makes. The "latest verdict wins" logic in particular is easy to get
subtly wrong, so it is tested from several angles.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.db import repo
from visualtaste.rec.labels import (
    ACTION_TO_TARGET,
    DEFAULT_ACTION_WEIGHTS,
    build_label_set,
    label_count,
    label_weights,
    latest_events,
    seen_image_ids,
)


@pytest.fixture
def db(tmp_path):
    connection = repo.init_db(tmp_path / "t.db")
    yield connection
    connection.close()


@pytest.fixture
def profile(db):
    return repo.create_profile(db, note="test")


def add_image(db, image_id: str) -> str:
    repo.upsert_image(
        db,
        image_id=image_id,
        source="pd12m",
        source_id=image_id,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        orig_path="p",
    )
    return image_id


def swipe(db, profile: int, image_id: str, action: str, *, dwell_ms: int = 1000) -> int:
    """Record one event, creating the image row if needed."""
    add_image(db, image_id)
    cursor = db.execute(
        """
        INSERT INTO events (profile_id, image_id, action, dwell_ms)
        VALUES (?, ?, ?, ?)
        """,
        (profile, image_id, action, dwell_ms),
    )
    db.commit()
    return int(cursor.lastrowid)


def supersede(db, old_event_id: int, new_event_id: int) -> None:
    """Mark an earlier verdict as replaced, as the History page does."""
    db.execute(
        "UPDATE events SET undone = 1, superseded_by = ? WHERE event_id = ?",
        (new_event_id, old_event_id),
    )
    db.commit()


def hex_id(n: int) -> str:
    return f"{n:016x}"


class TestWeightTable:
    def test_the_mapping_matches_the_brief(self):
        assert ACTION_TO_TARGET == {"like": 1, "save": 1, "dislike": 0, "skip": 0}
        assert DEFAULT_ACTION_WEIGHTS == {
            "like": 1.0,
            "save": 2.0,
            "dislike": 1.0,
            "skip": 0.25,
        }

    def test_a_save_is_worth_two_likes(self):
        # Saving means "I want to find this again", which is stronger and rarer
        # than a flick to the right.
        assert DEFAULT_ACTION_WEIGHTS["save"] == 2 * DEFAULT_ACTION_WEIGHTS["like"]

    def test_a_skip_is_a_weak_negative_not_a_full_one(self):
        assert 0 < DEFAULT_ACTION_WEIGHTS["skip"] < DEFAULT_ACTION_WEIGHTS["dislike"]

    def test_the_skip_weight_is_overridable_for_the_phase_9_comparison(self):
        assert label_weights(skip_weight=0.0)["skip"] == 0.0
        assert label_weights(skip_weight=0.5)["skip"] == 0.5
        # Everything else is untouched.
        assert label_weights(skip_weight=0.0)["save"] == 2.0

    def test_overriding_does_not_mutate_the_default_table(self):
        label_weights(skip_weight=0.9, save_weight=5.0)

        assert DEFAULT_ACTION_WEIGHTS["skip"] == 0.25
        assert DEFAULT_ACTION_WEIGHTS["save"] == 2.0


class TestBuildLabelSet:
    def test_maps_each_action_to_its_target_and_weight(self, db, profile):
        swipe(db, profile, hex_id(1), "like")
        swipe(db, profile, hex_id(2), "save")
        swipe(db, profile, hex_id(3), "dislike")
        swipe(db, profile, hex_id(4), "skip")

        labels = build_label_set(db, profile)

        assert len(labels) == 4
        by_id = dict(zip(labels.image_ids, zip(labels.targets, labels.weights, strict=True), strict=True))
        assert by_id[hex_id(1)] == (1.0, 1.0)
        assert by_id[hex_id(2)] == (1.0, 2.0)
        assert by_id[hex_id(3)] == (0.0, 1.0)
        assert by_id[hex_id(4)] == (0.0, 0.25)

    def test_labels_come_out_in_time_order(self, db, profile):
        # The validation split is the last 15% chronologically and replay cuts
        # the history at increasing points, so shuffling here would quietly
        # turn an honest time-ordered evaluation into a leaky random one.
        for n in range(6):
            swipe(db, profile, hex_id(n), "like")

        labels = build_label_set(db, profile)

        assert labels.image_ids == [hex_id(n) for n in range(6)]

    def test_an_empty_history_is_an_empty_label_set(self, db, profile):
        labels = build_label_set(db, profile)

        assert len(labels) == 0
        assert labels.targets.shape == (0,)
        assert labels.positives == []

    def test_another_profile_s_events_are_not_included(self, db, profile):
        other = repo.create_profile(db, note="other", is_simulated=True)
        swipe(db, profile, hex_id(1), "like")
        swipe(db, other, hex_id(2), "like")

        labels = build_label_set(db, profile)

        assert labels.image_ids == [hex_id(1)]

    def test_targets_and_weights_are_float32(self, db, profile):
        swipe(db, profile, hex_id(1), "like")

        labels = build_label_set(db, profile)

        assert labels.targets.dtype == np.float32
        assert labels.weights.dtype == np.float32


class TestLatestVerdictWins:
    def test_a_changed_verdict_replaces_the_earlier_one(self, db, profile):
        first = swipe(db, profile, hex_id(1), "like")
        second = swipe(db, profile, hex_id(1), "dislike")
        supersede(db, first, second)

        labels = build_label_set(db, profile)

        assert len(labels) == 1, "one image must contribute exactly one example"
        assert labels.targets[0] == 0.0
        assert labels.actions[0] == "dislike"

    def test_both_rows_still_exist_in_the_log(self, db, profile):
        # The log is a factual record: replay has to be able to reconstruct what
        # was known at any past moment, so nothing is deleted.
        first = swipe(db, profile, hex_id(1), "like")
        second = swipe(db, profile, hex_id(1), "dislike")
        supersede(db, first, second)

        rows = repo.query(db, "SELECT * FROM events ORDER BY event_id")

        assert len(rows) == 2
        assert rows[0]["undone"] == 1
        assert rows[0]["superseded_by"] == second

    def test_an_undone_event_is_excluded(self, db, profile):
        event_id = swipe(db, profile, hex_id(1), "like")
        db.execute("UPDATE events SET undone = 1 WHERE event_id = ?", (event_id,))
        db.commit()

        assert len(build_label_set(db, profile)) == 0

    def test_three_successive_verdicts_leave_only_the_last(self, db, profile):
        first = swipe(db, profile, hex_id(1), "like")
        second = swipe(db, profile, hex_id(1), "skip")
        supersede(db, first, second)
        third = swipe(db, profile, hex_id(1), "save")
        supersede(db, second, third)

        labels = build_label_set(db, profile)

        assert len(labels) == 1
        assert labels.actions == ["save"]

    def test_ordering_uses_event_id_not_the_timestamp(self, db, profile):
        # created_at has second resolution and twenty swipes can land in the
        # same second, which would make "time order" arbitrary within it.
        ids = [swipe(db, profile, hex_id(n), "like") for n in range(5)]
        db.execute("UPDATE events SET created_at = '2026-01-01 00:00:00'")
        db.commit()

        rows = latest_events(db, profile)

        assert [row["event_id"] for row in rows] == ids


class TestSkipHandling:
    def test_skips_can_be_dropped_entirely(self, db, profile):
        # The third arm of the Phase 9 comparison.
        swipe(db, profile, hex_id(1), "like")
        swipe(db, profile, hex_id(2), "skip")

        labels = build_label_set(db, profile, exclude_skips=True)

        assert labels.image_ids == [hex_id(1)]

    def test_a_zero_weight_action_is_dropped_rather_than_carried(self, db, profile):
        # A zero-weight example contributes nothing to any loss, so carrying it
        # as a row of zeros would only waste space and confuse the counts.
        swipe(db, profile, hex_id(1), "like")
        swipe(db, profile, hex_id(2), "skip")

        labels = build_label_set(db, profile, weights=label_weights(skip_weight=0.0))

        assert labels.image_ids == [hex_id(1)]

    def test_skips_are_kept_by_default(self, db, profile):
        swipe(db, profile, hex_id(1), "skip")

        assert len(build_label_set(db, profile)) == 1


class TestDerivedViews:
    def test_positives_are_likes_and_saves(self, db, profile):
        swipe(db, profile, hex_id(1), "like")
        swipe(db, profile, hex_id(2), "save")
        swipe(db, profile, hex_id(3), "dislike")

        labels = build_label_set(db, profile)

        assert labels.positives == [hex_id(1), hex_id(2)]

    def test_negatives_are_dislikes_and_skips(self, db, profile):
        swipe(db, profile, hex_id(1), "dislike")
        swipe(db, profile, hex_id(2), "skip")
        swipe(db, profile, hex_id(3), "like")

        labels = build_label_set(db, profile)

        assert labels.negatives == [hex_id(1), hex_id(2)]

    def test_strong_negatives_exclude_skips(self, db, profile):
        # Rocchio and kNN subtract a *mean* of negative vectors, and a mean has
        # no notion of weight -- so ambiguous skips at full strength would
        # distort the preference vector.
        swipe(db, profile, hex_id(1), "dislike")
        swipe(db, profile, hex_id(2), "skip")

        labels = build_label_set(db, profile)

        assert labels.strong_negatives == [hex_id(1)]

    def test_saves_are_listed_separately(self, db, profile):
        swipe(db, profile, hex_id(1), "like")
        swipe(db, profile, hex_id(2), "save")

        assert build_label_set(db, profile).saves == [hex_id(2)]

    def test_counts_and_class_balance(self, db, profile):
        swipe(db, profile, hex_id(1), "like")
        swipe(db, profile, hex_id(2), "like")
        swipe(db, profile, hex_id(3), "dislike")

        labels = build_label_set(db, profile)

        assert labels.counts() == {"dislike": 1, "like": 2}
        assert labels.class_balance() == (2, 1)


class TestSeenImages:
    def test_includes_undone_events(self, db, profile):
        # An image rated and then un-rated has still been seen; showing it again
        # as if new would look like a bug.
        event_id = swipe(db, profile, hex_id(1), "like")
        db.execute("UPDATE events SET undone = 1 WHERE event_id = ?", (event_id,))
        db.commit()

        assert seen_image_ids(db, profile) == {hex_id(1)}
        assert len(build_label_set(db, profile)) == 0

    def test_lists_every_image_acted_on_once_each(self, db, profile):
        first = swipe(db, profile, hex_id(1), "like")
        second = swipe(db, profile, hex_id(1), "dislike")
        supersede(db, first, second)
        swipe(db, profile, hex_id(2), "skip")

        assert seen_image_ids(db, profile) == {hex_id(1), hex_id(2)}

    def test_is_empty_for_a_fresh_profile(self, db, profile):
        assert seen_image_ids(db, profile) == set()


class TestLabelCount:
    def test_counts_distinct_images_not_events(self, db, profile):
        # The rung ladder keys off this number, so a re-rated image must count
        # once -- otherwise changing your mind twice would "unlock" R3.
        first = swipe(db, profile, hex_id(1), "like")
        second = swipe(db, profile, hex_id(1), "dislike")
        supersede(db, first, second)

        assert label_count(db, profile) == 1

    def test_excludes_undone_events(self, db, profile):
        event_id = swipe(db, profile, hex_id(1), "like")
        db.execute("UPDATE events SET undone = 1 WHERE event_id = ?", (event_id,))
        db.commit()

        assert label_count(db, profile) == 0

    def test_counts_up_as_swipes_arrive(self, db, profile):
        for n in range(5):
            swipe(db, profile, hex_id(n), "like")

        assert label_count(db, profile) == 5

    def test_is_zero_for_a_fresh_profile(self, db, profile):
        assert label_count(db, profile) == 0


class TestDwellTime:
    def test_dwell_is_recorded_on_the_event(self, db, profile):
        swipe(db, profile, hex_id(1), "like", dwell_ms=4200)

        assert repo.query(db, "SELECT dwell_ms FROM events")[0]["dwell_ms"] == 4200

    def test_dwell_never_reaches_the_label_set(self, db, profile):
        # Brief §7.5: dwell is logged, never used as a label. It correlates with
        # interest but also with distraction, a slow network, and having walked
        # away from the laptop.
        swipe(db, profile, hex_id(1), "like", dwell_ms=50)
        swipe(db, profile, hex_id(2), "like", dwell_ms=30_000)

        labels = build_label_set(db, profile)

        assert list(labels.weights) == [1.0, 1.0], "dwell leaked into the weights"
        assert not hasattr(labels, "dwell_ms")
