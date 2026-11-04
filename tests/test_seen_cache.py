"""The feed's per-profile seen-set cache returns exactly what a fresh read would.

`candidate_positions` keeps each profile's "not yet seen" mask between
requests and reads only new events. These tests compare it against a rebuild
from scratch after every change that could make the cache wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualtaste.db import repo
from visualtaste.rec import feed
from visualtaste.rec.base import FeatureBundle
from visualtaste.rec.labels import seen_image_ids

POOL = 40


def _fresh(connection, features, profile_id) -> np.ndarray:
    seen = {features.position(i) for i in seen_image_ids(connection, profile_id)}
    return np.array([r for r in range(len(features)) if r not in seen], dtype=np.int64)


@pytest.fixture
def pool(tmp_path):
    feed._SEEN_CACHE.clear()
    connection = repo.init_db(tmp_path / "seen.db")
    ids = [f"{i:016x}" for i in range(POOL)]
    for image_id in ids:
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
    profile_id = connection.execute(
        "INSERT INTO profiles (started_at, note) VALUES ('2026-11-04', 'seen-cache test')"
    ).lastrowid
    connection.commit()
    vectors = np.eye(POOL, dtype=np.float32)
    features = FeatureBundle(image_ids=ids, siglip=vectors, cluster64=np.zeros(POOL, np.int32))
    yield connection, features, profile_id
    connection.close()
    feed._SEEN_CACHE.clear()


def _act(connection, profile_id, image_ids, action="like"):
    connection.executemany(
        "INSERT INTO events (profile_id, image_id, action) VALUES (?, ?, ?)",
        [(profile_id, i, action) for i in image_ids],
    )
    connection.commit()


class TestSeenCache:
    def test_matches_a_fresh_read_as_events_arrive(self, pool):
        connection, features, profile_id = pool
        for step in range(5):
            got = feed.candidate_positions(connection, features, profile_id)
            np.testing.assert_array_equal(got, _fresh(connection, features, profile_id))
            _act(connection, profile_id, features.image_ids[step * 3 : step * 3 + 3])

    def test_a_repeat_request_with_no_new_events_is_unchanged(self, pool):
        connection, features, profile_id = pool
        _act(connection, profile_id, features.image_ids[:4])
        first = feed.candidate_positions(connection, features, profile_id)
        second = feed.candidate_positions(connection, features, profile_id)
        np.testing.assert_array_equal(first, second)
        assert 0 not in second and 4 in second

    def test_a_database_that_goes_backwards_is_rebuilt(self, pool):
        # A restore from backup can leave fewer events than the cache saw.
        connection, features, profile_id = pool
        _act(connection, profile_id, features.image_ids[:6])
        feed.candidate_positions(connection, features, profile_id)
        connection.execute("DELETE FROM events WHERE image_id IN (?, ?)", features.image_ids[4:6])
        connection.commit()
        got = feed.candidate_positions(connection, features, profile_id)
        np.testing.assert_array_equal(got, _fresh(connection, features, profile_id))
        assert 4 in got and 5 in got

    def test_a_new_bundle_is_never_served_an_old_mask(self, pool):
        connection, features, profile_id = pool
        _act(connection, profile_id, features.image_ids[:3])
        feed.candidate_positions(connection, features, profile_id)
        reordered = FeatureBundle(
            image_ids=list(reversed(features.image_ids)),
            siglip=features.siglip,
            cluster64=features.cluster64,
        )
        got = feed.candidate_positions(connection, reordered, profile_id)
        np.testing.assert_array_equal(got, _fresh(connection, reordered, profile_id))

    def test_profiles_do_not_share_a_mask(self, pool):
        connection, features, profile_id = pool
        other = connection.execute(
            "INSERT INTO profiles (started_at, note) VALUES ('2026-11-04', 'other')"
        ).lastrowid
        _act(connection, profile_id, features.image_ids[:5])
        _act(connection, other, features.image_ids[10:12])
        mine = feed.candidate_positions(connection, features, profile_id)
        theirs = feed.candidate_positions(connection, features, other)
        np.testing.assert_array_equal(mine, _fresh(connection, features, profile_id))
        np.testing.assert_array_equal(theirs, _fresh(connection, features, other))
