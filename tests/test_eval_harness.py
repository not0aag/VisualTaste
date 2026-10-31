"""Tests for replay, IPS, interleaving and the backup routines."""

from __future__ import annotations

import json

import numpy as np
import pytest

from visualtaste.config import Settings
from visualtaste.db import repo
from visualtaste.db.backup import (
    SNAPSHOTS_KEPT,
    export_events_jsonl,
    prune_snapshots,
    run_backup,
    snapshot_database,
)
from visualtaste.eval.interleave import (
    InterleavedBatch,
    InterleavingResult,
    finish_experiment,
    result_from_impressions,
    start_experiment,
    team_draft,
)
from visualtaste.eval.propensity import (
    DEFAULT_WEIGHT_CAP,
    monte_carlo_propensity,
    rewards_from_actions,
    snips,
)
from visualtaste.eval.replay import (
    CUT_EVERY,
    FIRST_CUT,
    compare,
    cut_points,
    learning_curve,
    replay,
)


class FixedCoin:
    """A generator stub, so the coin flip is controllable in a test."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestCutPoints:
    def test_the_schedule_matches_the_brief(self):
        # First cut at 200, then every 100.
        assert cut_points(650) == [200, 300, 400, 500]
        assert FIRST_CUT == 200
        assert CUT_EVERY == 100

    def test_a_history_too_short_for_a_test_block_yields_no_cuts(self):
        # Better than a degenerate single cut whose test block is three events.
        assert cut_points(250) == []
        assert cut_points(299) == []
        assert cut_points(300) == [200]

    def test_no_history_yields_no_cuts(self):
        assert cut_points(0) == []


class TestReplay:
    def _history(self, n=600, seed=1729):
        rng = np.random.default_rng(seed)
        truth = rng.random(n)
        actions = ["like" if rng.random() < p else "dislike" for p in truth]
        return actions, truth

    def test_a_model_that_knows_the_truth_scores_well(self):
        actions, truth = self._history()

        result = replay(actions, lambda _tr, te: truth[te] * 0.9 + 0.05, label="oracle")

        assert result.mean("auc") > 0.8
        assert result.usable_cuts == len(cut_points(len(actions)))

    def test_a_coin_flip_scores_exactly_chance(self):
        actions, _ = self._history()

        result = replay(actions, lambda _tr, te: np.full(len(te), 0.5), label="coin")

        assert result.mean("auc") == pytest.approx(0.5, abs=0.01)
        # And its log loss is ln 2, the cost of refusing to guess.
        assert result.mean("log_loss") == pytest.approx(np.log(2), abs=1e-6)

    def test_each_cut_trains_only_on_earlier_events(self):
        # The property the whole module exists for: no information may flow
        # backwards, or the reported AUC would be unachievable in reality.
        actions, _ = self._history(500)
        seen_train_sizes: list[tuple[int, int, int]] = []

        def fit_and_score(train, test):
            seen_train_sizes.append((len(train), int(train.max()), int(test.min())))
            return np.full(len(test), 0.5)

        replay(actions, fit_and_score, label="probe")

        for train_size, last_train, first_test in seen_train_sizes:
            assert last_train < first_test, "a test event preceded the training cut"
            assert train_size == first_test

    def test_a_model_returning_the_wrong_number_of_scores_fails_loudly(self):
        actions, _ = self._history(400)

        with pytest.raises(ValueError, match="scores for"):
            replay(actions, lambda _tr, te: np.full(len(te) + 1, 0.5), label="broken")

    def test_a_single_class_test_block_is_marked_unusable(self):
        # Happens genuinely when a good model serves a hundred liked images.
        actions = ["like"] * 400

        result = replay(actions, lambda _tr, te: np.full(len(te), 0.5), label="all likes")

        assert result.points
        assert result.usable_cuts == 0
        assert np.isnan(result.mean("auc"))

    def test_the_learning_curve_is_indexed_by_training_size(self):
        actions, truth = self._history(600)
        result = replay(actions, lambda _tr, te: truth[te], label="oracle")

        curve = learning_curve(result)

        assert [size for size, _ in curve] == [200, 300, 400, 500]


class TestCompare:
    def _results(self, seed=1729):
        rng = np.random.default_rng(seed)
        truth = rng.random(900)
        actions = ["like" if rng.random() < p else "dislike" for p in truth]
        good = replay(actions, lambda _tr, te: truth[te] * 0.9 + 0.05, label="good")
        coin = replay(actions, lambda _tr, te: np.full(len(te), 0.5), label="coin")
        return good, coin

    def test_a_real_difference_is_detected(self):
        good, coin = self._results()

        result = compare(good, coin, metric="auc")

        assert result.detectable
        assert result.difference > 0
        assert "higher" in result.summarise()

    def test_a_model_compared_with_itself_shows_no_difference(self):
        good, _ = self._results()

        result = compare(good, good, metric="auc")

        assert result.difference == pytest.approx(0.0)
        assert not result.detectable
        assert "no detectable difference" in result.summarise()

    def test_the_summary_says_so_when_there_is_nothing_to_report(self):
        good, _ = self._results()

        text = compare(good, good, metric="auc").summarise()

        assert "includes zero" in text

    def test_only_cuts_usable_by_both_are_compared(self):
        # Dropping a cut from one side only would silently compare different
        # sets of cuts and break the pairing.
        actions = ["like" if i % 3 else "dislike" for i in range(600)]
        truth = np.array([1.0 if a == "like" else 0.0 for a in actions])
        good = replay(actions, lambda _tr, te: truth[te] * 0.8 + 0.1, label="good")
        broken = replay(actions, lambda _tr, te: np.full(len(te), 0.5), label="flat")

        result = compare(good, broken)

        assert result.cuts == min(good.usable_cuts, broken.usable_cuts)

    def test_nothing_usable_gives_a_nan_comparison(self):
        actions = ["like"] * 400
        flat = replay(actions, lambda _tr, te: np.full(len(te), 0.5), label="flat")

        result = compare(flat, flat)

        assert result.cuts == 0
        assert np.isnan(result.difference)


# ---------------------------------------------------------------------------
# Propensity
# ---------------------------------------------------------------------------


class TestSnips:
    def test_identical_policies_reproduce_the_logged_rate(self):
        rewards = np.array([1.0, 0.0, 1.0, 1.0])
        propensities = np.full(4, 0.5)

        estimate = snips(rewards, propensities, propensities)

        assert estimate.value == pytest.approx(0.75)
        assert estimate.logged_value == pytest.approx(0.75)

    def test_upweighting_the_liked_images_raises_the_estimate(self):
        rewards = np.array([1.0, 0.0, 1.0, 0.0])
        logged = np.full(4, 0.5)
        # A new policy that prefers exactly the images that were liked.
        proposed = np.array([0.9, 0.1, 0.9, 0.1])

        estimate = snips(rewards, logged, proposed)

        assert estimate.value > estimate.logged_value

    def test_the_estimate_stays_within_the_reward_range(self):
        # The reason for self-normalising: plain IPS can return a "like rate"
        # of 1.4, which is not a possible value.
        rewards = np.array([1.0, 0.0, 1.0])
        logged = np.array([0.05, 0.5, 0.05])
        proposed = np.array([0.9, 0.1, 0.9])

        estimate = snips(rewards, logged, proposed)

        assert 0.0 <= estimate.value <= 1.0

    def test_extreme_weights_are_clipped_and_counted(self):
        rewards = np.array([1.0, 0.0])
        logged = np.array([0.002, 0.5])
        proposed = np.array([0.9, 0.5])

        estimate = snips(rewards, logged, proposed)

        assert estimate.clipped == 1
        assert estimate.max_weight <= DEFAULT_WEIGHT_CAP

    def test_impressions_without_a_usable_propensity_are_dropped(self):
        rewards = np.array([1.0, 0.0, 1.0])
        logged = np.array([np.nan, 0.5, 1e-9])
        proposed = np.array([0.5, 0.5, 0.5])

        estimate = snips(rewards, logged, proposed)

        assert estimate.dropped == 2
        assert estimate.samples == 1

    def test_the_effective_sample_size_exposes_a_fragile_estimate(self):
        # One impression carrying almost all the weight means the estimate
        # rests on one observation, however many rows went in.
        rewards = np.concatenate([[1.0], np.zeros(199)])
        logged = np.concatenate([[0.01], np.full(199, 0.9)])
        proposed = np.concatenate([[0.9], np.full(199, 0.01)])

        estimate = snips(rewards, logged, proposed)

        assert estimate.effective_fraction < 0.1
        assert not estimate.trustworthy
        assert "Too few effective samples" in estimate.summarise()

    def test_a_healthy_estimate_is_reported_with_its_caveat(self):
        rng = np.random.default_rng(1729)
        rewards = (rng.random(500) < 0.4).astype(float)
        logged = np.full(500, 0.5)

        estimate = snips(rewards, logged, np.full(500, 0.55))

        assert estimate.trustworthy
        assert "Noisy by construction" in estimate.summarise()

    def test_an_empty_log_is_nan_rather_than_zero(self):
        estimate = snips(np.array([]), np.array([]), np.array([]))

        assert np.isnan(estimate.value)
        assert estimate.samples == 0

    def test_mismatched_lengths_fail_loudly(self):
        with pytest.raises(ValueError, match="must all align"):
            snips(np.array([1.0, 0.0]), np.array([0.5]), np.array([0.5, 0.5]))

    def test_rewards_count_likes_and_saves(self):
        rewards = rewards_from_actions(["like", "save", "skip", "dislike"])

        np.testing.assert_array_equal(rewards, [1.0, 1.0, 0.0, 0.0])


class TestMonteCarloPropensity:
    def test_a_deterministic_sampler_gives_probability_one(self):
        probabilities = monte_carlo_propensity(lambda: [7, 8, 9], slots=3, trials=50)

        assert probabilities[(0, 7)] == pytest.approx(1.0)
        assert probabilities[(2, 9)] == pytest.approx(1.0)

    def test_a_coin_flip_sampler_gives_roughly_a_half(self):
        rng = np.random.default_rng(1729)

        probabilities = monte_carlo_propensity(
            lambda: [1] if rng.random() < 0.5 else [2], slots=1, trials=2000
        )

        assert probabilities[(0, 1)] == pytest.approx(0.5, abs=0.05)
        assert probabilities[(0, 1)] + probabilities[(0, 2)] == pytest.approx(1.0)

    def test_probabilities_sum_to_one_per_slot(self):
        rng = np.random.default_rng(7)

        probabilities = monte_carlo_propensity(
            lambda: list(rng.permutation(4)[:2]), slots=2, trials=500
        )

        for slot in (0, 1):
            total = sum(p for (s, _), p in probabilities.items() if s == slot)
            assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Interleaving
# ---------------------------------------------------------------------------


class TestTeamDraft:
    def test_the_worked_example_from_the_docstring(self):
        batch = team_draft([1, 2, 3], [2, 1, 4], slots=4, rng=FixedCoin(0.0))

        assert batch.first_pick == "a"
        assert batch.chosen == [1, 2, 3, 4]
        assert batch.arms == ["a", "b", "a", "b"]

    def test_the_coin_flip_decides_who_owns_slot_one(self):
        # Without this, one model would always own the most-attended slot and
        # the measured difference would partly be a measurement of position.
        a_first = team_draft([1, 2], [3, 4], slots=2, rng=FixedCoin(0.0))
        b_first = team_draft([1, 2], [3, 4], slots=2, rng=FixedCoin(0.9))

        assert a_first.arms[0] == "a"
        assert b_first.arms[0] == "b"
        assert a_first.chosen[0] == 1
        assert b_first.chosen[0] == 3

    def test_slots_alternate_between_the_arms(self):
        batch = team_draft([1, 2, 3], [4, 5, 6], slots=6, rng=FixedCoin(0.0))

        assert batch.arms == ["a", "b", "a", "b", "a", "b"]

    def test_no_candidate_appears_twice(self):
        batch = team_draft([1, 2, 3, 4], [1, 2, 3, 4], slots=4, rng=FixedCoin(0.0))

        assert len(set(batch.chosen)) == len(batch.chosen)

    def test_identical_rankings_split_the_batch_evenly(self):
        batch = team_draft([1, 2, 3, 4], [1, 2, 3, 4], slots=4, rng=FixedCoin(0.0))

        assert batch.arms.count("a") == 2
        assert batch.arms.count("b") == 2

    def test_a_short_batch_is_returned_rather_than_padded(self):
        batch = team_draft([1], [1], slots=5, rng=FixedCoin(0.0))

        assert batch.chosen == [1]
        assert batch.arms == ["a"]

    def test_an_empty_ranking_on_one_side_still_works(self):
        batch = team_draft([1, 2, 3], [], slots=3, rng=FixedCoin(0.0))

        assert batch.chosen == [1, 2, 3]
        assert set(batch.arms) == {"a"}


class TestInterleavingResult:
    def _batch(self):
        return InterleavedBatch(
            batch_id="b1", chosen=[0, 1, 2, 3], arms=["a", "b", "a", "b"], first_pick="a"
        )

    def test_likes_are_credited_to_the_arm_that_picked_the_slot(self):
        result = InterleavingResult()

        result.record_batch(self._batch(), ["like", "skip", "save", "dislike"])

        assert result.likes_a == 2
        assert result.likes_b == 0
        assert result.wins_a == 1

    def test_an_equal_batch_is_a_tie_and_carries_no_information(self):
        result = InterleavingResult()

        result.record_batch(self._batch(), ["like", "like", "skip", "skip"])

        assert result.ties == 1
        assert result.comparisons == 0

    def test_a_batch_with_no_likes_is_a_tie(self):
        result = InterleavingResult()

        result.record_batch(self._batch(), ["skip", "skip", "dislike", "dislike"])

        assert result.ties == 1

    def test_like_rates_are_per_impression_not_per_batch(self):
        result = InterleavingResult()
        result.record_batch(self._batch(), ["like", "skip", "like", "skip"])

        assert result.like_rate_a == pytest.approx(1.0)
        assert result.like_rate_b == pytest.approx(0.0)

    def test_a_consistent_winner_becomes_significant(self):
        result = InterleavingResult()
        for _ in range(10):
            result.record_batch(self._batch(), ["like", "skip", "like", "skip"])

        assert result.wins_a == 10
        assert result.p_value < 0.01
        assert "p = " in result.summarise("R3", "R4")

    def test_a_narrow_lead_is_reported_as_inconclusive(self):
        result = InterleavingResult()
        for _ in range(3):
            result.record_batch(self._batch(), ["like", "skip", "skip", "skip"])
        for _ in range(2):
            result.record_batch(self._batch(), ["skip", "like", "skip", "skip"])

        text = result.summarise("R3", "R4")

        assert "not yet distinguishable from chance" in text

    def test_all_ties_says_there_is_no_comparison_yet(self):
        result = InterleavingResult()
        result.record_batch(self._batch(), ["skip"] * 4)

        assert "No comparison yet" in result.summarise()

    def test_a_short_action_list_credits_only_the_slots_that_were_seen(self):
        result = InterleavingResult()

        result.record_batch(self._batch(), ["like"])

        assert result.impressions_a == 1
        assert result.impressions_b == 0


class TestExperimentRecords:
    def test_an_experiment_round_trips_through_the_database(self, tmp_path):
        connection = repo.init_db(tmp_path / "t.db")

        experiment_id = start_experiment(connection, version_a="r3_x", version_b="r4_y")
        result = InterleavingResult(
            wins_a=7, wins_b=3, likes_a=20, likes_b=12, impressions_a=50, impressions_b=50
        )
        finish_experiment(connection, experiment_id, result)

        row = repo.query(connection, "SELECT * FROM experiments")[0]
        stored = json.loads(row["result_json"])
        assert row["method"] == "team_draft"
        assert row["version_a"] == "r3_x"
        assert row["ended_at"] is not None
        assert stored["wins_a"] == 7
        assert stored["p_value"] == pytest.approx(result.p_value)
        connection.close()

    def test_the_result_can_be_rebuilt_from_the_logged_rows(self, tmp_path):
        # Recomputed from impressions and events, so the Model Lab's number
        # comes from the same rows as everything else and survives a restart.
        connection = repo.init_db(tmp_path / "t.db")
        profile = repo.create_profile(connection)
        experiment_id = start_experiment(connection, version_a="a", version_b="b")

        for index, (arm, action) in enumerate(
            [("a", "like"), ("b", "skip"), ("a", "save"), ("b", "dislike")]
        ):
            image_id = f"{index:016x}"
            repo.upsert_image(
                connection,
                image_id=image_id,
                source="pd12m",
                source_id=image_id,
                license_url="https://cc0",
                orig_path="p",
            )
            cursor = connection.execute(
                """
                INSERT INTO impressions (profile_id, image_id, experiment_id, arm,
                                         batch_id, rank)
                VALUES (?, ?, ?, ?, 'batch1', ?)
                """,
                (profile, image_id, experiment_id, arm, index),
            )
            connection.execute(
                """
                INSERT INTO events (impression_id, profile_id, image_id, action)
                VALUES (?, ?, ?, ?)
                """,
                (cursor.lastrowid, profile, image_id, action),
            )
        connection.commit()

        result = result_from_impressions(connection, experiment_id)

        assert result.likes_a == 2
        assert result.likes_b == 0
        assert result.wins_a == 1
        connection.close()

    def test_undone_events_do_not_count_towards_a_result(self, tmp_path):
        connection = repo.init_db(tmp_path / "t.db")
        profile = repo.create_profile(connection)
        experiment_id = start_experiment(connection, version_a="a", version_b="b")

        repo.upsert_image(
            connection,
            image_id="0" * 16,
            source="pd12m",
            source_id="x",
            license_url="https://cc0",
            orig_path="p",
        )
        cursor = connection.execute(
            """
            INSERT INTO impressions (profile_id, image_id, experiment_id, arm, batch_id, rank)
            VALUES (?, ?, ?, 'a', 'batch1', 0)
            """,
            (profile, "0" * 16, experiment_id),
        )
        connection.execute(
            """
            INSERT INTO events (impression_id, profile_id, image_id, action, undone)
            VALUES (?, ?, ?, 'like', 1)
            """,
            (cursor.lastrowid, profile, "0" * 16),
        )
        connection.commit()

        result = result_from_impressions(connection, experiment_id)

        assert result.likes_a == 0
        connection.close()


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------


@pytest.fixture
def backup_settings(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        min_free_gb=0,
    )
    settings.ensure_dirs()
    connection = repo.init_db(settings.db_path)
    profile = repo.create_profile(connection, note="real")
    for index in range(5):
        image_id = f"{index:016x}"
        repo.upsert_image(
            connection,
            image_id=image_id,
            source="pd12m",
            source_id=image_id,
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            orig_path="p",
        )
        connection.execute(
            "INSERT INTO events (profile_id, image_id, action) VALUES (?, ?, 'like')",
            (profile, image_id),
        )
    connection.commit()
    connection.close()
    return settings


class TestSnapshots:
    def test_a_snapshot_is_a_readable_database(self, backup_settings):
        path = snapshot_database(backup_settings)

        assert path is not None and path.exists()
        copy = repo.connect(path, read_only=True)
        assert len(repo.query(copy, "SELECT * FROM events")) == 5
        copy.close()

    def test_the_snapshot_lands_in_the_archive(self, backup_settings):
        path = snapshot_database(backup_settings)

        assert path.parent == backup_settings.backups_dir / "db"

    def test_no_part_file_is_left_behind(self, backup_settings):
        snapshot_database(backup_settings)

        assert list((backup_settings.backups_dir / "db").glob(".*.part")) == []

    def test_snapshotting_without_a_database_is_not_an_error(self, tmp_path):
        settings = Settings(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            archive_root=tmp_path / "archive",
            min_free_gb=0,
        )

        assert snapshot_database(settings) is None

    def test_only_the_newest_fourteen_are_kept(self, backup_settings):
        directory = backup_settings.backups_dir / "db"
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(20):
            (directory / f"visualtaste_2026010{index // 10}_{index:04d}.db").write_bytes(b"x")

        pruned = prune_snapshots(backup_settings)

        assert pruned == 20 - SNAPSHOTS_KEPT
        assert len(list(directory.glob("visualtaste_*.db"))) == SNAPSHOTS_KEPT

    def test_pruning_keeps_the_newest_by_name(self, backup_settings):
        directory = backup_settings.backups_dir / "db"
        for stamp in ("20260101_0001", "20260102_0001", "20260103_0001"):
            (directory / f"visualtaste_{stamp}.db").write_bytes(b"x")

        prune_snapshots(backup_settings, keep=1)

        remaining = [p.name for p in directory.glob("visualtaste_*.db")]
        assert remaining == ["visualtaste_20260103_0001.db"]


class TestEventsExport:
    def test_every_event_is_exported_one_per_line(self, backup_settings):
        path = export_events_jsonl(backup_settings)

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 5
        first = json.loads(lines[0])
        assert first["action"] == "like"
        assert "license_url" in first, "the export should be readable without the DB"

    def test_exporting_twice_does_not_duplicate_lines(self, backup_settings):
        # Append-only and idempotent, so it is safe to call on every shutdown.
        export_events_jsonl(backup_settings)
        path = export_events_jsonl(backup_settings)

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 5

    def test_new_events_are_appended_to_the_existing_file(self, backup_settings):
        export_events_jsonl(backup_settings)

        connection = repo.init_db(backup_settings.db_path)
        repo.upsert_image(
            connection,
            image_id="f" * 16,
            source="pd12m",
            source_id="new",
            license_url="https://cc0",
            orig_path="p",
        )
        connection.execute(
            "INSERT INTO events (profile_id, image_id, action) VALUES (1, ?, 'save')",
            ("f" * 16,),
        )
        connection.commit()
        connection.close()

        path = export_events_jsonl(backup_settings)

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 6
        assert json.loads(lines[-1])["action"] == "save"

    def test_a_truncated_final_line_does_not_stop_the_export(self, backup_settings):
        # An append interrupted by a crash leaves a partial line. Re-exporting
        # one event is a far better outcome than refusing to back anything up.
        path = export_events_jsonl(backup_settings)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"event_id": 6, "act')

        again = export_events_jsonl(backup_settings)

        assert again is not None

    def test_run_backup_does_both(self, backup_settings):
        snapshot, export = run_backup(backup_settings)

        assert snapshot is not None and snapshot.exists()
        assert export is not None and export.exists()
