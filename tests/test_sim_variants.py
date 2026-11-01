"""The Part 2 simulator variants and the study machinery built on them."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_simulator_session import corpus  # noqa: F401 - pytest fixture
from visualtaste.eval.sim import (
    DISCOVERY_HITS,
    LABEL_NOISE,
    VARIANTS,
    SimulatedUser,
    calibrate_conjunctive,
    make_user,
    make_variant_user,
    simulated_dwell,
    warmth_of,
)
from visualtaste.eval.study import (
    PairedComparison,
    RungSpec,
    SeedResult,
    compare,
    labels_from_history,
    run_seed,
    run_study,
)
from visualtaste.rec.base import FeatureBundle
from visualtaste.rec.labels import label_weights


def colour_bundle(n: int = 600, seed: int = 0) -> FeatureBundle:
    """SigLIP-like, DINOv2-like and colour64 blocks, all with the right norms."""
    rng = np.random.default_rng(seed)

    def unit(width):
        v = rng.normal(size=(n, width)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    histograms = rng.dirichlet(np.full(64, 0.3), size=n)
    return FeatureBundle(
        image_ids=[f"{i:016x}" for i in range(n)],
        siglip=unit(24),
        dinov2=unit(16),
        colour=np.sqrt(histograms).astype(np.float32),
        cluster64=(np.arange(n) % 8).astype(np.int32),
    )


class TestWarmth:
    def test_the_docstring_example(self):
        histogram = np.zeros(64)
        histogram[10] = 0.5  # L 0, a 2, b 2: warm
        histogram[5] = 0.5  # L 0, a 1, b 1: cool
        assert warmth_of(np.sqrt(histogram)) == pytest.approx(0.5)

    def test_a_fully_warm_image_is_one(self):
        histogram = np.zeros(64)
        histogram[[10, 11, 14, 15, 58]] = 0.2
        assert warmth_of(np.sqrt(histogram)) == pytest.approx(1.0)


class TestCalibration:
    @pytest.mark.parametrize("name", sorted(VARIANTS))
    def test_every_variant_likes_about_half_the_pool(self, name):
        features = colour_bundle()
        user = make_variant_user(name, features, seed=3, cluster_labels=features.cluster64)

        rows = np.arange(len(features))
        vectors = np.asarray(getattr(features, user.space))
        p = user.like_probability(vectors)
        if user.warmth is not None:
            semantic = user.affinity(vectors) - user.offset
            warm = user.warmth.scale * (warmth_of(features.colour) - user.warmth.offset)
            p = 1 / (1 + np.exp(-user.slope * np.minimum(semantic, warm)))

        assert len(rows) > 0
        assert (p > 0.5).mean() == pytest.approx(0.5, abs=0.03)
        assert user.label_noise == LABEL_NOISE
        assert user.interest_count == VARIANTS[name].interests

    def test_conjunctive_needs_both_conditions(self):
        affinity = np.array([0.9, 0.9, 0.1, 0.1] * 25)
        warmth = np.array([0.9, 0.1, 0.9, 0.1] * 25)
        offset, condition = calibrate_conjunctive(affinity, warmth, target_like_rate=0.25)
        both = (affinity > offset) & (warmth > condition.offset)
        assert both.mean() == pytest.approx(0.25)
        # Neither condition alone is enough.
        assert not ((affinity > offset) & ~(warmth > condition.offset))[both].any()

    def test_other_spaces_get_a_rescaled_slope(self):
        features = colour_bundle()
        semantic = make_variant_user("S1_semantic", features, seed=3)
        colour = make_variant_user("S3_colour", features, seed=3)
        assert semantic.slope == 10.0
        assert colour.slope != 10.0


class TestReactions:
    def test_react_rows_matches_react_for_the_phase8_user(self):
        # Reproducibility of the Phase 8 study rests on this.
        features = colour_bundle()
        first = make_user(features.siglip, seed=5, cluster_labels=features.cluster64)
        second = make_user(features.siglip, seed=5, cluster_labels=features.cluster64)
        rows = np.arange(40)

        assert first.react(features.siglip[rows]) == second.react_rows(features, rows)

    def test_label_noise_flips_about_five_percent(self):
        centroids = np.eye(4, dtype=np.float32)[:1]
        noisy = SimulatedUser(
            centroids=centroids, rng=np.random.default_rng(0), offset=-10, label_noise=0.05
        )
        vectors = np.tile(np.eye(4, dtype=np.float32)[:1], (20_000, 1))
        actions = np.array(noisy.react(vectors))

        verdicts = actions[actions != "skip"]
        assert (verdicts == "dislike").mean() == pytest.approx(0.05, abs=0.01)

    def test_discovery_is_recorded_at_the_fifth_like(self):
        user = SimulatedUser(
            centroids=np.eye(4, dtype=np.float32)[:2], rng=np.random.default_rng(0), offset=-10
        )
        vectors = np.tile(np.eye(4, dtype=np.float32)[:1], (40, 1))
        actions = user.react(vectors)

        likes = [i for i, a in enumerate(actions) if a in ("like", "save")]
        assert user.discovered_at[0] == likes[DISCOVERY_HITS - 1] + 1
        assert user.discovery_times()[1] is None  # censored
        assert user.interests_found(DISCOVERY_HITS) == 1

    def test_s5_replaces_interest_zero_once(self):
        features = colour_bundle()
        user = make_variant_user("S5_drift", features, seed=3, cluster_labels=features.cluster64)
        user.switch_at = 10
        before = user.centroids[0].copy()

        user.react_rows(features, np.arange(5))
        assert user.switched_at is None
        user.react_rows(features, np.arange(5, 10))
        user.react_rows(features, np.arange(10, 15))

        assert user.switched_at == 10
        assert user.interest_ids == [3, 1, 2]
        assert not np.allclose(user.centroids[0], before)


class TestDwell:
    def test_the_docstring_medians(self):
        rng = np.random.default_rng(0)
        base = simulated_dwell(np.full(20_000, 0.5), np.zeros(20_000), rng)
        novel = simulated_dwell(np.full(20_000, 0.5), np.ones(20_000), rng)
        assert np.median(base) == pytest.approx(2_500, rel=0.03)
        assert np.median(novel) == pytest.approx(3_730, rel=0.03)

    def test_it_is_capped(self):
        dwell = simulated_dwell(np.ones(1000), np.ones(1000), np.random.default_rng(0))
        assert dwell.max() <= 30_000


class TestSpecs:
    def test_parsing(self):
        assert RungSpec.parse("r3").blocks is None
        assert RungSpec.parse("r3:sig+dino").blocks == ("sig", "dino")
        assert RungSpec.parse("r3:skip=exclude").skip == "exclude"
        assert RungSpec.parse("r3:skip=1.0").skip == 1.0
        assert "/" not in RungSpec.parse("r3:sig+dino+col").slug

    def test_a_zero_skip_weight_is_the_same_as_excluding_skips(self):
        # Why the skip study compares 0.25 / 1.0 / exclude rather than 0 /
        # 0.25 / exclude: a zero-weight example adds nothing to the loss or to
        # S, so it is mathematically identical to dropping it (#20).
        ids = [f"{i:016x}" for i in range(6)]
        actions = ["like", "skip", "dislike", "skip", "like", "dislike"]
        zero = labels_from_history(ids, actions, label_weights(0.0), False)
        excluded = labels_from_history(ids, actions, label_weights(0.25), True)
        assert zero.image_ids == excluded.image_ids


class TestStatistics:
    def _results(self, first, second):
        def seed_result(spec, seed, value):
            fields = dict.fromkeys(SeedResult.__dataclass_fields__)
            fields.update(
                spec=spec, seed=seed, like_rate=value, discovery={}, image_ids=[], actions=[]
            )
            fields.update(served_rungs=[], dwell_ms=[], true_probabilities=[])
            return SeedResult(**fields)

        return {
            ("S1", "a"): [seed_result("a", s, v) for s, v in enumerate(first)],
            ("S1", "b"): [seed_result("b", s, v) for s, v in enumerate(second)],
        }

    def test_a_clear_win_is_supported(self):
        results = self._results([0.6] * 10, [0.5] * 10)
        c = compare(results, variant="S1", first="a", second="b", metric="like_rate")
        assert c.verdict == "supported"
        assert (c.wins, c.losses) == (10, 0)
        assert c.p_value == pytest.approx(2 / 2**10)

    def test_an_interval_crossing_zero_is_inconclusive(self):
        results = self._results([0.6, 0.4, 0.55, 0.45], [0.5, 0.5, 0.5, 0.5])
        c = compare(results, variant="S1", first="a", second="b", metric="like_rate")
        assert c.verdict == "inconclusive"
        assert c.losing_seeds == [1, 3]

    def test_lower_is_better_flips_the_verdict(self):
        results = self._results([0.6] * 10, [0.5] * 10)
        c = compare(
            results, variant="S1", first="a", second="b", metric="like_rate", expect="lower"
        )
        assert c.verdict == "not supported"
        assert c.losses == 10

    def test_the_row_names_the_losing_seeds(self):
        c = PairedComparison("m", "a", "b", "S1", 0.1, -0.1, 0.2, 3, 1, 0.6, [7], 4, "higher")
        assert "| 7 |" in c.row() and "inconclusive" in c.row()


class TestRunSeed:
    def test_a_seed_is_cached_and_reused(self, corpus, tmp_path):  # noqa: F811
        settings, connection, features = corpus
        kwargs = {
            "study": "t",
            "variant": "S1_semantic",
            "spec": RungSpec.parse("r1"),
            "seed": 11,
            "swipes": 60,
            "cluster_labels": features.cluster64,
            "cache_root": tmp_path / "eval",
            "with_replay": False,
        }
        first = run_seed(connection, features, settings, **kwargs)
        second = run_seed(connection, features, settings, **kwargs)

        assert (tmp_path / "eval" / "t" / "S1_semantic" / "r1" / "seed_11.json").exists()
        assert second == first or second.to_json() == first.to_json()
        assert len(first.actions) == 60 and len(first.dwell_ms) == 60
        assert first.served_rungs[0] == "r0"  # the bootstrap covered the start

    def test_a_study_pairs_seeds_across_specs(self, corpus, tmp_path):  # noqa: F811
        settings, connection, features = corpus
        results = run_study(
            connection,
            features,
            settings,
            study="t",
            variants=["S1_semantic"],
            specs=["r0", "r1"],
            seeds=2,
            swipes=40,
            cache_root=tmp_path / "eval",
            with_replay=False,
        )
        seeds = {spec: [r.seed for r in rs] for (_, spec), rs in results.items()}
        assert seeds["r0"] == seeds["r1"] == [settings.seed, settings.seed + 1]


class TestPhase10Machinery:
    def test_r4_options_reach_the_model(self, corpus):  # noqa: F811
        settings, _, _ = corpus
        model = RungSpec.parse("r4:tau=0.5,mmr=0.9,quota=0").build(settings, 3)
        assert (model.tau_post, model.mmr_lambda, model.explore_every) == (0.5, 0.9, None)
        assert model.seed == settings.seed  # R4 keeps the configured seed; R0 takes the session's
        with pytest.raises(ValueError):
            RungSpec.parse("r1:tau=2").build(settings, 3)

    def test_an_r4_session_with_propensities_and_a_shadow(self, corpus, tmp_path):  # noqa: F811
        settings, connection, features = corpus
        result = run_seed(
            connection,
            features,
            settings,
            study="t",
            variant="S1_semantic",
            spec=RungSpec.parse("r4"),
            seed=21,
            swipes=120,
            cluster_labels=features.cluster64,
            cache_root=tmp_path / "eval",
            with_replay=False,
            with_propensities=True,
            shadow="r3",
        )
        served = [i for i, rung in enumerate(result.served_rungs) if rung == "r4"]
        assert served, "R4 never took over from the bootstrap"
        assert all(0.005 <= result.propensities[i] <= 1 for i in served)
        assert set(result.shadow_shown) <= {0.0, 1.0}
        assert len(result.shadow_shown) == len(result.actions)

        from visualtaste.eval.study import ips_check

        check = ips_check([result], [result])
        assert check.logged_cards == len(served)

    def test_an_interleaved_session_credits_both_arms(self, corpus, tmp_path):  # noqa: F811
        from visualtaste.eval.study import run_interleaved_seed

        settings, connection, features = corpus
        outcome = run_interleaved_seed(
            connection,
            features,
            settings,
            variant="S1_semantic",
            first="r3",
            second="r4",
            seed=5,
            swipes=120,
            cluster_labels=features.cluster64,
            cache_root=tmp_path / "eval",
        )
        assert outcome.wins_a + outcome.wins_b + outcome.ties > 0
        rows = connection.execute(
            "SELECT DISTINCT arm FROM impressions WHERE arm IS NOT NULL"
        ).fetchall()
        assert {row["arm"] for row in rows} == {"a", "b"}

    def test_gain_retention(self):
        from visualtaste.eval.study import gain_retention

        def rows(spec, rates):
            fields = dict.fromkeys(SeedResult.__dataclass_fields__)
            out = []
            for seed, rate in enumerate(rates):
                f = dict(fields)
                f.update(
                    spec=spec, seed=seed, like_rate=rate, discovery={}, image_ids=[], actions=[]
                )
                f.update(served_rungs=[], dwell_ms=[], true_probabilities=[])
                out.append(SeedResult(**f))
            return out

        results = {
            ("S1", "r0"): rows("r0", [0.5] * 6),
            ("S1", "r3"): rows("r3", [0.7] * 6),
            ("S1", "r4"): rows("r4", [0.65] * 6),
        }
        ratio, low, high = gain_retention(
            results, variant="S1", candidate="r4", reference="r3", baseline="r0"
        )
        assert ratio == pytest.approx(0.75)
        assert low == pytest.approx(0.75) and high == pytest.approx(0.75)


class TestDefaultRungChoice:
    def test_the_best_mean_rank_on_the_combined_score_wins(self):
        from visualtaste.jobs.evaluate import CHOICE_VARIANTS, default_rung_choice

        def seeds(spec, like, found, cover):
            fields = dict.fromkeys(SeedResult.__dataclass_fields__)
            rows = []
            for seed in range(3):
                f = dict(fields)
                f.update(spec=spec, seed=seed, like_rate=like, interests_found=found)
                f.update(coverage_last_500=cover, discovery={}, image_ids=[], actions=[])
                f.update(served_rungs=[], dwell_ms=[], true_probabilities=[])
                rows.append(SeedResult(**f))
            return rows

        results = {}
        for variant in CHOICE_VARIANTS:
            results[(variant, "r3")] = seeds("r3", 0.62, 2.0, 0.3)  # high like rate, narrow
            results[(variant, "r4")] = seeds("r4", 0.58, 3.0, 0.9)  # broad, a little lower
            results[(variant, "r1")] = seeds("r1", 0.50, 2.0, 0.2)
        choice, combined, mean_rank = default_rung_choice(results, eligible=("r1", "r3", "r4"))

        assert choice == "r4"
        assert mean_rank == {"r4": 1.0, "r3": 2.0, "r1": 3.0}
        assert combined["S1_semantic"]["r4"] > combined["S1_semantic"]["r3"]


class TestLearningCurvePoint:
    def test_a_replay_metric_maps_to_its_seed_result_field(self):
        # Regression: phase9_h2 passed the replay point's `log_loss` straight
        # into SeedResult, which names the field `replay_log_loss`.
        from visualtaste.eval.study import _point_result

        result = _point_result("r3", 1729, 0.61, "replay_log_loss")
        assert result.metric("replay_log_loss") == pytest.approx(0.61)
        with pytest.raises(TypeError):
            _point_result("r3", 1729, 0.61, "log_loss")
