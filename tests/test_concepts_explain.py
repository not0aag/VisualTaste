"""Tests for "taste in words" and for the reason-code sentences.

The z-scoring in `concepts.py` is the part that can silently go wrong: raw
cosines also produce a plausible-looking list of words, just the same list for
everybody. Most of these tests are therefore about whether the correction is
actually doing anything, not only about whether the function returns strings.
"""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pytest

from visualtaste.rec import concepts as concepts_module
from visualtaste.rec.concepts import (
    MIN_INTERESTING_Z,
    PROMPT_TEMPLATE,
    ConceptCache,
    describe,
    describe_negative,
    label_for,
    load_concepts,
    score_concepts,
)
from visualtaste.rec.explain import (
    INTEREST_PREFIX,
    REASON_CODES,
    interest_id_of,
    interest_reason,
    is_known_reason,
    sentence_for,
)


def unit(rows) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# The vocabulary file
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_it_loads(self):
        assert len(load_concepts()) > 300

    def test_every_entry_is_a_bare_phrase(self):
        # The template adds "a photo of ... .", so a trailing full stop or a
        # leading article would be duplicated into the prompt.
        for concept in load_concepts():
            assert concept == concept.strip()
            assert not concept.endswith(".")
            assert not concept.startswith(("a ", "an ", "the "))

    def test_there_are_no_duplicates(self):
        # A duplicate would be counted twice in the mean and the sd, which
        # biases every z-score slightly towards that concept's baseline.
        loaded = load_concepts()

        assert len(loaded) == len(set(loaded))

    def test_comments_and_blanks_are_dropped(self, tmp_path):
        path = tmp_path / "concepts.txt"
        path.write_text("# styles\nbrutalist\n\n   \ncyanotype\n", encoding="utf-8")

        assert load_concepts(path) == ["brutalist", "cyanotype"]

    def test_an_empty_file_fails_loudly(self, tmp_path):
        path = tmp_path / "concepts.txt"
        path.write_text("# nothing but a comment\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no concepts"):
            load_concepts(path)

    def test_the_prompt_template_is_siglips_own(self):
        assert PROMPT_TEMPLATE == "a photo of {}."


# ---------------------------------------------------------------------------
# Z-scoring: the modality-gap correction
# ---------------------------------------------------------------------------


class TestScoreConcepts:
    #: The worked example from the module docstring: raw similarities 0.18,
    #: 0.12, 0.04 against a mean of 0.11 and an sd that makes the arithmetic
    #: come out at +2.3 / +0.3 / -2.3.
    WORKED_SIMILARITIES: ClassVar[list[float]] = [0.18, 0.12, 0.04]
    WORKED_NAMES: ClassVar[list[str]] = ["photograph", "brutalist", "cyanotype"]

    def _direction_and_vectors(self, similarities):
        """Build a direction and concept vectors with exactly these cosines.

        In 2-D with a direction of (1, 0), a concept vector of (s, sqrt(1-s^2))
        has cosine exactly s. That gives a fixture with chosen similarities
        without having to run an encoder.
        """
        direction = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[s, float(np.sqrt(1 - s * s))] for s in similarities], dtype=np.float32)
        return direction, vectors

    def test_the_fixture_really_has_the_intended_cosines(self):
        direction, vectors = self._direction_and_vectors(self.WORKED_SIMILARITIES)

        scored = score_concepts(direction, vectors, self.WORKED_NAMES)

        for item, expected in zip(scored, self.WORKED_SIMILARITIES, strict=True):
            assert item.raw_similarity == pytest.approx(expected, abs=1e-6)

    def test_the_worked_example_from_the_docstring(self):
        # mean = 0.1133, population sd = 0.0573. The docstring rounds to
        # "mean 0.11, sd 0.03" for readability, so the exact z-scores differ;
        # what the example claims and what is asserted here is the *shape*:
        # photograph strongly positive, brutalist near zero, cyanotype
        # strongly negative, and symmetric about the mean.
        direction, vectors = self._direction_and_vectors(self.WORKED_SIMILARITIES)

        scored = score_concepts(direction, vectors, self.WORKED_NAMES)
        by_name = {item.concept: item.z_score for item in scored}

        assert by_name["photograph"] > 1.0
        assert abs(by_name["brutalist"]) < 0.3
        assert by_name["cyanotype"] < -1.0

    def test_z_scores_are_centred_and_scaled(self):
        # The defining property: mean 0, sd 1. If this ever fails, every
        # threshold in the module (MIN_INTERESTING_Z) stops meaning what it says.
        direction, vectors = self._direction_and_vectors([0.18, 0.12, 0.04, 0.09, 0.2])

        z = np.array([item.z_score for item in score_concepts(direction, vectors, list("abcde"))])

        assert z.mean() == pytest.approx(0.0, abs=1e-6)
        assert z.std() == pytest.approx(1.0, abs=1e-6)

    def test_it_separates_two_tastes_that_raw_cosines_do_not(self):
        """The actual claim the correction makes.

        Taste 1 scores photograph 0.18, brutalist 0.12. Taste 2 scores them
        0.17 and 0.19. Both tastes rank photograph-ish words highly in raw
        terms; only after z-scoring does taste 2's ordering flip to brutalist.
        """
        names = ["photograph", "brutalist", "cyanotype"]
        d1, v1 = self._direction_and_vectors([0.18, 0.12, 0.04])
        d2, v2 = self._direction_and_vectors([0.17, 0.19, 0.05])

        first = {i.concept: i.z_score for i in score_concepts(d1, v1, names)}
        second = {i.concept: i.z_score for i in score_concepts(d2, v2, names)}

        assert first["photograph"] > first["brutalist"]
        assert second["brutalist"] > second["photograph"]

    def test_a_non_unit_direction_is_normalised(self):
        # A preference vector arriving un-normalised must not scale every
        # cosine and thereby every z-score.
        direction, vectors = self._direction_and_vectors([0.18, 0.12, 0.04])

        plain = score_concepts(direction, vectors, self.WORKED_NAMES)
        scaled = score_concepts(direction * 7.0, vectors, self.WORKED_NAMES)

        for a, b in zip(plain, scaled, strict=True):
            assert a.z_score == pytest.approx(b.z_score, abs=1e-5)

    def test_a_zero_direction_fails_loudly(self):
        _, vectors = self._direction_and_vectors([0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="zero-length"):
            score_concepts(np.zeros(2), vectors, list("abc"))

    def test_a_length_mismatch_fails_loudly(self):
        direction, vectors = self._direction_and_vectors([0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="concept"):
            score_concepts(direction, vectors, ["only", "two"])

    def test_no_spread_returns_zeros_rather_than_infinities(self):
        # Dividing by a zero sd would produce inf/nan z-scores that then sort
        # to the top and read as findings.
        direction, vectors = self._direction_and_vectors([0.2, 0.2, 0.2])

        scored = score_concepts(direction, vectors, list("abc"))

        assert [item.z_score for item in scored] == [0.0, 0.0, 0.0]

    def test_a_concept_score_prints_readably(self):
        direction, vectors = self._direction_and_vectors([0.18, 0.12, 0.04])

        text = str(score_concepts(direction, vectors, self.WORKED_NAMES)[0])

        assert text.startswith("photograph (z=+")


class TestDescribe:
    def _fixture(self, similarities, names):
        direction = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[s, float(np.sqrt(1 - s * s))] for s in similarities], dtype=np.float32)
        return direction, vectors, names

    def test_it_returns_the_most_distinctive_concepts_first(self):
        direction, vectors, names = self._fixture(
            [0.05, 0.30, 0.10, 0.20], ["low", "highest", "mid", "high"]
        )

        top = describe(direction, vectors, names, top_n=2, minimum_z=0.0)

        assert [item.concept for item in top] == ["highest", "high"]

    def test_uninteresting_concepts_are_dropped_not_padded(self):
        # A taste the vocabulary cannot describe should return few words rather
        # than a full list of shrugs.
        direction, vectors, names = self._fixture(
            [0.100, 0.101, 0.099, 0.300], ["a", "b", "c", "distinctive"]
        )

        top = describe(direction, vectors, names, top_n=4)

        assert [item.concept for item in top] == ["distinctive"]

    def test_the_default_threshold_is_the_documented_one(self):
        assert MIN_INTERESTING_Z == 0.75

    def test_describe_negative_finds_the_opposite_end(self):
        direction, vectors, names = self._fixture(
            [0.05, 0.30, 0.10, 0.20], ["lowest", "highest", "mid", "high"]
        )

        bottom = describe_negative(direction, vectors, names, top_n=1, minimum_z=0.0)

        assert [item.concept for item in bottom] == ["lowest"]

    def test_describe_and_describe_negative_do_not_overlap(self):
        direction, vectors, names = self._fixture(
            [0.05, 0.30, 0.10, 0.20, 0.02], ["a", "b", "c", "d", "e"]
        )

        top = {i.concept for i in describe(direction, vectors, names, top_n=2)}
        bottom = {i.concept for i in describe_negative(direction, vectors, names, top_n=2)}

        assert not (top & bottom)

    def test_label_for_joins_the_top_words(self):
        direction, vectors, names = self._fixture(
            [0.30, 0.25, 0.20, 0.02], ["first", "second", "third", "last"]
        )

        assert label_for(direction, vectors, names, words=3) == "first, second, third"

    def test_label_for_never_returns_an_empty_string(self):
        # It names an interest card in the UI, so a blank is worse than a
        # placeholder. minimum_z=0 inside `label_for` is what guarantees this.
        direction, vectors, names = self._fixture([0.1, 0.1, 0.1], list("abc"))

        assert label_for(direction, vectors, names) != ""


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class TestConceptCache:
    def test_a_cold_cache_reports_nothing(self, tmp_path):
        assert ConceptCache(tmp_path).load() is None

    def test_it_round_trips(self, tmp_path):
        cache = ConceptCache(tmp_path)
        vectors = unit([[1.0, 0.0], [0.0, 1.0]])

        cache.save(["brutalist", "cyanotype"], vectors)
        loaded = cache.load()

        assert loaded is not None
        names, restored = loaded
        assert names == ["brutalist", "cyanotype"]
        np.testing.assert_allclose(restored, vectors)

    def test_it_stores_float32_even_from_float64(self, tmp_path):
        cache = ConceptCache(tmp_path)

        cache.save(["a"], np.array([[1.0, 0.0]], dtype=np.float64))
        loaded = cache.load()

        assert loaded is not None
        assert loaded[1].dtype == np.float32

    def test_a_different_prompt_template_invalidates_it(self, tmp_path):
        # The whole point of recording the template: similarities under a
        # different prompt are not comparable, so reuse would be silently wrong.
        cache = ConceptCache(tmp_path)
        cache.save(["a"], unit([[1.0, 0.0]]))

        meta = json.loads(cache.meta_path.read_text(encoding="utf-8"))
        meta["prompt_template"] = "an image of {}."
        cache.meta_path.write_text(json.dumps(meta), encoding="utf-8")

        assert cache.load() is None

    def test_a_count_mismatch_invalidates_it(self, tmp_path):
        cache = ConceptCache(tmp_path)
        cache.save(["a", "b"], unit([[1.0, 0.0], [0.0, 1.0]]))

        meta = json.loads(cache.meta_path.read_text(encoding="utf-8"))
        meta["concepts"] = ["a", "b", "c"]
        cache.meta_path.write_text(json.dumps(meta), encoding="utf-8")

        assert cache.load() is None

    def test_ensure_builds_once_and_then_never_loads_the_encoder(self, tmp_path):
        """What keeps API startup from paying for the text encoder."""
        calls = []

        class FakeEncoder:
            def encode_texts(self, prompts):
                calls.append(list(prompts))
                return unit(np.arange(len(prompts) * 4, dtype=np.float32).reshape(-1, 4) + 1)

        cache = ConceptCache(tmp_path)

        def factory():
            return FakeEncoder()

        names, vectors = cache.ensure(factory)
        again_names, again_vectors = cache.ensure(factory)

        assert len(calls) == 1
        assert names == again_names
        np.testing.assert_allclose(vectors, again_vectors)

    def test_ensure_sends_prompts_through_the_template(self, tmp_path):
        seen = []

        class FakeEncoder:
            def encode_texts(self, prompts):
                seen.extend(prompts)
                return unit(
                    np.ones((len(prompts), 4), dtype=np.float32)
                    * np.arange(1, len(prompts) + 1).reshape(-1, 1)
                )

        ConceptCache(tmp_path).ensure(FakeEncoder)

        assert seen
        assert all(p.startswith("a photo of ") and p.endswith(".") for p in seen)

    def test_leftover_part_files_are_not_mistaken_for_the_cache(self, tmp_path):
        # Atomic writes: a killed save leaves `.part` files, and `load` must
        # report a cold cache rather than reading half a file.
        cache = ConceptCache(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / ".concept_vectors.npy.part").write_bytes(b"truncated")

        assert cache.load() is None

    def test_it_writes_the_dimension_into_the_metadata(self, tmp_path):
        cache = ConceptCache(tmp_path)
        cache.save(["a"], unit([[1.0, 0.0, 0.0, 0.0]]))

        meta = json.loads(cache.meta_path.read_text(encoding="utf-8"))

        assert meta["dim"] == 4
        assert meta["count"] == 1


def test_the_default_concepts_file_is_the_one_shipped():
    # Guards against the module picking up a stray file from the cwd.
    assert concepts_module.CONCEPTS_FILE.name == "concepts.txt"
    assert concepts_module.CONCEPTS_FILE.parent.name == "rec"


# ---------------------------------------------------------------------------
# Reason-code sentences
# ---------------------------------------------------------------------------


class TestSentences:
    def test_every_fixed_code_has_a_sentence(self):
        for reason in REASON_CODES:
            sentence = sentence_for(reason)

            assert sentence
            assert sentence[0].isupper()
            assert sentence.endswith(".")

    def test_an_interest_code_has_a_sentence(self):
        assert sentence_for(interest_reason(2)).endswith(".")

    def test_an_interest_sentence_names_the_interest_when_it_has_a_label(self):
        sentence = sentence_for(interest_reason(1), {"label": "brutalist, concrete"})

        assert "brutalist, concrete" in sentence

    def test_similar_likes_counts_its_neighbours(self):
        assert "three" in sentence_for("similar_likes", {"neighbours": ["a", "b", "c"]})
        assert "an image" in sentence_for("similar_likes", {"neighbours": ["a"]})

    def test_similar_likes_with_no_neighbours_still_reads(self):
        assert sentence_for("similar_likes", {"neighbours": []}).endswith(".")

    def test_the_explore_sentence_says_it_is_an_experiment(self):
        # The wording that makes a dislike in an exploration slot informative
        # rather than annoying.
        sentence = sentence_for("explore")

        assert "on purpose" in sentence
        assert "useful" in sentence

    def test_the_explore_sentence_can_name_the_cluster(self):
        sentence = sentence_for("explore", {"cluster_label": "aerial photography"})

        assert "aerial photography" in sentence

    def test_model_confident_reports_a_percentage(self):
        assert "82%" in sentence_for("model_confident", {"probability": 0.8231})

    def test_model_confident_without_a_probability_does_not_invent_one(self):
        sentence = sentence_for("model_confident", {})

        assert "%" not in sentence

    def test_no_sentence_flatters_or_overclaims(self):
        # §7.6: no flattery, no false precision. Asserted rather than trusted,
        # because this is the kind of copy that drifts.
        banned = ("love", "perfect", "amazing", "definitely", "certainly", "!")
        reasons = [*REASON_CODES, interest_reason(0)]

        for reason in reasons:
            sentence = sentence_for(reason).lower()
            for word in banned:
                assert word not in sentence, f"{reason!r} says {word!r}"

    def test_an_unknown_code_still_produces_a_sentence(self, caplog):
        # A missing sentence must not render an empty card, but it is a bug.
        with caplog.at_level("WARNING"):
            sentence = sentence_for("something_new")

        assert sentence.endswith(".")
        assert "something_new" in caplog.text

    def test_detail_may_be_omitted_entirely(self):
        assert sentence_for("similar_likes").endswith(".")


class TestReasonCodes:
    def test_interest_codes_round_trip(self):
        assert interest_id_of(interest_reason(7)) == 7

    def test_a_non_interest_code_parses_as_none(self):
        assert interest_id_of("cold_start") is None

    def test_a_malformed_interest_code_parses_as_none(self):
        assert interest_id_of(f"{INTEREST_PREFIX}abc") is None

    @pytest.mark.parametrize("reason", [*REASON_CODES, "interest:0", "interest:12"])
    def test_known_codes_are_recognised(self, reason):
        assert is_known_reason(reason)

    @pytest.mark.parametrize("reason", ["", "interest:", "interest:x", "made_up"])
    def test_unknown_codes_are_not(self, reason):
        assert not is_known_reason(reason)

    def test_the_fixed_codes_are_the_briefs_codes(self):
        # §7.6 lists five; four are fixed and the fifth is the dynamic
        # `interest:<k>`. `text_seed` is the one addition since (ROADMAP D4:
        # a cold-start card aimed at the person's own description).
        assert set(REASON_CODES) == {
            "cold_start",
            "similar_likes",
            "explore",
            "model_confident",
            "text_seed",
        }


class TestSentencesAreDerivedNotDuplicated:
    """The rungs produce a reason code and evidence; phrasing lives in one place.

    If a rung ever hard-codes its own sentence again, these tests still pass --
    they cannot detect that. What they do pin down is that the *default* path
    goes through `sentence_for`, so the single-source-of-truth property is
    exercised by the models the feed actually serves rather than only asserted
    in a docstring.
    """

    def test_an_explanation_fills_in_its_own_sentence(self):
        from visualtaste.rec.base import Explanation

        explanation = Explanation(reason="explore")

        assert explanation.sentence == sentence_for("explore")

    def test_the_sentence_reflects_the_detail(self):
        from visualtaste.rec.base import Explanation

        one = Explanation(reason="similar_likes", detail={"neighbours": ["a"]})
        three = Explanation(reason="similar_likes", detail={"neighbours": ["a", "b", "c"]})

        assert "an image" in one.sentence
        assert "three images" in three.sentence

    def test_an_explicit_sentence_still_wins(self):
        # Kept as an escape hatch, and asserted so a future rung with a genuinely
        # unusual explanation is not forced to extend the shared vocabulary.
        from visualtaste.rec.base import Explanation

        explanation = Explanation(reason="explore", sentence="Bespoke wording.")

        assert explanation.sentence == "Bespoke wording."

    def test_r1_and_r2_phrase_themselves_through_the_shared_sentences(self):
        import numpy as np

        from visualtaste.rec.base import FeatureBundle
        from visualtaste.rec.labels import ACTION_TO_TARGET, DEFAULT_ACTION_WEIGHTS, LabelSet
        from visualtaste.rec.r1_rocchio import RocchioRecommender
        from visualtaste.rec.r2_knn import KnnRecommender

        vectors = unit([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0], [0.2, 0.98]] * 3)
        ids = [f"{n:016x}" for n in range(len(vectors))]
        features = FeatureBundle(image_ids=ids, siglip=vectors)
        actions = ["like"] * 10 + ["dislike", "dislike"]
        labels = LabelSet(
            image_ids=ids,
            targets=np.array([ACTION_TO_TARGET[a] for a in actions], dtype=np.float32),
            weights=np.array([DEFAULT_ACTION_WEIGHTS[a] for a in actions], dtype=np.float32),
            actions=actions,
            timestamps=["2026-10-31T00:00:00"] * len(actions),
        )

        for model in (
            RocchioRecommender().fit(labels, features),
            KnnRecommender().fit(labels, features),
        ):
            explanation = model.explain(features, 0)

            assert explanation.sentence == sentence_for(explanation.reason, explanation.detail)
            assert "three images you liked" in explanation.sentence
