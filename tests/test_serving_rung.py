"""The app's R4 options apply where the app builds models, and nowhere else.

`serving_rung` adds `settings.r4_options` (DECISIONS #31); `load_rung`, which
the simulator uses, must stay exactly as it was, or every cached study would
silently change meaning.
"""

from __future__ import annotations

from visualtaste.api.deps import load_rung, serving_rung
from visualtaste.config import Settings
from visualtaste.rec.rerank import MMR_LAMBDA, SHORTLIST


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        min_free_gb=0,
        **overrides,
    )


class TestServingRung:
    def test_the_app_default_carries_the_tuned_settings(self, tmp_path):
        model = serving_rung("r4+r5", _settings(tmp_path))
        assert model.rung == "r4+r5"
        assert (model.tau_halflife, model.mmr_shortlist, model.mmr_lambda) == (200.0, 50, 0.9)

    def test_plain_r4_gets_them_too(self, tmp_path):
        model = serving_rung("r4", _settings(tmp_path))
        assert (model.tau_halflife, model.mmr_shortlist, model.mmr_lambda) == (200.0, 50, 0.9)

    def test_the_simulators_loader_is_unchanged(self, tmp_path):
        model = load_rung("r4+r5", _settings(tmp_path))
        assert model.tau_halflife is None
        assert model.mmr_shortlist == SHORTLIST
        assert model.mmr_lambda == MMR_LAMBDA

    def test_an_empty_setting_turns_it_off(self, tmp_path):
        model = serving_rung("r4+r5", _settings(tmp_path, r4_options=""))
        assert model.tau_halflife is None and model.mmr_shortlist == SHORTLIST

    def test_other_rungs_are_untouched(self, tmp_path):
        assert not hasattr(serving_rung("r3", _settings(tmp_path)), "tau_halflife")
