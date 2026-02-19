"""Tests for supplement module — Hill-function dose-response curves."""
import pytest


class TestHillFunction:
    def test_zero_dose_zero_effect(self):
        from supplement_module import hill_effect
        assert hill_effect(0.0, 1.0, 0.5) == pytest.approx(0.0)

    def test_half_max_gives_half_effect(self):
        from supplement_module import hill_effect
        assert hill_effect(0.5, 1.0, 0.5) == pytest.approx(0.5)

    def test_high_dose_saturates(self):
        from supplement_module import hill_effect
        assert hill_effect(10.0, 1.0, 0.5) > 0.9

    def test_diminishing_returns(self):
        from supplement_module import hill_effect
        low = hill_effect(0.25, 1.0, 0.5)
        mid = hill_effect(0.5, 1.0, 0.5)
        high = hill_effect(0.75, 1.0, 0.5)
        assert (mid - low) > (high - mid)


class TestSupplementEffects:
    def test_no_supplements_no_effect(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({})
        assert effects['nad_boost'] == pytest.approx(0.0)
        assert effects['inflammation_reduction'] == pytest.approx(0.0)
        assert effects['mitophagy_boost'] == pytest.approx(0.0)

    def test_nr_boosts_nad(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({'nr_dose': 0.8})
        assert effects['nad_boost'] > 0

    def test_dha_reduces_inflammation(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({'dha_dose': 0.8})
        assert effects['inflammation_reduction'] > 0

    def test_resveratrol_boosts_mitophagy(self):
        from supplement_module import compute_supplement_effects
        effects = compute_supplement_effects({'resveratrol_dose': 0.8})
        assert effects['mitophagy_boost'] > 0

    def test_gut_health_modulates_nad(self):
        from supplement_module import compute_supplement_effects
        low_gut = compute_supplement_effects({'nr_dose': 0.8}, gut_health=0.3)
        high_gut = compute_supplement_effects({'nr_dose': 0.8}, gut_health=0.9)
        assert high_gut['nad_boost'] > low_gut['nad_boost']
