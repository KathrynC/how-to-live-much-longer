"""Tests for lifestyle module — alcohol, diet, coffee, fasting effects."""
import pytest


class TestAlcoholEffects:
    def test_zero_alcohol_no_effect(self):
        from lifestyle_module import compute_alcohol_effects
        effects = compute_alcohol_effects(alcohol_intake=0.0, apoe_sensitivity=1.0)
        assert effects['inflammation_delta'] == pytest.approx(0.0)
        assert effects['nad_multiplier'] == pytest.approx(1.0)

    def test_alcohol_increases_inflammation(self):
        from lifestyle_module import compute_alcohol_effects
        effects = compute_alcohol_effects(alcohol_intake=0.5, apoe_sensitivity=1.0)
        assert effects['inflammation_delta'] > 0

    def test_apoe4_amplifies_alcohol(self):
        from lifestyle_module import compute_alcohol_effects
        normal = compute_alcohol_effects(alcohol_intake=0.5, apoe_sensitivity=1.0)
        apoe4 = compute_alcohol_effects(alcohol_intake=0.5, apoe_sensitivity=1.3)
        assert apoe4['inflammation_delta'] > normal['inflammation_delta']


class TestCoffeeEffects:
    def test_zero_coffee_no_effect(self):
        from lifestyle_module import compute_coffee_effects
        effects = compute_coffee_effects(cups=0, coffee_type='filtered', sex='M', apoe_genotype=0)
        assert effects['nad_boost'] == pytest.approx(0.0)

    def test_three_cups_provides_nad_boost(self):
        from lifestyle_module import compute_coffee_effects
        effects = compute_coffee_effects(cups=3, coffee_type='filtered', sex='M', apoe_genotype=0)
        assert effects['nad_boost'] > 0

    def test_diminishing_returns_past_three(self):
        from lifestyle_module import compute_coffee_effects
        three = compute_coffee_effects(cups=3, coffee_type='filtered', sex='M', apoe_genotype=0)
        five = compute_coffee_effects(cups=5, coffee_type='filtered', sex='M', apoe_genotype=0)
        assert five['nad_boost'] == pytest.approx(three['nad_boost'])

    def test_filtered_better_than_instant(self):
        from lifestyle_module import compute_coffee_effects
        filtered = compute_coffee_effects(cups=2, coffee_type='filtered', sex='M', apoe_genotype=0)
        instant = compute_coffee_effects(cups=2, coffee_type='instant', sex='M', apoe_genotype=0)
        assert filtered['nad_boost'] > instant['nad_boost']


class TestDietEffects:
    def test_standard_diet_neutral(self):
        from lifestyle_module import compute_diet_effects
        effects = compute_diet_effects(diet_type='standard', fasting_regimen=0.0)
        assert effects['demand_multiplier'] == pytest.approx(1.0)
        assert effects['mitophagy_boost'] == pytest.approx(0.0)

    def test_keto_reduces_demand(self):
        from lifestyle_module import compute_diet_effects
        effects = compute_diet_effects(diet_type='keto', fasting_regimen=0.0)
        assert effects['demand_multiplier'] < 1.0

    def test_fasting_boosts_mitophagy(self):
        from lifestyle_module import compute_diet_effects
        effects = compute_diet_effects(diet_type='standard', fasting_regimen=0.8)
        assert effects['mitophagy_boost'] > 0
