"""tests/test_protocol_classifier.py"""
import pytest


class TestRuleClassifier:
    """Test rule-based outcome classification."""

    def test_thriving(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.85, final_het=0.25, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "thriving"
        assert result["confidence"] >= 0.8

    def test_stable(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.55, final_het=0.55, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "stable"

    def test_declining(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.35, final_het=0.65, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "declining"

    def test_collapsed(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.10, final_het=0.90, baseline_atp=0.6, baseline_het=0.30)
        assert result["outcome_class"] == "collapsed"

    def test_paradoxical(self):
        from protocol_classifier import rule_classify
        result = rule_classify(final_atp=0.50, final_het=0.40, baseline_atp=0.55, baseline_het=0.30)
        assert result["outcome_class"] == "paradoxical"


class TestAnalyticsFitClassifier:
    """Test analytics-based prototype distance classification."""

    def test_computes_distances(self):
        from protocol_classifier import analytics_fit_classify, CLASS_PROTOTYPES
        analytics = {
            "energy": {"final_atp": 0.85, "mean_atp": 0.80},
            "damage": {"final_het": 0.20},
        }
        result = analytics_fit_classify(analytics)
        assert result["outcome_class"] in CLASS_PROTOTYPES
        assert 0.0 <= result["confidence"] <= 1.0
        assert "distances" in result

    def test_thriving_closest_to_thriving(self):
        from protocol_classifier import analytics_fit_classify
        analytics = {
            "energy": {"final_atp": 0.90, "mean_atp": 0.85},
            "damage": {"final_het": 0.15},
        }
        result = analytics_fit_classify(analytics)
        assert result["outcome_class"] == "thriving"


class TestMultiClassify:
    """Test the multi-method classification pipeline."""

    def test_pipeline_returns_all_methods(self):
        from protocol_classifier import multi_classify
        result = multi_classify(
            final_atp=0.85, final_het=0.25,
            baseline_atp=0.6, baseline_het=0.30,
            analytics={
                "energy": {"final_atp": 0.85, "mean_atp": 0.80},
                "damage": {"final_het": 0.25},
            },
            pipeline=["rule", "analytics_fit"],
        )
        assert "outcome_class" in result
        assert "confidence" in result
        assert "rule" in result["methods"]
        assert "analytics_fit" in result["methods"]

    def test_agreement_boosts_confidence(self):
        from protocol_classifier import multi_classify
        result = multi_classify(
            final_atp=0.85, final_het=0.20,
            baseline_atp=0.6, baseline_het=0.30,
            analytics={
                "energy": {"final_atp": 0.85, "mean_atp": 0.80},
                "damage": {"final_het": 0.20},
            },
            pipeline=["rule", "analytics_fit"],
        )
        # Both methods should agree on "thriving", boosting confidence
        assert result["outcome_class"] == "thriving"
        assert result["confidence"] >= 0.85

    def test_rule_only_pipeline(self):
        from protocol_classifier import multi_classify
        result = multi_classify(
            final_atp=0.85, final_het=0.20,
            baseline_atp=0.6, baseline_het=0.30,
            pipeline=["rule"],
        )
        assert result["outcome_class"] == "thriving"
        assert "rule" in result["methods"]
