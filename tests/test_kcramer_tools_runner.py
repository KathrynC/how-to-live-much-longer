"""Tests for kcramer_tools_runner CLI integration helpers."""

from __future__ import annotations

import kcramer_tools_runner as runner


class _FakeSim:
    """Minimal simulator stub for runner tests."""

    def __init__(self):
        self.calls = []

    def run(self, params: dict) -> dict:
        self.calls.append(params)
        return {"final_atp": 0.5}


def test_resilience_mode_merges_patient_into_protocols(monkeypatch):
    """Resilience mode should pass full 12D protocol vectors."""
    captured = {}

    monkeypatch.setattr(runner, "MitoSimulator", _FakeSim)

    def _fake_run_resilience_analysis(sim, protocols, output_key):
        captured["protocols"] = protocols
        captured["output_key"] = output_key
        return {"ok": True}

    monkeypatch.setattr(runner, "run_resilience_analysis", _fake_run_resilience_analysis)

    out = runner.run_mode(
        mode="resilience",
        patient_profile="near_cliff_80",
        output_key="final_atp",
        protocol="moderate",
    )

    assert out["result"] == {"ok": True}
    assert captured["output_key"] == "final_atp"
    # Ensure patient keys were merged into every protocol vector.
    for proto in captured["protocols"].values():
        assert "baseline_age" in proto
        assert "baseline_heteroplasmy" in proto


def test_vulnerability_mode_merges_selected_protocol(monkeypatch):
    """Vulnerability mode should pass one merged protocol dict."""
    captured = {}

    monkeypatch.setattr(runner, "MitoSimulator", _FakeSim)

    def _fake_run_vulnerability_analysis(sim, protocol, output_key):
        captured["protocol"] = protocol
        captured["output_key"] = output_key
        return [{"scenario": "x", "impact": 1.0}]

    monkeypatch.setattr(runner, "run_vulnerability_analysis", _fake_run_vulnerability_analysis)

    out = runner.run_mode(
        mode="vulnerability",
        patient_profile="post_chemo_55",
        output_key="final_atp",
        protocol="moderate",
    )

    assert isinstance(out["result"], list)
    assert captured["output_key"] == "final_atp"
    assert "baseline_age" in captured["protocol"]
    assert "rapamycin_dose" in captured["protocol"]


def test_compare_mode_uses_profiled_scalar(monkeypatch):
    """Compare mode should evaluate scalar analysis with selected profile."""
    monkeypatch.setattr(runner, "MitoSimulator", _FakeSim)
    captured = {}

    def _fake_run_scenario_comparison(analysis_fn, sim, scenarios, extract, **kwargs):
        value = analysis_fn(sim, **kwargs)
        captured["value"] = value
        captured["kwargs"] = kwargs
        return {"baseline": {"value": value}}

    monkeypatch.setattr(runner, "run_scenario_comparison", _fake_run_scenario_comparison)

    out = runner.run_mode(
        mode="compare",
        patient_profile="near_cliff_80",
        output_key="final_atp",
        protocol="moderate",
    )

    assert out["result"]["baseline"]["value"] == 0.5
    assert captured["value"] == 0.5
    assert captured["kwargs"]["output_key"] == "final_atp"
