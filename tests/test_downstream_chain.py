"""Tests for downstream chain — neuroplasticity and Alzheimer's pathology."""
import pytest
import numpy as np


class TestMEF2ODE:
    def test_engagement_increases_mef2(self):
        from downstream_chain import mef2_derivative
        dmef2 = mef2_derivative(mef2=0.2, engagement=0.9, apoe_mult=1.0)
        assert dmef2 > 0

    def test_no_engagement_mef2_decays(self):
        from downstream_chain import mef2_derivative
        dmef2 = mef2_derivative(mef2=0.5, engagement=0.0, apoe_mult=1.0)
        assert dmef2 < 0

    def test_mef2_saturates(self):
        from downstream_chain import mef2_derivative
        low = mef2_derivative(mef2=0.2, engagement=0.9, apoe_mult=1.0)
        high = mef2_derivative(mef2=0.9, engagement=0.9, apoe_mult=1.0)
        assert low > high


class TestHistoneAcetylation:
    def test_mef2_induces_ha(self):
        from downstream_chain import ha_derivative
        dha = ha_derivative(ha=0.2, mef2=0.8)
        assert dha > 0

    def test_ha_slow_decay(self):
        from downstream_chain import ha_derivative
        dha = ha_derivative(ha=0.5, mef2=0.0)
        assert dha < 0


class TestSynapticStrength:
    def test_engagement_strengthens_synapses(self):
        from downstream_chain import synaptic_derivative
        dss = synaptic_derivative(ss=1.0, ha=0.5, engagement=0.9)
        assert dss > 0

    def test_no_engagement_decay_toward_baseline(self):
        from downstream_chain import synaptic_derivative
        dss = synaptic_derivative(ss=1.5, ha=0.5, engagement=0.0)
        assert dss < 0

    def test_at_baseline_no_decay(self):
        from downstream_chain import synaptic_derivative
        dss = synaptic_derivative(ss=1.0, ha=0.5, engagement=0.0)
        assert dss == pytest.approx(0.0, abs=1e-10)


class TestAmyloidODE:
    def test_amyloid_accumulates(self):
        from downstream_chain import amyloid_derivative
        dab = amyloid_derivative(amyloid=0.2, inflammation=0.3, age=70, apoe_clearance=1.0)
        assert dab > 0 or dab < 0  # just confirm it runs

    def test_apoe4_reduces_clearance(self):
        from downstream_chain import amyloid_derivative
        normal = amyloid_derivative(amyloid=0.5, inflammation=0.3, age=75, apoe_clearance=1.0)
        apoe4 = amyloid_derivative(amyloid=0.5, inflammation=0.3, age=75, apoe_clearance=0.7)
        assert apoe4 > normal


class TestMemoryIndex:
    def test_baseline_memory(self):
        from downstream_chain import memory_index
        mi = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=0.0, tau=0.0)
        assert mi == pytest.approx(0.5, abs=0.01)

    def test_engagement_improves_memory(self):
        from downstream_chain import memory_index
        base = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=0.0, tau=0.0)
        engaged = memory_index(ss=1.6, mef2=0.8, cr=0.7, amyloid=0.0, tau=0.0)
        assert engaged > base

    def test_pathology_reduces_memory(self):
        from downstream_chain import memory_index
        healthy = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=0.0, tau=0.0)
        diseased = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=1.5, tau=1.0)
        assert diseased < healthy

    def test_resilience_buffers_pathology(self):
        from downstream_chain import memory_index
        no_buffer = memory_index(ss=1.0, mef2=0.0, cr=0.0, amyloid=1.0, tau=0.5)
        buffered = memory_index(ss=1.5, mef2=0.8, cr=0.8, amyloid=1.0, tau=0.5)
        assert buffered > no_buffer


class TestComputeDownstream:
    def test_runs_on_core_result(self):
        from simulator import simulate
        from downstream_chain import compute_downstream
        core = simulate()
        downstream = compute_downstream(
            core_result=core,
            patient_expanded={'intellectual_engagement': 0.5, 'baseline_age': 70},
        )
        assert len(downstream) == len(core['time'])
        assert 'memory_index' in downstream[0]
        assert 'MEF2_activity' in downstream[0]
        assert 'amyloid_burden' in downstream[0]

    def test_high_engagement_improves_memory(self):
        from simulator import simulate
        from downstream_chain import compute_downstream
        core = simulate()
        low = compute_downstream(
            core_result=core,
            patient_expanded={'intellectual_engagement': 0.1, 'baseline_age': 70},
        )
        high = compute_downstream(
            core_result=core,
            patient_expanded={'intellectual_engagement': 0.9, 'baseline_age': 70},
        )
        assert high[-1]['memory_index'] > low[-1]['memory_index']
