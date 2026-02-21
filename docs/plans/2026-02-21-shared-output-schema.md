# Shared Simulator Output Schema Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define a common JSON envelope for all 4 ODE simulators (LEMURS, mito, grief, stock) so that cross-simulator analysis tools, visualizations, and comparisons can consume simulator output without per-simulator special-casing.

**Architecture:** A single `SimulatorOutput` dataclass (or plain dict factory) defined in zimmerman-toolkit that wraps any simulator's output into a standardized envelope: `{metadata, trajectory, analytics, parameters}`. Each simulator gets a thin `to_standard_output()` adapter function in its own bridge module. The zimmerman-toolkit provides `validate_output()` and `compare_outputs()` utilities.

**Tech Stack:** Python 3.11+, numpy-only, JSON-serializable via NumpyEncoder

---

## Context

All 4 ODE simulators produce structurally similar output — a time-series trajectory (ndarray), nested analytics (4-pillar dicts), and input parameters. But the details diverge:

| Dimension | LEMURS | Mito | Grief | Stock |
|-----------|--------|------|-------|-------|
| Time key | `"times"` | `"time"` | `"times"` | implicit index |
| Time units | weeks | years | years | years (252 steps/yr) |
| State key | `"states"` (106,14) | `"states"` (3001,8) | `"states"` (3651,11) | trajectory (2521,7) |
| Pillar names | sleep_quality, stress_anxiety, physiological, intervention_response | energy, damage, dynamics, intervention | grief_trajectory, physiological_stress, immune_inflammatory, health_outcomes | returns, risk, drawdown, regime_adaptation |
| Flat key convention | `pillar_metric` | `pillar_metric` | `pillar_metric` | not flattened |
| Inf handling | cap 999 | cap ±999 | cap 999 | raw |
| Has baseline comparison | yes | yes | no | no |
| Extra arrays | — | heteroplasmy, deletion_het | — | trajectory embedded in run() |

These differences make cross-simulator tools (visualization, comparison, trajectory metrics) require per-simulator code. A shared envelope eliminates this.

**What the schema is NOT:**
- It does NOT replace each simulator's native output format
- It does NOT change the Zimmerman protocol (`run()` → flat dict)
- It does NOT normalize units across simulators (hours stay hours, years stay years)
- It is a **read-side** adapter, not a write-side constraint

---

## Reference Files

**Simulators to adapt:**
- `~/lemurs-simulator/simulator.py` — `simulate()` returns `{states, times, intervention, patient}`
- `~/how-to-live-much-longer/simulator.py` — `simulate()` returns `{time, states, heteroplasmy, ...}`
- `~/grief-simulator/simulator.py` — `simulate()` returns `{states, times, intervention, patient}`
- `~/stock-simulator/stock_simulator/dynamics.py` — `simulate_ode()` returns ndarray

**Analytics to wrap:**
- `~/lemurs-simulator/analytics.py` — `compute_all()` → 4-pillar nested dict
- `~/how-to-live-much-longer/analytics.py` — `compute_all()` → 4-pillar nested dict
- `~/grief-simulator/analytics.py` — `compute_all()` → 4-pillar nested dict
- `~/stock-simulator/stock_simulator/analytics.py` — `compute_all()` → 4-pillar nested dict

**Existing NumpyEncoder pattern:**
- `~/lemurs-simulator/analytics.py` — `NumpyEncoder` class for JSON serialization
- `~/how-to-live-much-longer/analytics.py` — same `NumpyEncoder`

---

## Schema Definition

```python
{
    # ── Metadata ─────────────────────────────────────────────────────────
    "schema_version": "1.0",
    "simulator": {
        "name": str,               # "lemurs" | "mito" | "grief" | "stock"
        "description": str,        # human-readable one-liner
        "state_dim": int,          # number of state variables
        "param_dim": int,          # number of input parameters
        "state_names": list[str],  # ordered state variable names
        "time_unit": str,          # "weeks" | "years" | "seconds"
        "time_horizon": float,     # total simulation duration in time_unit
    },

    # ── Trajectory ───────────────────────────────────────────────────────
    "trajectory": {
        "times": list[float],      # length T, in simulator's native time_unit
        "states": list[list[float]],  # T × D, row-major
        "n_steps": int,
        "dt": float,               # mean timestep in time_unit
        # Optional extra arrays (simulator-specific):
        "extra": {                 # e.g., {"heteroplasmy": [...], ...}
            str: list[float],
        },
    },

    # ── Analytics ────────────────────────────────────────────────────────
    "analytics": {
        "pillars": {
            str: {                 # pillar_name → {metric_name → value}
                str: float,
            },
        },
        "pillar_names": list[str],  # ordered pillar names
        "flat": {                   # flattened pillar_metric → value
            str: float,
        },
    },

    # ── Parameters ───────────────────────────────────────────────────────
    "parameters": {
        "input": dict,             # full input params as passed to simulator
        "bounds": dict,            # {param_name: [lo, hi]} from param_spec()
    },
}
```

This schema is JSON-serializable (using NumpyEncoder for ndarray→list conversion) and self-describing (metadata tells you how to interpret everything else).

---

## Tasks

### Task 1: Define schema module in zimmerman-toolkit

**Files:**
- Create: `~/zimmerman-toolkit/zimmerman/output_schema.py`
- Test: `~/zimmerman-toolkit/tests/test_output_schema.py`

**Step 1: Write tests**

Create `~/zimmerman-toolkit/tests/test_output_schema.py`:

```python
"""Tests for shared simulator output schema."""
import numpy as np
import json
import pytest
from zimmerman.output_schema import (
    SimulatorOutput,
    validate_output,
    NumpyEncoder,
)


def _make_example_output():
    """Create a valid SimulatorOutput for testing."""
    return SimulatorOutput(
        simulator_name="test",
        simulator_description="Test simulator",
        state_dim=3,
        param_dim=2,
        state_names=["x", "y", "z"],
        time_unit="years",
        time_horizon=10.0,
        times=np.linspace(0, 10, 101),
        states=np.random.default_rng(42).random((101, 3)),
        input_params={"a": 1.0, "b": 2.0},
        param_bounds={"a": (0.0, 5.0), "b": (0.0, 10.0)},
        pillars={
            "outcome": {"metric_a": 1.5, "metric_b": 2.3},
            "risk": {"metric_c": 0.1},
        },
    )


class TestSimulatorOutput:
    """Test SimulatorOutput construction and serialization."""

    def test_construction(self):
        out = _make_example_output()
        assert out.simulator_name == "test"
        assert out.state_dim == 3

    def test_to_dict(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["simulator"]["name"] == "test"
        assert d["simulator"]["state_dim"] == 3
        assert d["trajectory"]["n_steps"] == 101
        assert len(d["trajectory"]["times"]) == 101
        assert len(d["trajectory"]["states"]) == 101
        assert len(d["trajectory"]["states"][0]) == 3
        assert "outcome" in d["analytics"]["pillars"]
        assert d["analytics"]["flat"]["outcome_metric_a"] == 1.5
        assert d["parameters"]["input"]["a"] == 1.0

    def test_json_serializable(self):
        out = _make_example_output()
        d = out.to_dict()
        # Should not raise
        s = json.dumps(d, cls=NumpyEncoder)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["schema_version"] == "1.0"

    def test_pillar_names_preserved(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["analytics"]["pillar_names"] == ["outcome", "risk"]

    def test_flat_keys_correct(self):
        out = _make_example_output()
        d = out.to_dict()
        assert "outcome_metric_a" in d["analytics"]["flat"]
        assert "risk_metric_c" in d["analytics"]["flat"]

    def test_extra_arrays(self):
        out = _make_example_output()
        out.extra_arrays = {"heteroplasmy": np.linspace(0, 0.5, 101)}
        d = out.to_dict()
        assert "heteroplasmy" in d["trajectory"]["extra"]
        assert len(d["trajectory"]["extra"]["heteroplasmy"]) == 101

    def test_bounds_format(self):
        out = _make_example_output()
        d = out.to_dict()
        assert d["parameters"]["bounds"]["a"] == [0.0, 5.0]


class TestValidation:
    """Test validate_output checks."""

    def test_valid_output_passes(self):
        out = _make_example_output()
        d = out.to_dict()
        errors = validate_output(d)
        assert len(errors) == 0

    def test_missing_schema_version(self):
        out = _make_example_output()
        d = out.to_dict()
        del d["schema_version"]
        errors = validate_output(d)
        assert any("schema_version" in e for e in errors)

    def test_wrong_state_dim(self):
        out = _make_example_output()
        d = out.to_dict()
        d["simulator"]["state_dim"] = 5  # wrong, should be 3
        errors = validate_output(d)
        assert any("state_dim" in e for e in errors)

    def test_mismatched_times_states(self):
        out = _make_example_output()
        d = out.to_dict()
        d["trajectory"]["times"] = d["trajectory"]["times"][:50]  # truncate
        errors = validate_output(d)
        assert any("times" in e.lower() or "steps" in e.lower() for e in errors)
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_output_schema.py -v`
Expected: ImportError

**Step 3: Implement output_schema.py**

Create `~/zimmerman-toolkit/zimmerman/output_schema.py`:

```python
"""Shared output schema for ODE simulator results.

Defines a common JSON envelope that any ODE simulator can produce,
enabling cross-simulator analysis, visualization, and comparison
without per-simulator special-casing.

Usage:
    from zimmerman.output_schema import SimulatorOutput

    output = SimulatorOutput(
        simulator_name="lemurs",
        state_dim=14,
        ...
    )
    d = output.to_dict()   # JSON-serializable dict
    errors = validate_output(d)  # structural validation
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


@dataclass
class SimulatorOutput:
    """Standardized simulator output envelope.

    Wraps trajectory data, analytics, and parameters into a common
    format that cross-simulator tools can consume.
    """

    # Metadata
    simulator_name: str
    simulator_description: str = ""
    state_dim: int = 0
    param_dim: int = 0
    state_names: list[str] = field(default_factory=list)
    time_unit: str = "years"
    time_horizon: float = 0.0

    # Trajectory
    times: np.ndarray | None = None
    states: np.ndarray | None = None
    extra_arrays: dict[str, np.ndarray] = field(default_factory=dict)

    # Analytics
    pillars: dict[str, dict[str, float]] = field(default_factory=dict)

    # Parameters
    input_params: dict = field(default_factory=dict)
    param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict following the shared schema."""
        times_list = self.times.tolist() if self.times is not None else []
        states_list = self.states.tolist() if self.states is not None else []
        n_steps = len(times_list)

        # Compute dt
        if self.times is not None and len(self.times) > 1:
            dt = float(np.mean(np.diff(self.times)))
        else:
            dt = 0.0

        # Flatten analytics
        flat = {}
        for pillar_name, metrics in self.pillars.items():
            for metric_name, value in metrics.items():
                flat[f"{pillar_name}_{metric_name}"] = value

        # Extra arrays
        extra = {}
        for name, arr in self.extra_arrays.items():
            extra[name] = arr.tolist() if isinstance(arr, np.ndarray) else list(arr)

        # Bounds as lists (JSON doesn't have tuples)
        bounds = {k: list(v) for k, v in self.param_bounds.items()}

        return {
            "schema_version": "1.0",
            "simulator": {
                "name": self.simulator_name,
                "description": self.simulator_description,
                "state_dim": self.state_dim,
                "param_dim": self.param_dim,
                "state_names": list(self.state_names),
                "time_unit": self.time_unit,
                "time_horizon": self.time_horizon,
            },
            "trajectory": {
                "times": times_list,
                "states": states_list,
                "n_steps": n_steps,
                "dt": dt,
                "extra": extra,
            },
            "analytics": {
                "pillars": dict(self.pillars),
                "pillar_names": list(self.pillars.keys()),
                "flat": flat,
            },
            "parameters": {
                "input": dict(self.input_params),
                "bounds": bounds,
            },
        }

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), cls=NumpyEncoder, **kwargs)


def validate_output(d: dict) -> list[str]:
    """Validate a dict against the shared output schema.

    Returns a list of error messages. Empty list = valid.
    """
    errors = []

    # Top-level keys
    if "schema_version" not in d:
        errors.append("Missing required key: schema_version")
    for section in ["simulator", "trajectory", "analytics", "parameters"]:
        if section not in d:
            errors.append(f"Missing required section: {section}")

    if errors:
        return errors  # can't validate further without top-level structure

    # Simulator metadata
    sim = d["simulator"]
    if "name" not in sim:
        errors.append("Missing simulator.name")
    if "state_dim" not in sim:
        errors.append("Missing simulator.state_dim")

    # Trajectory consistency
    traj = d["trajectory"]
    times = traj.get("times", [])
    states = traj.get("states", [])

    if len(times) != len(states):
        errors.append(
            f"Trajectory length mismatch: times has {len(times)} steps, "
            f"states has {len(states)} steps"
        )

    if states and sim.get("state_dim"):
        actual_dim = len(states[0]) if states[0] else 0
        if actual_dim != sim["state_dim"]:
            errors.append(
                f"state_dim mismatch: simulator says {sim['state_dim']}, "
                f"trajectory has {actual_dim} columns"
            )

    # Analytics
    analytics = d["analytics"]
    if "pillars" not in analytics:
        errors.append("Missing analytics.pillars")
    if "flat" not in analytics:
        errors.append("Missing analytics.flat")

    # Check flat keys match pillars
    if "pillars" in analytics and "flat" in analytics:
        expected_flat = set()
        for pillar_name, metrics in analytics["pillars"].items():
            for metric_name in metrics:
                expected_flat.add(f"{pillar_name}_{metric_name}")
        actual_flat = set(analytics["flat"].keys())
        if expected_flat != actual_flat:
            missing = expected_flat - actual_flat
            extra = actual_flat - expected_flat
            if missing:
                errors.append(f"Flat keys missing from pillars: {missing}")
            if extra:
                errors.append(f"Flat keys not in pillars: {extra}")

    return errors


def compare_outputs(*outputs: dict) -> dict:
    """Compare multiple simulator outputs sharing the same schema.

    Args:
        *outputs: SimulatorOutput.to_dict() results.

    Returns:
        Comparison dict with per-metric deltas and rankings.
    """
    if len(outputs) < 2:
        return {"error": "Need at least 2 outputs to compare"}

    names = [o["simulator"]["name"] for o in outputs]

    # Find shared flat metric keys
    all_keys = [set(o["analytics"]["flat"].keys()) for o in outputs]
    shared_keys = all_keys[0]
    for ks in all_keys[1:]:
        shared_keys &= ks

    comparison = {
        "simulators": names,
        "shared_metrics": sorted(shared_keys),
        "per_metric": {},
    }

    for key in sorted(shared_keys):
        values = [o["analytics"]["flat"][key] for o in outputs]
        comparison["per_metric"][key] = {
            "values": dict(zip(names, values)),
            "range": float(max(values) - min(values)),
        }

    return comparison
```

**Step 4: Run tests**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_output_schema.py -v`
Expected: All tests pass.

**Step 5: Register in __init__.py**

Add to `zimmerman/__init__.py`:
```python
from zimmerman.output_schema import SimulatorOutput, validate_output, compare_outputs, NumpyEncoder
```

Add to `__all__`:
```python
"SimulatorOutput", "validate_output", "compare_outputs", "NumpyEncoder",
```

**Step 6: Commit**

```bash
cd ~/zimmerman-toolkit
git add zimmerman/output_schema.py tests/test_output_schema.py zimmerman/__init__.py
git commit -m "feat: add shared SimulatorOutput schema for cross-simulator interop"
```

---

### Task 2: LEMURS adapter

**Files:**
- Modify: `~/lemurs-simulator/lemurs_simulator.py`
- Test: `~/lemurs-simulator/tests/test_lemurs_simulator.py`

**Step 1: Write tests for to_standard_output**

Add to `~/lemurs-simulator/tests/test_lemurs_simulator.py` (or create if needed):

```python
class TestStandardOutput:
    """Test shared output schema adapter."""

    def test_to_standard_output_returns_valid_schema(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimmerman-toolkit"))
        from zimmerman.output_schema import validate_output

        sim = LEMURSSimulator()
        result = sim.run({})
        std = sim.to_standard_output({})
        errors = validate_output(std)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_standard_output_metadata(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimmerman-toolkit"))

        sim = LEMURSSimulator()
        std = sim.to_standard_output({})
        assert std["simulator"]["name"] == "lemurs"
        assert std["simulator"]["state_dim"] == 14
        assert std["simulator"]["time_unit"] == "weeks"
        assert len(std["simulator"]["state_names"]) == 14

    def test_standard_output_trajectory(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimmerman-toolkit"))

        sim = LEMURSSimulator()
        std = sim.to_standard_output({})
        assert std["trajectory"]["n_steps"] == 106
        assert len(std["trajectory"]["states"][0]) == 14

    def test_standard_output_analytics(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimmerman-toolkit"))

        sim = LEMURSSimulator()
        std = sim.to_standard_output({})
        assert "sleep_quality" in std["analytics"]["pillars"]
        assert "stress_anxiety" in std["analytics"]["pillars"]
```

**Step 2: Implement to_standard_output in LEMURSSimulator**

Add method to `LEMURSSimulator` class in `~/lemurs-simulator/lemurs_simulator.py`:

```python
def to_standard_output(self, params: dict) -> dict:
    """Run simulation and return shared output schema dict.

    This produces the zimmerman-toolkit SimulatorOutput format
    for cross-simulator analysis and comparison.
    """
    import sys
    _ZT_PATH = str(Path(__file__).resolve().parent.parent / "zimmerman-toolkit")
    if _ZT_PATH not in sys.path:
        sys.path.insert(0, _ZT_PATH)
    from zimmerman.output_schema import SimulatorOutput

    # Resolve params
    intervention, patient = self._split_params(params)
    result = simulate(intervention, patient)
    analytics_result = compute_all(result["states"], result["times"])

    output = SimulatorOutput(
        simulator_name="lemurs",
        simulator_description="14-state college student biopsychosocial ODE (15-week semester)",
        state_dim=14,
        param_dim=12,
        state_names=list(STATE_NAMES),
        time_unit="weeks",
        time_horizon=15.0,
        times=result["times"],
        states=result["states"],
        pillars=analytics_result,
        input_params=params,
        param_bounds=dict(self.param_spec()),
    )
    return output.to_dict()
```

**Step 3: Run tests**

Run: `cd ~/lemurs-simulator && PYTHONPATH=../zimmerman-toolkit python -m pytest tests/test_lemurs_simulator.py::TestStandardOutput -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
cd ~/lemurs-simulator
git add lemurs_simulator.py tests/test_lemurs_simulator.py
git commit -m "feat: add to_standard_output() for shared output schema"
```

---

### Task 3: Mito adapter

**Files:**
- Modify: `~/how-to-live-much-longer/zimmerman_bridge.py` (or `mito_simulator.py`, whichever has MitoSimulator)
- Test: add test for to_standard_output

**Step 1: Implement to_standard_output**

Add `to_standard_output()` method to MitoSimulator, following the LEMURS pattern but with:
- `simulator_name="mito"`
- `state_dim=8`
- `time_unit="years"`
- `time_horizon=30.0`
- `extra_arrays={"heteroplasmy": result["heteroplasmy"], "deletion_heteroplasmy": result["deletion_heteroplasmy"]}`

**Step 2: Write tests and verify**

Run: `cd ~/how-to-live-much-longer && python -m pytest tests/ -v`

**Step 3: Commit**

```bash
cd ~/how-to-live-much-longer
git add zimmerman_bridge.py tests/
git commit -m "feat: add to_standard_output() for shared output schema"
```

---

### Task 4: Grief adapter

**Files:**
- Modify: `~/grief-simulator/grief_simulator.py`

**Step 1: Implement to_standard_output**

Same pattern:
- `simulator_name="grief"`
- `state_dim=11`
- `time_unit="years"`
- `time_horizon=10.0`

**Step 2: Write tests and verify**

**Step 3: Commit**

```bash
cd ~/grief-simulator
git add grief_simulator.py tests/
git commit -m "feat: add to_standard_output() for shared output schema"
```

---

### Task 5: Cross-simulator integration test

**Files:**
- Create: `~/zimmerman-toolkit/tests/test_output_schema_integration.py`

**Step 1: Write integration tests**

```python
"""Integration tests: shared output schema across real simulators."""
import sys
import pytest
from pathlib import Path

from zimmerman.output_schema import validate_output, compare_outputs

# Try to import each simulator
simulators = {}
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lemurs-simulator"))
    from lemurs_simulator import LEMURSSimulator
    simulators["lemurs"] = LEMURSSimulator()
except ImportError:
    pass

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "how-to-live-much-longer"))
    from zimmerman_bridge import MitoSimulator
    simulators["mito"] = MitoSimulator()
except ImportError:
    pass

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "grief-simulator"))
    from grief_simulator import GriefSimulator
    simulators["grief"] = GriefSimulator()
except ImportError:
    pass


@pytest.mark.skipif(len(simulators) == 0, reason="No simulators available")
class TestCrossSimulatorSchema:
    @pytest.mark.parametrize("name", list(simulators.keys()))
    def test_valid_schema(self, name):
        sim = simulators[name]
        if not hasattr(sim, "to_standard_output"):
            pytest.skip(f"{name} has no to_standard_output()")
        d = sim.to_standard_output({})
        errors = validate_output(d)
        assert len(errors) == 0, f"{name} validation errors: {errors}"

    @pytest.mark.parametrize("name", list(simulators.keys()))
    def test_schema_version(self, name):
        sim = simulators[name]
        if not hasattr(sim, "to_standard_output"):
            pytest.skip(f"{name} has no to_standard_output()")
        d = sim.to_standard_output({})
        assert d["schema_version"] == "1.0"

    def test_compare_outputs(self):
        outputs = []
        for name, sim in simulators.items():
            if hasattr(sim, "to_standard_output"):
                outputs.append(sim.to_standard_output({}))
        if len(outputs) < 2:
            pytest.skip("Need at least 2 simulators for comparison")
        result = compare_outputs(*outputs)
        assert "simulators" in result
```

**Step 2: Run tests**

Run: `cd ~/zimmerman-toolkit && python -m pytest tests/test_output_schema_integration.py -v`

**Step 3: Commit**

```bash
cd ~/zimmerman-toolkit
git add tests/test_output_schema_integration.py
git commit -m "test: add cross-simulator output schema integration tests"
```

---

### Task 6: Documentation

**Files:**
- Create: `~/zimmerman-toolkit/docs/OutputSchema.md`

**Step 1: Write documentation**

```markdown
# Shared Simulator Output Schema

A common JSON envelope for all ODE simulators in the workspace.

## Why

Every ODE simulator (LEMURS, mito, grief, stock) produces trajectory data,
4-pillar analytics, and input parameters in slightly different formats.
The shared schema wraps them into a uniform envelope so that cross-simulator
tools can consume them generically.

## Usage

```python
from zimmerman.output_schema import SimulatorOutput, validate_output

# Any simulator with to_standard_output():
from lemurs_simulator import LEMURSSimulator
sim = LEMURSSimulator()
output = sim.to_standard_output({"nature_rx": 0.8})

# Validate
errors = validate_output(output)
assert len(errors) == 0

# Access uniformly
print(output["simulator"]["name"])           # "lemurs"
print(output["trajectory"]["n_steps"])       # 106
print(output["analytics"]["flat"].keys())    # all scalar metrics
```

## Schema Structure

[Schema definition as documented in the plan Context section]

## Cross-Simulator Comparison

```python
from zimmerman.output_schema import compare_outputs

lemurs_out = lemurs_sim.to_standard_output({})
mito_out = mito_sim.to_standard_output({})
comparison = compare_outputs(lemurs_out, mito_out)
# Shows shared metrics and value ranges
```
```

**Step 2: Commit**

```bash
cd ~/zimmerman-toolkit
git add docs/OutputSchema.md
git commit -m "docs: add shared output schema documentation"
```

---

## Verification

After all tasks:
```bash
# Zimmerman-toolkit tests
cd ~/zimmerman-toolkit
python -m pytest tests/test_output_schema.py -v
python -m pytest tests/ -v   # all ~285+ tests pass

# LEMURS standard output
cd ~/lemurs-simulator
PYTHONPATH=../zimmerman-toolkit python -c "
from lemurs_simulator import LEMURSSimulator
sim = LEMURSSimulator()
d = sim.to_standard_output({})
print(f'Schema v{d[\"schema_version\"]}: {d[\"simulator\"][\"name\"]} '
      f'{d[\"simulator\"][\"state_dim\"]}D, {d[\"trajectory\"][\"n_steps\"]} steps, '
      f'{len(d[\"analytics\"][\"flat\"])} metrics')
"
# Expected: Schema v1.0: lemurs 14D, 106 steps, ~30 metrics
```

## Design Rationale

**Why zimmerman-toolkit, not a new shared library:**
The zimmerman-toolkit already serves as the cross-simulator interoperability layer (Simulator protocol). Adding the output schema there keeps the cross-cutting concern in the right place.

**Why a dataclass, not just a dict:**
`SimulatorOutput` enforces required fields at construction time, provides `.to_dict()` and `.to_json()` methods, and documents the schema in code. But it's trivially convertible to a plain dict for consumers that don't want the class dependency.

**Why not normalize units:**
Normalizing would lose the domain-specific semantics (GAD-7 of 10 means "clinical threshold" — normalizing to 0-1 hides this). The schema preserves native units and includes `time_unit` metadata so consumers can handle conversion if needed.

**Why read-side adapter, not write-side constraint:**
Each simulator's `simulate()` function works perfectly as-is. Forcing them to produce the shared schema would break backward compatibility. Instead, `to_standard_output()` is a separate method that wraps the native output — zero-cost if you don't call it.

**Why flat analytics alongside nested pillars:**
The Zimmerman protocol expects flat dicts (`run()` → `{key: float}`). The nested pillars preserve domain structure (which metrics belong to which pillar). Including both avoids redundant flattening in consumers.
