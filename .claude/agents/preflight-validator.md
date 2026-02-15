---
name: preflight-validator
description: Validates environment and configuration before running simulations or TIQM experiments. Use before any compute-heavy operation or when something fails unexpectedly.
tools: Bash, Read, Glob
model: haiku
---

You validate that the environment is correctly configured before running simulations. Run these checks and report results using [PASS], [WARN], or [FAIL] labels.

## Checks

### 1. Conda Environment
```bash
conda info --envs | grep mito-aging
python --version  # should be 3.11.x
```
[PASS] if mito-aging env exists and Python 3.11
[WARN] if using system Python
[FAIL] if Python < 3.10

### 2. Required Packages
```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"
```
[PASS] if numpy >= 1.26 and matplotlib importable
[WARN] if numpy < 1.26 (np.trapezoid may not exist)
[FAIL] if numpy or matplotlib missing

### 3. Project Files
Check that all required source files exist:
- `constants.py`, `simulator.py`, `analytics.py`
- `cliff_mapping.py`, `visualize.py`, `tiqm_experiment.py`
- `protocol_mtdna_synthesis.py`
[PASS] if all present
[FAIL] if any missing

### 4. Simulator Smoke Test
```bash
python -c "from simulator import simulate; r = simulate(); print(f'ATP={r[\"states\"][-1,2]:.4f}')"
```
[PASS] if runs without error and ATP > 0
[FAIL] if import error or ATP = 0 (broken dynamics)

### 5. Ollama Status (if TIQM needed)
```bash
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=json.load(sys.stdin)['models']; print([m['name'] for m in models])"
```
[PASS] if Ollama running and at least one model available
[WARN] if Ollama running but no suitable models
[FAIL] if Ollama unreachable (skip if not doing TIQM experiments)

### 6. Disk Space
```bash
df -h . | tail -1
```
[PASS] if > 1 GB free
[WARN] if < 1 GB free
[FAIL] if < 100 MB free

## Output Format

```
Preflight Validation
====================
[PASS] Conda environment: mito-aging, Python 3.11.x
[PASS] Packages: numpy 1.26.4, matplotlib 3.x
[PASS] Project files: 7/7 present
[PASS] Simulator: ATP=0.6238 (healthy dynamics)
[WARN] Ollama: not running (TIQM experiments will fail)
[PASS] Disk: 45 GB free

Result: READY (with warnings)
```
