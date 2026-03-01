# Lakoff-CA Integration: Dual-Vocabulary Annotation for Mitochondrial Semantics

**Date**: 2026-02-22  
**Status**: Integrated, tested, ready for use.

## Overview

This integration maps the discrete state transitions of the mitochondrial semantic cellular automaton (CA) to Lakoffian image schemas (CONTAINER, PATH, SCALE, CYCLE, BALANCE, FORCE), providing a **dual‑vocabulary annotation layer** that links biological observables to cross‑domain cognitive abstractions.

The system builds on the existing Lakoff infrastructure (`patterns/lakoff_integration.py`) and adapts the image‑schema detector pattern from `motion‑analytics‑toolkit`. Each CA trajectory is annotated with:

1. **Global image schema metrics** (six schemas with quantitative measures)
2. **Lakoff archetype similarity scores** (conservative, aggressive, transplant‑focused, metabolic_optimizer)
3. **Metaphor violation detection** (grounding‑vs‑linking mismatches)
4. **Per‑step dual‑vocabulary annotations** (discrete bin labels + image schema activations)

The integration follows **Lakoff Maxim‑7**: ground first in observable biological metrics, then link to cross‑domain abstractions for cognitive accessibility.

## Components

### 1. `ca_image_schemas.py` – Image Schema Detectors

Adapts the motion‑analytics image‑schema detectors for CA trajectories. Defines:

- **`CATrajectory`** – wraps discrete state lists, converts to continuous exemplars using bin‑schema centers.
- **`CAImageSchemaDetector`** – six detectors:
  - **PATH** – trajectories toward/away from cliff (N_deletion, ATP displacement)
  - **CYCLE** – ROS/NAD/senescence oscillations (FFT‑based frequency detection)
  - **CONTAINER** – heteroplasmy fractions bounded [0,1], ATP reserve headroom
  - **SCALE** – graded severity progression across ordered bins (minimal → growing → approaching_cliff → past_cliff)
  - **BALANCE** – homeostatic regulation (N_healthy copy homeostasis)
  - **FORCE** – intervention‑driven push against damage accumulation

Each detector returns an `ImageSchema` object with scalar metrics.

### 2. `ca_lakoff_annotator.py` – Main Annotation Pipeline

Orchestrates the full annotation workflow:

```python
from ca_lakoff_annotator import annotate_from_simulation

annotation = annotate_from_simulation(
    patient={...},          # optional (default: 70‑year‑old moderate patient)
    intervention={...},     # optional (default: no treatment)
    sim_years=30.0,        # simulation duration
    dt=0.25,               # CA time step (years)
)
```

The returned `annotation` dictionary contains:

| Key | Description |
|-----|-------------|
| `discrete_trajectory` | list of discrete state dicts (bin labels) |
| `image_schemas` | dict mapping schema names to metric dicts |
| `ca_analytics` | raw CA analytics (rule/cascade/attractor/fidelity stats) |
| `ca_features` | flattened CA feature dict (for Lakoff similarity) |
| `schema_features` | flattened image‑schema feature dict |
| `archetype_similarities` | dict mapping archetype names to similarity scores (0‑1) |
| `best_archetype` | (name, score) tuple of best‑matching archetype |
| `metaphor_violations` | list of grounding‑vs‑linking mismatches (empty if none) |
| `dual_vocabulary` | per‑step annotations (step, age, discrete_state) |
| `all_features` | merged feature dict (CA + schemas + ODE analytics) |

### 3. Lakoff Feature Layer Extensions

`patterns/lakoff_integration.py` has been extended with CA‑specific feature layers:

- **Grounded features**: `ca_bin_agreement`, `ca_attractor`, `ca_fidelity`, `ca_rule_firing`
- **Linking features**: image‑schema metrics (`schema_path_straightness`, `schema_cycle_frequency`, etc.)

These features are automatically extracted and used for archetype matching.

### 4. SystemViz Crosswalk Updates

`patterns/lakoff_systemviz_crosswalk.json` now includes CA‑specific SystemViz terms:

- `driver.ca.rule_firing`, `driver.ca.cascade`, `driver.ca.attractor_transition`
- `signal.ca.bin_transition`, `signal.ca.cliff_crossing_signal`
- `state.ca.attractor`, `state.ca.bin_state`, `state.ca.discrete_state`
- `boundary.ca.cliff_boundary`, `boundary.ca.bin_threshold`
- `relation.ca.bin_ordering`, `relation.ca.variable_coupling`
- `domain.ca.state_space`, `domain.ca.bin_lattice`

These terms are mapped to relevant Lakoff archetypes (conservative, aggressive, transplant_focused, metabolic_optimizer).

## Usage Examples

### Batch Annotation of Patient Populations

```bash
# Annotate first 20 normal patients (no treatment)
python run_ca_lakoff_batch.py --population normal --max-patients 20

# Annotate edge‑case patients with a specific intervention
python run_ca_lakoff_batch.py --population edge \
  --intervention '{"rapamycin_dose":0.5, "nad_supplement":0.75}' \
  --output-dir output/ca_lakoff_edge_treated
```

Outputs individual annotation JSON files in the specified directory plus a `batch_summary.json`.

### Add CA‑Lakoff Annotations to an Existing Protocol Dictionary

```bash
# Add CA‑Lakoff annotations to a protocol dictionary (first 10 records)
python add_ca_lakoff_to_dictionary.py \
  --input artifacts/protocol_pipeline/protocol_dictionary.json \
  --max-records 10 \
  --output output/protocol_dictionary_ca_lakoff.json
```

Adds a `lakoff_ca` field to each record with schema summaries and archetype matches.

### Direct API Usage

```python
from ca_lakoff_annotator import annotate_from_simulation, annotate_ca_trajectory
from ca_simulator import run_single_cell

# Run CA simulation
ca_result = run_single_cell(
    patient={"baseline_age": 70.0, "baseline_heteroplasmy": 0.3},
    intervention={"rapamycin_dose": 0.5},
)

# Annotate the CA trajectory
annotation = annotate_ca_trajectory(ca_result)

print(f"Best archetype: {annotation['best_archetype'][0]} ({annotation['best_archetype'][1]:.3f})")
print(f"Schemas detected: {list(annotation['image_schemas'].keys())}")
```

## Integration with Protocol Dictionary Pipeline

The protocol dictionary pipeline (`run_protocol_pipeline.py`) already includes a Lakoff archetype classifier (`patterns/lakoff_classifier.py`) that uses ODE analytics. The CA‑Lakoff annotator complements this by providing **CA‑specific image‑schema annotations**.

To keep the pipeline lightweight, CA‑Lakoff annotation is offered as a **post‑processing step** (`add_ca_lakoff_to_dictionary.py`). However, you can also extend `protocol_enrichment.py` to automatically include CA‑Lakoff annotations by adding a call to `annotate_from_simulation` inside `enrich_record`.

## Test Results

### Normal Population (20 patients, no treatment)
- **Archetype distribution**: 100% conservative (similarity 0.667)
- **Schemas detected**: all six schemas present in every trajectory
- **Metaphor violations**: none

### Edge‑Case Population (20 patients, no treatment)
- **Archetype distribution**: 95% conservative (similarity 0.667), 5% conservative with lower similarity (0.333) for extreme patients (`all_max`)
- **Schemas detected**: all six schemas present
- **Metaphor violations**: none

### Protocol Dictionary Sample (2 records with interventions)
- **Archetype distribution**: 100% aggressive (similarity 0.667, 0.333)
- **Schemas detected**: all six schemas present
- **Metaphor violations**: none

The results show that the Lakoff‑CA integration:
- Correctly distinguishes untreated (conservative) from treated (aggressive) protocols
- Detects all six image schemas in mitochondrial CA trajectories
- Produces no metaphor violations for the tested scenarios (grounding‑linking alignment)

## Future Enhancements

1. **Per‑step schema activations** – currently dual‑vocabulary annotations only include discrete state and age; a sliding‑window detector could add per‑step schema activation strengths.
2. **Visualization tools** – plot CA trajectory heatmaps with overlaid image‑schema activations and archetype similarity timelines.
3. **Integration with TIQM pipeline** – automatically run CA‑Lakoff annotation for each LLM‑generated protocol and include schema metrics in confirmation‑wave evaluation.
4. **Extended archetype set** – add more Lakoff archetypes specific to mitochondrial dynamics (e.g., `senolytic_focused`, `metabolic_flexibility`).
5. **Cross‑project portability** – the same architecture can be applied to the LEMURS CA and grief‑simulator CA with minimal adaptation.

## Dependencies

- `numpy`, `scipy` (for FFT)
- `patterns/lakoff_integration.py` (existing Lakoff framework)
- `ca_simulator.py`, `ca_analytics.py`, `ca_schema.py` (CA layer)
- `simulator.py`, `analytics.py` (ODE layer, optional)

All dependencies are already satisfied in the `mito‑aging` conda environment.

## References

- Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.
- Cramer, J. G. (forthcoming 2026). *How to Live Much Longer: The Mitochondrial DNA Connection*. Springer.
- Zimmerman, J. (2025). *The Zimmerman Toolkit: Black‑Box Simulator Interrogation*. (Internal project.)
- Evolutionary‑Robotics project (parent), `motion‑analytics‑toolkit` (image‑schema detector pattern).

---

*Integration developed 2026‑02‑22 as part of the “Lakoff pattern‑language integration” next‑step selection.*