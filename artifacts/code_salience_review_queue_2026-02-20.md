# Code Salience Review Queue

**Date:** 2026-02-20
**Purpose:** Every file in this repo was evaluated for relevance to the mitochondrial aging simulation project. Files below are flagged for your review. Nothing here is a stray from another git repo — everything was built for this project — but significant dead code and orphans were found.

**Verdict: 0 strays from another project. 12 files flagged for review (unused, orphaned, or dead-end).**

---

## Category 1: Dead-End Optimizer Suite (5 files + 3 supporting)

These optimizer modules were built for this project but are **never used in production** — only imported by their test files. Meanwhile, the sibling `~/ea-toolkit` project provides a mature, tested optimizer library (HillClimber, ES, DE, CMA-ES, NoveltySeeker, Ensemble) that is already integrated via `ea_optimizer.py`.

| File | Lines | What It Does | Used By (production) | Test |
|------|-------|-------------|---------------------|------|
| `bayesian_optimizer.py` | ~150 | Surrogate + UCB acquisition optimizer | Nothing | `test_search_addons_tools.py` |
| `active_learning_optimizer.py` | ~120 | Active learning surrogate optimizer | Nothing | `test_search_addons_tools.py` |
| `map_elites_optimizer.py` | ~140 | MAP-Elites quality-diversity optimizer | Nothing | `test_search_addons_tools.py` |
| `nsga2_optimizer.py` | ~180 | NSGA-II multi-objective search | Nothing | `test_search_addons_tools.py` |
| `robust_optimizer.py` | ~120 | Robust optimizer across patient uncertainty | Nothing | `test_search_addons_tools.py` |
| `ml_prefilter_runner.py` | ~100 | ML prefilter before true simulation | Nothing | `test_ml_prefilter_runner.py` |

**Supporting modules** (would become orphans if the above are removed):

| File | Lines | What It Does | Used By (production) |
|------|-------|-------------|---------------------|
| `search_addons.py` | ~80 | Shared helpers: `random_protocol()`, `gaussian_mutation()`, etc. | Only the 5 dead-end modules above |
| `surrogate_optimizer.py` | ~100 | KNN surrogate model | Only `bayesian_optimizer`, `active_learning_optimizer`, `ml_prefilter_runner` |
| `gradient_refiner.py` | ~120 | Gradient-based local refinement | Only `ml_prefilter_runner` |

**Decision options:**
- **Keep:** These represent exploratory optimization work. They're tested (21 tests in `test_search_addons_tools.py` + `test_surrogate_optimizer.py` + `test_gradient_refiner.py` + `test_ml_prefilter_runner.py`), so they don't hurt test counts. They just take up space.
- **Archive:** Move to `archive/optimizers/` to declutter the root directory.
- **Remove:** Delete all 8 files + their 4 test files. This would reduce the codebase by ~12 files and ~70 tests. `ea_optimizer.py` (which uses the external ea-toolkit) fully covers optimization needs.

---

## Category 2: Orphan Modules (3 files)

These files are **never imported by anything** in the project. No test, no production code references them.

| File | What It Does | Why It Might Be Here |
|------|-------------|---------------------|
| `protocol_mtdna_synthesis.py` | Prints a 9-step mtDNA synthesis protocol as text | Reference documentation masquerading as code. Has no imports from this project. Could be a markdown file instead. |
| `layer_viz.py` | Population-layer comparison plots (core vs agroecology vs grief) | Built for cohort analysis but never wired into any pipeline. Has a `__main__` CLI entry point but nothing calls it. |
| `visualize_transplant_comparison.py` | Side-by-side plot from a specific 2026-02-18 artifact | One-off visualization for a specific experiment. Hardcodes the artifact filename. |

**Decision options:**
- `protocol_mtdna_synthesis.py`: Convert to markdown in `docs/` or keep as runnable reference.
- `layer_viz.py`: Wire into `visualize.py` or the CLI, or archive.
- `visualize_transplant_comparison.py`: Archive or delete — its purpose was served when it generated the plot.

---

## Category 3: CLAUDE.md Inconsistency (not a file issue)

CLAUDE.md extensively references `cramer_bridge.py` (imports, code patterns, architecture docs). **This file does not exist.** The actual file is `kcramer_bridge.py` (using the `kcramer` namespace to disambiguate from John G. Cramer). All real imports in the codebase use `kcramer_bridge` correctly.

Similarly, CLAUDE.md references `cramer-toolkit` importing as `cramer`, but the actual import namespace is `kcramer`.

**Recommendation:** Update CLAUDE.md to use the correct filenames. This is a documentation bug, not a code issue. The code itself is consistent.

---

## Category 4: Cross-Project Data Dependencies (2 files)

These files read data from the sibling Evolutionary Robotics project. They don't import Python code from ER — they read a TSV file as seed data for experiments.

| File | External Data | Purpose |
|------|--------------|---------|
| `character_seed_experiment.py` | `~/pybullet_test/Evolutionary-Robotics/archetypometrics_characters.tsv` | Uses 2000 fictional character archetypes as seeds for LLM-generated intervention protocols |
| `pds_mapping.py` | Same file | Maps Zimmerman's Power/Danger/Structure dimensions to patient parameters |

**Assessment:** These are legitimate cross-project data sharing. The character dataset is a shared resource. The files fail gracefully if the TSV is missing. No action needed, but documenting the dependency is good practice.

---

## Not Flagged (everything else)

The remaining ~50 Python files all serve clear roles in the project architecture:

- **Core layer** (5 files): `constants.py`, `simulator.py`, `analytics.py`, `schemas.py`, `llm_common.py`
- **Precision medicine expansion** (10 files): genetics, lifestyle, supplements, parameter resolver, downstream chain, scenario framework
- **Protocol dictionary pipeline** (7 files): record, dictionary, enrichment, classifier, rewrite rules, pattern language, review
- **Experiment scripts** (14 files): Tiers 1–5, all import from core
- **Visualization** (5 active files): `visualize.py`, `cliff_mapping.py`, `scenario_plot.py`, `resilience_viz.py`, `grief_mito_viz.py`, `zimmerman_viz.py`
- **Integration bridges** (6 files): zimmerman, kcramer, grief, ea-toolkit bridges + runners
- **Resilience suite** (2 files): `disturbances.py`, `resilience_metrics.py`
- **Utilities** (4 files): `generate_patients.py`, `pest_control.py`, `prompt_templates.py`, `lit_spider.py`
- **Tests** (34 files): All have active production counterparts

The `ten_types_audit.py` is generic in concept (innovation framework audit) but is specifically configured for this repository, has its own test file (6 tests), and produces useful meta-analysis output. It belongs here.

The `scripts/zotero_index.py` is a bibliography tool that indexes Zotero PDFs for literature validation. Standalone but purposeful for a research project grounded in published literature.
