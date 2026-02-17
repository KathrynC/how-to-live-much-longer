# Compliance Impact Report: Deletion Replication Advantage 1.10 -> 1.21

Date: 2026-02-17

## Purpose

Quantify behavioral impact of the strict Appendix-2 compliance update that raised `DELETION_REPLICATION_ADVANTAGE` from `1.10` to `1.21`.

This change was made for conformance with source text wording ("at least 21% faster"), not as an independent re-calibration pass.

## Method

- Compare two settings in identical simulations: `1.10` (pre-compliance) vs `1.21` (compliance).
- Deterministic simulator runs (`stochastic=False`), 30-year horizon.
- Spotlight scenarios + 80 randomized patients per protocol for 5 protocol families.
- Metrics: `final_heteroplasmy`, `final_deletion_heteroplasmy`, `final_atp`, deletion-cliff crossing.

## Spotlight Scenarios

| Scenario | final_het (1.10) | final_het (1.21) | Δhet | final_del_het (1.10) | final_del_het (1.21) | Δdel_het | final_atp (1.10) | final_atp (1.21) | ΔATP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_no_treatment | 0.4200 | 0.4200 | +0.0000 | 0.2317 | 0.2317 | +0.0000 | 0.6859 | 0.6859 | -0.0000 |
| default_cocktail | 0.2428 | 0.2432 | +0.0005 | 0.0984 | 0.0989 | +0.0005 | 0.7696 | 0.7696 | -0.0000 |
| near_cliff_no_treatment | 0.7047 | 0.7055 | +0.0007 | 0.3975 | 0.3988 | +0.0013 | 0.6217 | 0.6215 | -0.0002 |
| near_cliff_aggressive | 0.1618 | 0.1618 | +0.0000 | 0.0449 | 0.0449 | +0.0000 | 0.6609 | 0.6609 | +0.0000 |
| near_cliff_transplant_focused | 0.1937 | 0.1937 | +0.0000 | 0.0436 | 0.0436 | +0.0000 | 0.6900 | 0.6900 | +0.0000 |

## Protocol-Level Population Sweep (N=80 patients per protocol)

| Protocol | mean Δhet | median Δhet | mean Δdel_het | median Δdel_het | mean ΔATP | median ΔATP | cliff-loss rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_treatment | +0.0004 | +0.0003 | +0.0010 | +0.0008 | -0.0002 | -0.0000 | 0.000 |
| conservative | +0.0009 | +0.0009 | +0.0017 | +0.0013 | -0.0001 | -0.0000 | 0.000 |
| moderate | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | 0.000 |
| aggressive | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.0000 | +0.0000 | 0.000 |
| transplant_focused | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.000 |

## Aggregate Effect (across protocol means)

- Mean Δfinal_heteroplasmy: +0.0003
- Mean Δfinal_deletion_heteroplasmy: +0.0006
- Mean Δfinal_atp: -0.0000

## Interpretation

- As expected, increasing deletion replication advantage raises deletion burden and total heteroplasmy, and lowers ATP on average.
- Magnitude varies by protocol and patient baseline state; near-cliff and weak-treatment regimes are most sensitive.
- This report quantifies *conformance cost*: stricter textual compliance increases model aggressiveness in late-life deterioration dynamics.

## Assumptions and Scientific Grounding

- Ground truth for this conformance update is Cramer Appendix 2 wording (forthcoming Springer 2026).
- The comparison isolates one parameter change (`1.10` -> `1.21`) while holding all other settings fixed.
- Results are simulation-model effects, not direct clinical outcome predictions.