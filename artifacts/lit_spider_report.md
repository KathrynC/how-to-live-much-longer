# Lit Spider Report

**Date:** 2026-02-16 13:12:32 UTC
**Model:** keyword-only
**Parameters searched:** 2
**PubMed queries:** 2
**Abstracts fetched:** 9
**LLM extractions:** 0
**Elapsed:** 4s

## Summary

| Parameter | Current | Priority | Assessment | Discrepancy | Lit Range | Papers |
|-----------|---------|----------|------------|-------------|-----------|--------|
| `heteroplasmy_cliff` | 0.7 fraction | medium | sparse | minor | 75–75 | 1 |
| `doubling_time_young` | 11.8 years | low | conflicting | ok | 10–81 | 2 |

## Medium Priority (refinable)

### `heteroplasmy_cliff`

- **Current value:** 0.7 fraction
- **Location:** `constants.py:HETEROPLASMY_CLIFF`
- **Citation:** Rossignol et al. 2003
- **Assessment:** sparse
- **Discrepancy:** minor
- **Literature range:** 75 – 75 (median 75, n=1)

**Key papers:**

- Fu Y, Land M, Kavlashvili T (2025). *Engineering mtDNA deletions by reconstituting end joining in human mitochondria.* [PMID:40068680](https://pubmed.ncbi.nlm.nih.gov/40068680/) [DOI](https://doi.org/10.1016/j.cell.2025.02.009)

**Extracted values:**

- 75.0 % (low) — keyword extraction (no semantic understanding)

## Low Priority (verify)

### `doubling_time_young`

- **Current value:** 11.8 years
- **Location:** `constants.py:DOUBLING_TIME_YOUNG`
- **Citation:** Vandiver et al. 2023 (Cramer Appendix 2 p.155)
- **Assessment:** conflicting
- **Discrepancy:** none
- **Literature range:** 10 – 81 (median 10, n=4)

**Key papers:**

- Shammas MK, Nie Y, Gilsrud A (2023). *CHCHD10 mutations induce tissue-specific mitochondrial DNA deletions with a distinct signature.* [PMID:37815936](https://pubmed.ncbi.nlm.nih.gov/37815936/) [DOI](https://doi.org/10.1093/hmg/ddad161)
- Vandiver AR, Hoang AN, Herbst A (2023). *Nanopore sequencing identifies a higher frequency and expanded spectrum of mitochondrial DNA deletion mutations in human aging.* [PMID:37132288](https://pubmed.ncbi.nlm.nih.gov/37132288/) [DOI](https://doi.org/10.1111/acel.13842)

**Extracted values:**

- 10.0 mu (low) — keyword extraction (no semantic understanding)
- 10.0 mu (low) — keyword extraction (no semantic understanding)
- 10.0 mu (low) — keyword extraction (no semantic understanding)
- 81.0 years (low) — keyword extraction (no semantic understanding)

## Assessment Categories

### Major Discrepancies (action needed)
- (none)

### Well Supported
- (none)

### Sparse Evidence
- `heteroplasmy_cliff`

### No Data Found
- (none)
