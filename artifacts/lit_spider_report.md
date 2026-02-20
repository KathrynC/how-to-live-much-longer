# Lit Spider Report

**Date:** 2026-02-20 01:37:36 UTC
**Model:** keyword-only
**Parameters searched:** 26
**PubMed queries:** 26
**Abstracts fetched:** 363
**LLM extractions:** 0
**Elapsed:** 44s

## Summary

| Parameter | Current | Priority | Assessment | Discrepancy | Lit Range | Papers |
|-----------|---------|----------|------------|-------------|-----------|--------|
| `senescence_ros_multiplier` | 2.0 multiplier | high | sparse | **MAJOR** | 0.1–0.1 | 1 |
| `base_replication_rate` | 0.1 per year | high | conflicting | minor | 0.1–200 | 7 |
| `ros_damage_coupling` | 0.15 dimensionless coupling strength | high | well-supported | minor | 35–100 | 1 |
| `apoptosis_rate` | 0.02 per year (energy-gated) | high | conflicting | minor | 0.5–95 | 3 |
| `exercise_biogenesis` | 0.03 dimensionless coupling strength | high | conflicting | minor | 1–66 | 6 |
| `nad_quality_control_boost` | 0.03 dimensionless (NAD-dependent mitophagy term) | high | sparse | minor | 2–2 | 1 |
| `atp_relaxation_time` | 1.0 years | high | conflicting | minor | 1–830 | 6 |
| `ros_relaxation_time` | 1.0 years | high | well-supported | minor | 0.24–30 | 2 |
| `nad_relaxation_time` | 0.3 years | high | conflicting | minor | 0.1–331 | 7 |
| `yamanaka_repair_rate` | 0.06 dimensionless (yama * 0.06 * n_d * energy_available) | high | sparse | minor | 7–7 | 1 |
| `senolytic_clearance_rate` | 0.2 per year (dose-dependent) | high | well-supported | minor | 15–70 | 4 |
| `rapamycin_mitophagy_boost` | 0.08 dimensionless (added to baseline mitophagy) | high | no-data | ok |  | 0 |
| `cliff_steepness` | 15.0 dimensionless (sigmoid steepness parameter) | medium | conflicting | **MAJOR** | 0–1.2e+03 | 1 |
| `heteroplasmy_cliff` | 0.7 fraction | medium | well-supported | minor | 5–100 | 3 |
| `damaged_replication_advantage` | 1.05 multiplier (vs healthy mtDNA replication rate) | medium | well-supported | minor | 0.5–97 | 4 |
| `tissue_ros_sensitivity_cardiac` | 1.2 multiplier (vs default tissue) | medium | conflicting | minor | 2–95 | 2 |
| `tissue_biogenesis_brain` | 0.3 multiplier (vs default tissue) | medium | well-supported | minor | 2–67 | 4 |
| `tissue_biogenesis_muscle` | 1.5 multiplier (vs default tissue) | medium | conflicting | minor | 3–90 | 4 |
| `cd38_base_survival` | 0.4 fraction surviving CD38 degradation | medium | well-supported | minor | 35–87 | 2 |
| `tissue_ros_sensitivity_brain` | 1.5 multiplier (vs default tissue) | medium | no-data | ok |  | 0 |
| `doubling_time_young` | 11.8 years | low | conflicting | minor | 0.02–93 | 7 |
| `doubling_time_old` | 3.06 years | low | conflicting | minor | 0.00015–100 | 9 |
| `nad_decline_rate` | 0.01 per year | low | conflicting | minor | 3–700 | 5 |
| `baseline_ros` | 0.1 normalized | low | conflicting | minor | 0.15–750 | 7 |
| `ros_per_damaged` | 0.3 dimensionless coupling strength | low | sparse | minor | 25–160 | 2 |
| `senescence_rate` | 0.005 per year | low | no-data | ok |  | 0 |

## High Priority (no citation)

### `senescence_ros_multiplier`

- **Current value:** 2.0 multiplier
- **Location:** `simulator.py:454 (1 + 2*ros term in senescence)`
- **Citation:** None
- **Assessment:** sparse
- **Discrepancy:** major
- **Literature range:** 0.1 – 0.1 (median 0.1, n=1)

**Key papers:**

- Tang Q, Tang K, Markby GR (2025). *Autophagy regulates cellular senescence by mediating the degradation of CDKN1A/p21 and CDKN2A/p16 through SQSTM1/p62-mediated selective autophagy in myxomatous mitral valve degeneration.* [PMID:39988732](https://pubmed.ncbi.nlm.nih.gov/39988732/) [DOI](https://doi.org/10.1080/15548627.2025.2469315)

**Extracted values:**

- 0.1 % (low) — keyword extraction (no semantic understanding)

### `base_replication_rate`

- **Current value:** 0.1 per year
- **Location:** `simulator.py:297 (embedded in derivatives())`
- **Citation:** None
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 0.1 – 200 (median 30, n=17)

**Key papers:**

- Lin C, Luo L, Xun Z (2024). *Novel function of MOTS-c in mitochondrial remodelling contributes to its antiviral role during HBV infection.* [PMID:37788894](https://pubmed.ncbi.nlm.nih.gov/37788894/) [DOI](https://doi.org/10.1136/gutjnl-2023-330389)
- An J, Nam CH, Kim R (2024). *Mitochondrial DNA mosaicism in normal human somatic cells.* [PMID:39039280](https://pubmed.ncbi.nlm.nih.gov/39039280/) [DOI](https://doi.org/10.1038/s41588-024-01838-z)
- Berardo A, Domínguez-González C, Engelstad K (2022). *Advances in Thymidine Kinase 2 Deficiency: Clinical Aspects, Translational Progress, and Emerging Therapies.* [PMID:35094997](https://pubmed.ncbi.nlm.nih.gov/35094997/) [DOI](https://doi.org/10.3233/JND-210786)
- Jiang Z-z, Chu M, Yan L-n (2024). *SFTSV nucleoprotein mediates DNA sensor cGAS degradation to suppress cGAS-dependent antiviral responses.* [PMID:38712963](https://pubmed.ncbi.nlm.nih.gov/38712963/) [DOI](https://doi.org/10.1128/spectrum.03796-23)
- Giannoulis SV, Chenoweth MJ, Saquilayan P (2022). *Examining the role of mitochondrial genetic variation in nicotine dependence.* [PMID:35227992](https://pubmed.ncbi.nlm.nih.gov/35227992/) [DOI](https://doi.org/10.1016/j.psychres.2022.114452)

**Extracted values:**

- 70.0 % (low) — keyword extraction (no semantic understanding)
- 0.3 % (low) — keyword extraction (no semantic understanding)
- 6.0 % (low) — keyword extraction (no semantic understanding)
- 5.0 × (low) — keyword extraction (no semantic understanding)
- 20.0 per year (low) — keyword extraction (no semantic understanding)
- 2.0 years (low) — keyword extraction (no semantic understanding)
- 30.0 % (low) — keyword extraction (no semantic understanding)
- 30.0 % (low) — keyword extraction (no semantic understanding)

### `ros_damage_coupling`

- **Current value:** 0.15 dimensionless coupling strength
- **Location:** `simulator.py:310 (ROS→damage conversion)`
- **Citation:** None
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 35 – 100 (median 38, n=3)

**Key papers:**

- Ramírez-Martín N, Buigues A, Rodríguez-Varela C (2025). *Nicotinamide mononucleotide supplementation improves oocyte developmental competence in different ovarian damage conditions.* [PMID:39923879](https://pubmed.ncbi.nlm.nih.gov/39923879/) [DOI](https://doi.org/10.1016/j.ajog.2025.02.006)

**Extracted values:**

- 35.0 years (low) — keyword extraction (no semantic understanding)
- 100.0 µM (low) — keyword extraction (no semantic understanding)
- 38.0 years (low) — keyword extraction (no semantic understanding)

### `apoptosis_rate`

- **Current value:** 0.02 per year (energy-gated)
- **Location:** `simulator.py:336 (embedded in derivatives())`
- **Citation:** None
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 0.5 – 95 (median 15.5, n=4)

**Key papers:**

- Jiang XL, Tai H, Xiao XS (2022). *Cangfudaotan decoction inhibits mitochondria-dependent apoptosis of granulosa cells in rats with polycystic ovarian syndrome.* [PMID:36465612](https://pubmed.ncbi.nlm.nih.gov/36465612/) [DOI](https://doi.org/10.3389/fendo.2022.962154)
- Lee TL, Shen WC, Chen YC (2025). *None* [PMID:39245438](https://pubmed.ncbi.nlm.nih.gov/39245438/) [DOI](https://doi.org/10.1080/15548627.2024.2395799)
- Del Nagro C, Xiao Y, Rangell L (2014). *Depletion of the central metabolite NAD leads to oncosis-mediated cell death.* [PMID:25355314](https://pubmed.ncbi.nlm.nih.gov/25355314/) [DOI](https://doi.org/10.1074/jbc.M114.580159)

**Extracted values:**

- 0.5 % (low) — keyword extraction (no semantic understanding)
- 30.0 min (low) — keyword extraction (no semantic understanding)
- 1.0 % (low) — keyword extraction (no semantic understanding)
- 95.0 % (low) — keyword extraction (no semantic understanding)

### `exercise_biogenesis`

- **Current value:** 0.03 dimensionless coupling strength
- **Location:** `simulator.py:333 (exercise → biogenesis term)`
- **Citation:** None
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 1 – 66 (median 30.7, n=23)

**Key papers:**

- Mengozzi A, Armenia S, De Biase N (2025). *Circulating mitochondrial DNA signature in cardiometabolic patients.* [PMID:40045401](https://pubmed.ncbi.nlm.nih.gov/40045401/) [DOI](https://doi.org/10.1186/s12933-025-02656-1)
- Hinkle JS, Rivera CN, Vaughan RA (2022). *AICAR stimulates mitochondrial biogenesis and BCAA catabolic enzyme expression in C2C12 myotubes.* [PMID:34798200](https://pubmed.ncbi.nlm.nih.gov/34798200/) [DOI](https://doi.org/10.1016/j.biochi.2021.11.004)
- Benite-Ribeiro SA, Lucas-Lima KL, Jones JN (2020). *Transcription of mtDNA and dyslipidemia are ameliorated by aerobic exercise in type 2 diabetes.* [PMID:32804305](https://pubmed.ncbi.nlm.nih.gov/32804305/) [DOI](https://doi.org/10.1007/s11033-020-05725-8)
- Latimer LE, Constantin-Teodosiu D, Popat B (2022). *Whole-body and muscle responses to aerobic exercise training and withdrawal in ageing and COPD.* [PMID:34588196](https://pubmed.ncbi.nlm.nih.gov/34588196/) [DOI](https://doi.org/10.1183/13993003.01507-2021)
- Rivera ME, Lyon ES, Vaughan RA (2020). *Effect of metformin on myotube BCAA catabolism.* [PMID:31385363](https://pubmed.ncbi.nlm.nih.gov/31385363/) [DOI](https://doi.org/10.1002/jcb.29327)

**Extracted values:**

- 21.0 mm (low) — keyword extraction (no semantic understanding)
- 42.0 % (low) — keyword extraction (no semantic understanding)
- 33.0 % (low) — keyword extraction (no semantic understanding)
- 25.0 % (low) — keyword extraction (no semantic understanding)
- 35.0 % (low) — keyword extraction (no semantic understanding)
- 1.0 mM (low) — keyword extraction (no semantic understanding)
- 60.0 % (low) — keyword extraction (no semantic understanding)
- 65.0 % (low) — keyword extraction (no semantic understanding)

### `nad_quality_control_boost`

- **Current value:** 0.03 dimensionless (NAD-dependent mitophagy term)
- **Location:** `simulator.py:366 (nad_supp * 0.03 term)`
- **Citation:** None
- **Assessment:** sparse
- **Discrepancy:** minor
- **Literature range:** 2 – 2 (median 2, n=1)

**Key papers:**

- Yu H, Gan D, Luo Z (2024). *α-Ketoglutarate improves cardiac insufficiency through NAD* [PMID:38254035](https://pubmed.ncbi.nlm.nih.gov/38254035/) [DOI](https://doi.org/10.1186/s10020-024-00783-1)

**Extracted values:**

- 2.0 % (low) — keyword extraction (no semantic understanding)

### `atp_relaxation_time`

- **Current value:** 1.0 years
- **Location:** `simulator.py:396 (1.0 * (atp_target - atp) term)`
- **Citation:** None
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 1 – 830 (median 40, n=15)

**Key papers:**

- Weyand CM, Goronzy JJ (2025). *Metabolic checkpoints in rheumatoid arthritis.* [PMID:39550308](https://pubmed.ncbi.nlm.nih.gov/39550308/) [DOI](https://doi.org/10.1016/j.semarthrit.2024.152586)
- Consolini AE, Ragone MI, Bonazzola P (2017). *Mitochondrial Bioenergetics During Ischemia and Reperfusion.* [PMID:28551786](https://pubmed.ncbi.nlm.nih.gov/28551786/) [DOI](https://doi.org/10.1007/978-3-319-55330-6_8)
- Okatan EN, Olgar Y, Tuncay E (2019). *Azoramide improves mitochondrial dysfunction in palmitate-induced insulin resistant H9c2 cells.* [PMID:31327095](https://pubmed.ncbi.nlm.nih.gov/31327095/) [DOI](https://doi.org/10.1007/s11010-019-03590-z)
- Bombicino SS, Iglesias DE, Rukavina-Mikusic IA (2017). *Hydrogen peroxide, nitric oxide and ATP are molecules involved in cardiac mitochondrial biogenesis in Diabetes.* [PMID:28756312](https://pubmed.ncbi.nlm.nih.gov/28756312/) [DOI](https://doi.org/10.1016/j.freeradbiomed.2017.07.027)
- Layec G, Bringard A, Le Fur Y (2016). *Mitochondrial Coupling and Contractile Efficiency in Humans with High and Low V˙O2peaks.* [PMID:26694849](https://pubmed.ncbi.nlm.nih.gov/26694849/) [DOI](https://doi.org/10.1249/MSS.0000000000000858)

**Extracted values:**

- 1.0 % (low) — keyword extraction (no semantic understanding)
- 10.0 mM (low) — keyword extraction (no semantic understanding)
- 36.0 mM (low) — keyword extraction (no semantic understanding)
- 50.0 % (low) — keyword extraction (no semantic understanding)
- 47.0 % (low) — keyword extraction (no semantic understanding)
- 30.0 % (low) — keyword extraction (no semantic understanding)
- 52.0 % (low) — keyword extraction (no semantic understanding)
- 23.0 % (low) — keyword extraction (no semantic understanding)

### `ros_relaxation_time`

- **Current value:** 1.0 years
- **Location:** `simulator.py:417 (1.0 * (ros_eq - ros) term)`
- **Citation:** None
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 0.24 – 30 (median 20, n=3)

**Key papers:**

- Carlström M, Rannier Ribeiro Antonino Carvalho L, Guimaraes D (2024). *Dimethyl malonate preserves renal and mitochondrial functions following ischemia-reperfusion via inhibition of succinate dehydrogenase.* [PMID:38061207](https://pubmed.ncbi.nlm.nih.gov/38061207/) [DOI](https://doi.org/10.1016/j.redox.2023.102984)
- Rani S, Sahoo RK, Kumar V (2023). *None* [PMID:36306447](https://pubmed.ncbi.nlm.nih.gov/36306447/) [DOI](https://doi.org/10.1021/acs.molpharmaceut.2c00752)

**Extracted values:**

- 30.0 min (low) — keyword extraction (no semantic understanding)
- 20.0 min (low) — keyword extraction (no semantic understanding)
- 0.24 nm (low) — keyword extraction (no semantic understanding)

### `nad_relaxation_time`

- **Current value:** 0.3 years
- **Location:** `simulator.py:440 (0.3 * (nad_target - nad) term)`
- **Citation:** None
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 0.1 – 331 (median 28, n=18)

**Key papers:**

- Poljsak B (2018). *NAMPT-Mediated NAD Biosynthesis as the Internal Timing Mechanism: In NAD+ World, Time Is Running in Its Own Way.* [PMID:28756747](https://pubmed.ncbi.nlm.nih.gov/28756747/) [DOI](https://doi.org/10.1089/rej.2017.1975)
- Lambeth JD, McCaslin DR, Kamin H (1976). *Adrenodoxin reductase-adrenodexin complex.* [PMID:12171](https://pubmed.ncbi.nlm.nih.gov/12171/)
- Rechsteiner M, Hillyard D, Olivera BM (1976). *Turnover at nicotinamide adenine dinucleotide in cultures of human cells.* [PMID:178671](https://pubmed.ncbi.nlm.nih.gov/178671/) [DOI](https://doi.org/10.1002/jcp.1040880210)
- Jewett SL, Rocklin AM (1993). *Variation of one unit of activity with oxidation rate of organic substrate in indirect superoxide dismutase assays.* [PMID:8214600](https://pubmed.ncbi.nlm.nih.gov/8214600/) [DOI](https://doi.org/10.1006/abio.1993.1368)
- Grivennikova VG, Vinogradov AD (2006). *Generation of superoxide by the mitochondrial Complex I.* [PMID:16678117](https://pubmed.ncbi.nlm.nih.gov/16678117/) [DOI](https://doi.org/10.1016/j.bbabio.2006.03.013)

**Extracted values:**

- 24.0 hours (low) — keyword extraction (no semantic understanding)
- 331.0 mV (low) — keyword extraction (no semantic understanding)
- 40.0 mV (low) — keyword extraction (no semantic understanding)
- 18.0 minutes (low) — keyword extraction (no semantic understanding)
- 24.0 hours (low) — keyword extraction (no semantic understanding)
- 50.0 % (low) — keyword extraction (no semantic understanding)
- 1.0 mM (low) — keyword extraction (no semantic understanding)
- 1.0 mM (low) — keyword extraction (no semantic understanding)

### `yamanaka_repair_rate`

- **Current value:** 0.06 dimensionless (yama * 0.06 * n_d * energy_available)
- **Location:** `simulator.py:329 (Yamanaka repair term)`
- **Citation:** None
- **Assessment:** sparse
- **Discrepancy:** minor
- **Literature range:** 7 – 7 (median 7, n=1)

**Key papers:**

- Goya RG, Lehmann M, Chiavellini P (2018). *Rejuvenation by cell reprogramming: a new horizon in gerontology.* [PMID:30558644](https://pubmed.ncbi.nlm.nih.gov/30558644/) [DOI](https://doi.org/10.1186/s13287-018-1075-y)

**Extracted values:**

- 7.0 years (low) — keyword extraction (no semantic understanding)

### `senolytic_clearance_rate`

- **Current value:** 0.2 per year (dose-dependent)
- **Location:** `simulator.py:457 (seno * 0.2 * sen term)`
- **Citation:** None
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 15 – 70 (median 55, n=14)

**Key papers:**

- Yan H, Miranda EAD, Jin S (2024). *Primary oocytes with cellular senescence features are involved in ovarian aging in mice.* [PMID:38871781](https://pubmed.ncbi.nlm.nih.gov/38871781/) [DOI](https://doi.org/10.1038/s41598-024-64441-6)
- Yan H, Miranda EAD, Jin S (2024). *Primary oocytes with cellular senescence features are involved in ovarian aging in mice.* [PMID:38260383](https://pubmed.ncbi.nlm.nih.gov/38260383/) [DOI](https://doi.org/10.1101/2024.01.08.574768)
- Ghamar Talepoor A, Khosropanah S, Doroudchi M (2021). *Partial recovery of senescence in circulating follicular helper T cells after Dasatinib treatment.* [PMID:33631598](https://pubmed.ncbi.nlm.nih.gov/33631598/) [DOI](https://doi.org/10.1016/j.intimp.2021.107465)
- Alsuraih M, O'Hara SP, Woodrum JE (2021). *Genetic or pharmacological reduction of cholangiocyte senescence improves inflammation and fibrosis in the * [PMID:33870156](https://pubmed.ncbi.nlm.nih.gov/33870156/) [DOI](https://doi.org/10.1016/j.jhepr.2021.100250)

**Extracted values:**

- 15.0 % (low) — keyword extraction (no semantic understanding)
- 15.0 % (low) — keyword extraction (no semantic understanding)
- 50.0 % (low) — keyword extraction (no semantic understanding)
- 50.0 % (low) — keyword extraction (no semantic understanding)
- 35.0 % (low) — keyword extraction (no semantic understanding)
- 70.0 % (low) — keyword extraction (no semantic understanding)
- 60.0 % (low) — keyword extraction (no semantic understanding)
- 40.0 % (low) — keyword extraction (no semantic understanding)

### `rapamycin_mitophagy_boost`

- **Current value:** 0.08 dimensionless (added to baseline mitophagy)
- **Location:** `simulator.py:366 (rapa * 0.08 term)`
- **Citation:** None
- **Assessment:** no-data
- **Discrepancy:** none
- **Literature range:** No numerical values found

## Medium Priority (refinable)

### `cliff_steepness`

- **Current value:** 15.0 dimensionless (sigmoid steepness parameter)
- **Location:** `constants.py:CLIFF_STEEPNESS`
- **Citation:** Simulation calibration
- **Assessment:** conflicting
- **Discrepancy:** major
- **Literature range:** 0 – 1200 (median 2.054, n=9)

**Key papers:**

- Niu Z, Ye ZW, Huang Q (2025). *Accuracy of photorespiration and mitochondrial respiration in the light fitted by CO* [PMID:40933716](https://pubmed.ncbi.nlm.nih.gov/40933716/) [DOI](https://doi.org/10.3389/fpls.2025.1455533)

**Extracted values:**

- 1200.0 μmol (low) — keyword extraction (no semantic understanding)
- 1.033 μmol (low) — keyword extraction (no semantic understanding)
- 2.054 μmol (low) — keyword extraction (no semantic understanding)
- 0.063 μmol (low) — keyword extraction (no semantic understanding)
- 0.312 μmol (low) — keyword extraction (no semantic understanding)
- 600.0 μmol (low) — keyword extraction (no semantic understanding)
- 0.0 μmol (low) — keyword extraction (no semantic understanding)
- 400.0 μmol (low) — keyword extraction (no semantic understanding)

### `heteroplasmy_cliff`

- **Current value:** 0.7 fraction
- **Location:** `constants.py:HETEROPLASMY_CLIFF`
- **Citation:** Rossignol et al. 2003
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 5 – 100 (median 82.5, n=4)

**Key papers:**

- Fu Y, Land M, Kavlashvili T (2025). *Engineering mtDNA deletions by reconstituting end joining in human mitochondria.* [PMID:40068680](https://pubmed.ncbi.nlm.nih.gov/40068680/) [DOI](https://doi.org/10.1016/j.cell.2025.02.009)
- DiMauro S, Moraes CT (1993). *Mitochondrial encephalomyopathies.* [PMID:8215979](https://pubmed.ncbi.nlm.nih.gov/8215979/) [DOI](https://doi.org/10.1001/archneur.1993.00540110075008)
- Yokota M, Hatakeyama H, Okabe S (2015). *Mitochondrial respiratory dysfunction caused by a heteroplasmic mitochondrial DNA mutation blocks cellular reprogramming.* [PMID:26025377](https://pubmed.ncbi.nlm.nih.gov/26025377/) [DOI](https://doi.org/10.1093/hmg/ddv201)

**Extracted values:**

- 75.0 % (low) — keyword extraction (no semantic understanding)
- 5.0 years (low) — keyword extraction (no semantic understanding)
- 90.0 % (low) — keyword extraction (no semantic understanding)
- 100.0 % (low) — keyword extraction (no semantic understanding)

### `damaged_replication_advantage`

- **Current value:** 1.05 multiplier (vs healthy mtDNA replication rate)
- **Location:** `constants.py:DAMAGED_REPLICATION_ADVANTAGE`
- **Citation:** Vandiver et al. 2023 (Cramer Appendix 2 pp.154-155)
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 0.5 – 97 (median 70.5, n=8)

**Key papers:**

- Bradshaw E, Yoshida M, Ling F (2017). *Regulation of Small Mitochondrial DNA Replicative Advantage by Ribonucleotide Reductase in * [PMID:28717049](https://pubmed.ncbi.nlm.nih.gov/28717049/) [DOI](https://doi.org/10.1534/g3.117.043851)
- Dubie JJ, Caraway AR, Stout MM (2020). *The conflict within: origin, proliferation and persistence of a spontaneously arising selfish mitochondrial genome.* [PMID:31787044](https://pubmed.ncbi.nlm.nih.gov/31787044/) [DOI](https://doi.org/10.1098/rstb.2019.0174)
- Zhu W, Yoshida M, Ling F (2025). *Caloric restriction enhances inheritance of wild-type mitochondrial DNA in Saccharomyces cerevisiae.* [PMID:41249289](https://pubmed.ncbi.nlm.nih.gov/41249289/) [DOI](https://doi.org/10.1038/s41598-025-23888-x)
- Chabi B, Mousson de Camaret B, Duborjal H (2003). *Quantification of mitochondrial DNA deletion, depletion, and overreplication: application to diagnosis.* [PMID:12881447](https://pubmed.ncbi.nlm.nih.gov/12881447/) [DOI](https://doi.org/10.1373/49.8.1309)

**Extracted values:**

- 95.0 % (low) — keyword extraction (no semantic understanding)
- 96.0 % (low) — keyword extraction (no semantic understanding)
- 1.0 mu (low) — keyword extraction (no semantic understanding)
- 65.0 % (low) — keyword extraction (no semantic understanding)
- 52.0 % (low) — keyword extraction (no semantic understanding)
- 0.5 % (low) — keyword extraction (no semantic understanding)
- 76.0 % (low) — keyword extraction (no semantic understanding)
- 97.0 % (low) — keyword extraction (no semantic understanding)

### `tissue_ros_sensitivity_cardiac`

- **Current value:** 1.2 multiplier (vs default tissue)
- **Location:** `constants.py:TISSUE_PROFILES['cardiac']['ros_sensitivity']`
- **Citation:** Cramer Ch. VII (qualitative)
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 2 – 95 (median 19.8, n=8)

**Key papers:**

- Rudokas MW, McKay M, Toksoy Z (2024). *Mitochondrial network remodeling of the diabetic heart: implications to ischemia related cardiac dysfunction.* [PMID:39026280](https://pubmed.ncbi.nlm.nih.gov/39026280/) [DOI](https://doi.org/10.1186/s12933-024-02357-1)
-  (2021). *Effects of pre-operative isolation on postoperative pulmonary complications after elective surgery: an international prospective cohort study.* [PMID:34371522](https://pubmed.ncbi.nlm.nih.gov/34371522/) [DOI](https://doi.org/10.1111/anae.15560)

**Extracted values:**

- 40.0 % (low) — keyword extraction (no semantic understanding)
- 27.9 % (low) — keyword extraction (no semantic understanding)
- 2.0 % (low) — keyword extraction (no semantic understanding)
- 11.7 % (low) — keyword extraction (no semantic understanding)
- 2.1 % (low) — keyword extraction (no semantic understanding)
- 2.0 % (low) — keyword extraction (no semantic understanding)
- 95.0 % (low) — keyword extraction (no semantic understanding)
- 95.0 % (low) — keyword extraction (no semantic understanding)

### `tissue_biogenesis_brain`

- **Current value:** 0.3 multiplier (vs default tissue)
- **Location:** `constants.py:TISSUE_PROFILES['brain']['biogenesis_rate']`
- **Citation:** Post-mitotic neuron biology (qualitative)
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 2 – 67 (median 67, n=7)

**Key papers:**

- Mendelev N, Mehta SL, Idris H (2012). *Selenite stimulates mitochondrial biogenesis signaling and enhances mitochondrial functional performance in murine hippocampal neuronal cells.* [PMID:23110128](https://pubmed.ncbi.nlm.nih.gov/23110128/) [DOI](https://doi.org/10.1371/journal.pone.0047910)
- Qin J, Li Y, Wang K (2019). *Propofol induces impairment of mitochondrial biogenesis through inhibiting the expression of peroxisome proliferator-activated receptor-γ coactivator-1α.* [PMID:31190345](https://pubmed.ncbi.nlm.nih.gov/31190345/) [DOI](https://doi.org/10.1002/jcb.29138)
- Cui Y, Meng S, Zhang N (2024). *High-concentration hydrogen inhalation mitigates sepsis-associated encephalopathy in mice by improving mitochondrial dynamics.* [PMID:39258790](https://pubmed.ncbi.nlm.nih.gov/39258790/) [DOI](https://doi.org/10.1111/cns.70021)
- Fields JA, Serger E, Campos S (2016). *HIV alters neuronal mitochondrial fission/fusion in the brain during HIV-associated neurocognitive disorders.* [PMID:26611103](https://pubmed.ncbi.nlm.nih.gov/26611103/) [DOI](https://doi.org/10.1016/j.nbd.2015.11.015)

**Extracted values:**

- 24.0 hours (low) — keyword extraction (no semantic understanding)
- 2.0 % (low) — keyword extraction (no semantic understanding)
- 67.0 % (low) — keyword extraction (no semantic understanding)
- 67.0 % (low) — keyword extraction (no semantic understanding)
- 67.0 % (low) — keyword extraction (no semantic understanding)
- 67.0 % (low) — keyword extraction (no semantic understanding)
- 50.0 % (low) — keyword extraction (no semantic understanding)

### `tissue_biogenesis_muscle`

- **Current value:** 1.5 multiplier (vs default tissue)
- **Location:** `constants.py:TISSUE_PROFILES['muscle']['biogenesis_rate']`
- **Citation:** PGC-1alpha biology (qualitative)
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 3 – 90 (median 5, n=14)

**Key papers:**

- Mølmen KS, Almquist NW, Skattebo Ø (2025). *Effects of Exercise Training on Mitochondrial and Capillary Growth in Human Skeletal Muscle: A Systematic Review and Meta-Regression.* [PMID:39390310](https://pubmed.ncbi.nlm.nih.gov/39390310/) [DOI](https://doi.org/10.1007/s40279-024-02120-2)
- Reisman EG, Caruana NJ, Bishop DJ (2024). *Exercise training and changes in skeletal muscle mitochondrial proteins: from blots to "omics".* [PMID:39288086](https://pubmed.ncbi.nlm.nih.gov/39288086/) [DOI](https://doi.org/10.1080/10409238.2024.2383408)
- Safdar A, Little JP, Stokl AJ (2011). *Exercise increases mitochondrial PGC-1alpha content and promotes nuclear-mitochondrial cross-talk to coordinate mitochondrial biogenesis.* [PMID:21245132](https://pubmed.ncbi.nlm.nih.gov/21245132/) [DOI](https://doi.org/10.1074/jbc.M110.211466)
- Kang C, Chung E, Diffee G (2013). *Exercise training attenuates aging-associated mitochondrial dysfunction in rat skeletal muscle: role of PGC-1α.* [PMID:23994518](https://pubmed.ncbi.nlm.nih.gov/23994518/) [DOI](https://doi.org/10.1016/j.exger.2013.08.004)

**Extracted values:**

- 5.0 % (low) — keyword extraction (no semantic understanding)
- 5.0 % (low) — keyword extraction (no semantic understanding)
- 7.0 % (low) — keyword extraction (no semantic understanding)
- 3.0 % (low) — keyword extraction (no semantic understanding)
- 4.0 % (low) — keyword extraction (no semantic understanding)
- 11.0 % (low) — keyword extraction (no semantic understanding)
- 3.0 % (low) — keyword extraction (no semantic understanding)
- 4.0 % (low) — keyword extraction (no semantic understanding)

### `cd38_base_survival`

- **Current value:** 0.4 fraction surviving CD38 degradation
- **Location:** `constants.py:CD38_BASE_SURVIVAL`
- **Citation:** Cramer Ch. VI.A.3 p.73 (qualitative)
- **Assessment:** well-supported
- **Discrepancy:** minor
- **Literature range:** 35 – 87 (median 87, n=3)

**Key papers:**

- Sun C, Liu X, Wang B (2019). *Endocytosis-mediated mitochondrial transplantation: Transferring normal human astrocytic mitochondria into glioma cells rescues aerobic respiration and enhances radiosensitivity.* [PMID:31281500](https://pubmed.ncbi.nlm.nih.gov/31281500/) [DOI](https://doi.org/10.7150/thno.33100)
- Hu Y, Wang H, Wang Q (2014). *Overexpression of CD38 decreases cellular NAD levels and alters the expression of proteins involved in energy metabolism and antioxidant defense.* [PMID:24295520](https://pubmed.ncbi.nlm.nih.gov/24295520/) [DOI](https://doi.org/10.1021/pr4010597)

**Extracted values:**

- 87.0 x (low) — keyword extraction (no semantic understanding)
- 87.0 x (low) — keyword extraction (no semantic understanding)
- 35.0 % (low) — keyword extraction (no semantic understanding)

### `tissue_ros_sensitivity_brain`

- **Current value:** 1.5 multiplier (vs default tissue)
- **Location:** `constants.py:TISSUE_PROFILES['brain']['ros_sensitivity']`
- **Citation:** Cramer Ch. V.J p.65 (qualitative)
- **Assessment:** no-data
- **Discrepancy:** none
- **Literature range:** No numerical values found

## Low Priority (verify)

### `doubling_time_young`

- **Current value:** 11.8 years
- **Location:** `constants.py:DOUBLING_TIME_YOUNG`
- **Citation:** Vandiver et al. 2023 (Cramer Appendix 2 p.155)
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 0.02 – 93 (median 17, n=23)

**Key papers:**

- Shammas MK, Nie Y, Gilsrud A (2023). *CHCHD10 mutations induce tissue-specific mitochondrial DNA deletions with a distinct signature.* [PMID:37815936](https://pubmed.ncbi.nlm.nih.gov/37815936/) [DOI](https://doi.org/10.1093/hmg/ddad161)
- Vandiver AR, Hoang AN, Herbst A (2023). *Nanopore sequencing identifies a higher frequency and expanded spectrum of mitochondrial DNA deletion mutations in human aging.* [PMID:37132288](https://pubmed.ncbi.nlm.nih.gov/37132288/) [DOI](https://doi.org/10.1111/acel.13842)
- Hayakawa M, Torii K, Sugiyama S (1991). *Age-associated accumulation of 8-hydroxydeoxyguanosine in mitochondrial DNA of human diaphragm.* [PMID:1898383](https://pubmed.ncbi.nlm.nih.gov/1898383/) [DOI](https://doi.org/10.1016/0006-291x(91)91921-x)
- Hayakawa M, Hattori K, Sugiyama S (1992). *Age-associated oxygen damage and mutations in mitochondrial DNA in human hearts.* [PMID:1472070](https://pubmed.ncbi.nlm.nih.gov/1472070/) [DOI](https://doi.org/10.1016/0006-291x(92)92300-m)
- Wei YH, Lee CF, Lee HC (2001). *Increases of mitochondrial mass and mitochondrial genome in association with enhanced oxidative stress in human cells harboring 4,977 BP-deleted mitochondrial DNA.* [PMID:11795533](https://pubmed.ncbi.nlm.nih.gov/11795533/) [DOI](https://doi.org/10.1111/j.1749-6632.2001.tb05640.x)

**Extracted values:**

- 10.0 mu (low) — keyword extraction (no semantic understanding)
- 10.0 mu (low) — keyword extraction (no semantic understanding)
- 10.0 mu (low) — keyword extraction (no semantic understanding)
- 81.0 years (low) — keyword extraction (no semantic understanding)
- 0.02 % (low) — keyword extraction (no semantic understanding)
- 0.25 % (low) — keyword extraction (no semantic understanding)
- 10.0 years (low) — keyword extraction (no semantic understanding)
- 0.51 % (low) — keyword extraction (no semantic understanding)

### `doubling_time_old`

- **Current value:** 3.06 years
- **Location:** `constants.py:DOUBLING_TIME_OLD`
- **Citation:** Vandiver et al. 2023 (Cramer Appendix 2 p.155)
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 0.00015 – 100 (median 20, n=39)

**Key papers:**

- Ramírez-Martín N, Buigues A, Rodríguez-Varela C (2025). *Nicotinamide mononucleotide supplementation improves oocyte developmental competence in different ovarian damage conditions.* [PMID:39923879](https://pubmed.ncbi.nlm.nih.gov/39923879/) [DOI](https://doi.org/10.1016/j.ajog.2025.02.006)
- Shadyab AH, LaCroix AZ (2015). *Genetic factors associated with longevity: a review of recent findings.* [PMID:25446805](https://pubmed.ncbi.nlm.nih.gov/25446805/) [DOI](https://doi.org/10.1016/j.arr.2014.10.005)
- Kimoloi S, Sen A, Guenther S (2022). *Combined fibre atrophy and decreased muscle regeneration capacity driven by mitochondrial DNA alterations underlie the development of sarcopenia.* [PMID:35765148](https://pubmed.ncbi.nlm.nih.gov/35765148/) [DOI](https://doi.org/10.1002/jcsm.13026)
- Lee Y, Lee SM, Choi J (2021). *Mitochondrial DNA Haplogroup Related to the Prevalence of * [PMID:34572132](https://pubmed.ncbi.nlm.nih.gov/34572132/) [DOI](https://doi.org/10.3390/cells10092482)
- Yu-Wai-Man P, Lai-Cheong J, Borthwick GM (2010). *Somatic mitochondrial DNA deletions accumulate to high levels in aging human extraocular muscles.* [PMID:20164450](https://pubmed.ncbi.nlm.nih.gov/20164450/) [DOI](https://doi.org/10.1167/iovs.09-4660)

**Extracted values:**

- 35.0 years (low) — keyword extraction (no semantic understanding)
- 100.0 µM (low) — keyword extraction (no semantic understanding)
- 38.0 years (low) — keyword extraction (no semantic understanding)
- 20.0 % (low) — keyword extraction (no semantic understanding)
- 35.0 % (low) — keyword extraction (no semantic understanding)
- 0.07673 % (low) — keyword extraction (no semantic understanding)
- 0.00015 % (low) — keyword extraction (no semantic understanding)
- 3.03 % (low) — keyword extraction (no semantic understanding)

### `nad_decline_rate`

- **Current value:** 0.01 per year
- **Location:** `constants.py:NAD_DECLINE_RATE`
- **Citation:** Camacho-Pereira et al. 2016 (Cramer Ch. VI.A.3)
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 3 – 700 (median 34, n=27)

**Key papers:**

- Li T, Wang Y, Yu Y (2024). *The NAD* [PMID:39460833](https://pubmed.ncbi.nlm.nih.gov/39460833/) [DOI](https://doi.org/10.1007/s10815-024-03263-x)
- de Guia RM, Agerholm M, Nielsen TS (2019). *Aerobic and resistance exercise training reverses age-dependent decline in NAD* [PMID:31207144](https://pubmed.ncbi.nlm.nih.gov/31207144/) [DOI](https://doi.org/10.14814/phy2.14139)
- Ketron GL, Grun F, Grill JD (2025). *Pharmacokinetic and pharmacodynamic assessment of oral nicotinamide in the NEAT clinical trial for early Alzheimer's disease.* [PMID:40069789](https://pubmed.ncbi.nlm.nih.gov/40069789/) [DOI](https://doi.org/10.1186/s13195-025-01693-y)
- Bai X, Wang P (2022). *Relationship between sperm NAD + concentration and reproductive aging in normozoospermia men:A Cohort study.* [PMID:36182928](https://pubmed.ncbi.nlm.nih.gov/36182928/) [DOI](https://doi.org/10.1186/s12894-022-01107-3)
- Hara N, Osago H, Hiyoshi M (2019). *Quantitative analysis of the effects of nicotinamide phosphoribosyltransferase induction on the rates of NAD+ synthesis and breakdown in mammalian cells using stable isotope-labeling combined with mass spectrometry.* [PMID:30875389](https://pubmed.ncbi.nlm.nih.gov/30875389/) [DOI](https://doi.org/10.1371/journal.pone.0214000)

**Extracted values:**

- 200.0 μM (low) — keyword extraction (no semantic understanding)
- 35.0 years (low) — keyword extraction (no semantic understanding)
- 55.0 years (low) — keyword extraction (no semantic understanding)
- 12.0 % (low) — keyword extraction (no semantic understanding)
- 28.0 % (low) — keyword extraction (no semantic understanding)
- 25.0 % (low) — keyword extraction (no semantic understanding)
- 30.0 % (low) — keyword extraction (no semantic understanding)
- 52.0 μM (low) — keyword extraction (no semantic understanding)

### `baseline_ros`

- **Current value:** 0.1 normalized
- **Location:** `constants.py:BASELINE_ROS`
- **Citation:** Cramer Ch. IV.B p.53, Ch. II.H p.14
- **Assessment:** conflicting
- **Discrepancy:** minor
- **Literature range:** 0.15 – 750 (median 10, n=17)

**Key papers:**

- Cadenas S (2018). *Mitochondrial uncoupling, ROS generation and cardioprotection.* [PMID:29859845](https://pubmed.ncbi.nlm.nih.gov/29859845/) [DOI](https://doi.org/10.1016/j.bbabio.2018.05.019)
- Martins WK, Santos NF, Rocha CS (2019). *Parallel damage in mitochondria and lysosomes is an efficient way to photoinduce cell death.* [PMID:30176156](https://pubmed.ncbi.nlm.nih.gov/30176156/) [DOI](https://doi.org/10.1080/15548627.2018.1515609)
- St-Pierre J, Buckingham JA, Roebuck SJ (2002). *Topology of superoxide production from different sites in the mitochondrial electron transport chain.* [PMID:12237311](https://pubmed.ncbi.nlm.nih.gov/12237311/) [DOI](https://doi.org/10.1074/jbc.M207217200)
- Grivennikova VG, Kareyeva AV, Vinogradov AD (2018). *Oxygen-dependence of mitochondrial ROS production as detected by Amplex Red assay.* [PMID:29702406](https://pubmed.ncbi.nlm.nih.gov/29702406/) [DOI](https://doi.org/10.1016/j.redox.2018.04.014)
- Fang J, Wong HS, Brand MD (2020). *Production of superoxide and hydrogen peroxide in the mitochondrial matrix is dominated by site I* [PMID:32971363](https://pubmed.ncbi.nlm.nih.gov/32971363/) [DOI](https://doi.org/10.1016/j.redox.2020.101722)

**Extracted values:**

- 25.0 % (low) — keyword extraction (no semantic understanding)
- 10.0 nM (low) — keyword extraction (no semantic understanding)
- 2.0 µM (low) — keyword extraction (no semantic understanding)
- 0.15 % (low) — keyword extraction (no semantic understanding)
- 750.0 μM (low) — keyword extraction (no semantic understanding)
- 3.0 pmol (low) — keyword extraction (no semantic understanding)
- 84.0 pmol (low) — keyword extraction (no semantic understanding)
- 7.0 % (low) — keyword extraction (no semantic understanding)

### `ros_per_damaged`

- **Current value:** 0.3 dimensionless coupling strength
- **Location:** `constants.py:ROS_PER_DAMAGED`
- **Citation:** Cramer Ch. II.H p.14, Appendix 2 pp.152-153
- **Assessment:** sparse
- **Discrepancy:** minor
- **Literature range:** 25 – 160 (median 92.5, n=2)

**Key papers:**

- Cadenas S (2018). *Mitochondrial uncoupling, ROS generation and cardioprotection.* [PMID:29859845](https://pubmed.ncbi.nlm.nih.gov/29859845/) [DOI](https://doi.org/10.1016/j.bbabio.2018.05.019)
- Shan H, Li X, Ouyang C (2022). *Salidroside prevents PM2.5-induced BEAS-2B cell apoptosis via SIRT1-dependent regulation of ROS and mitochondrial function.* [PMID:35026589](https://pubmed.ncbi.nlm.nih.gov/35026589/) [DOI](https://doi.org/10.1016/j.ecoenv.2022.113170)

**Extracted values:**

- 25.0 % (low) — keyword extraction (no semantic understanding)
- 160.0 µM (low) — keyword extraction (no semantic understanding)

### `senescence_rate`

- **Current value:** 0.005 per year
- **Location:** `constants.py:SENESCENCE_RATE`
- **Citation:** Cramer Ch. VII.A pp.89-92 (qualitative)
- **Assessment:** no-data
- **Discrepancy:** none
- **Literature range:** No numerical values found

## Assessment Categories

### Major Discrepancies (action needed)
- `senescence_ros_multiplier`
- `cliff_steepness`

### Well Supported
- `ros_damage_coupling`
- `ros_relaxation_time`
- `senolytic_clearance_rate`
- `heteroplasmy_cliff`
- `damaged_replication_advantage`
- `tissue_biogenesis_brain`
- `cd38_base_survival`

### Sparse Evidence
- `nad_quality_control_boost`
- `senescence_ros_multiplier`
- `yamanaka_repair_rate`
- `ros_per_damaged`

### No Data Found
- `rapamycin_mitophagy_boost`
- `tissue_ros_sensitivity_brain`
- `senescence_rate`
