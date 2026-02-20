"""9-Step Protocol: Synthetic mtDNA Ring Assembly and Mitochondrial Transplant.

A comprehensive protocol for synthesizing haplogroup-matched mtDNA rings,
replacing damaged mtDNA in isolated mitochondria via platelet-derived mitlets,
and infusing modified mitochondria into a recipient.

Based on principles from:
    Cramer, J.G. (2025). *How to Live Much Longer: The Mitochondrial
    DNA Connection*. ISBN 979-8-9928220-0-4.

DISCLAIMER: This is a theoretical/computational protocol for research and
simulation purposes. It describes procedures that would require extensive
regulatory approval, safety testing, and ethical review before any
clinical application. No actual laboratory work should be performed based
solely on this document without proper institutional oversight.

Each step includes:
    - Procedure description
    - Reagents and materials
    - Estimated costs
    - Suggested vendors
    - Quality control checkpoints
"""


def get_protocol():
    """Return the complete 9-step protocol as a structured dict.

    Returns:
        Dict with "title", "version", "steps" (list of 9 step dicts),
        and "total_estimated_cost".
    """
    return {
        "title": "Synthetic mtDNA Ring Assembly and Mitochondrial Transplant "
                 "via Platelet-Derived Mitlets",
        "version": "1.0",
        "reference": "Cramer, J.G. (2025). How to Live Much Longer: The "
                     "Mitochondrial DNA Connection. ISBN 979-8-9928220-0-4.",
        "disclaimer": (
            "Theoretical/computational protocol for research and simulation "
            "purposes. Requires regulatory approval, safety testing, and "
            "ethical review before any clinical application."
        ),

        "steps": [
            # ── Step 1 ───────────────────────────────────────────────────
            {
                "step": 1,
                "title": "Platelet-Derived Mitlet Extraction",
                "description": (
                    "Extract mitochondria-containing extracellular vesicles "
                    "(mitlets) from expired blood bank platelets. Platelets "
                    "are anucleate cells rich in functional mitochondria. "
                    "Expired platelet concentrates (5-7 days post-collection) "
                    "are an accessible source that would otherwise be "
                    "discarded."
                ),
                "procedure": [
                    "1. Obtain expired platelet concentrate units from blood "
                    "bank (type-matched to recipient, ideally haplogroup-"
                    "matched via prior mtDNA sequencing of donor registry).",
                    "2. Centrifuge at 1,500×g for 15 min to pellet platelets.",
                    "3. Resuspend pellet in mitochondria isolation buffer "
                    "(250 mM sucrose, 10 mM Tris-HCl pH 7.4, 1 mM EDTA).",
                    "4. Homogenize using nitrogen cavitation (800 psi, 20 min) "
                    "to lyse platelets while preserving mitochondrial "
                    "membrane integrity.",
                    "5. Differential centrifugation: 600×g (10 min) to remove "
                    "debris, then 10,000×g (15 min) to pellet mitochondria.",
                    "6. Resuspend mitochondrial pellet in respiration buffer "
                    "(125 mM KCl, 2 mM MgCl₂, 2.5 mM KH₂PO₄, 20 mM "
                    "HEPES pH 7.2).",
                    "7. Verify mitochondrial integrity via JC-1 staining "
                    "(red/green fluorescence ratio >3 indicates intact ΔΨ).",
                ],
                "reagents": [
                    "Expired platelet concentrates (blood bank)",
                    "Sucrose (molecular biology grade)",
                    "Tris-HCl buffer pH 7.4",
                    "EDTA (0.5 M stock)",
                    "KCl, MgCl₂, KH₂PO₄, HEPES buffers",
                    "JC-1 mitochondrial membrane potential probe",
                    "Liquid nitrogen (for cavitation)",
                ],
                "estimated_cost_usd": 500,
                "vendors": [
                    "Blood bank / Red Cross (platelet concentrates)",
                    "Sigma-Aldrich / MilliporeSigma (buffers, reagents)",
                    "Thermo Fisher (JC-1 probe, Cat# T3168)",
                    "Parr Instrument Co. (nitrogen cavitation vessel)",
                ],
                "quality_control": [
                    "JC-1 red/green ratio >3.0 (intact membrane potential)",
                    "Protein concentration via BCA assay (target: 5-10 mg/mL)",
                    "Electron microscopy spot-check (intact double membranes)",
                    "Citrate synthase activity assay (mitochondrial viability)",
                ],
            },

            # ── Step 2 ───────────────────────────────────────────────────
            {
                "step": 2,
                "title": "Synthetic mtDNA Ring Assembly",
                "description": (
                    "Synthesize the complete 16,569 bp human mitochondrial "
                    "genome as a circular DNA molecule from a digital "
                    "sequence template. The reference sequence (rCRS, "
                    "NC_012920.1) is divided into overlapping fragments "
                    "for synthesis and assembled via Gibson assembly into "
                    "a complete circular genome."
                ),
                "procedure": [
                    "1. Obtain recipient's mtDNA sequence via whole-genome "
                    "sequencing or targeted mtDNA enrichment sequencing "
                    "(to match haplogroup and personal variants).",
                    "2. Design synthesis as 8-10 overlapping fragments of "
                    "~2,000 bp each, with 40 bp overlaps for Gibson "
                    "assembly.",
                    "3. Order synthetic DNA fragments from vendor (gene "
                    "synthesis service). Total: ~16,600 bp across fragments.",
                    "4. Perform Gibson assembly: combine equimolar fragments "
                    "with Gibson Assembly Master Mix (T5 exonuclease, "
                    "Phusion polymerase, Taq ligase) at 50°C for 60 min.",
                    "5. Transform assembly into competent E. coli for "
                    "amplification (optional: use cell-free rolling circle "
                    "amplification instead).",
                    "6. Verify circular topology via gel electrophoresis "
                    "(supercoiled band) and restriction digest mapping.",
                    "7. Sequence-verify the complete assembly via Oxford "
                    "Nanopore long-read sequencing (single reads spanning "
                    "the entire 16.6 kb circle).",
                ],
                "reagents": [
                    "Synthetic DNA fragments (8-10 × ~2 kb)",
                    "Gibson Assembly Master Mix (NEB)",
                    "Competent E. coli cells (NEB 10-beta or similar)",
                    "LB media and plates with appropriate antibiotic",
                    "Miniprep kit for plasmid isolation",
                    "Restriction enzymes for verification",
                    "Oxford Nanopore sequencing kit",
                ],
                "estimated_cost_usd": 8000,
                "vendors": [
                    "Twist Bioscience (gene fragments, $0.07/bp × 16,600 bp "
                    "≈ $1,200)",
                    "IDT (Integrated DNA Technologies) — alternative "
                    "fragment synthesis",
                    "GenScript — full gene synthesis service",
                    "New England Biolabs (Gibson Assembly, Cat# E2611)",
                    "Oxford Nanopore Technologies (MinION sequencing)",
                ],
                "quality_control": [
                    "Gel electrophoresis: single supercoiled band at ~16.6 kb",
                    "Restriction digest: expected fragment pattern (≥3 enzymes)",
                    "Nanopore sequencing: >99.5% identity to target sequence",
                    "Topology confirmation: S1 nuclease sensitivity (linear "
                    "contaminant removal)",
                ],
            },

            # ── Step 3 ───────────────────────────────────────────────────
            {
                "step": 3,
                "title": "Synonymous Base Substitution (Watermarking)",
                "description": (
                    "Introduce synonymous (silent) codon changes in 5-6 "
                    "protein-coding regions of the synthetic mtDNA. These "
                    "mutations do not change the encoded proteins but create "
                    "unique restriction sites that distinguish synthetic from "
                    "original mtDNA. This enables selective destruction of "
                    "original (damaged) mtDNA while preserving synthetic "
                    "copies. Critically, the D-loop (control region), tRNA "
                    "genes, rRNA genes, and duplicated sequences are left "
                    "unmodified to preserve regulatory function."
                ),
                "procedure": [
                    "1. Identify 5-6 protein-coding genes distributed across "
                    "the mtDNA circle: ND1 (3,307-4,262), CO1 (5,904-7,445), "
                    "ND4 (10,760-12,137), ND5 (12,337-14,148), CYTB "
                    "(14,747-15,887), and ATP6 (8,527-9,207).",
                    "2. For each gene, identify 2-3 positions where a third-"
                    "codon wobble substitution creates a new restriction site "
                    "not present in the original sequence.",
                    "3. Verify that each substitution is: (a) synonymous "
                    "(same amino acid), (b) uses mitochondrial codon table "
                    "(NCBI translation table 2), (c) does not disrupt any "
                    "overlapping reading frame or regulatory element.",
                    "4. Design substitutions to create unique 6-8 bp "
                    "restriction sites (e.g., NotI: GCGGCCGC, PacI: "
                    "TTAATTAA) that are absent from human mtDNA.",
                    "5. Incorporate substitutions into the synthetic fragment "
                    "design (Step 2) before ordering.",
                    "6. Verify by restriction digest that synthetic mtDNA "
                    "cuts at designed sites and original mtDNA does not.",
                ],
                "reagents": [
                    "Bioinformatics tools (custom scripts for codon "
                    "optimization)",
                    "Restriction enzymes for verification (NotI, PacI, etc.)",
                    "Gel electrophoresis materials",
                ],
                "estimated_cost_usd": 500,
                "vendors": [
                    "New England Biolabs (restriction enzymes)",
                    "Benchling or SnapGene (sequence design software)",
                ],
                "quality_control": [
                    "In silico verification: no amino acid changes "
                    "(mitochondrial codon table)",
                    "D-loop, tRNA, rRNA regions confirmed unmodified",
                    "No overlapping reading frame disruption",
                    "Restriction digest confirms synthetic-specific cutting",
                    "Protein expression test (in vitro translation) confirms "
                    "functional proteins",
                ],
                "regions_to_avoid": [
                    "D-loop / control region (16,024-576): replication origin, "
                    "promoters",
                    "tRNA genes (22 genes scattered throughout): critical "
                    "secondary structure",
                    "rRNA genes (12S: 648-1,601; 16S: 1,671-3,229): ribosome "
                    "function",
                    "Heavy-strand origin of replication (OH): within D-loop",
                    "Light-strand origin of replication (OL: 5,721-5,798)",
                    "Any region with overlapping genes on opposite strands",
                ],
            },

            # ── Step 4 ───────────────────────────────────────────────────
            {
                "step": 4,
                "title": "PCR Amplification and Size Selection",
                "description": (
                    "Amplify the synthetic mtDNA ring to therapeutic "
                    "quantities (~10⁴× amplification) and purify full-length "
                    "circular molecules by size selection. Rolling circle "
                    "amplification (RCA) is preferred over standard PCR to "
                    "maintain circular topology."
                ),
                "procedure": [
                    "1. Perform rolling circle amplification (RCA) using "
                    "phi29 DNA polymerase with random hexamer primers. "
                    "This amplifies circular templates isothermally while "
                    "maintaining topology.",
                    "2. Incubate at 30°C for 16-18 hours (yields ~10⁴× "
                    "amplification from nanogram input).",
                    "3. Heat-inactivate phi29 at 65°C for 10 min.",
                    "4. Digest concatenated RCA product with a single-cut "
                    "restriction enzyme to regenerate unit-length linear "
                    "molecules.",
                    "5. Gel-purify full-length 16.6 kb band by pulse-field "
                    "gel electrophoresis (PFGE) or BluePippin automated "
                    "size selection (15-18 kb window).",
                    "6. Recircularize purified linear molecules using T4 DNA "
                    "ligase at dilute concentration (promotes intramolecular "
                    "ligation).",
                    "7. Remove remaining linear molecules with Plasmid-Safe "
                    "ATP-dependent DNase (Epicentre).",
                    "8. Quantify by Qubit fluorometry and verify by gel.",
                ],
                "reagents": [
                    "phi29 DNA polymerase (NEB or Thermo)",
                    "Random hexamer primers",
                    "dNTP mix (10 mM each)",
                    "Restriction enzyme (single-cutter for linearization)",
                    "T4 DNA ligase",
                    "Plasmid-Safe DNase (Lucigen/Epicentre)",
                    "ATP (for Plasmid-Safe DNase)",
                    "BluePippin cassettes (15-18 kb)",
                    "Qubit dsDNA BR assay kit",
                ],
                "estimated_cost_usd": 3000,
                "vendors": [
                    "New England Biolabs (phi29, Cat# M0269; T4 ligase)",
                    "Thermo Fisher (phi29, Qubit assays)",
                    "Lucigen/Epicentre (Plasmid-Safe DNase, Cat# E3101K)",
                    "Sage Science (BluePippin for size selection)",
                ],
                "quality_control": [
                    "Qubit quantification: target yield >1 μg per reaction",
                    "Gel electrophoresis: single band at ~16.6 kb (post-"
                    "linearization) or supercoiled (post-recircularization)",
                    "Nanopore sequencing of amplified product (verify no "
                    "mutations introduced by phi29)",
                    "A260/A280 ratio: 1.8-2.0 (pure DNA)",
                ],
            },

            # ── Step 5 ───────────────────────────────────────────────────
            {
                "step": 5,
                "title": "Custom Restriction Enzyme Synthesis",
                "description": (
                    "Synthesize or obtain programmable nucleases that "
                    "selectively cut the ORIGINAL mitochondrial sequences "
                    "at sites that are disrupted in the synthetic version "
                    "(by the synonymous substitutions from Step 3). This "
                    "enables selective destruction of damaged endogenous "
                    "mtDNA while sparing the synthetic replacement. "
                    "Mitochondria-targeted CRISPR-free nucleases (e.g., "
                    "mitoTALENs or ZFNs) are preferred due to challenges "
                    "of importing guide RNA into mitochondria."
                ),
                "procedure": [
                    "1. Design mitoTALEN pairs targeting 2-3 sites in the "
                    "original mtDNA sequence that are disrupted by synonymous "
                    "substitutions in the synthetic version.",
                    "2. Each mitoTALEN pair consists of: (a) mitochondrial "
                    "targeting sequence (MTS) from COX8A, (b) TALE DNA-"
                    "binding domain recognizing 15-20 bp flanking the cut "
                    "site, (c) FokI nuclease domain.",
                    "3. Order mitoTALEN expression constructs from synthesis "
                    "vendor or assemble from TALE repeat modules.",
                    "4. Express mitoTALEN proteins in E. coli or cell-free "
                    "expression system.",
                    "5. Purify by Ni-NTA affinity chromatography (His-tagged).",
                    "6. Validate specificity in vitro: incubate with both "
                    "original and synthetic mtDNA, verify cutting of original "
                    "only.",
                    "7. Alternatively, use mitochondria-targeted zinc finger "
                    "nucleases (mtZFNs) with the same targeting strategy.",
                ],
                "reagents": [
                    "mitoTALEN expression constructs (custom synthesis)",
                    "E. coli expression strain (BL21-DE3 or similar)",
                    "IPTG for induction",
                    "Ni-NTA agarose for purification",
                    "Dialysis and concentration supplies",
                    "Original and synthetic mtDNA (for validation)",
                ],
                "estimated_cost_usd": 12000,
                "vendors": [
                    "Addgene (mitoTALEN backbone vectors, if available)",
                    "Thermo Fisher (GeneArt gene synthesis for custom TALE "
                    "constructs)",
                    "GenScript (custom protein expression service)",
                    "Sigma-Aldrich (CompoZr zinc finger nucleases, custom)",
                    "QIAGEN (Ni-NTA purification)",
                ],
                "quality_control": [
                    "In vitro cleavage assay: >90% cutting of original mtDNA",
                    "Specificity assay: <1% cutting of synthetic mtDNA",
                    "SDS-PAGE: correct protein size",
                    "Western blot: anti-FokI or anti-FLAG confirmation",
                    "Mitochondrial import assay (in isolated mitochondria)",
                ],
            },

            # ── Step 6 ───────────────────────────────────────────────────
            {
                "step": 6,
                "title": "PolG and MGME1 Cleanup Protein Synthesis",
                "description": (
                    "Produce mitochondrial DNA polymerase gamma (PolG) and "
                    "mitochondrial genome maintenance exonuclease 1 (MGME1) "
                    "for degradation of linearized (cut) original mtDNA "
                    "fragments. After the restriction enzymes (Step 5) "
                    "linearize the original mtDNA, these cleanup enzymes "
                    "disassemble the fragments into nucleotides that are "
                    "recycled by the mitochondrial nucleotide pool."
                ),
                "procedure": [
                    "1. Clone human POLG (catalytic subunit, UniProt P54098) "
                    "and MGME1 (UniProt Q9BQP7) coding sequences with "
                    "N-terminal mitochondrial targeting sequences.",
                    "2. Add His6-tag for purification and TEV cleavage site "
                    "for tag removal.",
                    "3. Express in E. coli BL21(DE3) at 18°C (slow induction "
                    "for soluble expression) or use baculovirus/insect cell "
                    "system for proper folding.",
                    "4. Purify: Ni-NTA → size exclusion chromatography → "
                    "ion exchange (Q-Sepharose).",
                    "5. Remove His-tag with TEV protease, re-purify.",
                    "6. Verify exonuclease activity on linearized DNA "
                    "substrates (3'→5' for PolG, 5'→3' for MGME1).",
                    "7. Verify that circular (intact synthetic) mtDNA is "
                    "resistant to degradation.",
                    "8. Formulate in storage buffer with 50% glycerol, "
                    "store at -80°C.",
                ],
                "reagents": [
                    "POLG and MGME1 expression constructs",
                    "E. coli BL21(DE3) or Sf9 insect cells",
                    "IPTG or baculovirus for induction",
                    "Ni-NTA, SEC, and IEX chromatography columns",
                    "TEV protease",
                    "Linear DNA substrates for activity assay",
                    "Glycerol, storage buffer components",
                ],
                "estimated_cost_usd": 8000,
                "vendors": [
                    "GenScript (gene synthesis and protein expression service)",
                    "Addgene (if POLG/MGME1 expression plasmids available)",
                    "GE Healthcare / Cytiva (chromatography columns and media)",
                    "NEB (TEV protease, Cat# P8112)",
                    "Thermo Fisher (Sf9 cells, baculovirus expression system)",
                ],
                "quality_control": [
                    "SDS-PAGE: correct molecular weight (PolG ~140 kDa "
                    "catalytic, MGME1 ~40 kDa)",
                    "Exonuclease activity: >90% degradation of linear DNA "
                    "in 30 min at 37°C",
                    "Circular DNA protection: <5% degradation of supercoiled "
                    "plasmid",
                    "Mass spectrometry: confirm protein identity",
                    "Endotoxin test: <0.1 EU/mL (for downstream cell work)",
                ],
            },

            # ── Step 7 ───────────────────────────────────────────────────
            {
                "step": 7,
                "title": "Mitochondria-Targeted Liposome Encapsulation",
                "description": (
                    "Encapsulate synthetic mtDNA rings plus cleanup enzymes "
                    "(mitoTALENs, PolG, MGME1) in liposomes engineered for "
                    "dual-membrane fusion with mitochondria. The liposome "
                    "must fuse with both the outer mitochondrial membrane "
                    "(OMM) and inner mitochondrial membrane (IMM) to deliver "
                    "cargo to the matrix. This is achieved using a dual-"
                    "layer liposome with cardiolipin-enriched inner leaflet "
                    "and mitochondria-targeting peptides on the surface."
                ),
                "procedure": [
                    "1. Prepare lipid mixture: DOPC (40%), DOPE (25%), "
                    "cardiolipin (20%), cholesterol (10%), DSPE-PEG2000-"
                    "TPP (5%). The TPP (triphenylphosphonium) moiety "
                    "targets the negative ΔΨ of mitochondria.",
                    "2. Dissolve lipids in chloroform:methanol (2:1), "
                    "evaporate to thin film under nitrogen, vacuum "
                    "desiccate overnight.",
                    "3. Hydrate lipid film with cargo solution: synthetic "
                    "mtDNA (50 ng/μL) + mitoTALEN proteins (100 nM) + "
                    "PolG (50 nM) + MGME1 (50 nM) in 10 mM HEPES, "
                    "150 mM NaCl, pH 7.4.",
                    "4. Extrude through 200 nm polycarbonate membranes "
                    "(21 passes) using mini-extruder.",
                    "5. Remove unencapsulated material by size-exclusion "
                    "chromatography (Sepharose CL-4B).",
                    "6. Concentrate by ultracentrifugation (100,000×g, 1 hr).",
                    "7. Characterize: DLS for size (target 150-250 nm), "
                    "zeta potential (should be slightly positive due to "
                    "TPP), encapsulation efficiency by DNA quantification "
                    "of lysed vs intact liposomes.",
                ],
                "reagents": [
                    "DOPC, DOPE, cardiolipin (Avanti Polar Lipids)",
                    "Cholesterol (Sigma)",
                    "DSPE-PEG2000-TPP (custom synthesis or Avanti)",
                    "Polycarbonate extrusion membranes (200 nm)",
                    "Mini-extruder apparatus",
                    "Sepharose CL-4B column",
                    "HEPES, NaCl buffers",
                ],
                "estimated_cost_usd": 5000,
                "vendors": [
                    "Avanti Polar Lipids (all lipids, extrusion equipment)",
                    "Sigma-Aldrich (cholesterol, buffers)",
                    "Cytiva (Sepharose CL-4B)",
                    "Malvern Panalytical (DLS instrument, Zetasizer)",
                    "Custom synthesis services (DSPE-PEG-TPP conjugate)",
                ],
                "quality_control": [
                    "DLS: mean diameter 150-250 nm, PDI <0.2",
                    "Zeta potential: +5 to +20 mV",
                    "Encapsulation efficiency: >30% for DNA, >20% for proteins",
                    "Cryo-EM: unilamellar vesicles with cargo visible",
                    "Stability: <10% size change over 24 hr at 4°C",
                    "Sterility test (for downstream cell application)",
                ],
            },

            # ── Step 8 ───────────────────────────────────────────────────
            {
                "step": 8,
                "title": "Mitlet + Liposome Combination and mtDNA Replacement",
                "description": (
                    "Combine the cargo-loaded liposomes (Step 7) with the "
                    "isolated mitochondria/mitlets (Step 1) to achieve "
                    "mtDNA replacement ex vivo. The liposomes fuse with "
                    "mitochondrial membranes, delivering mitoTALENs (which "
                    "cut original mtDNA), cleanup enzymes (which degrade "
                    "the fragments), and synthetic mtDNA rings (which "
                    "replace the originals). Verify replacement by "
                    "restriction digest and sequencing."
                ),
                "procedure": [
                    "1. Mix cargo-loaded liposomes with isolated mitochondria "
                    "at a 100:1 liposome-to-mitochondrion ratio in "
                    "respiration buffer at 37°C.",
                    "2. Incubate for 30 min with gentle agitation (orbital "
                    "shaker, 100 rpm) to allow membrane fusion.",
                    "3. Add fresh respiration buffer with succinate (5 mM) "
                    "and ADP (1 mM) to maintain mitochondrial function "
                    "during the replacement process.",
                    "4. Allow 2 hours for: (a) mitoTALEN cutting of original "
                    "mtDNA, (b) PolG/MGME1 degradation of fragments, "
                    "(c) synthetic mtDNA establishment.",
                    "5. Wash mitochondria 3× by pelleting (10,000×g, 10 min) "
                    "and resuspending to remove excess liposomes and free "
                    "enzymes.",
                    "6. Extract mtDNA from a small aliquot for verification.",
                    "7. Restriction digest with synthetic-specific enzyme: "
                    "presence of expected bands confirms synthetic mtDNA; "
                    "absence of original-specific bands confirms replacement.",
                    "8. Quantitative PCR targeting original vs synthetic "
                    "sequences to measure replacement efficiency.",
                    "9. JC-1 re-staining to confirm mitochondria remain "
                    "functional after the replacement procedure.",
                ],
                "reagents": [
                    "Cargo-loaded liposomes (from Step 7)",
                    "Isolated mitochondria/mitlets (from Step 1)",
                    "Respiration buffer with succinate and ADP",
                    "mtDNA extraction kit (QIAGEN or similar)",
                    "Restriction enzymes (original-specific and "
                    "synthetic-specific)",
                    "qPCR primers and probes (TaqMan, targeting SNP sites)",
                    "JC-1 probe",
                ],
                "estimated_cost_usd": 2000,
                "vendors": [
                    "QIAGEN (mtDNA extraction, Cat# 67543)",
                    "NEB (restriction enzymes)",
                    "Thermo Fisher (TaqMan qPCR assays, JC-1)",
                    "Bio-Rad (qPCR instrument)",
                ],
                "quality_control": [
                    "Replacement efficiency: >80% synthetic mtDNA by qPCR",
                    "Restriction digest: clean synthetic-specific bands",
                    "JC-1 red/green ratio: >2.5 (maintained membrane potential)",
                    "Oxygen consumption rate (Seahorse or Clark electrode): "
                    ">70% of pre-treatment rate",
                    "Absence of free DNA in supernatant (no leakage)",
                    "Nanopore sequencing of extracted mtDNA: >95% reads "
                    "match synthetic sequence",
                ],
            },

            # ── Step 9 ───────────────────────────────────────────────────
            {
                "step": 9,
                "title": "Modified Mitochondria Infusion into Recipient",
                "description": (
                    "Infuse the mtDNA-replaced mitochondria into the "
                    "recipient patient. Mitochondrial transplant has been "
                    "demonstrated in cardiac ischemia-reperfusion models "
                    "(McCully et al., 2009). The modified mitochondria are "
                    "delivered via direct tissue injection or intravenous "
                    "infusion depending on the target tissue. Haplogroup "
                    "matching (confirmed in Steps 1-2) minimizes immune "
                    "recognition."
                ),
                "procedure": [
                    "1. Perform final QC on modified mitochondria: viability, "
                    "mtDNA replacement verification, sterility.",
                    "2. Resuspend in clinical-grade infusion buffer (Plasma-"
                    "Lyte A or equivalent isotonic solution).",
                    "3. For cardiac applications: direct injection into "
                    "myocardium (0.5-1.0 × 10⁹ mitochondria in 1 mL, "
                    "multiple injection sites).",
                    "4. For systemic applications: IV infusion of 1-5 × 10⁹ "
                    "mitochondria in 50 mL over 30 min. Mitochondria are "
                    "taken up by cells via macropinocytosis and tunneling "
                    "nanotubes.",
                    "5. For neurological applications: intrathecal injection "
                    "or intranasal delivery (mitochondria can cross the "
                    "blood-brain barrier when sufficiently small).",
                    "6. Monitor recipient for 24 hours: vital signs, "
                    "inflammatory markers (CRP, IL-6), cardiac enzymes "
                    "(if cardiac target).",
                    "7. Follow-up at 1 week, 1 month, 3 months: measure "
                    "heteroplasmy in accessible tissue (blood cells), "
                    "functional assessments specific to the treated "
                    "condition.",
                    "8. Long-term monitoring: repeat mtDNA sequencing at "
                    "6 and 12 months to assess durability of replacement.",
                ],
                "reagents": [
                    "Clinical-grade infusion buffer (Plasma-Lyte A)",
                    "Sterile syringes and needles (cardiac: 27G)",
                    "IV infusion sets",
                    "Monitoring equipment (pulse ox, ECG, BP)",
                ],
                "estimated_cost_usd": 5000,
                "vendors": [
                    "Baxter (Plasma-Lyte A)",
                    "BD (syringes, needles, IV sets)",
                    "Clinical laboratory (monitoring, blood tests)",
                ],
                "quality_control": [
                    "Pre-infusion: sterility confirmed (no growth in 72 hr "
                    "culture)",
                    "Pre-infusion: endotoxin <5 EU/kg body weight",
                    "Pre-infusion: viability >80% (JC-1)",
                    "Post-infusion 24 hr: no adverse inflammatory response "
                    "(CRP <10 mg/L)",
                    "Post-infusion 1 month: heteroplasmy reduction in "
                    "blood cells (qPCR)",
                    "Post-infusion 3 months: functional improvement "
                    "(tissue-specific assessments)",
                ],
            },
        ],

        "total_estimated_cost_usd": 44000,
        "cost_breakdown": {
            "step_1_mitlet_extraction": 500,
            "step_2_mtdna_synthesis": 8000,
            "step_3_synonymous_substitution": 500,
            "step_4_amplification": 3000,
            "step_5_restriction_enzyme": 12000,
            "step_6_cleanup_proteins": 8000,
            "step_7_liposome_encapsulation": 5000,
            "step_8_mtdna_replacement": 2000,
            "step_9_infusion": 5000,
        },
        "notes": [
            "Costs are estimates for reagents and materials only; do not "
            "include equipment, personnel, or facility costs.",
            "Haplogroup matching is critical for immune tolerance; the "
            "recipient's own mtDNA sequence should be used as the synthesis "
            "template.",
            "The entire procedure from platelet collection to infusion "
            "should ideally be completed within 48-72 hours to minimize "
            "mitochondrial degradation.",
            "Multiple treatment cycles may be needed for durable "
            "heteroplasmy reduction in solid tissues.",
        ],
    }


def print_protocol():
    """Print the protocol in a human-readable format."""
    protocol = get_protocol()

    print("=" * 74)
    print(f"  {protocol['title']}")
    print("=" * 74)
    print(f"\nVersion: {protocol['version']}")
    print(f"Reference: {protocol['reference']}")
    print(f"\nDISCLAIMER: {protocol['disclaimer']}")
    print()

    for step in protocol["steps"]:
        print("-" * 74)
        print(f"  STEP {step['step']}: {step['title']}")
        print("-" * 74)
        print(f"\n{step['description']}\n")

        print("  PROCEDURE:")
        for line in step["procedure"]:
            print(f"    {line}")

        print(f"\n  REAGENTS:")
        for r in step["reagents"]:
            print(f"    - {r}")

        print(f"\n  ESTIMATED COST: ${step['estimated_cost_usd']:,}")

        print(f"\n  SUGGESTED VENDORS:")
        for v in step["vendors"]:
            print(f"    - {v}")

        print(f"\n  QUALITY CONTROL:")
        for qc in step["quality_control"]:
            print(f"    ✓ {qc}")

        if "regions_to_avoid" in step:
            print(f"\n  REGIONS TO AVOID:")
            for r in step["regions_to_avoid"]:
                print(f"    ✗ {r}")

        print()

    print("=" * 74)
    print(f"  TOTAL ESTIMATED COST: ${protocol['total_estimated_cost_usd']:,}")
    print("=" * 74)

    print("\n  COST BREAKDOWN:")
    for step_name, cost in protocol["cost_breakdown"].items():
        print(f"    {step_name:40s} ${cost:>6,}")

    print("\n  NOTES:")
    for note in protocol["notes"]:
        print(f"    • {note}")

    print()


if __name__ == "__main__":
    print_protocol()
