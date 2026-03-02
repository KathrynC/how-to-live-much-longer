"""Family genomics module — WGS-informed priors for mitochondrial aging.

Provides whole genome sequencing (WGS) data integration for the 7-member
Cramer family. Replaces population-average priors (N_healthy=0.75,
N_deletion=0.15, N_point=0.10) with direct mtDNA heteroplasmy measurements,
haplogroup-specific vulnerability, and cross-generational calibration.

Family structure (from family_ecosystem_report.py):
    John Jr. (91) ──── [wife]
                        │
                  John III (62) ──── Kathryn (64)
                                      │
                       ┌──────────────┼──────────┬──────────┐
                    Peter (28)    Jasper (24)  Ratio (23)  Selena (26)

mtDNA is maternal-only: John Jr. has his mother's lineage; Kathryn's
children share Kathryn's mtDNA. Nuclear DNA: ~50% parent-child,
~25% grandparent-grandchild.

References:
    Cramer (forthcoming 2026), Appendix 2: haplogroup deletion frequencies
    Kazuno et al. 2006: common deletion frequency by haplogroup
    DNAComplete / Nebula Genomics: WGS depths (1x, 30x, 100x Elite)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from constants import (
    HAPLOGROUP_VULNERABILITY,
    HAPLOGROUP_VULNERABILITY_DEFAULT,
    WGS_HET_CONFIDENCE,
    PGX_CYP3A4_MULTIPLIER,
    PGX_CYP2D6_MULTIPLIER,
    DIRECT_REPEAT_VULNERABILITY_SLOPE,
    DIRECT_REPEAT_BASELINE_COUNT,
    IBD_SHARING,
    SENSOR_PRIOR_N_HEALTHY,
    SENSOR_PRIOR_N_DELETION,
    SENSOR_PRIOR_N_POINT,
    SENSOR_PRIOR_MEMBRANE_POTENTIAL,
)


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class GenomicProfile:
    """WGS-derived genomic data for one individual."""
    haplogroup: str = "unknown"              # e.g. "H2a", "J1c", "U5a"
    # mtDNA measurements (from WGS)
    mtdna_deletion_burden: float = 0.15      # fraction deletion-bearing copies
    mtdna_point_mutations: float = 0.10      # fraction point-mutation copies
    mtdna_copy_number: float = 1.0           # normalized total copies
    direct_repeat_count: int = 0             # D-loop direct repeats (→ vulnerability)
    # Nuclear genotypes (from WGS)
    apoe_genotype: int = 0                   # 0/1/2 APOE4 alleles
    foxo3_protective: int = 0                # 1 if CC at rs9486902
    cd38_risk: int = 0                       # 1 if risk variant at rs6449197
    # Pharmacogenomics
    cyp2d6_status: str = "normal"            # poor/intermediate/normal/ultra
    cyp3a4_status: str = "normal"            # poor/intermediate/normal/ultra
    # Sequencing metadata
    sequencing_depth: float = 30.0           # 1x, 30x, or 100x
    het_call_confidence: float = 0.85        # higher at deeper sequencing


@dataclass
class MedicalHistoryEntry:
    """A single lab value or clinical measurement from medical records."""
    age_at_measurement: float                # patient age when measured
    variable: str                            # e.g. "hs_crp", "glucose", "atp_proxy"
    value: float                             # measured value
    unit: str = ""                           # e.g. "mg/L", "mg/dL"
    source: str = ""                         # e.g. "labcorp_2024", "mayo_clinic"


@dataclass
class FamilyMember:
    """One member of the family pedigree with genomic and clinical data."""
    name: str
    age: float
    sex: str                                 # "M" or "F"
    relationship: str                        # e.g. "proband", "son", "spouse", "grandchild"
    genomic_profile: Optional[GenomicProfile] = None
    medical_history: list[MedicalHistoryEntry] = field(default_factory=list)
    # Existing model parameters (from family_ecosystem_report.py)
    patient_overrides: dict = field(default_factory=dict)
    intervention_overrides: dict = field(default_factory=dict)


# ── FamilyPedigree class ───────────────────────────────────────────────────


# Kinship map for the Cramer family.
# Key: (name_a, name_b) or (name_b, name_a) → relationship type.
_CRAMER_KINSHIP = {}

def _add_kinship(a: str, b: str, rel: str) -> None:
    _CRAMER_KINSHIP[(a, b)] = rel
    _CRAMER_KINSHIP[(b, a)] = rel

# Parent-child
_add_kinship("John Jr.", "John III", "parent_child")
# Grandparent-grandchild
_add_kinship("John Jr.", "Peter", "grandparent_grandchild")
_add_kinship("John Jr.", "Jasper", "grandparent_grandchild")
_add_kinship("John Jr.", "Ratio", "grandparent_grandchild")
_add_kinship("John Jr.", "Selena Shea", "grandparent_grandchild")
_add_kinship("Kathryn", "Peter", "parent_child")
_add_kinship("Kathryn", "Jasper", "parent_child")
_add_kinship("Kathryn", "Ratio", "parent_child")
_add_kinship("Kathryn", "Selena Shea", "parent_child")
_add_kinship("John III", "Peter", "parent_child")
_add_kinship("John III", "Jasper", "parent_child")
_add_kinship("John III", "Ratio", "parent_child")
_add_kinship("John III", "Selena Shea", "parent_child")
# Spouse
_add_kinship("John III", "Kathryn", "spouse")
# Siblings
_add_kinship("Peter", "Jasper", "sibling")
_add_kinship("Peter", "Ratio", "sibling")
_add_kinship("Peter", "Selena Shea", "sibling")
_add_kinship("Jasper", "Ratio", "sibling")
_add_kinship("Jasper", "Selena Shea", "sibling")
_add_kinship("Ratio", "Selena Shea", "sibling")

# Maternal mtDNA sharing groups.
# Kathryn's children share her mtDNA; John Jr. has his mother's lineage.
_MATERNAL_GROUPS = {
    "kathryn_line": {"Kathryn", "Peter", "Jasper", "Ratio", "Selena Shea"},
    "john_sr_line": {"John Jr."},
    "john_iii_line": {"John III"},  # has his mother's mtDNA (John Jr.'s wife)
}


class FamilyPedigree:
    """Collection of family members with kinship analysis."""

    def __init__(self, members: list[FamilyMember]):
        self.members = {m.name: m for m in members}

    def get_member(self, name: str) -> FamilyMember:
        if name not in self.members:
            raise KeyError(f"Unknown family member: {name}")
        return self.members[name]

    def shared_nuclear_fraction(self, name_a: str, name_b: str) -> float:
        """Returns expected IBD sharing fraction between two members."""
        if name_a == name_b:
            return 1.0
        rel = _CRAMER_KINSHIP.get((name_a, name_b))
        if rel is None:
            return 0.0
        return IBD_SHARING.get(rel, 0.0)

    def shared_mtdna(self, name_a: str, name_b: str) -> bool:
        """Returns True if two members share the same maternal mtDNA line."""
        for group in _MATERNAL_GROUPS.values():
            if name_a in group and name_b in group:
                return True
        return False

    def relatives_with_allele(self, allele: str, value: int) -> list[FamilyMember]:
        """Return members whose genomic profile has the given allele value."""
        result = []
        for m in self.members.values():
            if m.genomic_profile is not None:
                if getattr(m.genomic_profile, allele, None) == value:
                    result.append(m)
        return result

    def members_by_age(self) -> list[FamilyMember]:
        """Return members sorted oldest to youngest."""
        return sorted(self.members.values(), key=lambda m: m.age, reverse=True)


# ── Core computation functions ─────────────────────────────────────────────


def _haplogroup_vulnerability(haplogroup: str) -> float:
    """Look up genetic vulnerability for a haplogroup string.

    Tries exact match first, then prefix match (e.g. "H2a" → try "H2a",
    then "H2", then "H"). Falls back to HAPLOGROUP_VULNERABILITY_DEFAULT.
    """
    # Exact match
    if haplogroup in HAPLOGROUP_VULNERABILITY:
        return HAPLOGROUP_VULNERABILITY[haplogroup]
    # Try progressively shorter prefixes
    for length in range(len(haplogroup) - 1, 0, -1):
        prefix = haplogroup[:length]
        if prefix in HAPLOGROUP_VULNERABILITY:
            return HAPLOGROUP_VULNERABILITY[prefix]
    return HAPLOGROUP_VULNERABILITY_DEFAULT


def compute_family_priors(
    target: FamilyMember,
    pedigree: FamilyPedigree,
) -> dict[str, float]:
    """Compute family-informed priors for the 8D state vector.

    For the target member:
    - If WGS available: use direct mtDNA measurements for N_healthy, N_deletion, N_point
    - If haplogroup known: use HAPLOGROUP_VULNERABILITY for genetic_vulnerability
    - Membrane potential: estimate from relatives' SpO2/hs-CRP histories if available
    - For observable vars (ATP, ROS, NAD, Sen): use as weighted initial estimates

    Returns dict with keys:
        'n_healthy', 'n_deletion', 'n_point', 'membrane_potential',
        'genetic_vulnerability', 'atp_prior', 'ros_prior', 'nad_prior', 'sen_prior',
        'prior_confidence'  # 0-1, higher = more WGS data available
    """
    profile = target.genomic_profile
    confidence = 0.0

    if profile is not None:
        # Direct mtDNA heteroplasmy from WGS
        n_del = profile.mtdna_deletion_burden
        n_pt = profile.mtdna_point_mutations
        n_h = max(0.0, 1.0 - n_del - n_pt) * profile.mtdna_copy_number
        psi = SENSOR_PRIOR_MEMBRANE_POTENTIAL  # default; refined below

        # Confidence based on sequencing depth
        confidence = profile.het_call_confidence

        # Haplogroup → genetic vulnerability
        vuln = _haplogroup_vulnerability(profile.haplogroup)

        # Adjust for direct repeat count if measured
        if profile.direct_repeat_count > 0:
            dr_excess = profile.direct_repeat_count - DIRECT_REPEAT_BASELINE_COUNT
            vuln *= (1.0 + dr_excess * DIRECT_REPEAT_VULNERABILITY_SLOPE)
    else:
        # No WGS: fall back to population priors
        n_h = SENSOR_PRIOR_N_HEALTHY
        n_del = SENSOR_PRIOR_N_DELETION
        n_pt = SENSOR_PRIOR_N_POINT
        psi = SENSOR_PRIOR_MEMBRANE_POTENTIAL
        vuln = HAPLOGROUP_VULNERABILITY_DEFAULT

    # Calibration from relatives
    cal = calibrate_from_relatives(target, pedigree)
    if cal['confidence'] > 0.0:
        vuln *= cal['vulnerability_adjustment']
        # Blend confidence
        confidence = max(confidence, cal['confidence'] * 0.5)

    # Estimate membrane potential from medical history if available
    for entry in target.medical_history:
        if entry.variable == "membrane_potential":
            psi = entry.value
            break

    # Observable-variable priors from medical history (for initial estimates)
    atp_prior = None
    ros_prior = None
    nad_prior = None
    sen_prior = None
    for entry in target.medical_history:
        if entry.variable == "atp_proxy" and atp_prior is None:
            atp_prior = entry.value
        elif entry.variable == "ros_proxy" and ros_prior is None:
            ros_prior = entry.value
        elif entry.variable == "nad_blood" and nad_prior is None:
            # Normalize to 0-1 scale (blood NAD baseline ~30 μmol/L)
            nad_prior = min(1.0, entry.value / 30.0)
        elif entry.variable == "senescent_proxy" and sen_prior is None:
            sen_prior = entry.value

    result = {
        'n_healthy': n_h,
        'n_deletion': n_del,
        'n_point': n_pt,
        'membrane_potential': psi,
        'genetic_vulnerability': vuln,
        'prior_confidence': min(1.0, confidence),
    }
    if atp_prior is not None:
        result['atp_prior'] = atp_prior
    if ros_prior is not None:
        result['ros_prior'] = ros_prior
    if nad_prior is not None:
        result['nad_prior'] = nad_prior
    if sen_prior is not None:
        result['sen_prior'] = sen_prior

    return result


def compute_wgs_genetic_modifiers(profile: GenomicProfile) -> dict[str, float]:
    """Like compute_genetic_modifiers() but from actual WGS genotypes.

    Returns same output shape as genetics_module.compute_genetic_modifiers()
    (10 modifier keys), plus pharmacogenomic adjustments:
    - cyp3a4 poor metabolizer → reduce rapamycin_dose effectiveness
    - cyp2d6 status → adjust supplement metabolism
    """
    from genetics_module import compute_genetic_modifiers

    # Start with standard genetic modifiers from actual WGS genotypes
    mods = compute_genetic_modifiers(
        apoe_genotype=profile.apoe_genotype,
        foxo3_protective=profile.foxo3_protective,
        cd38_risk=profile.cd38_risk,
    )

    # Haplogroup-based vulnerability adjustment
    haplo_vuln = _haplogroup_vulnerability(profile.haplogroup)
    mods['vulnerability'] *= haplo_vuln

    # Direct repeat count adjustment
    if profile.direct_repeat_count > 0:
        dr_excess = profile.direct_repeat_count - DIRECT_REPEAT_BASELINE_COUNT
        mods['vulnerability'] *= (1.0 + dr_excess * DIRECT_REPEAT_VULNERABILITY_SLOPE)

    # Pharmacogenomic adjustments
    mods['cyp3a4_multiplier'] = PGX_CYP3A4_MULTIPLIER.get(
        profile.cyp3a4_status, 1.0)
    mods['cyp2d6_multiplier'] = PGX_CYP2D6_MULTIPLIER.get(
        profile.cyp2d6_status, 1.0)

    return mods


def calibrate_from_relatives(
    target: FamilyMember,
    pedigree: FamilyPedigree,
) -> dict[str, float]:
    """Use relatives' medical histories to calibrate target's progression.

    For each relative who shares a risk allele with the target and has
    medical history at an older age, compute the empirical trajectory
    of that variable. Weight by IBD sharing fraction.

    Returns:
        'vulnerability_adjustment': float,  # empirical correction to vulnerability
        'progression_rate_factor': float,   # faster/slower than model predicts
        'confidence': float,                # based on data density
    """
    if target.genomic_profile is None:
        return {
            'vulnerability_adjustment': 1.0,
            'progression_rate_factor': 1.0,
            'confidence': 0.0,
        }

    vuln_adjustments = []
    rate_adjustments = []
    total_weight = 0.0

    for name, member in pedigree.members.items():
        if name == target.name:
            continue
        if not member.medical_history:
            continue

        ibd = pedigree.shared_nuclear_fraction(target.name, name)
        if ibd <= 0.0:
            continue

        # Check for shared risk alleles
        shared_alleles = 0
        if member.genomic_profile is not None and target.genomic_profile is not None:
            for allele in ('apoe_genotype', 'foxo3_protective', 'cd38_risk'):
                if (getattr(member.genomic_profile, allele) ==
                        getattr(target.genomic_profile, allele)):
                    shared_alleles += 1

        if shared_alleles == 0:
            continue

        # Weight by IBD sharing * number of shared alleles
        weight = ibd * (shared_alleles / 3.0)

        # Look at relative's medical history for trajectory indicators
        het_measurements = [
            e for e in member.medical_history
            if e.variable in ('heteroplasmy', 'deletion_burden')
        ]
        if het_measurements:
            # Compare observed heteroplasmy at their age to expected
            latest = max(het_measurements, key=lambda e: e.age_at_measurement)
            # Simple heuristic: if relative's het is higher than expected
            # for their age, vulnerability is higher than baseline
            expected_het = 0.05 + 0.005 * latest.age_at_measurement
            observed_het = latest.value
            if expected_het > 0.01:
                ratio = observed_het / expected_het
                vuln_adjustments.append(ratio)
                rate_adjustments.append(ratio)
                total_weight += weight

    if total_weight > 0.0 and vuln_adjustments:
        # Weighted average of adjustments
        weights = np.linspace(0.5, 1.0, len(vuln_adjustments))
        weights = weights / weights.sum()
        avg_vuln = float(np.average(vuln_adjustments, weights=weights))
        avg_rate = float(np.average(rate_adjustments, weights=weights))
        conf = min(1.0, total_weight * 0.5)
    else:
        avg_vuln = 1.0
        avg_rate = 1.0
        conf = 0.0

    return {
        'vulnerability_adjustment': avg_vuln,
        'progression_rate_factor': avg_rate,
        'confidence': conf,
    }


# ── Cramer family factory ─────────────────────────────────────────────────


def build_cramer_family(
    genomic_profiles: Optional[dict[str, GenomicProfile]] = None,
    medical_histories: Optional[dict[str, list[MedicalHistoryEntry]]] = None,
) -> FamilyPedigree:
    """Build the Cramer family pedigree.

    If genomic_profiles/medical_histories are None, uses defaults
    matching the existing family_ecosystem_report.py parameters.
    This allows the module to work before WGS data arrives.
    """
    if genomic_profiles is None:
        genomic_profiles = {}
    if medical_histories is None:
        medical_histories = {}

    members = [
        FamilyMember(
            name="John Jr.",
            age=91,
            sex="M",
            relationship="grandfather",
            genomic_profile=genomic_profiles.get("John Jr.", GenomicProfile(
                sequencing_depth=100.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(100.0, 0.95),
                mtdna_deletion_burden=0.45,
                mtdna_point_mutations=0.20,
                apoe_genotype=0,
            )),
            medical_history=medical_histories.get("John Jr.", []),
            patient_overrides={"baseline_heteroplasmy": 0.65},
            intervention_overrides={"transplant_rate": 0.9},
        ),
        FamilyMember(
            name="John III",
            age=62,
            sex="M",
            relationship="father",
            genomic_profile=genomic_profiles.get("John III", GenomicProfile(
                sequencing_depth=30.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(30.0, 0.85),
                apoe_genotype=0,
            )),
            medical_history=medical_histories.get("John III", []),
            patient_overrides={"sex": "male"},
            intervention_overrides={"rapamycin_dose": 0.4},
        ),
        FamilyMember(
            name="Kathryn",
            age=64,
            sex="F",
            relationship="mother",
            genomic_profile=genomic_profiles.get("Kathryn", GenomicProfile(
                sequencing_depth=30.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(30.0, 0.85),
                apoe_genotype=1,
            )),
            medical_history=medical_histories.get("Kathryn", []),
            patient_overrides={"sex": "female", "osteopenia": True},
            intervention_overrides={"rapamycin_dose": 0.4},
        ),
        FamilyMember(
            name="Peter",
            age=28,
            sex="M",
            relationship="grandchild",
            genomic_profile=genomic_profiles.get("Peter", GenomicProfile(
                sequencing_depth=30.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(30.0, 0.85),
            )),
            medical_history=medical_histories.get("Peter", []),
            patient_overrides={"structural_drag_override": 1.35},
            intervention_overrides={"nr_dose": 1.0},
        ),
        FamilyMember(
            name="Jasper",
            age=24,
            sex="M",
            relationship="grandchild",
            genomic_profile=genomic_profiles.get("Jasper", GenomicProfile(
                sequencing_depth=30.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(30.0, 0.85),
            )),
            medical_history=medical_histories.get("Jasper", []),
            patient_overrides={"structural_drag_override": 1.45},
            intervention_overrides={"nr_dose": 1.0},
        ),
        FamilyMember(
            name="Ratio",
            age=23,
            sex="M",
            relationship="grandchild",
            genomic_profile=genomic_profiles.get("Ratio", GenomicProfile(
                sequencing_depth=30.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(30.0, 0.85),
            )),
            medical_history=medical_histories.get("Ratio", []),
            patient_overrides={"structural_drag_override": 1.15, "seizure_vulnerability": 0.8},
            intervention_overrides={"magnesium_dose": 1.0},
        ),
        FamilyMember(
            name="Selena Shea",
            age=26,
            sex="F",
            relationship="grandchild",
            genomic_profile=genomic_profiles.get("Selena Shea", GenomicProfile(
                sequencing_depth=30.0,
                het_call_confidence=WGS_HET_CONFIDENCE.get(30.0, 0.85),
            )),
            medical_history=medical_histories.get("Selena Shea", []),
            patient_overrides={"structural_drag_override": 1.30},
            intervention_overrides={"nr_dose": 1.0},
        ),
    ]

    return FamilyPedigree(members)
