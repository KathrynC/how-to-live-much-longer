"""Tests for family_genomics module — WGS-informed priors for mitochondrial aging."""
import pytest
import numpy as np


# ── TestGenomicProfile ─────────────────────────────────────────────────────

class TestGenomicProfile:

    def test_default_values(self):
        from family_genomics import GenomicProfile
        gp = GenomicProfile()
        assert gp.haplogroup == "unknown"
        assert gp.mtdna_deletion_burden == 0.15
        assert gp.mtdna_point_mutations == 0.10
        assert gp.apoe_genotype == 0
        assert gp.cyp3a4_status == "normal"

    def test_deep_sequencing_confidence(self):
        from family_genomics import GenomicProfile
        gp = GenomicProfile(sequencing_depth=100.0, het_call_confidence=0.95)
        assert gp.het_call_confidence == 0.95
        assert gp.sequencing_depth == 100.0

    def test_haplogroup_assignment(self):
        from family_genomics import GenomicProfile
        gp = GenomicProfile(haplogroup="H2a")
        assert gp.haplogroup == "H2a"


# ── TestFamilyMember ───────────────────────────────────────────────────────

class TestFamilyMember:

    def test_construction_with_genomic_profile(self):
        from family_genomics import FamilyMember, GenomicProfile
        gp = GenomicProfile(haplogroup="J1c", apoe_genotype=1)
        fm = FamilyMember(name="Test", age=65, sex="F", relationship="proband",
                          genomic_profile=gp)
        assert fm.genomic_profile.haplogroup == "J1c"
        assert fm.genomic_profile.apoe_genotype == 1

    def test_construction_without_genomic_profile(self):
        from family_genomics import FamilyMember
        fm = FamilyMember(name="Test", age=65, sex="F", relationship="proband")
        assert fm.genomic_profile is None
        assert fm.medical_history == []

    def test_medical_history_entries(self):
        from family_genomics import FamilyMember, MedicalHistoryEntry
        entries = [
            MedicalHistoryEntry(age_at_measurement=60.0, variable="hs_crp",
                                value=1.5, unit="mg/L"),
            MedicalHistoryEntry(age_at_measurement=62.0, variable="glucose",
                                value=95.0, unit="mg/dL"),
        ]
        fm = FamilyMember(name="Test", age=65, sex="F", relationship="proband",
                          medical_history=entries)
        assert len(fm.medical_history) == 2
        assert fm.medical_history[0].variable == "hs_crp"


# ── TestFamilyPedigree ─────────────────────────────────────────────────────

class TestFamilyPedigree:

    @pytest.fixture
    def pedigree(self):
        from family_genomics import build_cramer_family
        return build_cramer_family()

    def test_build_cramer_family_returns_7_members(self, pedigree):
        assert len(pedigree.members) == 7

    def test_all_expected_members_present(self, pedigree):
        expected = {"John Jr.", "John III", "Kathryn", "Peter",
                    "Jasper", "Ratio", "Selena Shea"}
        assert set(pedigree.members.keys()) == expected

    def test_shared_nuclear_fraction_parent_child(self, pedigree):
        assert pedigree.shared_nuclear_fraction("Kathryn", "Peter") == 0.50
        assert pedigree.shared_nuclear_fraction("John III", "Jasper") == 0.50

    def test_shared_nuclear_fraction_grandparent_grandchild(self, pedigree):
        assert pedigree.shared_nuclear_fraction("John Jr.", "Peter") == 0.25
        assert pedigree.shared_nuclear_fraction("John Jr.", "Selena Shea") == 0.25

    def test_shared_nuclear_fraction_spouse(self, pedigree):
        assert pedigree.shared_nuclear_fraction("John III", "Kathryn") == 0.0

    def test_shared_nuclear_fraction_sibling(self, pedigree):
        assert pedigree.shared_nuclear_fraction("Peter", "Jasper") == 0.50
        assert pedigree.shared_nuclear_fraction("Ratio", "Selena Shea") == 0.50

    def test_shared_nuclear_fraction_self(self, pedigree):
        assert pedigree.shared_nuclear_fraction("Kathryn", "Kathryn") == 1.0

    def test_shared_mtdna_kathryn_children(self, pedigree):
        """Kathryn's children share her mtDNA."""
        assert pedigree.shared_mtdna("Kathryn", "Peter") is True
        assert pedigree.shared_mtdna("Kathryn", "Jasper") is True
        assert pedigree.shared_mtdna("Peter", "Ratio") is True

    def test_shared_mtdna_john_jr_not_with_john_iii(self, pedigree):
        """John Jr. doesn't share mtDNA with John III (different maternal lines)."""
        assert pedigree.shared_mtdna("John Jr.", "John III") is False

    def test_shared_mtdna_john_jr_not_with_grandchildren(self, pedigree):
        """John Jr. doesn't share mtDNA with grandchildren (paternal line)."""
        assert pedigree.shared_mtdna("John Jr.", "Peter") is False

    def test_relatives_with_allele_apoe(self, pedigree):
        """Kathryn is the only APOE4 het carrier in defaults."""
        carriers = pedigree.relatives_with_allele("apoe_genotype", 1)
        names = {m.name for m in carriers}
        assert "Kathryn" in names

    def test_members_by_age(self, pedigree):
        ordered = pedigree.members_by_age()
        ages = [m.age for m in ordered]
        assert ages == sorted(ages, reverse=True)
        assert ordered[0].name == "John Jr."

    def test_get_member_raises_on_unknown(self, pedigree):
        with pytest.raises(KeyError):
            pedigree.get_member("Unknown Person")


# ── TestComputeFamilyPriors ────────────────────────────────────────────────

class TestComputeFamilyPriors:

    @pytest.fixture
    def pedigree(self):
        from family_genomics import build_cramer_family
        return build_cramer_family()

    def test_with_wgs_data_uses_measured_heteroplasmy(self, pedigree):
        from family_genomics import compute_family_priors
        john_jr = pedigree.get_member("John Jr.")
        priors = compute_family_priors(john_jr, pedigree)
        # John Jr. has genomic profile with mtdna_deletion_burden=0.45
        assert priors['n_deletion'] == 0.45
        assert priors['n_point'] == 0.20

    def test_without_wgs_data_falls_back(self):
        from family_genomics import (
            FamilyMember, FamilyPedigree, compute_family_priors,
        )
        from constants import SENSOR_PRIOR_N_HEALTHY, SENSOR_PRIOR_N_DELETION
        member = FamilyMember(name="NoProfie", age=50, sex="M",
                              relationship="proband")
        ped = FamilyPedigree([member])
        priors = compute_family_priors(member, ped)
        assert priors['n_healthy'] == SENSOR_PRIOR_N_HEALTHY
        assert priors['n_deletion'] == SENSOR_PRIOR_N_DELETION

    def test_haplogroup_vulnerability_mapping(self):
        from family_genomics import (
            FamilyMember, FamilyPedigree, GenomicProfile, compute_family_priors,
        )
        gp = GenomicProfile(haplogroup="J1", apoe_genotype=0)
        member = FamilyMember(name="Test", age=50, sex="M",
                              relationship="proband", genomic_profile=gp)
        ped = FamilyPedigree([member])
        priors = compute_family_priors(member, ped)
        # J haplogroup has vulnerability 0.85
        assert priors['genetic_vulnerability'] == pytest.approx(0.85)

    def test_prior_confidence_reflects_depth(self, pedigree):
        from family_genomics import compute_family_priors
        john_jr = pedigree.get_member("John Jr.")
        priors = compute_family_priors(john_jr, pedigree)
        # 100x sequencing → het_call_confidence=0.95
        assert priors['prior_confidence'] >= 0.90

    def test_priors_for_john_jr(self, pedigree):
        from family_genomics import compute_family_priors
        john_jr = pedigree.get_member("John Jr.")
        priors = compute_family_priors(john_jr, pedigree)
        # John Jr. has baseline_heteroplasmy=0.65 in family_ecosystem_report
        # WGS profile: mtdna_deletion_burden=0.45, mtdna_point_mutations=0.20
        assert priors['n_deletion'] == 0.45
        assert priors['n_point'] == 0.20
        # n_healthy = 1.0 - 0.45 - 0.20 = 0.35
        assert priors['n_healthy'] == pytest.approx(0.35)

    def test_priors_contain_expected_keys(self, pedigree):
        from family_genomics import compute_family_priors
        member = pedigree.get_member("Kathryn")
        priors = compute_family_priors(member, pedigree)
        for key in ('n_healthy', 'n_deletion', 'n_point', 'membrane_potential',
                     'genetic_vulnerability', 'prior_confidence'):
            assert key in priors


# ── TestComputeWgsGeneticModifiers ─────────────────────────────────────────

class TestComputeWgsGeneticModifiers:

    def test_output_shape_matches_genetic_modifiers(self):
        from family_genomics import GenomicProfile, compute_wgs_genetic_modifiers
        from genetics_module import compute_genetic_modifiers
        gp = GenomicProfile(apoe_genotype=1, foxo3_protective=0, cd38_risk=0)
        wgs_mods = compute_wgs_genetic_modifiers(gp)
        base_mods = compute_genetic_modifiers(apoe_genotype=1)
        # WGS mods should have all keys from base + PGx keys
        for key in base_mods:
            assert key in wgs_mods
        assert 'cyp3a4_multiplier' in wgs_mods
        assert 'cyp2d6_multiplier' in wgs_mods

    def test_pgx_poor_metabolizer(self):
        from family_genomics import GenomicProfile, compute_wgs_genetic_modifiers
        gp = GenomicProfile(cyp3a4_status="poor", cyp2d6_status="poor")
        mods = compute_wgs_genetic_modifiers(gp)
        assert mods['cyp3a4_multiplier'] == 0.5
        assert mods['cyp2d6_multiplier'] == 0.6

    def test_pgx_normal_metabolizer(self):
        from family_genomics import GenomicProfile, compute_wgs_genetic_modifiers
        gp = GenomicProfile()
        mods = compute_wgs_genetic_modifiers(gp)
        assert mods['cyp3a4_multiplier'] == 1.0
        assert mods['cyp2d6_multiplier'] == 1.0

    def test_haplogroup_affects_vulnerability(self):
        from family_genomics import GenomicProfile, compute_wgs_genetic_modifiers
        gp_j = GenomicProfile(haplogroup="J")
        gp_i = GenomicProfile(haplogroup="I")
        mods_j = compute_wgs_genetic_modifiers(gp_j)
        mods_i = compute_wgs_genetic_modifiers(gp_i)
        # J (0.85) should have lower vulnerability than I (1.10)
        assert mods_j['vulnerability'] < mods_i['vulnerability']


# ── TestCalibrateFromRelatives ─────────────────────────────────────────────

class TestCalibrateFromRelatives:

    def test_with_mock_medical_histories(self):
        from family_genomics import (
            FamilyMember, FamilyPedigree, GenomicProfile,
            MedicalHistoryEntry, calibrate_from_relatives,
        )
        parent = FamilyMember(
            name="Parent", age=65, sex="F", relationship="parent",
            genomic_profile=GenomicProfile(apoe_genotype=1),
            medical_history=[
                MedicalHistoryEntry(age_at_measurement=60, variable="heteroplasmy",
                                    value=0.40),
            ],
        )
        child = FamilyMember(
            name="Child", age=35, sex="M", relationship="child",
            genomic_profile=GenomicProfile(apoe_genotype=1),
        )
        # Need to register kinship — use a custom pedigree
        ped = FamilyPedigree([parent, child])
        cal = calibrate_from_relatives(child, ped)
        # No registered kinship between "Parent" and "Child" in the default
        # Cramer kinship map, so confidence should be 0
        assert cal['confidence'] == 0.0
        assert cal['vulnerability_adjustment'] == 1.0

    def test_no_relatives_neutral_adjustment(self):
        from family_genomics import (
            FamilyMember, FamilyPedigree, GenomicProfile, calibrate_from_relatives,
        )
        solo = FamilyMember(
            name="Solo", age=50, sex="M", relationship="proband",
            genomic_profile=GenomicProfile(),
        )
        ped = FamilyPedigree([solo])
        cal = calibrate_from_relatives(solo, ped)
        assert cal['vulnerability_adjustment'] == 1.0
        assert cal['progression_rate_factor'] == 1.0
        assert cal['confidence'] == 0.0

    def test_no_genomic_profile_returns_neutral(self):
        from family_genomics import (
            FamilyMember, FamilyPedigree, calibrate_from_relatives,
        )
        member = FamilyMember(name="None", age=50, sex="M",
                              relationship="proband")
        ped = FamilyPedigree([member])
        cal = calibrate_from_relatives(member, ped)
        assert cal['confidence'] == 0.0


# ── TestEstimateStateWithFamilyPriors ──────────────────────────────────────

class TestEstimateStateWithFamilyPriors:

    @pytest.fixture
    def obs_model(self):
        from wearable_sensors import WearableObservationModel
        return WearableObservationModel(seed=42)

    @pytest.fixture
    def healthy_state(self):
        return np.array([0.90, 0.05, 0.95, 0.10, 0.90, 0.02, 0.95, 0.03])

    def test_family_priors_none_gives_identical(self, obs_model, healthy_state):
        """family_priors=None gives identical results to current behavior."""
        readings = obs_model.observe(healthy_state, t=0.0)
        est_default = obs_model.estimate_state(readings)
        est_none = obs_model.estimate_state(readings, family_priors=None)
        np.testing.assert_array_equal(est_default, est_none)

    def test_family_priors_override_unobservable(self, obs_model, healthy_state):
        """Family priors should override est[0,1,6,7]."""
        readings = obs_model.observe(healthy_state, t=0.0)
        priors = {
            'n_healthy': 0.60,
            'n_deletion': 0.30,
            'n_point': 0.10,
            'membrane_potential': 0.80,
        }
        est = obs_model.estimate_state(readings, family_priors=priors)
        assert est[0] == 0.60  # N_healthy
        assert est[1] == 0.30  # N_deletion
        assert est[6] == 0.80  # ΔΨ
        assert est[7] == 0.10  # N_point

    def test_observable_vars_still_from_sensors(self, obs_model, healthy_state):
        """Observable variables (est[2-5]) should still come from sensors."""
        from constants import BASELINE_ATP
        readings = obs_model.observe(healthy_state, t=0.0)
        priors = {'n_healthy': 0.60, 'n_deletion': 0.30}
        est_priors = obs_model.estimate_state(readings, family_priors=priors)
        est_default = obs_model.estimate_state(readings)
        # ATP estimate should be the same regardless of family priors
        assert est_priors[2] == est_default[2]
        # ROS estimate should be the same
        assert est_priors[3] == est_default[3]


# ── TestScenarioG ──────────────────────────────────────────────────────────

class TestScenarioG:

    def test_scenario_g_exists(self):
        from scenario_definitions import get_example_scenarios
        scenarios = get_example_scenarios()
        assert len(scenarios) == 7
        assert scenarios[6].name.startswith("G")

    def test_scenario_g_has_family_priors(self):
        from scenario_definitions import get_example_scenarios
        scenarios = get_example_scenarios()
        g_config = scenarios[6].sensor_config
        assert g_config is not None
        assert 'family_priors' in g_config
        priors = g_config['family_priors']
        assert 'n_healthy' in priors
        assert 'n_deletion' in priors
        assert 'n_point' in priors

    def test_scenario_g_has_5_devices(self):
        from scenario_definitions import get_example_scenarios
        scenarios = get_example_scenarios()
        g_config = scenarios[6].sensor_config
        assert len(g_config['devices']) == 5

    def test_scenario_g_runs_without_error(self):
        from scenario_definitions import get_example_scenarios
        from scenario_runner import run_scenario
        scenarios = get_example_scenarios()
        result = run_scenario(scenarios[6], years=5)
        assert 'core' in result
        assert 'downstream' in result
