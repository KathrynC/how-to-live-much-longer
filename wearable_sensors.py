"""Wearable sensor observation model for the mitochondrial aging simulator.

Formalizes the Phase 10 prototype (telemetry_bridge_local.py) into a proper
observation layer. Three consumer wearable devices — Apple Watch Series 11,
Oura Ring 4, Dexcom Stelo CGM — provide noisy, partial observations of the
8D hidden ODE state vector.

Evolutionary Robotics connection: without sensors (proprioception), robots
suffer "sensory death" and freeze. Closed-loop control with sensors
outperforms open-loop by 84%. The bottleneck is sensor-motor bandwidth.
This module implements the analogous observation layer for the mitochondrial
aging controller.

Three layers:
    A. Device specifications (dataclasses)
    B. Observation functions (hidden state -> noisy sensor readings)
    C. Unified WearableObservationModel class

Completely unobservable variables (population-average priors only):
    - N_healthy, N_deletion, N_point — require tissue biopsy + deep sequencing
    - Membrane_potential — only indirectly proxied via SpO2 (noisy,
      multi-cause: senescence and ROS also affect SpO2 independently of ΔΨ)

Reference:
    Cramer, J.G. (forthcoming 2026). *How to Live Much Longer: The
    Mitochondrial DNA Connection*. Springer. ISBN 978-3-032-17740-7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from constants import (
    # Sensor baselines
    SENSOR_BASELINE_HR, SENSOR_BASELINE_HRV, SENSOR_BASELINE_SPO2,
    SENSOR_BASELINE_TEMP, SENSOR_BASELINE_SYS_BP, SENSOR_BASELINE_DIA_BP,
    SENSOR_BASELINE_GLUCOSE, SENSOR_BASELINE_RESP_RATE, SENSOR_BASELINE_ACTIVITY,
    SENSOR_BASELINE_LACTATE, SENSOR_BASELINE_HS_CRP, SENSOR_BASELINE_GDF15,
    SENSOR_BASELINE_NAD_BLOOD, SENSOR_BASELINE_8OHDG,
    # Sensitivity coefficients
    SENSOR_HR_ATP_SENSITIVITY, SENSOR_HR_ROS_SENSITIVITY, SENSOR_HR_AGE_DRIFT,
    SENSOR_HRV_ATP_SENSITIVITY, SENSOR_HRV_ROS_SENSITIVITY,
    SENSOR_HRV_SEN_SENSITIVITY, SENSOR_HRV_AGE_DECLINE,
    SENSOR_SPO2_PSI_SENSITIVITY, SENSOR_SPO2_SEN_SENSITIVITY,
    SENSOR_TEMP_ROS_SENSITIVITY, SENSOR_TEMP_SEN_SENSITIVITY,
    SENSOR_TEMP_PSI_SENSITIVITY,
    SENSOR_BP_SEN_SENSITIVITY, SENSOR_BP_ROS_SENSITIVITY, SENSOR_BP_AGE_DRIFT,
    SENSOR_GLUCOSE_NAD_SENSITIVITY, SENSOR_GLUCOSE_ATP_SENSITIVITY,
    SENSOR_RESP_ATP_SENSITIVITY, SENSOR_RESP_ROS_SENSITIVITY,
    SENSOR_ACTIVITY_ATP_SCALING,
    SENSOR_LACTATE_ATP_SENSITIVITY, SENSOR_LACTATE_ROS_SENSITIVITY,
    SENSOR_LACTATE_SEN_SENSITIVITY, SENSOR_LACTATE_EXERCISE_BOOST,
    SENSOR_HS_CRP_SEN_SENSITIVITY, SENSOR_HS_CRP_ROS_SENSITIVITY,
    SENSOR_GDF15_ATP_SENSITIVITY, SENSOR_GDF15_ROS_SENSITIVITY,
    SENSOR_GDF15_NAD_SENSITIVITY, SENSOR_GDF15_SEN_SENSITIVITY,
    SENSOR_NAD_BLOOD_SENSITIVITY,
    SENSOR_8OHDG_ROS_SENSITIVITY,
    # Noise
    SENSOR_NOISE_APPLE_HR, SENSOR_NOISE_APPLE_HRV, SENSOR_NOISE_APPLE_SPO2,
    SENSOR_NOISE_APPLE_TEMP, SENSOR_NOISE_APPLE_BP, SENSOR_NOISE_APPLE_ACTIVITY,
    SENSOR_NOISE_OURA_HR, SENSOR_NOISE_OURA_HRV, SENSOR_NOISE_OURA_SPO2,
    SENSOR_NOISE_OURA_TEMP, SENSOR_NOISE_OURA_RESP,
    SENSOR_NOISE_DEXCOM_GLUCOSE,
    SENSOR_NOISE_LINGO_LACTATE,
    SENSOR_NOISE_HS_CRP, SENSOR_NOISE_GDF15, SENSOR_NOISE_NAD_BLOOD,
    SENSOR_NOISE_8OHDG,
    # CGM / Lingo cycle
    CGM_ACTIVE_DAYS, CGM_CHANGEOVER_DAYS, CGM_CYCLE_DAYS,
    LINGO_ACTIVE_DAYS, LINGO_CHANGEOVER_DAYS, LINGO_CYCLE_DAYS,
    # Biomarker interval
    BIOMARKER_INTERVAL_DAYS,
    # Ranges
    SENSOR_RANGE_HR, SENSOR_RANGE_HRV, SENSOR_RANGE_SPO2,
    SENSOR_RANGE_TEMP, SENSOR_RANGE_SYS_BP, SENSOR_RANGE_DIA_BP,
    SENSOR_RANGE_GLUCOSE, SENSOR_RANGE_RESP_RATE, SENSOR_RANGE_ACTIVITY,
    SENSOR_RANGE_LACTATE, SENSOR_RANGE_HS_CRP, SENSOR_RANGE_GDF15,
    SENSOR_RANGE_NAD_BLOOD, SENSOR_RANGE_8OHDG,
    # Priors
    SENSOR_PRIOR_N_HEALTHY, SENSOR_PRIOR_N_DELETION, SENSOR_PRIOR_N_POINT,
    SENSOR_PRIOR_MEMBRANE_POTENTIAL,
    # State info
    BASELINE_ATP, BASELINE_ROS, BASELINE_NAD,
    BASELINE_SENESCENT, BASELINE_MEMBRANE_POTENTIAL,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer A — Device Specifications
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorChannel:
    """A single sensor channel on a wearable device."""
    name: str               # e.g. "heart_rate"
    unit: str               # e.g. "bpm"
    range_min: float        # hardware minimum
    range_max: float        # hardware maximum
    noise_std: float        # Gaussian noise standard deviation
    sampling_interval: float  # seconds between samples
    availability: float     # fraction of time channel produces valid data (0-1)


@dataclass
class DeviceSpec:
    """Specification for a wearable device."""
    name: str
    channels: Dict[str, SensorChannel] = field(default_factory=dict)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels.keys())

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def apple_watch_11_spec() -> DeviceSpec:
    """Apple Watch Series 11 (2025) sensor specification."""
    return DeviceSpec(
        name="Apple Watch Series 11",
        channels={
            "heart_rate": SensorChannel(
                "heart_rate", "bpm",
                SENSOR_RANGE_HR[0], SENSOR_RANGE_HR[1],
                SENSOR_NOISE_APPLE_HR, 5.0, 0.98,
            ),
            "hrv": SensorChannel(
                "hrv", "ms_rmssd",
                SENSOR_RANGE_HRV[0], SENSOR_RANGE_HRV[1],
                SENSOR_NOISE_APPLE_HRV, 300.0, 0.95,
            ),
            "spo2": SensorChannel(
                "spo2", "percent",
                SENSOR_RANGE_SPO2[0], SENSOR_RANGE_SPO2[1],
                SENSOR_NOISE_APPLE_SPO2, 60.0, 0.90,
            ),
            "temperature": SensorChannel(
                "temperature", "celsius",
                SENSOR_RANGE_TEMP[0], SENSOR_RANGE_TEMP[1],
                SENSOR_NOISE_APPLE_TEMP, 60.0, 0.95,
            ),
            "blood_pressure_sys": SensorChannel(
                "blood_pressure_sys", "mmHg",
                SENSOR_RANGE_SYS_BP[0], SENSOR_RANGE_SYS_BP[1],
                SENSOR_NOISE_APPLE_BP, 3600.0, 0.85,
            ),
            "blood_pressure_dia": SensorChannel(
                "blood_pressure_dia", "mmHg",
                SENSOR_RANGE_DIA_BP[0], SENSOR_RANGE_DIA_BP[1],
                SENSOR_NOISE_APPLE_BP, 3600.0, 0.85,
            ),
            "activity": SensorChannel(
                "activity", "g",
                SENSOR_RANGE_ACTIVITY[0], SENSOR_RANGE_ACTIVITY[1],
                SENSOR_NOISE_APPLE_ACTIVITY, 0.1, 0.99,
            ),
        },
    )


def oura_ring_4_spec() -> DeviceSpec:
    """Oura Ring 4 (2024) sensor specification."""
    return DeviceSpec(
        name="Oura Ring 4",
        channels={
            "heart_rate": SensorChannel(
                "heart_rate", "bpm",
                SENSOR_RANGE_HR[0], SENSOR_RANGE_HR[1],
                SENSOR_NOISE_OURA_HR, 5.0, 0.97,
            ),
            "hrv": SensorChannel(
                "hrv", "ms_rmssd",
                SENSOR_RANGE_HRV[0], SENSOR_RANGE_HRV[1],
                SENSOR_NOISE_OURA_HRV, 300.0, 0.95,
            ),
            "spo2": SensorChannel(
                "spo2", "percent",
                SENSOR_RANGE_SPO2[0], SENSOR_RANGE_SPO2[1],
                SENSOR_NOISE_OURA_SPO2, 60.0, 0.88,
            ),
            "temperature": SensorChannel(
                "temperature", "celsius",
                SENSOR_RANGE_TEMP[0], SENSOR_RANGE_TEMP[1],
                SENSOR_NOISE_OURA_TEMP, 60.0, 0.97,
            ),
            "respiratory_rate": SensorChannel(
                "respiratory_rate", "brpm",
                SENSOR_RANGE_RESP_RATE[0], SENSOR_RANGE_RESP_RATE[1],
                SENSOR_NOISE_OURA_RESP, 60.0, 0.92,
            ),
        },
    )


def dexcom_stelo_spec() -> DeviceSpec:
    """Dexcom Stelo CGM (2024) sensor specification."""
    return DeviceSpec(
        name="Dexcom Stelo",
        channels={
            "glucose": SensorChannel(
                "glucose", "mg/dL",
                SENSOR_RANGE_GLUCOSE[0], SENSOR_RANGE_GLUCOSE[1],
                SENSOR_NOISE_DEXCOM_GLUCOSE, 300.0, 1.0,
                # availability handled by CGM wear cycle, not this field
            ),
        },
    )


def abbott_lingo_spec() -> DeviceSpec:
    """Abbott Lingo continuous lactate monitor specification."""
    return DeviceSpec(
        name="Abbott Lingo",
        channels={
            "lactate": SensorChannel(
                "lactate", "mmol/L",
                SENSOR_RANGE_LACTATE[0], SENSOR_RANGE_LACTATE[1],
                SENSOR_NOISE_LINGO_LACTATE, 300.0, 1.0,
                # availability handled by Lingo wear cycle, not this field
            ),
        },
    )


def periodic_biomarker_spec() -> DeviceSpec:
    """Periodic blood biomarker panel (quarterly venous draw)."""
    return DeviceSpec(
        name="Periodic Biomarker Panel",
        channels={
            "hs_crp": SensorChannel(
                "hs_crp", "mg/L",
                SENSOR_RANGE_HS_CRP[0], SENSOR_RANGE_HS_CRP[1],
                SENSOR_NOISE_HS_CRP, 86400.0, 1.0,
                # availability handled by quarterly schedule, not this field
            ),
            "gdf15": SensorChannel(
                "gdf15", "pg/mL",
                SENSOR_RANGE_GDF15[0], SENSOR_RANGE_GDF15[1],
                SENSOR_NOISE_GDF15, 86400.0, 1.0,
            ),
            "nad_blood": SensorChannel(
                "nad_blood", "μmol/L",
                SENSOR_RANGE_NAD_BLOOD[0], SENSOR_RANGE_NAD_BLOOD[1],
                SENSOR_NOISE_NAD_BLOOD, 86400.0, 1.0,
            ),
            "ohdg_8": SensorChannel(
                "ohdg_8", "ng/mL",
                SENSOR_RANGE_8OHDG[0], SENSOR_RANGE_8OHDG[1],
                SENSOR_NOISE_8OHDG, 86400.0, 1.0,
            ),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Layer B — Observation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def observe_heart_rate(
    atp: float, ros: float, age: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_APPLE_HR,
) -> float:
    """Map ODE state to heart rate reading.

    Biology: low ATP → chronotropic compensation (heart beats faster to
    deliver more O2); elevated ROS → sympathetic nervous system activation.
    """
    atp_deficit = max(0.0, BASELINE_ATP - atp)
    ros_excess = max(0.0, ros - BASELINE_ROS)
    age_effect = max(0.0, age - 40.0) * SENSOR_HR_AGE_DRIFT

    hr = (SENSOR_BASELINE_HR
          + SENSOR_HR_ATP_SENSITIVITY * atp_deficit
          + SENSOR_HR_ROS_SENSITIVITY * ros_excess
          + age_effect)
    hr += rng.normal(0.0, noise_std)
    return _clamp(hr, *SENSOR_RANGE_HR)


def observe_hrv(
    atp: float, ros: float, sen: float, age: float,
    rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_APPLE_HRV,
) -> float:
    """Map ODE state to HRV (RMSSD) reading.

    Biology: vagal tone requires ATP for parasympathetic signaling;
    SASP from senescent cells suppresses parasympathetic drive.
    """
    age_effect = max(0.0, age - 40.0) * SENSOR_HRV_AGE_DECLINE

    hrv = (SENSOR_BASELINE_HRV
           + SENSOR_HRV_ATP_SENSITIVITY * (atp - BASELINE_ATP)
           + SENSOR_HRV_ROS_SENSITIVITY * max(0.0, ros - BASELINE_ROS)
           + SENSOR_HRV_SEN_SENSITIVITY * sen
           + age_effect)
    hrv += rng.normal(0.0, noise_std)
    return _clamp(hrv, *SENSOR_RANGE_HRV)


def observe_spo2(
    psi: float, sen: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_APPLE_SPO2,
) -> float:
    """Map ODE state to SpO2 reading.

    Biology: low membrane potential → ETC dysfunction → reduced O2
    utilization efficiency; senescence → tissue-level oxygen extraction
    impairment.
    """
    spo2 = (SENSOR_BASELINE_SPO2
            + SENSOR_SPO2_PSI_SENSITIVITY * (psi - BASELINE_MEMBRANE_POTENTIAL)
            + SENSOR_SPO2_SEN_SENSITIVITY * sen)
    spo2 += rng.normal(0.0, noise_std)
    return _clamp(spo2, *SENSOR_RANGE_SPO2)


def observe_temperature(
    ros: float, sen: float, psi: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_APPLE_TEMP,
) -> float:
    """Map ODE state to body temperature reading.

    Biology: SASP→IL-6→thermogenesis; ROS→inflammatory heat;
    low ΔΨ → uncoupled respiration → waste heat.
    """
    temp = (SENSOR_BASELINE_TEMP
            + SENSOR_TEMP_ROS_SENSITIVITY * max(0.0, ros - BASELINE_ROS)
            + SENSOR_TEMP_SEN_SENSITIVITY * sen
            + SENSOR_TEMP_PSI_SENSITIVITY * max(0.0, BASELINE_MEMBRANE_POTENTIAL - psi))
    temp += rng.normal(0.0, noise_std)
    return _clamp(temp, *SENSOR_RANGE_TEMP)


def observe_blood_pressure(
    sen: float, ros: float, age: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_APPLE_BP,
) -> Tuple[float, float]:
    """Map ODE state to systolic/diastolic blood pressure.

    Biology: endothelial senescence → reduced NO bioavailability →
    vasoconstriction; ROS → vascular inflammation.
    """
    age_effect = max(0.0, age - 40.0) * SENSOR_BP_AGE_DRIFT

    sys_bp = (SENSOR_BASELINE_SYS_BP
              + SENSOR_BP_SEN_SENSITIVITY * sen
              + SENSOR_BP_ROS_SENSITIVITY * max(0.0, ros - BASELINE_ROS)
              + age_effect)
    sys_bp += rng.normal(0.0, noise_std)

    # Diastolic tracks systolic with ~60% of deviations
    dia_delta = (sys_bp - SENSOR_BASELINE_SYS_BP) * 0.6
    dia_bp = SENSOR_BASELINE_DIA_BP + dia_delta
    dia_bp += rng.normal(0.0, noise_std * 0.7)

    return (
        _clamp(sys_bp, *SENSOR_RANGE_SYS_BP),
        _clamp(dia_bp, *SENSOR_RANGE_DIA_BP),
    )


def observe_glucose(
    nad: float, atp: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_DEXCOM_GLUCOSE,
) -> float:
    """Map ODE state to blood glucose reading.

    Biology: NAD→SIRT1→insulin sensitivity (higher NAD = better glucose
    control); ATP→pancreatic beta-cell secretion capacity.
    Cramer Ch. VI.A.3 pp.72-73: NAD+ supplementation effects on metabolic
    signaling pathways; SIRT1 activation improves insulin sensitivity.
    """
    glucose = (SENSOR_BASELINE_GLUCOSE
               + SENSOR_GLUCOSE_NAD_SENSITIVITY * (nad - BASELINE_NAD)
               + SENSOR_GLUCOSE_ATP_SENSITIVITY * (atp - BASELINE_ATP))
    glucose += rng.normal(0.0, noise_std)
    return _clamp(glucose, *SENSOR_RANGE_GLUCOSE)


def observe_respiratory_rate(
    atp: float, ros: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_OURA_RESP,
) -> float:
    """Map ODE state to respiratory rate reading.

    Biology: compensatory hyperventilation for ATP deficit;
    ROS → mild respiratory drive increase.
    """
    atp_deficit = max(0.0, BASELINE_ATP - atp)
    ros_excess = max(0.0, ros - BASELINE_ROS)

    resp = (SENSOR_BASELINE_RESP_RATE
            + SENSOR_RESP_ATP_SENSITIVITY * atp_deficit
            + SENSOR_RESP_ROS_SENSITIVITY * ros_excess)
    resp += rng.normal(0.0, noise_std)
    return _clamp(resp, *SENSOR_RANGE_RESP_RATE)


def observe_activity(
    exercise_level: float, atp: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_APPLE_ACTIVITY,
) -> float:
    """Map exercise intention × ATP capacity to accelerometer reading.

    Biology: fatigue from low ATP limits the ability to execute intended
    exercise. Activity = exercise_level × ATP scaling.
    """
    activity = exercise_level * SENSOR_ACTIVITY_ATP_SCALING * atp
    activity += rng.normal(0.0, noise_std)
    return _clamp(activity, *SENSOR_RANGE_ACTIVITY)


def observe_lactate(
    atp: float, ros: float, sen: float, exercise_level: float,
    rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_LINGO_LACTATE,
) -> float:
    """Map ODE state to interstitial lactate reading.

    Biology: low ATP → glycolytic compensation → lactate rises;
    exercise → direct lactate production; ROS and senescence contribute
    to mitochondrial dysfunction driving glycolytic overflow.
    Cramer Ch. VIII.A p.100: ATP deficit forces glycolytic overdrive.
    """
    ros_excess = max(0.0, ros - BASELINE_ROS)

    lactate = (SENSOR_BASELINE_LACTATE
               + SENSOR_LACTATE_ATP_SENSITIVITY * (atp - BASELINE_ATP)
               + SENSOR_LACTATE_ROS_SENSITIVITY * ros_excess
               + SENSOR_LACTATE_SEN_SENSITIVITY * sen
               + SENSOR_LACTATE_EXERCISE_BOOST * exercise_level)
    lactate += rng.normal(0.0, noise_std)
    return _clamp(lactate, *SENSOR_RANGE_LACTATE)


def observe_hs_crp(
    sen: float, ros: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_HS_CRP,
) -> float:
    """Map ODE state to high-sensitivity CRP reading.

    Biology: SASP→IL-6→hepatic CRP synthesis; ROS→inflammatory signaling.
    Cramer Ch. VII.A pp.89-92.
    """
    ros_excess = max(0.0, ros - BASELINE_ROS)

    crp = (SENSOR_BASELINE_HS_CRP
           + SENSOR_HS_CRP_SEN_SENSITIVITY * sen
           + SENSOR_HS_CRP_ROS_SENSITIVITY * ros_excess)
    crp += rng.normal(0.0, noise_std)
    return _clamp(crp, *SENSOR_RANGE_HS_CRP)


def observe_gdf15(
    atp: float, ros: float, nad: float, sen: float,
    rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_GDF15,
) -> float:
    """Map ODE state to GDF-15 reading.

    Biology: mitochondrial stress integrator — elevated by ATP deficit,
    ROS excess, NAD deficit, and senescence. Most information-rich
    single biomarker. Literature: Wiklund et al. 2010.
    """
    ros_excess = max(0.0, ros - BASELINE_ROS)

    gdf15 = (SENSOR_BASELINE_GDF15
             + SENSOR_GDF15_ATP_SENSITIVITY * (atp - BASELINE_ATP)
             + SENSOR_GDF15_ROS_SENSITIVITY * ros_excess
             + SENSOR_GDF15_NAD_SENSITIVITY * (nad - BASELINE_NAD)
             + SENSOR_GDF15_SEN_SENSITIVITY * sen)
    gdf15 += rng.normal(0.0, noise_std)
    return _clamp(gdf15, *SENSOR_RANGE_GDF15)


def observe_nad_blood(
    nad: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_NAD_BLOOD,
) -> float:
    """Map ODE state to whole-blood NAD+ reading.

    Biology: near-direct measurement of ODE state[4].
    Cramer Ch. VI.A.3 pp.72-73: NAD+ as master metabolic cofactor.
    """
    nad_blood = SENSOR_BASELINE_NAD_BLOOD + SENSOR_NAD_BLOOD_SENSITIVITY * (nad - BASELINE_NAD)
    nad_blood += rng.normal(0.0, noise_std)
    return _clamp(nad_blood, *SENSOR_RANGE_NAD_BLOOD)


def observe_8ohdg(
    ros: float, rng: np.random.Generator,
    noise_std: float = SENSOR_NOISE_8OHDG,
) -> float:
    """Map ODE state to 8-OHdG reading.

    Biology: oxidative DNA damage directly proportional to ROS exposure.
    """
    ohdg = SENSOR_BASELINE_8OHDG + SENSOR_8OHDG_ROS_SENSITIVITY * ros
    ohdg += rng.normal(0.0, noise_std)
    return _clamp(ohdg, *SENSOR_RANGE_8OHDG)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer C — Unified Observation Model
# ═══════════════════════════════════════════════════════════════════════════════

class WearableObservationModel:
    """Unified wearable sensor observation model.

    Manages device specifications, generates noisy sensor readings from
    hidden ODE state, and provides algebraic state estimation with
    information loss quantification.

    Args:
        devices: List of device names to include.
            Valid: "apple_watch", "oura_ring", "dexcom_stelo",
                   "abbott_lingo", "periodic_biomarker".
            Default: all three original wearables.
        patient_age: Patient baseline age (years). Used for age-dependent
            observation functions.
        seed: RNG seed for reproducible noise.
    """

    DEVICE_FACTORIES = {
        "apple_watch": apple_watch_11_spec,
        "oura_ring": oura_ring_4_spec,
        "dexcom_stelo": dexcom_stelo_spec,
        "abbott_lingo": abbott_lingo_spec,
        "periodic_biomarker": periodic_biomarker_spec,
    }

    def __init__(
        self,
        devices: Optional[List[str]] = None,
        patient_age: float = 63.0,
        seed: int = 42,
    ):
        if devices is None:
            devices = ["apple_watch", "oura_ring", "dexcom_stelo"]

        self.device_specs: Dict[str, DeviceSpec] = {}
        for name in devices:
            factory = self.DEVICE_FACTORIES.get(name)
            if factory is None:
                raise ValueError(
                    f"Unknown device '{name}'. "
                    f"Valid: {list(self.DEVICE_FACTORIES.keys())}"
                )
            self.device_specs[name] = factory()

        self.patient_age = patient_age
        self.rng = np.random.default_rng(seed)

    @property
    def all_channels(self) -> Dict[str, SensorChannel]:
        """All channels across all devices (last device wins on duplicates)."""
        channels = {}
        for spec in self.device_specs.values():
            channels.update(spec.channels)
        return channels

    def _cgm_available(self, t: float) -> bool:
        """Check if CGM is producing readings at simulation time t.

        Models the 15-day wear cycle: 14 days active, 1 day changeover.
        t is in years; convert to days.
        """
        if "dexcom_stelo" not in self.device_specs:
            return False
        day_in_cycle = (t * 365.25) % CGM_CYCLE_DAYS
        return day_in_cycle < CGM_ACTIVE_DAYS

    def _lingo_available(self, t: float) -> bool:
        """Check if Abbott Lingo is producing readings at simulation time t.

        Same 14-day active / 1-day changeover wear cycle as CGM.
        t is in years; convert to days.
        """
        if "abbott_lingo" not in self.device_specs:
            return False
        day_in_cycle = (t * 365.25) % LINGO_CYCLE_DAYS
        return day_in_cycle < LINGO_ACTIVE_DAYS

    def _biomarker_available(self, t: float) -> bool:
        """Check if blood biomarker panel results are available at time t.

        Returns True within a 1-day window of every 91.3 days (quarterly).
        t is in years; convert to days.
        """
        if "periodic_biomarker" not in self.device_specs:
            return False
        day = t * 365.25
        # Distance to nearest quarterly draw day
        remainder = day % BIOMARKER_INTERVAL_DAYS
        return remainder < 1.0 or (BIOMARKER_INTERVAL_DAYS - remainder) < 1.0

    def observe(
        self,
        state: np.ndarray,
        t: float,
        intervention: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Generate noisy sensor readings from hidden ODE state.

        Args:
            state: 8D ODE state vector
                [N_h, N_del, ATP, ROS, NAD, Sen, ΔΨ, N_pt]
            t: Simulation time (years from start).
            intervention: Current intervention dict (for exercise_level).

        Returns:
            Dict mapping channel names to sensor readings.
            NaN for unavailable channels (e.g., CGM during changeover).
        """
        n_h, n_del, atp, ros, nad, sen, psi, n_pt = state
        age = self.patient_age + t
        exercise = 0.0
        if intervention is not None:
            exercise = intervention.get("exercise_level", 0.0)

        readings: Dict[str, float] = {}
        all_ch = self.all_channels

        if "heart_rate" in all_ch:
            ch = all_ch["heart_rate"]
            readings["heart_rate"] = observe_heart_rate(
                atp, ros, age, self.rng, ch.noise_std)

        if "hrv" in all_ch:
            ch = all_ch["hrv"]
            readings["hrv"] = observe_hrv(
                atp, ros, sen, age, self.rng, ch.noise_std)

        if "spo2" in all_ch:
            ch = all_ch["spo2"]
            readings["spo2"] = observe_spo2(
                psi, sen, self.rng, ch.noise_std)

        if "temperature" in all_ch:
            ch = all_ch["temperature"]
            readings["temperature"] = observe_temperature(
                ros, sen, psi, self.rng, ch.noise_std)

        if "blood_pressure_sys" in all_ch:
            ch_sys = all_ch["blood_pressure_sys"]
            sys_bp, dia_bp = observe_blood_pressure(
                sen, ros, age, self.rng, ch_sys.noise_std)
            readings["blood_pressure_sys"] = sys_bp
            readings["blood_pressure_dia"] = dia_bp

        if "glucose" in all_ch:
            if self._cgm_available(t):
                ch = all_ch["glucose"]
                readings["glucose"] = observe_glucose(
                    nad, atp, self.rng, ch.noise_std)
            else:
                readings["glucose"] = float("nan")

        if "respiratory_rate" in all_ch:
            ch = all_ch["respiratory_rate"]
            readings["respiratory_rate"] = observe_respiratory_rate(
                atp, ros, self.rng, ch.noise_std)

        if "activity" in all_ch:
            ch = all_ch["activity"]
            readings["activity"] = observe_activity(
                exercise, atp, self.rng, ch.noise_std)

        # Lactate (Abbott Lingo) — 14-day wear cycle
        if "lactate" in all_ch:
            if self._lingo_available(t):
                ch = all_ch["lactate"]
                readings["lactate"] = observe_lactate(
                    atp, ros, sen, exercise, self.rng, ch.noise_std)
            else:
                readings["lactate"] = float("nan")

        # Blood biomarker panel — quarterly availability
        if "hs_crp" in all_ch:
            if self._biomarker_available(t):
                readings["hs_crp"] = observe_hs_crp(
                    sen, ros, self.rng, all_ch["hs_crp"].noise_std)
                readings["gdf15"] = observe_gdf15(
                    atp, ros, nad, sen, self.rng, all_ch["gdf15"].noise_std)
                readings["nad_blood"] = observe_nad_blood(
                    nad, self.rng, all_ch["nad_blood"].noise_std)
                readings["ohdg_8"] = observe_8ohdg(
                    ros, self.rng, all_ch["ohdg_8"].noise_std)
            else:
                readings["hs_crp"] = float("nan")
                readings["gdf15"] = float("nan")
                readings["nad_blood"] = float("nan")
                readings["ohdg_8"] = float("nan")

        return readings

    def estimate_state(
        self, observations: Dict[str, float],
        family_priors: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Estimate the 8D ODE state from noisy sensor readings.

        Uses algebraic inversion of the observation functions. For
        unobservable variables (N_healthy, N_deletion, N_point,
        Membrane_potential), returns population-average priors unless
        family_priors are provided (from WGS/family genomics data).

        This is intentionally a simple estimator — NOT a Kalman filter —
        to clearly demonstrate information loss from partial observability.

        Args:
            observations: Dict of sensor readings from observe().
            family_priors: Optional dict from family_genomics.compute_family_priors().
                Keys: 'n_healthy', 'n_deletion', 'n_point', 'membrane_potential'.
                When provided, these replace population-average priors for
                unobservable variables. Observable variables (ATP, ROS, NAD,
                Sen) are still estimated from sensors.

        Returns:
            Estimated 8D state vector in same order as ODE state:
            [N_h, N_del, ATP, ROS, NAD, Sen, ΔΨ, N_pt]
        """
        # Use family-informed priors when available, else population averages
        if family_priors is not None:
            n_h_prior = family_priors.get('n_healthy', SENSOR_PRIOR_N_HEALTHY)
            n_d_prior = family_priors.get('n_deletion', SENSOR_PRIOR_N_DELETION)
            n_p_prior = family_priors.get('n_point', SENSOR_PRIOR_N_POINT)
            psi_prior = family_priors.get('membrane_potential', SENSOR_PRIOR_MEMBRANE_POTENTIAL)
        else:
            n_h_prior = SENSOR_PRIOR_N_HEALTHY
            n_d_prior = SENSOR_PRIOR_N_DELETION
            n_p_prior = SENSOR_PRIOR_N_POINT
            psi_prior = SENSOR_PRIOR_MEMBRANE_POTENTIAL

        # Start with priors for unobservable variables
        est = np.array([
            n_h_prior,                     # [0] N_healthy — unobservable
            n_d_prior,                     # [1] N_deletion — unobservable
            BASELINE_ATP,                  # [2] ATP — estimated from sensors
            BASELINE_ROS,                  # [3] ROS — estimated from sensors
            BASELINE_NAD,                  # [4] NAD — estimated from sensors
            BASELINE_SENESCENT,            # [5] Senescent — estimated from sensors
            psi_prior,                     # [6] ΔΨ — poorly observable
            n_p_prior,                     # [7] N_point — unobservable
        ], dtype=np.float64)

        # Count available estimates for averaging
        atp_estimates = []
        ros_estimates = []
        sen_estimates = []
        nad_estimates = []

        # Invert heart rate → ATP estimate
        hr = observations.get("heart_rate")
        if hr is not None and not math.isnan(hr):
            # hr ≈ baseline + sensitivity * (1.0 - atp) + ...
            # Solve for atp (ignore ROS and age terms — partial info)
            atp_from_hr = BASELINE_ATP - (hr - SENSOR_BASELINE_HR) / SENSOR_HR_ATP_SENSITIVITY
            atp_estimates.append(atp_from_hr)

        # Invert HRV → ATP and senescence estimates
        hrv = observations.get("hrv")
        if hrv is not None and not math.isnan(hrv):
            # hrv ≈ baseline + atp_sens * (atp - 1) + sen_sens * sen
            # Two unknowns from one equation: use other info if available
            # Rough: attribute half deviation to ATP, half to senescence
            deviation = hrv - SENSOR_BASELINE_HRV
            atp_from_hrv = BASELINE_ATP + (deviation * 0.5) / SENSOR_HRV_ATP_SENSITIVITY
            atp_estimates.append(atp_from_hrv)
            if SENSOR_HRV_SEN_SENSITIVITY != 0:
                sen_from_hrv = (deviation * 0.5) / SENSOR_HRV_SEN_SENSITIVITY
                sen_estimates.append(max(0.0, sen_from_hrv))

        # Invert SpO2 → membrane potential estimate (indirect)
        spo2 = observations.get("spo2")
        if spo2 is not None and not math.isnan(spo2):
            if SENSOR_SPO2_PSI_SENSITIVITY != 0:
                psi_est = BASELINE_MEMBRANE_POTENTIAL + (spo2 - SENSOR_BASELINE_SPO2) / SENSOR_SPO2_PSI_SENSITIVITY
                est[6] = max(0.0, psi_est)

        # Invert temperature → ROS and senescence estimates
        temp = observations.get("temperature")
        if temp is not None and not math.isnan(temp):
            deviation = temp - SENSOR_BASELINE_TEMP
            # Split between ROS and senescence contributions
            if SENSOR_TEMP_ROS_SENSITIVITY != 0:
                ros_from_temp = BASELINE_ROS + (deviation * 0.5) / SENSOR_TEMP_ROS_SENSITIVITY
                ros_estimates.append(max(0.0, ros_from_temp))
            if SENSOR_TEMP_SEN_SENSITIVITY != 0:
                sen_from_temp = (deviation * 0.5) / SENSOR_TEMP_SEN_SENSITIVITY
                sen_estimates.append(max(0.0, sen_from_temp))

        # Invert blood pressure → senescence estimate
        sys_bp = observations.get("blood_pressure_sys")
        if sys_bp is not None and not math.isnan(sys_bp):
            if SENSOR_BP_SEN_SENSITIVITY != 0:
                sen_from_bp = (sys_bp - SENSOR_BASELINE_SYS_BP) / SENSOR_BP_SEN_SENSITIVITY
                sen_estimates.append(max(0.0, sen_from_bp))

        # Invert glucose → NAD estimate
        glucose = observations.get("glucose")
        if glucose is not None and not math.isnan(glucose):
            if SENSOR_GLUCOSE_NAD_SENSITIVITY != 0:
                nad_from_glucose = BASELINE_NAD + (glucose - SENSOR_BASELINE_GLUCOSE) / SENSOR_GLUCOSE_NAD_SENSITIVITY
                nad_estimates.append(max(0.0, nad_from_glucose))

        # Invert respiratory rate → ATP estimate
        resp = observations.get("respiratory_rate")
        if resp is not None and not math.isnan(resp):
            if SENSOR_RESP_ATP_SENSITIVITY != 0:
                resp_deficit = resp - SENSOR_BASELINE_RESP_RATE
                atp_from_resp = BASELINE_ATP - resp_deficit / SENSOR_RESP_ATP_SENSITIVITY
                atp_estimates.append(atp_from_resp)

        # Invert lactate → ATP estimate
        # Lactate rises with ATP deficit; attribute half to ATP, half to exercise
        lactate = observations.get("lactate")
        if lactate is not None and not math.isnan(lactate):
            deviation = lactate - SENSOR_BASELINE_LACTATE
            if SENSOR_LACTATE_ATP_SENSITIVITY != 0:
                # Use half the deviation for ATP (other half may be exercise)
                atp_from_lactate = BASELINE_ATP + (deviation * 0.5) / SENSOR_LACTATE_ATP_SENSITIVITY
                atp_estimates.append(atp_from_lactate)

        # Invert hs-CRP → senescence estimate
        hs_crp = observations.get("hs_crp")
        if hs_crp is not None and not math.isnan(hs_crp):
            if SENSOR_HS_CRP_SEN_SENSITIVITY != 0:
                sen_from_crp = (hs_crp - SENSOR_BASELINE_HS_CRP) / SENSOR_HS_CRP_SEN_SENSITIVITY
                sen_estimates.append(max(0.0, sen_from_crp))

        # Invert GDF-15 → ATP, ROS, NAD, Sen estimates (4-way split)
        gdf15 = observations.get("gdf15")
        if gdf15 is not None and not math.isnan(gdf15):
            deviation = gdf15 - SENSOR_BASELINE_GDF15
            # Split deviation 4 ways among ATP, ROS, NAD, Sen
            quarter = deviation * 0.25
            if SENSOR_GDF15_ATP_SENSITIVITY != 0:
                atp_from_gdf15 = BASELINE_ATP + quarter / SENSOR_GDF15_ATP_SENSITIVITY
                atp_estimates.append(atp_from_gdf15)
            if SENSOR_GDF15_ROS_SENSITIVITY != 0:
                ros_from_gdf15 = BASELINE_ROS + quarter / SENSOR_GDF15_ROS_SENSITIVITY
                ros_estimates.append(max(0.0, ros_from_gdf15))
            if SENSOR_GDF15_NAD_SENSITIVITY != 0:
                nad_from_gdf15 = BASELINE_NAD + quarter / SENSOR_GDF15_NAD_SENSITIVITY
                nad_estimates.append(max(0.0, nad_from_gdf15))
            if SENSOR_GDF15_SEN_SENSITIVITY != 0:
                sen_from_gdf15 = quarter / SENSOR_GDF15_SEN_SENSITIVITY
                sen_estimates.append(max(0.0, sen_from_gdf15))

        # Invert NAD+ blood → NAD estimate (near-direct, highest quality)
        nad_blood = observations.get("nad_blood")
        if nad_blood is not None and not math.isnan(nad_blood):
            if SENSOR_NAD_BLOOD_SENSITIVITY != 0:
                nad_from_blood = BASELINE_NAD + (nad_blood - SENSOR_BASELINE_NAD_BLOOD) / SENSOR_NAD_BLOOD_SENSITIVITY
                nad_estimates.append(max(0.0, nad_from_blood))

        # Invert 8-OHdG → ROS estimate (direct proportionality)
        ohdg = observations.get("ohdg_8")
        if ohdg is not None and not math.isnan(ohdg):
            if SENSOR_8OHDG_ROS_SENSITIVITY != 0:
                ros_from_ohdg = (ohdg - SENSOR_BASELINE_8OHDG) / SENSOR_8OHDG_ROS_SENSITIVITY
                ros_estimates.append(max(0.0, ros_from_ohdg))

        # Average estimates for each observable variable
        if atp_estimates:
            est[2] = max(0.0, sum(atp_estimates) / len(atp_estimates))
        if ros_estimates:
            est[3] = max(0.0, sum(ros_estimates) / len(ros_estimates))
        if nad_estimates:
            est[4] = max(0.0, sum(nad_estimates) / len(nad_estimates))
        if sen_estimates:
            est[5] = max(0.0, min(1.0, sum(sen_estimates) / len(sen_estimates)))

        return est

    def information_loss(
        self,
        true_state: np.ndarray,
        estimated_state: np.ndarray,
    ) -> Dict[str, float]:
        """Quantify information loss between true and estimated state.

        Args:
            true_state: Actual 8D ODE state vector.
            estimated_state: Estimated state from estimate_state().

        Returns:
            Dict with per-variable absolute error and total RMSE.
        """
        state_names = [
            "N_healthy", "N_deletion", "ATP", "ROS", "NAD",
            "Senescent_fraction", "Membrane_potential", "N_point",
        ]
        errors = {}
        for i, name in enumerate(state_names):
            errors[f"{name}_error"] = abs(true_state[i] - estimated_state[i])

        diff = true_state - estimated_state
        errors["total_rmse"] = float(np.sqrt(np.mean(diff ** 2)))

        return errors
