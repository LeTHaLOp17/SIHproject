"""
Slope Stability Physics Engine: Factor of Safety (FS) Formulations
Couples classical geotechnical engineering principles with neural network loss functions.

Formulation:
1. Infinite Slope Stability with Seepage (Terzaghi Effective Stress + Mohr-Coulomb Criterion)
2. Seismic Pseudostatic Acceleration Adjustment (IS 1893:2016 for NER Seismic Zone V)
"""

import math
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GeotechnicalParameters:
    cohesion_kpa: float            # c' (effective cohesion in kPa)
    friction_angle_deg: float      # phi' (effective internal friction angle in degrees)
    soil_unit_weight_kn_m3: float  # gamma (bulk unit weight of soil, typically 17-21 kN/m3)
    water_unit_weight_kn_m3: float = 9.81  # gamma_w (unit weight of water = 9.81 kN/m3)


@dataclass
class SlopeConditions:
    slope_angle_deg: float         # beta (inclination of slope in degrees)
    soil_depth_m: float            # z (depth to shear failure plane in meters)
    pore_water_pressure_kpa: float # u (positive pore water pressure in kPa)
    seismic_coeff_kh: float = 0.08 # k_h (horizontal seismic coefficient, 0.05-0.15 for NER)


class SlopeStabilityPhysics:
    """
    Computes deterministic Factor of Safety (FS) using geotechnical limit equilibrium equations.
    FS > 1.3: Stable (Safe)
    1.0 <= FS <= 1.3: Marginally Stable (Advisory / Watch)
    FS < 1.0: Active Mechanical Failure (Warning / Evacuation)
    """

    @staticmethod
    def calculate_infinite_slope_fs(params: GeotechnicalParameters, conditions: SlopeConditions) -> Dict[str, Any]:
        beta_rad = math.radians(conditions.slope_angle_deg)
        phi_rad = math.radians(params.friction_angle_deg)

        # Prevent division by zero on flat ground
        if beta_rad < math.radians(3.0):
            return {
                "factor_of_safety": 99.0,
                "stability_state": "COMPLETELY_STABLE_FLAT_TERRAIN",
                "is_critical": False,
                "effective_normal_stress_kpa": 100.0,
                "shear_strength_kpa": 100.0,
                "shear_stress_kpa": 1.0
            }

        gamma = params.soil_unit_weight_kn_m3
        z = max(0.5, conditions.soil_depth_m)
        u = max(0.0, conditions.pore_water_pressure_kpa)
        c = max(0.0, params.cohesion_kpa)
        k_h = conditions.seismic_coeff_kh

        # Total normal stress on failure plane: sigma = gamma * z * cos^2(beta)
        total_normal_stress = gamma * z * (math.cos(beta_rad) ** 2)

        # Terzaghi's effective normal stress: sigma' = sigma - u
        effective_normal_stress = max(0.0, total_normal_stress - u)

        # Resisting shear strength (Mohr-Coulomb): tau_f = c' + sigma' * tan(phi')
        shear_strength_resisting = c + (effective_normal_stress * math.tan(phi_rad))

        # Mobilized driving shear stress (including seismic inertial force):
        # tau_d = gamma * z * sin(beta) * cos(beta) + k_h * gamma * z * cos^2(beta)
        gravitational_driving = gamma * z * math.sin(beta_rad) * math.cos(beta_rad)
        seismic_driving = k_h * gamma * z * (math.cos(beta_rad) ** 2)
        total_driving_stress = max(0.1, gravitational_driving + seismic_driving)

        # Factor of Safety
        fs = shear_strength_resisting / total_driving_stress
        fs_clamped = round(float(fs), 3)

        if fs_clamped >= 1.30:
            state = "STABLE"
            is_critical = False
        elif fs_clamped >= 1.00:
            state = "MARGINALLY_STABLE"
            is_critical = False
        elif fs_clamped >= 0.70:
            state = "HIGH_FAILURE_RISK"
            is_critical = True
        else:
            state = "IMMINENT_MECHANICAL_COLLAPSE"
            is_critical = True

        return {
            "factor_of_safety": fs_clamped,
            "stability_state": state,
            "is_critical": is_critical,
            "effective_normal_stress_kpa": round(effective_normal_stress, 2),
            "shear_strength_kpa": round(shear_strength_resisting, 2),
            "shear_stress_kpa": round(total_driving_stress, 2),
            "pore_pressure_ratio_ru": round(u / (gamma * z), 3) if (gamma * z) > 0 else 0.0
        }
