"""
Hybrid Ensemble Model Fusion Core
Fuses Physics-Informed Neural Network (PINN), LSTM Rainfall Forecast, XGBoost Tabular Susceptibility, and Computer Vision Crack Tracking.
"""

from typing import Dict, Any
from app.physics.slope_stability import SlopeStabilityPhysics, GeotechnicalParameters, SlopeConditions
from app.models.pinn_model import PINNPredictor
from app.models.rainfall_lstm import RainfallForecaster
from app.models.tabular_xgboost import XGBoostSusceptibilityEngine
from app.models.crack_cv import CrackDisplacementAnalyzer


class HybridEnsembleFusionEngine:
    """
    Synthesizes multiple predictive modalities:
    Weighting Strategy:
      - 35% Deterministic / PINN Factor of Safety Physics
      - 30% XGBoost Multidimensional Conditioning Factors
      - 20% LSTM Future Precipitation / Moisture Saturation
      - 15% Field Crack Deformation Velocity (if reported)
    """

    def __init__(self):
        self.pinn = PINNPredictor()
        self.lstm = RainfallForecaster()
        self.xgboost = XGBoostSusceptibilityEngine()

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Physics Limit Equilibrium Computation
        geotech = GeotechnicalParameters(
            cohesion_kpa=float(input_data.get("cohesion_kpa", 12.0)),
            friction_angle_deg=float(input_data.get("friction_angle_deg", 28.0)),
            soil_unit_weight_kn_m3=float(input_data.get("unit_weight_kn_m3", 18.5))
        )
        conditions = SlopeConditions(
            slope_angle_deg=float(input_data.get("slope_deg", 32.0)),
            soil_depth_m=float(input_data.get("soil_depth_m", 2.5)),
            pore_water_pressure_kpa=float(input_data.get("current_pore_pressure_kpa", 25.0)),
            seismic_coeff_kh=float(input_data.get("seismic_coeff_kh", 0.08))
        )
        physics_res = SlopeStabilityPhysics.calculate_infinite_slope_fs(geotech, conditions)
        fs_val = physics_res["factor_of_safety"]

        # 2. PINN Neural Inference
        pinn_features = [
            conditions.slope_angle_deg / 60.0,
            float(input_data.get("elevation_m", 1500.0)) / 4000.0,
            conditions.soil_depth_m / 10.0,
            geotech.cohesion_kpa / 50.0,
            geotech.friction_angle_deg / 45.0,
            geotech.soil_unit_weight_kn_m3 / 25.0,
            conditions.pore_water_pressure_kpa / 100.0,
            float(input_data.get("rainfall_24h_mm", 50.0)) / 300.0,
            float(input_data.get("soil_moisture_pct", 65.0)) / 100.0
        ]
        pinn_res = self.pinn.predict(pinn_features)

        # 3. XGBoost Tabular Susceptibility
        xgb_res = self.xgboost.predict_susceptibility(input_data)
        xgb_prob = xgb_res["xgboost_landslide_probability"]

        # 4. LSTM Rainfall Forecaster
        rain_seq = input_data.get("rainfall_history_24h", [])
        lstm_res = self.lstm.forecast(rain_seq)
        rain_trigger_risk = min(1.0, (lstm_res["forecasted_rain_next_12h_mm"] / 120.0))

        # 5. Field / Citizen Crack Velocity (if available)
        crack_velocity = float(input_data.get("crack_velocity_mm_per_day", 0.0))
        crack_risk_contribution = min(1.0, crack_velocity / 15.0)

        # 6. Ensemble Synthesis
        # Convert FS to risk index: FS=1.5 -> risk=0.0, FS=1.0 -> risk=0.5, FS=0.5 -> risk=1.0
        fs_risk = min(1.0, max(0.0, (1.4 - fs_val) / 0.9))

        if crack_velocity > 0.0:
            composite_score = (
                0.30 * fs_risk +
                0.25 * xgb_prob +
                0.25 * pinn_res["pinn_failure_probability"] +
                0.10 * rain_trigger_risk +
                0.10 * crack_risk_contribution
            )
        else:
            composite_score = (
                0.35 * fs_risk +
                0.30 * xgb_prob +
                0.25 * pinn_res["pinn_failure_probability"] +
                0.10 * rain_trigger_risk
            )

        composite_score = round(float(composite_score), 4)

        # Hazard Tier Definition
        if composite_score >= 0.85 or fs_val < 0.80:
            hazard_level = "EVACUATION_MANDATE"
            recommended_action = "IMMEDIATE_MANDATORY_EVACUATION_AND_ROAD_CLOSURE"
            lead_time_h = 2.0
        elif composite_score >= 0.70 or fs_val < 1.00:
            hazard_level = "WARNING"
            recommended_action = "ISSUE_RED_ALERT_DEPLOY_RESCUE_TEAMS_RESTRICT_TRAFFIC"
            lead_time_h = 6.0
        elif composite_score >= 0.50 or fs_val <= 1.25:
            hazard_level = "WATCH"
            recommended_action = "ISSUE_ORANGE_ALERT_INSPECT_CULVERTS_PREPARE_SHELTERS"
            lead_time_h = 12.0
        elif composite_score >= 0.30:
            hazard_level = "ADVISORY"
            recommended_action = "ISSUE_YELLOW_ADVISORY_MONITOR_SENSORS"
            lead_time_h = 24.0
        else:
            hazard_level = "NORMAL"
            recommended_action = "ROUTINE_MONITORING"
            lead_time_h = 72.0

        return {
            "composite_risk_score": composite_score,
            "hazard_level": hazard_level,
            "recommended_action": recommended_action,
            "estimated_lead_time_hours": lead_time_h,
            "subsystem_outputs": {
                "physics_factor_of_safety": fs_val,
                "pinn_failure_probability": pinn_res["pinn_failure_probability"],
                "xgboost_susceptibility_probability": xgb_prob,
                "forecasted_rain_next_12h_mm": lstm_res["forecasted_rain_next_12h_mm"],
                "soil_saturation_state": lstm_res["saturation_level"]
            }
        }
