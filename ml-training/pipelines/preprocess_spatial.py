"""
Spatial Data Preprocessing Pipeline (GeoPandas & Rasterio)
Extracts geomorphological conditioning factors from CartoDEM, GSI Bhukosh Lithology, and IMD Grids.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple


def load_raw_conditioning_factors(csv_path: str) -> pd.DataFrame:
    """Loads raw conditioning factors table."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Conditioning factors file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[PREPROCESS] Loaded {len(df)} terrain grid samples from {csv_path}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs normalized machine learning features and geotechnical interaction terms.
    """
    df_feat = df.copy()

    # 1. Topographic Wetness Index (TWI) proxy: ln(Catchment / tan(slope))
    # TWI proxy = (rainfall_3d_mm * antecedent_moisture_pct / 100.0) / (np.tan(np.radians(np.maximum(5.0, df_feat['slope_deg']))) + 0.01)
    slope_rad = np.radians(np.maximum(3.0, df_feat['slope_deg']))
    df_feat['topographic_wetness_proxy'] = (df_feat['rainfall_3d_mm'] * (df_feat['antecedent_moisture_pct'] / 100.0)) / (np.tan(slope_rad) + 0.01)

    # 2. Road cut instability amplification factor: exp(-distance_to_road / 200m)
    df_feat['road_cut_influence'] = np.exp(-df_feat['distance_to_road_m'] / 200.0)

    # 3. Lithological vulnerability encoding
    litho_dict = {
        "PHYL": 0.92,
        "SHALE": 0.88,
        "MUDST": 0.80,
        "SANDST": 0.55,
        "SCHIST": 0.70,
        "QUARTZ": 0.35,
        "GNEISS": 0.30
    }
    df_feat['lithology_numeric'] = df_feat['lithology_code'].map(lambda x: litho_dict.get(str(x).upper(), 0.50))

    # 4. Infinite Slope Factor of Safety calculation (Physics column)
    gamma = df_feat['unit_weight_kn_m3']
    z = df_feat['soil_depth_m']
    c = df_feat['cohesion_kpa']
    phi_rad = np.radians(df_feat['friction_angle_deg'])

    # Approximate pore water pressure based on moisture percentage
    u_est = np.maximum(0.0, (df_feat['antecedent_moisture_pct'] - 65.0) * 1.5)

    normal_stress = gamma * z * (np.cos(slope_rad) ** 2)
    eff_normal_stress = np.maximum(0.5, normal_stress - u_est)
    shear_strength = c + (eff_normal_stress * np.tan(phi_rad))
    driving_shear = gamma * z * np.sin(slope_rad) * np.cos(slope_rad)

    df_feat['analytical_fs'] = np.clip(shear_strength / np.maximum(0.5, driving_shear), 0.1, 5.0)

    print(f"[PREPROCESS] Engineered {df_feat.shape[1]} features including analytical Factor of Safety.")
    return df_feat


if __name__ == "__main__":
    schema_path = os.path.join(os.path.dirname(__file__), "../data/schemas/conditioning_factors_template.csv")
    df = load_raw_conditioning_factors(schema_path)
    df_features = engineer_features(df)
    print("\nFeature Summary:")
    print(df_features[['grid_id', 'slope_deg', 'topographic_wetness_proxy', 'road_cut_influence', 'analytical_fs']])
