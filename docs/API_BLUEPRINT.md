# API Blueprint: REST & gRPC Endpoint Specification
## AI-Based Landslide Early Warning & Risk Monitoring Platform
**Base URL**: `https://api.ews-landslide.mdoner.gov.in/api/v1`  
**Security**: Bearer JWT (RS256) | mTLS for IoT Gateways | OpenAPI 3.1 & Protocol Buffers

---

## 1. REST Endpoints Overview

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `POST` | `/predict/slope` | Real-time PINN + Ensemble landslide inference for a specific coordinate | SDMA / DDMA / System |
| `POST` | `/predict/batch-grid` | Batch inference for geographical bounding box grid (GeoJSON/COG) | System / GIS Engine |
| `POST` | `/cv/displacement` | Optical Flow crack displacement measurement from paired photos | Public / Citizen / Field Eng |
| `GET`  | `/alerts` | Query active early warning alerts with spatial/severity filters | Public |
| `POST` | `/alerts/generate` | Operator or model-triggered emergency alert generation | SDMA / DDMA Admin |
| `GET`  | `/alerts/{id}/cap-xml` | Export alert in OASIS Common Alerting Protocol (CAP-IN v1.2) XML | NDMA SACHET / Public |
| `POST` | `/sync/mobile-push` | Mobile app delta upload of offline reports, photos, and sensor reads | Citizen / Field Eng |
| `POST` | `/sync/mobile-pull` | Mobile app delta download of updated hazard zones and active alerts | Citizen / Field Eng |
| `POST` | `/reports/citizen` | Submit citizen landslide observation with GPS, tags, and photo | Public (No login required) |
| `GET`  | `/dashboard/summary` | Aggregated statistics for state/district command centers | SDMA / DDMA |
| `GET`  | `/dashboard/isolation-index`| Graph Theory Village Isolation rankings and cutoff road links | SDMA / NDRF / District Mag |
| `POST` | `/dashboard/simulate-road-cut`| Simulate consequence of bridge/highway collapse on village access | Incident Commander |

---

## 2. Detailed Endpoint Specifications

### 2.1 Landslide Inference: `POST /predict/slope`

Evaluates real-time hazard using the Hybrid Ensemble Model (PINN + LSTM + XGBoost).

#### Request Headers
```http
Content-Type: application/json
Authorization: Bearer <JWT_TOKEN>
```

#### Request Payload
```json
{
  "latitude": 27.3389,
  "longitude": 88.6065,
  "elevation_m": 1650.0,
  "slope_deg": 38.5,
  "aspect_deg": 142.0,
  "lithology_code": "PHYL",
  "soil_type": "Sandy_Clay_Loam",
  "soil_depth_m": 2.4,
  "cohesion_kpa": 12.5,
  "friction_angle_deg": 28.0,
  "unit_weight_kn_m3": 18.5,
  "current_pore_pressure_kpa": 42.8,
  "rainfall_1h_mm": 18.4,
  "rainfall_24h_mm": 142.6,
  "soil_moisture_pct": 84.2,
  "distance_to_road_m": 45.0,
  "distance_to_fault_m": 210.0
}
```

#### Response Payload (`200 OK`)
```json
{
  "status": "SUCCESS",
  "timestamp": "2026-09-08T10:15:00Z",
  "coordinates": { "latitude": 27.3389, "longitude": 88.6065 },
  "physics_engine": {
    "factor_of_safety": 0.84,
    "stability_state": "UNSTABLE_MECHANICAL_FAILURE",
    "effective_normal_stress_kpa": 24.6,
    "shear_strength_kpa": 25.6,
    "shear_stress_kpa": 30.5
  },
  "temporal_engine": {
    "predicted_saturation_6h_pct": 92.4,
    "trigger_probability_next_12h": 0.89
  },
  "tabular_engine": {
    "susceptibility_class": "VERY_HIGH",
    "xgboost_probability": 0.93
  },
  "ensemble_result": {
    "composite_risk_score": 0.91,
    "hazard_level": "WARNING_RED",
    "recommended_action": "IMMEDIATE_TRAFFIC_SUSPENSION_AND_EVACUATION",
    "lead_time_hours": 3.5
  }
}
```

---

### 2.2 Computer Vision Crack Displacement: `POST /cv/displacement`

Computes relative surface displacement between two timestamps.

#### Request (Multipart Form-Data)
- `image_baseline`: File (JPEG/PNG) taken at \(t_1\)
- `image_followup`: File (JPEG/PNG) taken at \(t_2\)
- `metadata`: JSON string with time interval, GPS stamp, and optional fiducial calibration scale (pixels/mm).

#### Response Payload (`200 OK`)
```json
{
  "status": "SUCCESS",
  "displacement_analysis": {
    "time_elapsed_hours": 48.0,
    "scale_calibration_pixels_per_mm": 4.25,
    "max_displacement_mm": 18.4,
    "mean_displacement_mm": 12.1,
    "displacement_velocity_mm_per_day": 9.2,
    "strain_acceleration": "ACCELERATING_TERTIARY_CREEP",
    "critical_threshold_exceeded": true,
    "confidence_score": 0.94
  },
  "visual_overlay_url": "https://cdn.ews-landslide.mdoner.gov.in/cv/overlays/CRK-2024-9182.jpg"
}
```

---

### 2.3 NDMA Common Alerting Protocol: `GET /alerts/{id}/cap-xml`

Generates OASIS CAP-IN v1.2 compliant XML for seamless NDMA SACHET ingestion.

#### Response (`200 OK`, `Content-Type: application/xml`)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>IN-NER-SK-DDMA-2026-0042</identifier>
  <sender>ddma.gangtok@sikkim.gov.in</sender>
  <sent>2026-09-08T10:15:00+05:30</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <category>Geo</category>
    <event>Landslide Warning</event>
    <urgency>Immediate</urgency>
    <severity>Extreme</severity>
    <certainty>Observed</certainty>
    <eventCode>
      <valueName>NDMA_CODE</valueName>
      <value>LS-WARN-RED</value>
    </eventCode>
    <expires>2026-09-08T22:15:00+05:30</expires>
    <senderName>District Disaster Management Authority, East Sikkim</senderName>
    <headline>IMMEDIATE EVACUATION: Critical Landslide Threat along NH-10 Corridor</headline>
    <description>Heavy antecedent rainfall (142mm/24h) and pore-pressure sensors indicate immediate hillslope failure between KM 42 and KM 46. Factor of Safety has dropped to 0.84.</description>
    <instruction>All vehicular traffic on NH-10 is suspended. Residents of Lower Rongli are instructed to move immediately to designated higher ground at Govt Senior Secondary School.</instruction>
    <area>
      <areaDesc>NH-10 Km 42-46, East Sikkim</areaDesc>
      <circle>27.3389,88.6065,2.5</circle>
    </area>
  </info>
</alert>
```

---

### 2.4 Mobile Offline Sync: `POST /sync/mobile-push`

Uploads locally created citizen landslide logs and crack observations captured offline.

#### Request Payload
```json
{
  "device_id": "MBL-SM-A54-89104A",
  "app_version": "2.4.0",
  "client_timestamp": "2026-09-08T10:12:00Z",
  "offline_reports": [
    {
      "local_id": "loc_rep_38472910",
      "reported_at": "2026-09-08T06:30:00Z",
      "latitude": 25.5788,
      "longitude": 91.8933,
      "accuracy_m": 4.5,
      "observation_type": "TENSION_CRACK_ON_ROAD",
      "apparent_crack_width_cm": 15.0,
      "spring_water_turbidity": "HIGHLY_MUDDY",
      "local_tinyml_risk_score": 0.88,
      "photo_hashes": ["sha256_91823a0df..."]
    }
  ]
}
```

---

### 2.5 Graph Theory Village Isolation Index: `GET /dashboard/isolation-index`

Returns prioritized leaderboard of settlements at risk of becoming landlocked.

#### Request Parameters
- `district_id`: String (e.g., `SK_EAST_SIKKIM`)
- `min_risk_threshold`: Float (e.g., `0.70`)

#### Response Payload (`200 OK`)
```json
{
  "status": "SUCCESS",
  "evaluated_network": "East Sikkim Arterial Transport Graph v4.2",
  "total_settlements": 142,
  "isolated_settlements_count": 7,
  "critical_bottlenecks": [
    {
      "structure_id": "BRG-NH10-04",
      "type": "BRIDGE",
      "name": "Teesta River Crossing Mile 44",
      "slope_hazard_score": 0.91,
      "is_articulation_point": true,
      "dependent_villages_count": 4,
      "dependent_population_total": 18450
    }
  ],
  "prioritized_evacuation_leaderboard": [
    {
      "rank": 1,
      "village_id": "VILL-SK-042",
      "village_name": "Rongli Upper Basti",
      "population": 4200,
      "vulnerable_population": 980,
      "primary_access_road_id": "RD-SK-NH10-A",
      "isolation_index_score": 89.4,
      "days_of_medical_inventory_remaining": 1.5,
      "designated_helipad_coordinates": [88.6210, 27.3450],
      "alternate_egress_path": ["PATH-TRAIL-NORTH-02", "RD-SEC-41"]
    }
  ]
}
```
