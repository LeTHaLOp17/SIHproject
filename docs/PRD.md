# Product Requirements Document (PRD)
## AI-Based Landslide Early Warning & Risk Monitoring Platform
**Project**: Ministry of Development of North Eastern Region (MDoNER) - Problem Statement ID: 26001  
**Target Region**: North Eastern Region (NER) of India (Assam, Meghalaya, Arunachal Pradesh, Nagaland, Manipur, Mizoram, Tripura, Sikkim)  
**Classification**: Mission-Critical Disaster Management & Life-Safety Platform  
**Document Version**: 2.4.0 (Production Release)

---

## 1. Problem Statement & Operational Context
The North Eastern Region (NER) of India comprises over 262,000 square kilometers of geologically young, tectonically active Himalayan and Indo-Burman mountain belts. Characteristics of this operational theater include:
- **Intense Precipitation**: Rainfall exceeding 2,500 mm to 11,000 mm annually (e.g., Mawsynram and Cherrapunji in Meghalaya).
- **Critical Arterial Highways**: National Highways (NH-10, NH-29, NH-44, NH-102) serve as singular lifelines connecting entire state populations. A single slope failure isolates hundreds of thousands of citizens from medicine, food, and evacuation corridors.
- **Extreme Connectivity Voids**: Deep valley topography causes massive RF attenuation and total cellular blackouts ("shadow zones").
- **Linguistic Diversity**: Tribal populations speak distinct indigenous languages (Assamese, Bodo, Khasi, Mizo, Manipuri, Bengali, Hindi), necessitating intuitive, multilingual, and voice-assisted interfaces.
- **Sparse Ground Sensors**: Historical sensor data is sparse, causing standard pure machine-learning models to suffer from excessive false alarms or missed events.

---

## 2. User Personas

### Persona 1: District Disaster Management Officer (DDMA / SDMA)
- **Role**: District Magistrate / Emergency Operations Center Director (e.g., Gangtok, Kohima, Aizawl).
- **Core Needs**: Instant triage of high-risk hillslopes, automated CAP/NDMA compliance, map overlays of severed routes, and 1-click evacuation authorization.
- **Pain Points**: Information overload during torrential rains; false alarms erode public compliance; delayed warnings from manual synthesis.

### Persona 2: Field Geotechnical & Highway Engineer (GSI / BRO / NHIDCL)
- **Role**: Border Roads Organisation (BRO) Officer or Geological Survey of India (GSI) Geotechnical Specialist.
- **Core Needs**: Quantitative Factor of Safety (\(FS\)) slope values, pore pressure telemetry from piezometers, rate of tilt change (degrees/hr), and millimeters of crack widening.
- **Pain Points**: Inability to access cloud-only dashboards during field inspections in remote road-cut areas.

### Persona 3: Citizen & Village Headman (Gaon Bura / Rangbah Shnong)
- **Role**: Local resident in an isolated hillslope hamlet.
- **Core Needs**: Early warning sirens, spoken IVR warnings in mother tongue (e.g., Khasi, Bodo, Assamese), clear safe-haven evacuation directions, and ability to report slope cracks with photographs without cellular connectivity.
- **Pain Points**: Illiteracy or lack of English/Hindi fluency; apps that fail without internet; delayed help after village access road collapses.

### Persona 4: Emergency Response & Airlift Coordinator (NDRF / Assam Rifles / IAF)
- **Role**: National Disaster Response Force (NDRF) Battalion Commander.
- **Core Needs**: Real-time list of "landlocked" villages that have 0 road connectivity due to bridge or highway failure; prioritized evacuation lists based on population and vulnerable demographics; coordinates for helicopter airdrops.
- **Pain Points**: Discovering hours after a landslide that a village has been cut off and supplies cannot reach hospital clinics.

---

## 3. Product Epics & Scope

```
EPIC 1: Multi-Modal Sensor Telemetry & Ingestion
EPIC 2: Hybrid Physics-Informed & Ensemble AI Predictive Engine
EPIC 3: Graph Theory Road Collapse & Village Isolation Index Engine
EPIC 4: Multi-Channel Alert Dispatcher (CAP v1.2, CDAC SMS, IVR Voice, Push)
EPIC 5: Offline-First TinyML Multilingual Mobile Application
EPIC 6: High-Fidelity 3D GIS Command Center Dashboard (React + CesiumJS)
```

---

## 4. Functional Requirements

### Epic 1: Multi-Modal Sensor & Satellite Telemetry
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-1.1** | IoT Ingestion Gateway | P0 | Ingest EMQX MQTT sensor streams (pore pressure, soil moisture, tilt meter, rain gauge) at up to 10,000 msgs/sec. |
| **FR-1.2** | IMD Weather Harvesting | P0 | Automated cron harvesting of IMD Doppler radar and 3-hour precipitation forecasts via REST API. |
| **FR-1.3** | Satellite Surface Moisture Feed | P1 | Ingest ISRO Bhuvan / MOSDAC Sentinel-1 InSAR surface displacement and soil saturation raster layers. |
| **FR-1.4** | Sensor Fault Detection | P1 | Auto-flag dead nodes, battery drain (< 3.3V), or erratic sensor noise using Hampel filtering. |

### Epic 2: Hybrid Physics-Informed & Ensemble AI Engine
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-2.1** | PINN Slope Stability Model | P0 | Couple the Infinite Slope Factor of Safety (\(FS\)) with a deep neural network; output continuous \(FS\) score per 10m grid cell. |
| **FR-2.2** | Recall-Maximizing XGBoost | P0 | Tabular susceptibility inference with hyperparameter tuning enforcing **Recall > 0.90** for high-risk zones. |
| **FR-2.3** | Rainfall Forecaster (LSTM) | P0 | Sequence past 24-hour rainfall and 6-12 hour forecasted precipitation to predict time-to-critical soil saturation. |
| **FR-2.4** | CV Crack Displacement Tool | P1 | Compute displacement (in mm) between time-stamped citizen photographs of hill/road cracks using Lucas-Kanade Optical Flow. |

### Epic 3: Graph Theory Isolation Index Module ("Secret Sauce")
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-3.1** | Road Topology Graph Construction | P0 | Neo4j / NetworkX representation of all NH, State Highways, and rural arterial links with village dependency nodes. |
| **FR-3.2** | Bridge & Cut-Vertex Detection | P0 | Dynamic computation of articulation points and bridges; upon simulated or reported road cut, compute isolated village clusters within 250ms. |
| **FR-3.3** | Vulnerability-Weighted Index | P0 | Output ranked evacuation list prioritizing villages based on total population, vulnerable demographic ratio, and days of isolated supplies. |
| **FR-3.4** | Resilient Egress Routing | P1 | Generate alternate 4x4 or foot-track evacuation corridors avoiding all high-susceptibility slope segments. |

### Epic 4: Multi-Channel Alert Dispatcher & NDMA SACHET Integration
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-4.1** | OASIS CAP-IN v1.2 Serialization | P0 | Automated generation of valid Common Alerting Protocol XML alerts consumable by NDMA SACHET platform. |
| **FR-4.2** | CDAC SMS Gateway Dispatch | P0 | Geo-targeted bulk SMS push via CDAC telecom gateway with regional language templates. |
| **FR-4.3** | Automated IVR Voice Broadcast | P0 | Outbound telephony calls with pre-rendered voice files in Assamese, Bodo, Khasi, Bengali, Hindi, and English. |
| **FR-4.4** | Priority-Tiered Queuing | P0 | BullMQ Redis queuing with strict prioritization: Red/Evac alerts preempt green/yellow advisories. Exponential backoff retry for failed dispatches. |

### Epic 5: Offline-First TinyML Mobile Application (Flutter)
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-5.1** | Zero-Internet On-Device Inference | P0 | Embedded int8-quantized TensorFlow Lite model runs local risk inference on phone in < 45ms without network. |
| **FR-5.2** | Local SQLite Offline Storage | P0 | All user reports, photographs, GPS stamps, and received hazard polygons stored in local SQLite; synced seamlessly on reconnection. |
| **FR-5.3** | Multilingual UI Support | P0 | Full UI and audio alerts in Assamese, Bodo, Khasi, Bengali, Hindi, and English. |
| **FR-5.4** | Field Crack Photographic Logging | P1 | Built-in camera tool with on-screen scale calibration overlay for logging progressive crack displacement. |

### Epic 6: 3D GIS Command Dashboard (React + CesiumJS)
| ID | Requirement | Priority | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **FR-6.1** | 3D Himalayan Globe & Terrain | P0 | CesiumJS rendering of high-resolution digital elevation models (DEM), contour slopes, and extruded 3D hazard polygons. |
| **FR-6.2** | Real-Time IoT Telemetry Gauges | P0 | Live WebSocket updates of pore pressure, tilt meters, and rainfall histograms. |
| **FR-6.3** | Road Closure Simulator | P0 | Interactive UI permitting operators to toggle any road segment to "collapsed", instantly viewing isolated downstream communities. |

---

## 5. Non-Functional Requirements (NFRs)

| Attribute | Specification | Measurement & Verification |
| :--- | :--- | :--- |
| **AI Recall** | **\(\ge 0.92\)** on landslide event classification | Evaluated across historical holdout sets (Sikkim 2023, Manipur 2022 events). |
| **Latency (Inference)** | \(\le 150\text{ ms}\) for cloud REST; \(\le 50\text{ ms}\) for on-device TFLite | Benchmarked via Locust load test & Android profiling. |
| **Alert Dispatch Time** | P99 \(\le 12\text{ seconds}\) from hazard detection to SMS/CAP broadcast | Measured end-to-end through BullMQ metrics. |
| **Offline Resilience** | 100% of core mobile functions work without cellular or Wi-Fi | Validated by putting devices into Airplane mode for 72 hours. |
| **Concurrent Load** | \(\ge 50,000\) simultaneous mobile sync connections | Verified using distributed k6 stress test on NestJS API cluster. |
| **High Availability** | 99.95% uptime across Kubernetes cluster | Multi-zone replica deployments with automated pod recovery. |
| **Data Retention** | 10 years for time-series sensor data | Managed via TimescaleDB compression policies and data tiers. |
| **Accessibility & UX** | WCAG 2.1 AA compliant; tactile/audio alerts for emergency states | Color-blind safe palette (viridis / tactical emergency scale). |
