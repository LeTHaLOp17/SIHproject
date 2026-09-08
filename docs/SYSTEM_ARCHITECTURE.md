# System Architecture Specification: 6-Layer Disaster Monitoring & Early Warning Platform
## Ministry of Development of North Eastern Region (MDoNER) - Problem Statement ID: 26001

---

## 1. Overview of the 6-Layer Architecture

The system is organized into six functional layers designed to guarantee continuous operation, physics-grounded prediction, and emergency communication across the mountainous topography of the North Eastern Region of India.

```
+-----------------------------------------------------------------------------------+
| LAYER 6: PRESENTATION & EMERGENCY ALERT ENGINE                                    |
| - React 3D CesiumJS Command Center Dashboard | Flutter Offline-First Mobile App   |
| - OASIS CAP v1.2 / NDMA SACHET Gateway | C-DAC SMS Push | Multilingual IVR Voice |
+-----------------------------------------------------------------------------------+
                                        ^
                                        | (Real-time WebSockets, REST, CAP XML, SMS)
+-----------------------------------------------------------------------------------+
| LAYER 5: MLOPS, ORCHESTRATION & GRAPH ISOLATION ENGINE                            |
| - Graph Theory Isolation Index: Bridge & Cut-Vertex Failure Simulation            |
| - Evacuation Route & Airdrop Prioritization | MLflow Registry | TFLite Quantizer  |
+-----------------------------------------------------------------------------------+
                                        ^
                                        | (Processed Inferences & Hazard Polygons)
+-----------------------------------------------------------------------------------+
| LAYER 4: AI ENGINE & HYBRID ENSEMBLE CORE                                         |
| - Physics-Informed Neural Network (PINN: Slope Stability & Factor of Safety FS)   |
| - LSTM Rainfall Forecaster (6-12h Horizon) | XGBoost Tabular Classifier           |
| - Computer Vision Crack Displacement Engine (Lucas-Kanade / Farneback Flow)       |
+-----------------------------------------------------------------------------------+
                                        ^
                                        | (Feature Vectors, Time-Series Windows)
+-----------------------------------------------------------------------------------+
| LAYER 3: DISTRIBUTED DATA LAKE & SPATIAL REPOSITORIES                             |
| - PostgreSQL 16 + PostGIS 3.4 (Geometries, DEM, Administrative Polygons)          |
| - TimescaleDB (IoT Sensor Hypertables, Continuous Hourly Aggregates)              |
| - Neo4j 5.x Graph DB (Road Networks, Bridges, Village Connectivity Graph)        |
+-----------------------------------------------------------------------------------+
                                        ^
                                        | (Normalized Events & Streams)
+-----------------------------------------------------------------------------------+
| LAYER 2: INGESTION & MESSAGING FABRIC                                             |
| - EMQX MQTT Broker Cluster (mTLS Field Gateway)                                  |
| - Apache Kafka Event Bus (Topics: telemetry-raw, citizen-reports, inferences)     |
| - External Harvesters: IMD Weather Doppler, GSI Bhukosh, ISRO Bhuvan Satellite   |
+-----------------------------------------------------------------------------------+
                                        ^
                                        | (Sensor Telemetry & Citizen Data)
+-----------------------------------------------------------------------------------+
| LAYER 1: EDGE PERCEPTION & FIELD TELEMETRY                                        |
| - Geotechnical Sensor Nodes (Pore Pressure, Tilt-meter, Soil Moisture, Rain)     |
| - LoRaWAN / Cellular IoT Gateways (Solar + Battery Backup)                        |
| - Citizen Mobile App (Offline Edge AI: TinyML TFLite int8 on-device inference)    |
+-----------------------------------------------------------------------------------+
```

---

## 2. In-Depth Layer Breakdown

### Layer 1: Edge Perception & Field Telemetry
1. **In-Situ Geotechnical Instrumentation**:
   - **Vibrating Wire Piezometers**: Deployed in boreholes at 2m, 5m, and 10m depths to record pore water pressure (\(u\)), the primary trigger for effective stress reduction.
   - **Biaxial MEMS Inclinometers / Tilt Meters**: Mounted on retaining walls, highway cuttings, and anchor piles to detect microscopic creep (\(\Delta \theta / \Delta t\) in degrees/hr).
   - **Time-Domain Reflectometry (TDR) Soil Moisture Probes**: Quantifies volumetric water content (\(\theta_w\)).
   - **Tipping Bucket Rain Gauges**: Captures localized precipitation at 0.2mm resolution.
2. **LoRaWAN / Satellite Field Gateways**:
   - Deployed at high ridge lines with direct line-of-sight to sensor nodes across a 5km radius.
   - Dual-uplink failover: 4G LTE primary with fallback to ISRO INSAT-3DR satellite telemetry transponders for zero-cellular conditions.
3. **Citizen Field Perception (TinyML Mobile Edge)**:
   - Embedded int8 quantized neural network (`tiny_landslide_pinn.tflite`) running on mobile devices.
   - Uses device accelerometer and user-reported observations (spring discharge muddy water, tension cracks, leaning trees) to perform immediate on-device risk assessment.

### Layer 2: Ingestion & Messaging Fabric
1. **EMQX MQTT Broker Cluster**:
   - Terminates TLS 1.3 connections from field gateways on port 8883.
   - Handles device keep-alives, offline message queuing (QoS 1 & QoS 2), and payload validation.
2. **Apache Kafka Event Bus**:
   - High-throughput message retention with partitioned topics:
     - `ner.sensors.telemetry.v1`: Sensor data partitioned by `node_id`.
     - `ner.citizen.reports.v1`: Citizen uploaded photos and GPS points.
     - `ner.external.imd.v1`: Ingested rainfall forecasts and satellite grids.
3. **External Government API Connectors**:
   - **IMD Mausam / Radar**: Hourly rainfall rate, 24h cumulative, and 3h Doppler rainfall predictions.
   - **GSI Bhukosh**: Lithological boundaries, thrust fault lines (Main Central Thrust, Main Boundary Thrust), shear zones.
   - **ISRO Bhuvan / MOSDAC**: ALOS/Sentinel-1 InSAR ground deformation maps and CartoDEM elevation rasters.

### Layer 3: Distributed Data Lake & Polyglot Persistence
1. **PostgreSQL + PostGIS**:
   - Stores spatial layers: Administrative boundaries (state, district, block, village), road networks, river courses, and spatial grid cells.
   - Spatial indexing using GiST and R-tree for sub-millisecond point-in-polygon queries.
2. **TimescaleDB**:
   - Sensor time-series hypertables with automated chunking every 7 days.
   - Continuous aggregates computing rolling 1-hour, 3-hour, 24-hour, and 72-hour antecedent rainfall indices.
3. **Neo4j Graph Database**:
   - Models the North Eastern road and settlement topology.
   - Nodes represent settlements (with population and demographics), road intersections, highway bridges, and hospital relief hubs.
   - Edges represent physical road links with distance and dynamic landslide hazard ratings.
4. **Redis Cache & BullMQ Store**:
   - Stores active alert states, rate-limiting tokens, session state, and priority dispatch queues.

### Layer 4: AI & Analytical Engine (Hybrid Physics-Informed Core)
1. **Physics-Informed Neural Network (PINN)**:
   - Formulates loss as \(\mathcal{L} = \mathcal{L}_{supervised} + \lambda \mathcal{L}_{physics}\).
   - Incorporates the Mohr-Coulomb failure criterion and the infinite slope model for Factor of Safety (\(FS\)):
     \[
     FS = \frac{c' + (\gamma \cdot z - u) \cos^2\beta \tan\phi'}{\gamma \cdot z \sin\beta \cos\beta}
     \]
     where \(c'\) = effective cohesion (kPa), \(\phi'\) = internal friction angle (\(^\circ\)), \(\gamma\) = soil unit weight (\(\text{kN/m}^3\)), \(z\) = soil mantle depth (m), \(\beta\) = slope angle (\(^\circ\)), and \(u\) = pore water pressure (kPa).
   - Eliminates unphysical false alarms during periods when rain occurs on low-angle, structurally stable bedrock slopes.
2. **Temporal LSTM / Transformer Forecaster**:
   - Sequences historical antecedent rainfall and forecasts moisture saturation index over 6 to 12 hour lead windows.
3. **Gradient Boosted Decision Trees (XGBoost)**:
   - Evaluates static terrain features (slope, aspect, elevation, plan curvature, profile curvature, lithology, distance to road cut, distance to tectonic fault, NDVI).
   - Trained with scale positive weight adjustment to achieve **Recall > 0.90**.
4. **Computer Vision (U-Net + Lucas-Kanade Optical Flow)**:
   - Compares paired photographs of ground/road tension cracks captured at time \(t_1\) and \(t_2\).
   - Computes sub-pixel displacement vectors to calculate crack widening rate in mm/day.

### Layer 5: MLOps, Orchestration & Graph Theory Isolation Index
1. **Graph Theory Village Isolation Index Module**:
   - Executes single-point-of-failure analysis on the Himalayan arterial highway system.
   - Identifies which settlements become landlocked when specific road links or bridges fail.
   - Generates prioritized evacuation orders and identifies safe alternate walking tracks.
2. **MLOps Lifecycle (MLflow & Automated TFLite Export)**:
   - Retrains models monthly as new landslide inventories and sensor streams are validated by geotechnical engineers.
   - Automatic post-training quantization to int8 TFLite binaries for mobile deployment.

### Layer 6: Presentation & Multi-Channel Alert Engine
1. **React.js + CesiumJS 3D Command Center**:
   - Interactive 3D globe showing Himalayan terrain with extruded hazard zones, live sensor statuses, and simulated road cuts.
2. **NestJS Alert Microservice**:
   - Validates hazard triggers, serializes alerts into Common Alerting Protocol (CAP-IN v1.2) XML for NDMA SACHET, and dispatches bulk SMS via CDAC and IVR calls via telecom gateways.
3. **Flutter Mobile Client**:
   - Cross-platform Android/iOS client featuring full offline local SQLite storage, on-device TinyML risk inference, and multilingual user interface.
