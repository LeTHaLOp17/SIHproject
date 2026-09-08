-- =========================================================================================
-- AI LANDSLIDE EARLY WARNING SYSTEM (NER, INDIA) - TIMESCALEDB SENSOR TIME-SERIES SCHEMA
-- Designed for Problem Statement ID: 26001 (Ministry of MDoNER)
-- Target: PostgreSQL 16+ with TimescaleDB 2.14+ Extension
-- =========================================================================================

CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- -----------------------------------------------------------------------------------------
-- 1. IoT Sensor Registry (Hardware Metadata & Location)
-- -----------------------------------------------------------------------------------------

CREATE TABLE sensor_nodes (
    node_id VARCHAR(64) PRIMARY KEY,
    gateway_id VARCHAR(64) NOT NULL,
    state VARCHAR(64) NOT NULL,
    district VARCHAR(64) NOT NULL,
    slope_name VARCHAR(128) NOT NULL,
    installed_latitude NUMERIC(9,6) NOT NULL,
    installed_longitude NUMERIC(9,6) NOT NULL,
    installed_elevation_m NUMERIC(7,2) NOT NULL,
    piezometer_depth_m NUMERIC(4,2) DEFAULT 5.0,
    inclinometer_axis VARCHAR(16) DEFAULT 'BIAXIAL_XY',
    communication_channel VARCHAR(32) DEFAULT 'LORAWAN' CHECK (communication_channel IN ('LORAWAN', 'CELLULAR_4G', 'SATELLITE_INSAT')),
    battery_chemistry VARCHAR(32) DEFAULT 'LITHIUM_THIONYL',
    firmware_version VARCHAR(32) DEFAULT '1.4.2',
    last_ping_time TIMESTAMPTZ,
    status VARCHAR(32) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'MAINTENANCE_REQUIRED', 'CALIBRATING', 'DECOMMISSIONED')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sensor_district ON sensor_nodes(state, district);

-- -----------------------------------------------------------------------------------------
-- 2. Sensor Telemetry Hypertable (High-Frequency Time-Series)
-- -----------------------------------------------------------------------------------------

CREATE TABLE sensor_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(64) NOT NULL REFERENCES sensor_nodes(node_id),
    rainfall_10m_mm NUMERIC(5,2) DEFAULT 0.0,
    rainfall_1h_mm NUMERIC(6,2) DEFAULT 0.0,
    rainfall_24h_mm NUMERIC(7,2) DEFAULT 0.0,
    soil_moisture_pct NUMERIC(5,2) NOT NULL, -- 0.0 to 100.0%
    pore_water_pressure_kpa NUMERIC(7,2) NOT NULL, -- negative/positive pore pressure
    inclination_deg_x NUMERIC(6,3) NOT NULL, -- tilt angle on axis X
    inclination_deg_y NUMERIC(6,3) NOT NULL, -- tilt angle on axis Y
    tilt_velocity_deg_hr NUMERIC(7,4) DEFAULT 0.0, -- first derivative of tilt
    acceleration_magnitude_g NUMERIC(5,3) DEFAULT 1.000, -- seismic/dynamic vibration
    soil_temperature_c NUMERIC(4,1),
    battery_voltage_v NUMERIC(4,2) NOT NULL,
    signal_rssi_dbm INT NOT NULL,
    anomaly_flag BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (timestamp, node_id)
);

-- Convert to TimescaleDB hypertable with 7-day chunk interval
SELECT create_hypertable(
    'sensor_telemetry',
    'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_telemetry_node_time ON sensor_telemetry (node_id, timestamp DESC);

-- -----------------------------------------------------------------------------------------
-- 3. TimescaleDB Compression Policy (Cost-Efficient Long-Term Storage)
-- -----------------------------------------------------------------------------------------

ALTER TABLE sensor_telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'node_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Automatically compress chunks older than 14 days
SELECT add_compression_policy('sensor_telemetry', INTERVAL '14 days');

-- Retention policy: keep 5 years of historical raw sensor data
SELECT add_retention_policy('sensor_telemetry', INTERVAL '1825 days');

-- -----------------------------------------------------------------------------------------
-- 4. Continuous Materialized Aggregates (Sub-Second Dashboard Lookups)
-- -----------------------------------------------------------------------------------------

-- Continuous Aggregate 1: Hourly Summary per Node
CREATE MATERIALIZED VIEW sensor_hourly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket_time,
    node_id,
    SUM(rainfall_10m_mm) AS hourly_rainfall_total_mm,
    AVG(soil_moisture_pct) AS avg_soil_moisture_pct,
    MAX(soil_moisture_pct) AS max_soil_moisture_pct,
    AVG(pore_water_pressure_kpa) AS avg_pore_pressure_kpa,
    MAX(pore_water_pressure_kpa) AS max_pore_pressure_kpa,
    AVG(tilt_velocity_deg_hr) AS avg_tilt_velocity_deg_hr,
    MAX(ABS(tilt_velocity_deg_hr)) AS max_tilt_velocity_deg_hr,
    MIN(battery_voltage_v) AS min_battery_voltage_v
FROM sensor_telemetry
GROUP BY bucket_time, node_id
WITH NO DATA;

-- Refresh continuous aggregate every 15 minutes for current and past 2 hours
SELECT add_continuous_aggregate_policy(
    'sensor_hourly_summary',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- Continuous Aggregate 2: Daily Rolling Antecedent Rainfall Index (ARI)
CREATE MATERIALIZED VIEW sensor_daily_rainfall
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket_time,
    node_id,
    SUM(rainfall_10m_mm) AS daily_rainfall_mm,
    AVG(soil_moisture_pct) AS daily_mean_moisture_pct,
    AVG(pore_water_pressure_kpa) AS daily_mean_pore_pressure_kpa
FROM sensor_telemetry
GROUP BY bucket_time, node_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'sensor_daily_rainfall',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
