-- =========================================================================================
-- AI LANDSLIDE EARLY WARNING SYSTEM (NER, INDIA) - POSTGRESQL + POSTGIS SPATIAL SCHEMA
-- Designed for Problem Statement ID: 26001 (Ministry of MDoNER)
-- Target: PostgreSQL 16+ with PostGIS 3.4+ Extension
-- =========================================================================================

-- Enable required spatial and cryptographic extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "postgis_raster";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Set spatial reference identifier standard (WGS 84 / Geographic 2D)
SET standard_conforming_strings = on;

-- -----------------------------------------------------------------------------------------
-- 1. Administrative Boundaries (NER States, Districts, Blocks, Villages)
-- -----------------------------------------------------------------------------------------

CREATE TABLE administrative_regions (
    region_id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    state_name VARCHAR(64) NOT NULL CHECK (state_name IN (
        'Assam', 'Meghalaya', 'Arunachal Pradesh', 'Nagaland',
        'Manipur', 'Mizoram', 'Tripura', 'Sikkim'
    )),
    district_name VARCHAR(64) NOT NULL,
    subdivision VARCHAR(64),
    population INTEGER NOT NULL DEFAULT 0,
    vulnerable_pop_ratio NUMERIC(4,3) DEFAULT 0.200, -- elderly, infants, disabled
    medical_facility_count INT DEFAULT 0,
    emergency_contact_phone VARCHAR(32),
    geom GEOMETRY(MultiPolygon, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_admin_geom ON administrative_regions USING GIST(geom);
CREATE INDEX idx_admin_state_district ON administrative_regions(state_name, district_name);

-- -----------------------------------------------------------------------------------------
-- 2. Landslide Susceptibility Spatial 10m Grid Cells (Precomputed Conditioning Features)
-- -----------------------------------------------------------------------------------------

CREATE TABLE terrain_grid_cells (
    grid_id VARCHAR(64) PRIMARY KEY,
    region_id VARCHAR(32) REFERENCES administrative_regions(region_id),
    elevation_m NUMERIC(7,2) NOT NULL,
    slope_deg NUMERIC(5,2) NOT NULL,
    aspect_deg NUMERIC(5,2) NOT NULL,
    plan_curvature NUMERIC(6,4) DEFAULT 0.0,
    profile_curvature NUMERIC(6,4) DEFAULT 0.0,
    lithology_code VARCHAR(32) NOT NULL,
    soil_type VARCHAR(64) NOT NULL,
    soil_depth_m NUMERIC(4,2) DEFAULT 2.0,
    cohesion_kpa NUMERIC(6,2) NOT NULL DEFAULT 10.0,
    friction_angle_deg NUMERIC(4,2) NOT NULL DEFAULT 30.0,
    unit_weight_kn_m3 NUMERIC(4,2) NOT NULL DEFAULT 18.0,
    distance_to_road_m NUMERIC(8,2) DEFAULT 9999.0,
    distance_to_fault_m NUMERIC(8,2) DEFAULT 9999.0,
    distance_to_river_m NUMERIC(8,2) DEFAULT 9999.0,
    land_use_class VARCHAR(64) NOT NULL,
    ndvi NUMERIC(4,3) DEFAULT 0.500,
    historical_event_count INT DEFAULT 0,
    geom GEOMETRY(Polygon, 4326) NOT NULL,
    centroid GEOMETRY(Point, 4326) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_terrain_grid_geom ON terrain_grid_cells USING GIST(geom);
CREATE INDEX idx_terrain_grid_centroid ON terrain_grid_cells USING GIST(centroid);
CREATE INDEX idx_terrain_slope ON terrain_grid_cells(slope_deg);

-- -----------------------------------------------------------------------------------------
-- 3. Geological Survey of India (GSI) Historical Landslide Inventory Catalog
-- -----------------------------------------------------------------------------------------

CREATE TABLE landslide_inventory (
    event_id VARCHAR(64) PRIMARY KEY,
    gsi_code VARCHAR(64),
    state VARCHAR(64) NOT NULL,
    district VARCHAR(64) NOT NULL,
    event_date DATE NOT NULL,
    event_time TIME,
    severity VARCHAR(32) NOT NULL CHECK (severity IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL', 'CATASTROPHIC')),
    trigger_cause VARCHAR(64) NOT NULL CHECK (trigger_cause IN (
        'MONSOON_TORRENTIAL_RAIN', 'CLOUD_BURST', 'CONTINUOUS_SOIL_SATURATION',
        'ROAD_CUTTING_SLOPE_TOE', 'EARTHQUAKE_SEISMIC', 'DEFORESTATION', 'OTHER'
    )),
    landslide_type VARCHAR(64) DEFAULT 'DEBRIS_FLOW',
    area_sq_m NUMERIC(10,2),
    depth_m NUMERIC(5,2),
    fatalities INT DEFAULT 0,
    injuries INT DEFAULT 0,
    infrastructure_impact TEXT,
    road_highway_blocked VARCHAR(64),
    validation_status VARCHAR(32) DEFAULT 'VERIFIED_GSI' CHECK (validation_status IN ('UNVERIFIED', 'VERIFIED_CITIZEN', 'VERIFIED_FIELD', 'VERIFIED_GSI')),
    location GEOMETRY(Point, 4326) NOT NULL,
    runout_polygon GEOMETRY(Polygon, 4326),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_landslide_inv_loc ON landslide_inventory USING GIST(location);
CREATE INDEX idx_landslide_date ON landslide_inventory(event_date);
CREATE INDEX idx_landslide_state ON landslide_inventory(state, district);

-- -----------------------------------------------------------------------------------------
-- 4. Arterial Highway & Rural Transportation Network
-- -----------------------------------------------------------------------------------------

CREATE TABLE transportation_links (
    link_id VARCHAR(64) PRIMARY KEY,
    highway_number VARCHAR(32), -- e.g. NH-10, NH-29
    road_name VARCHAR(128) NOT NULL,
    road_category VARCHAR(32) NOT NULL CHECK (road_category IN ('NATIONAL_HIGHWAY', 'STATE_HIGHWAY', 'MAJOR_DISTRICT_ROAD', 'RURAL_LINK', 'EVACUATION_TRAIL')),
    length_km NUMERIC(6,2) NOT NULL,
    carriageway_width_m NUMERIC(4,1) DEFAULT 7.0,
    surface_type VARCHAR(32) DEFAULT 'ASPHALT',
    is_bridge BOOLEAN DEFAULT FALSE,
    bridge_id VARCHAR(32),
    slope_vulnerability_score NUMERIC(4,3) DEFAULT 0.100, -- 0.0 to 1.0 from ML
    is_currently_blocked BOOLEAN DEFAULT FALSE,
    blockage_cause VARCHAR(128),
    start_intersection_id VARCHAR(64) NOT NULL,
    end_intersection_id VARCHAR(64) NOT NULL,
    geom GEOMETRY(LineString, 4326) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_road_geom ON transportation_links USING GIST(geom);
CREATE INDEX idx_road_blocked ON transportation_links(is_currently_blocked);

-- -----------------------------------------------------------------------------------------
-- 5. Citizen & Field Geotechnical Reports (Crowd-Sourced & Mobile Offline Sync)
-- -----------------------------------------------------------------------------------------

CREATE TABLE citizen_field_reports (
    report_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id VARCHAR(64) NOT NULL,
    reporter_role VARCHAR(32) DEFAULT 'CITIZEN' CHECK (reporter_role IN ('CITIZEN', 'VILLAGE_HEAD', 'FIELD_ENGINEER', 'BRO_OFFICER', 'NDRF_PERSONNEL')),
    reporter_phone VARCHAR(32),
    observation_type VARCHAR(64) NOT NULL CHECK (observation_type IN (
        'TENSION_CRACK_ON_SLOPE', 'ROAD_SURFACE_SUBSIDENCE', 'MUDDY_SPRING_WATER',
        'LEANING_TREES_OR_POLES', 'ROCK_FALL', 'MINOR_SLUMP', 'MAJOR_DEBRIS_FLOW'
    )),
    apparent_crack_width_cm NUMERIC(6,2),
    apparent_length_m NUMERIC(6,2),
    slope_tilt_observed BOOLEAN DEFAULT FALSE,
    water_seepage_observed BOOLEAN DEFAULT FALSE,
    description TEXT,
    primary_image_url TEXT,
    followup_image_url TEXT,
    computed_displacement_mm NUMERIC(6,2),
    confidence_score NUMERIC(4,3),
    sync_status VARCHAR(32) DEFAULT 'SYNCED' CHECK (sync_status IN ('PENDING', 'SYNCED', 'REJECTED')),
    client_created_at TIMESTAMPTZ NOT NULL,
    server_received_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    geom GEOMETRY(Point, 4326) NOT NULL
);

CREATE INDEX idx_citizen_geom ON citizen_field_reports USING GIST(geom);
CREATE INDEX idx_citizen_sync ON citizen_field_reports(sync_status, server_received_at);

-- -----------------------------------------------------------------------------------------
-- 6. Disaster Early Warning Alerts (CAP-IN v1.2 Compliant)
-- -----------------------------------------------------------------------------------------

CREATE TABLE emergency_alerts (
    alert_id VARCHAR(64) PRIMARY KEY,
    cap_identifier VARCHAR(128) UNIQUE NOT NULL,
    sender_agency VARCHAR(128) NOT NULL DEFAULT 'SDMA_EWS_SYSTEM',
    severity VARCHAR(32) NOT NULL CHECK (severity IN ('Advisory', 'Watch', 'Warning', 'Evacuation')),
    urgency VARCHAR(32) NOT NULL CHECK (urgency IN ('Immediate', 'Expected', 'Future', 'Past')),
    certainty VARCHAR(32) NOT NULL CHECK (certainty IN ('Observed', 'Likely', 'Possible', 'Unlikely')),
    headline TEXT NOT NULL,
    description TEXT NOT NULL,
    instruction TEXT NOT NULL,
    headline_assamese TEXT,
    headline_bengali TEXT,
    headline_hindi TEXT,
    headline_bodo TEXT,
    headline_khasi TEXT,
    affected_villages_count INT DEFAULT 0,
    estimated_population_at_risk INT DEFAULT 0,
    affected_polygon GEOMETRY(MultiPolygon, 4326) NOT NULL,
    sent_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    cap_xml_payload TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_alerts_geom ON emergency_alerts USING GIST(affected_polygon);
CREATE INDEX idx_alerts_active ON emergency_alerts(is_active, expires_at);
