// =========================================================================================
// AI LANDSLIDE EARLY WARNING SYSTEM (NER, INDIA) - NEO4J ROAD & VILLAGE GRAPH SCHEMA
// Designed for Problem Statement ID: 26001 (Ministry of MDoNER)
// Target: Neo4j 5.x Graph Engine
// =========================================================================================

// -----------------------------------------------------------------------------------------
// 1. Constraints & Indexes
// -----------------------------------------------------------------------------------------

CREATE CONSTRAINT unique_village_id IF NOT EXISTS
FOR (v:Village) REQUIRE v.village_id IS UNIQUE;

CREATE CONSTRAINT unique_intersection_id IF NOT EXISTS
FOR (i:RoadIntersection) REQUIRE i.intersection_id IS UNIQUE;

CREATE CONSTRAINT unique_road_segment_id IF NOT EXISTS
FOR (r:RoadSegment) REQUIRE r.link_id IS UNIQUE;

CREATE CONSTRAINT unique_bridge_id IF NOT EXISTS
FOR (b:Bridge) REQUIRE b.bridge_id IS UNIQUE;

CREATE CONSTRAINT unique_hub_id IF NOT EXISTS
FOR (h:EmergencyHub) REQUIRE h.hub_id IS UNIQUE;

CREATE INDEX village_district_idx IF NOT EXISTS
FOR (v:Village) ON (v.state, v.district);

// -----------------------------------------------------------------------------------------
// 2. Sample Graph Topology for Critical Himalayan Arterial Corridors (NH-10 & NH-29)
// -----------------------------------------------------------------------------------------

// Emergency Command Centers / District Headquarters
MERGE (dhq1:EmergencyHub {
    hub_id: "DHQ-GANGTOK-01",
    name: "East Sikkim District HQ & STNM Multi-Specialty Hospital",
    type: "LEVEL_1_TRAUMA_CENTER",
    latitude: 27.3314,
    longitude: 88.6138,
    helipad_available: true,
    bed_capacity: 450,
    ration_stock_days: 45
});

MERGE (dhq2:EmergencyHub {
    hub_id: "DHQ-KOHIMA-01",
    name: "Kohima District HQ & State Disaster Relief Hub",
    type: "DISTRICT_HQ_AND_RELIEF_BASE",
    latitude: 25.6751,
    longitude: 94.1086,
    helipad_available: true,
    bed_capacity: 320,
    ration_stock_days: 60
});

// Strategic Road Intersections on National Highway Corridors
MERGE (int_nh10_km35:RoadIntersection {
    intersection_id: "INT-NH10-KM35",
    name: "Rangpo Border Junction",
    latitude: 27.1767,
    longitude: 88.5303,
    elevation_m: 350.0
});

MERGE (int_nh10_km44:RoadIntersection {
    intersection_id: "INT-NH10-KM44",
    name: "Singtam Highway Fork",
    latitude: 27.2344,
    longitude: 88.5002,
    elevation_m: 420.0
});

MERGE (int_rongli_f:RoadIntersection {
    intersection_id: "INT-RONGLI-FORK",
    name: "Rorathang-Rongli Mountain Pass Intersection",
    latitude: 27.2012,
    longitude: 88.6210,
    elevation_m: 1120.0
});

// Single Point of Failure Bridge Structure
MERGE (brg_teesta_04:Bridge {
    bridge_id: "BRG-TEESTA-04",
    name: "Teesta Bridge at Mile 44",
    length_m: 145.0,
    carrying_capacity_tons: 40.0,
    structural_health_status: "CRITICAL_SCOUR_WATCH",
    latitude: 27.2340,
    longitude: 88.5010
});

// Remote Hill Villages dependent on arterial corridor
MERGE (v1:Village {
    village_id: "VILL-SK-RONGLI-UPPER",
    name: "Rongli Upper Basti",
    state: "Sikkim",
    district: "Pakyong",
    population: 3450,
    elderly_count: 420,
    infants_count: 280,
    chronic_patients_count: 48,
    primary_medical_stock_days: 3.5,
    helipad_point: [88.6210, 27.2025]
});

MERGE (v2:Village {
    village_id: "VILL-SK-DOLEPCHEP",
    name: "Dolepchep Hamlet",
    state: "Sikkim",
    district: "Pakyong",
    population: 1820,
    elderly_count: 210,
    infants_count: 140,
    chronic_patients_count: 18,
    primary_medical_stock_days: 2.0,
    helipad_point: [88.6410, 27.2150]
});

MERGE (v3:Village {
    village_id: "VILL-SK-RHENOCK",
    name: "Rhenock Valley Settlement",
    state: "Sikkim",
    district: "Pakyong",
    population: 5900,
    elderly_count: 650,
    infants_count: 490,
    chronic_patients_count: 85,
    primary_medical_stock_days: 6.0,
    helipad_point: [88.6430, 27.1850]
});

// Establish Road Segments (Links) with dynamic landslide hazard weights
MERGE (int_nh10_km35)-[:ROAD_LINK {
    link_id: "RD-NH10-RANGPO-SINGTAM",
    name: "NH-10 Rangpo to Singtam Corridor",
    length_km: 12.4,
    hazard_score: 0.88, // high landslide probability
    is_blocked: false,
    speed_limit_kmh: 40
}]->(int_nh10_km44);

MERGE (int_nh10_km44)-[:ROAD_LINK {
    link_id: "RD-NH10-SINGTAM-GANGTOK",
    name: "NH-10 Singtam to Gangtok Main Link",
    length_km: 26.5,
    hazard_score: 0.45,
    is_blocked: false,
    speed_limit_kmh: 50
}]->(dhq1);

// Vulnerable single-artery branch off Singtam to Rongli pass
MERGE (int_nh10_km44)-[:ROAD_LINK {
    link_id: "RD-FEEDER-RONGLI-VALLEY",
    name: "Rorathang-Rongli Mountain Arterial Pass",
    length_km: 18.2,
    hazard_score: 0.94, // EXTREME RISK - Single Point of Failure
    is_blocked: false,
    has_bridge: true,
    speed_limit_kmh: 25
}]->(int_rongli_f);

MERGE (int_rongli_f)-[:CONNECTS_VILLAGE {distance_km: 2.1}]->(v1);
MERGE (int_rongli_f)-[:CONNECTS_VILLAGE {distance_km: 5.4}]->(v2);
MERGE (int_rongli_f)-[:CONNECTS_VILLAGE {distance_km: 7.8}]->(v3);

// -----------------------------------------------------------------------------------------
// 3. CORE ALGORITHM: Graph Theory Village Isolation Index Calculation Query
// -----------------------------------------------------------------------------------------
// This query simulates the collapse of road link 'RD-FEEDER-RONGLI-VALLEY' (or any blocked segment)
// and computes which villages become completely unreachable from District Hospital (DHQ)
// outputting the ranked Evacuation Priority Score.

MATCH (targetRoad:RoadSegment {link_id: "RD-FEEDER-RONGLI-VALLEY"})
SET targetRoad.is_blocked = true;

// Query to detect Isolated Villages:
MATCH (v:Village)
OPTIONAL MATCH p = shortestPath((v)-[:CONNECTS_VILLAGE|ROAD_LINK*]-(dhq:EmergencyHub {hub_id: "DHQ-GANGTOK-01"}))
WHERE ALL(r IN relationships(p) WHERE r.is_blocked IS NULL OR r.is_blocked = false)
WITH v, p
WHERE p IS NULL // No active non-blocked path exists to hospital
WITH v,
     v.population AS pop,
     (v.elderly_count + v.infants_count + v.chronic_patients_count) AS vulnerable_pop,
     v.primary_medical_stock_days AS med_days
// Isolation Index Formula:
// I_iso = (pop) * (1 + vulnerable_pop / pop) * (1 / max(0.5, med_days))
RETURN
    v.village_id AS village_id,
    v.name AS village_name,
    v.state AS state,
    v.district AS district,
    pop AS total_population,
    vulnerable_pop AS vulnerable_population,
    med_days AS days_medical_supplies_remaining,
    round((pop * (1.0 + (toFloat(vulnerable_pop) / pop)) * (1.0 / med_days)), 2) AS isolation_priority_index,
    v.helipad_point AS helicopter_airdrop_coordinates
ORDER BY isolation_priority_index DESC;

// -----------------------------------------------------------------------------------------
// 4. Query to Detect All "Bridge Edges" (Single Points of Failure) in Network
// -----------------------------------------------------------------------------------------
MATCH (u:RoadIntersection)-[r:ROAD_LINK]-(w:RoadIntersection)
WHERE NOT r.is_blocked = true
WITH u, w, r
// Check if removing edge r destroys connectivity between u and any Emergency Hub
MATCH p = shortestPath((u)-[:ROAD_LINK*]-(dhq:EmergencyHub))
WHERE NOT r IN relationships(p)
WITH u, w, r, count(p) AS alternate_paths
WHERE alternate_paths = 0
RETURN
    r.link_id AS critical_bottleneck_link_id,
    r.name AS road_name,
    r.hazard_score AS current_landslide_hazard,
    u.name AS from_junction,
    w.name AS to_junction,
    "SINGLE_POINT_OF_FAILURE_BRIDGE" AS vulnerability_type
ORDER BY r.hazard_score DESC;
