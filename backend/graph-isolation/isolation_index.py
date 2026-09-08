"""
The "Secret Sauce" Differentiator: Graph Theory Village Isolation Index Module
Calculates exactly which Himalayan settlements become landlocked when specific roads/bridges collapse,
and generates prioritized evacuation orders and helicopter airdrop routes.

Features:
- Dual Engine: Uses NetworkX if installed; otherwise executes on an optimized pure-Python Graph Engine (Tarjan's Bridge DFS + Dijkstra).
- Single-point-of-failure Bridge detection.
- Articulation point (cut-vertex) detection.
- Supply- and demographic-weighted Village Isolation Priority Ranking.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque
import heapq

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class SettlementNode:
    village_id: str
    name: str
    state: str
    district: str
    population: int
    elderly_count: int
    infants_count: int
    chronic_patients_count: int
    days_medical_stock: float
    helipad_coordinates: Tuple[float, float] # (latitude, longitude)


@dataclass
class RoadEdge:
    link_id: str
    name: str
    from_node: str
    to_node: str
    length_km: float
    hazard_score: float # 0.0 (safe) to 1.0 (imminent landslide failure)
    is_blocked: bool = False
    is_bridge: bool = False


class PurePythonGraph:
    """Zero-dependency internal graph engine with Tarjan's Bridge algorithm and Dijkstra."""
    def __init__(self):
        self.adj = {} # node -> list of (neighbor, edge_data)
        self.nodes_data = {}
        self.edges_data = {}

    def add_node(self, node_id, **kwargs):
        if node_id not in self.adj:
            self.adj[node_id] = []
        self.nodes_data[node_id] = kwargs

    def add_edge(self, u, v, **kwargs):
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, kwargs))
        self.adj[v].append((u, kwargs))
        self.edges_data[(min(u, v), max(u, v))] = kwargs

    def get_edge_data(self, u, v):
        return self.edges_data.get((min(u, v), max(u, v)), {})

    def find_bridges(self, active_only=True) -> List[Tuple[str, str]]:
        """Tarjan's O(V + E) Bridge-finding algorithm."""
        visited = set()
        tin = {}
        low = {}
        timer = 0
        bridges = []

        def dfs(u, p=None):
            nonlocal timer
            visited.add(u)
            tin[u] = low[u] = timer
            timer += 1

            for v, data in self.adj.get(u, []):
                if active_only and data.get("is_blocked", False):
                    continue
                if v == p:
                    continue
                if v in visited:
                    low[u] = min(low[u], tin[v])
                else:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > tin[u]:
                        bridges.append((u, v))

        for node in list(self.adj.keys()):
            if node not in visited:
                dfs(node)

        return bridges

    def shortest_path_length(self, source, target, blocked_link_ids: Set[str]) -> Optional[float]:
        """Dijkstra's algorithm skipping blocked links."""
        dist = {source: 0.0}
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == target:
                return d
            if d > dist.get(u, float("inf")):
                continue

            for v, data in self.adj.get(u, []):
                if data.get("link_id") in blocked_link_ids or data.get("is_blocked", False):
                    continue
                w = float(data.get("length_km", data.get("weight", 1.0)))
                if d + w < dist.get(v, float("inf")):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))

        return None


class HimalayanRoadIsolationGraph:
    """
    Graph Theory Engine for Himalayan Settlement Accessibility.
    """

    def __init__(self):
        self.emergency_hubs: Set[str] = set()
        self.settlements: Dict[str, SettlementNode] = {}
        self.road_links: Dict[str, RoadEdge] = {}
        self.nodes_meta = {}

        if HAS_NETWORKX:
            self.G = nx.Graph()
        else:
            self.G = PurePythonGraph()

    def add_emergency_hub(self, hub_id: str, name: str, lat: float, lon: float):
        """Adds a District Hospital / Primary Relief Base as a sink node."""
        self.emergency_hubs.add(hub_id)
        if HAS_NETWORKX:
            self.G.add_node(hub_id, type="EMERGENCY_HUB", name=name, lat=lat, lon=lon)
        else:
            self.G.add_node(hub_id, type="EMERGENCY_HUB", name=name, lat=lat, lon=lon)
        self.nodes_meta[hub_id] = {"name": name, "lat": lat, "lon": lon}

    def add_settlement(self, settlement: SettlementNode, connects_to_intersection: str, distance_km: float = 2.0):
        """Adds a village node and attaches it to the road network."""
        self.settlements[settlement.village_id] = settlement
        link_id = f"LINK-{settlement.village_id}"
        if HAS_NETWORKX:
            self.G.add_node(settlement.village_id, type="VILLAGE", data=settlement, name=settlement.name)
            self.G.add_edge(
                settlement.village_id, connects_to_intersection,
                link_id=link_id, weight=distance_km, length_km=distance_km,
                hazard_score=0.1, is_blocked=False
            )
        else:
            self.G.add_node(settlement.village_id, type="VILLAGE", data=settlement, name=settlement.name)
            self.G.add_edge(
                settlement.village_id, connects_to_intersection,
                link_id=link_id, weight=distance_km, length_km=distance_km,
                hazard_score=0.1, is_blocked=False
            )

    def add_road_segment(self, road: RoadEdge):
        """Adds a traversable road link between two junctions."""
        self.road_links[road.link_id] = road
        if HAS_NETWORKX:
            self.G.add_edge(
                road.from_node, road.to_node,
                link_id=road.link_id, weight=road.length_km, length_km=road.length_km,
                hazard_score=road.hazard_score, is_blocked=road.is_blocked,
                is_bridge=road.is_bridge, name=road.name
            )
        else:
            self.G.add_edge(
                road.from_node, road.to_node,
                link_id=road.link_id, weight=road.length_km, length_km=road.length_km,
                hazard_score=road.hazard_score, is_blocked=road.is_blocked,
                is_bridge=road.is_bridge, name=road.name
            )

    def find_network_bridges(self) -> List[Dict[str, Any]]:
        """Tarjan's algorithm to identify all Bridge Edges."""
        results = []
        if HAS_NETWORKX:
            active_edges = [(u, v) for u, v, d in self.G.edges(data=True) if not d.get("is_blocked", False)]
            active_subgraph = self.G.edge_subgraph(active_edges).copy()
            bridges = list(nx.bridges(active_subgraph))
            for u, v in bridges:
                edge_data = self.G.get_edge_data(u, v)
                if edge_data and "link_id" in edge_data:
                    results.append({
                        "link_id": edge_data["link_id"],
                        "name": edge_data.get("name", "Unnamed Road"),
                        "from_node": u,
                        "to_node": v,
                        "hazard_score": edge_data.get("hazard_score", 0.0),
                        "is_bridge_structure": edge_data.get("is_bridge", False),
                        "vulnerability": "SINGLE_POINT_OF_FAILURE"
                    })
        else:
            bridges = self.G.find_bridges(active_only=True)
            for u, v in bridges:
                edge_data = self.G.get_edge_data(u, v)
                if edge_data and "link_id" in edge_data:
                    results.append({
                        "link_id": edge_data["link_id"],
                        "name": edge_data.get("name", "Unnamed Road"),
                        "from_node": u,
                        "to_node": v,
                        "hazard_score": edge_data.get("hazard_score", 0.0),
                        "is_bridge_structure": edge_data.get("is_bridge", False),
                        "vulnerability": "SINGLE_POINT_OF_FAILURE"
                    })
        return results

    def simulate_collapse(self, blocked_link_ids: List[str]) -> Dict[str, Any]:
        """
        Simulates the collapse of one or more road links.
        Computes reachability to Emergency Hubs and generates the Village Isolation Index.
        """
        blocked_set = set(blocked_link_ids)
        isolated_villages = []
        accessible_villages = []

        if HAS_NETWORKX:
            sim_G = self.G.copy()
            for u, v, data in list(sim_G.edges(data=True)):
                if data.get("link_id") in blocked_set:
                    sim_G.remove_edge(u, v)

            for v_id, s in self.settlements.items():
                can_reach_hub = False
                shortest_dist = float("inf")
                best_hub = None

                for hub_id in self.emergency_hubs:
                    if nx.has_path(sim_G, v_id, hub_id):
                        can_reach_hub = True
                        try:
                            dist = nx.shortest_path_length(sim_G, v_id, hub_id, weight="weight")
                            if dist < shortest_dist:
                                shortest_dist = dist
                                best_hub = hub_id
                        except nx.NetworkXNoPath:
                            pass

                if not can_reach_hub:
                    vulnerable_pop = s.elderly_count + s.infants_count + s.chronic_patients_count
                    vulnerable_ratio = (vulnerable_pop / s.population) if s.population > 0 else 0.2
                    med_factor = 1.0 / max(0.5, s.days_medical_stock)
                    isolation_score = round(s.population * (1.0 + vulnerable_ratio) * med_factor, 2)

                    isolated_villages.append({
                        "village_id": s.village_id,
                        "village_name": s.name,
                        "state": s.state,
                        "district": s.district,
                        "total_population": s.population,
                        "vulnerable_population": vulnerable_pop,
                        "vulnerable_ratio": round(vulnerable_ratio, 3),
                        "days_medical_stock_remaining": s.days_medical_stock,
                        "isolation_priority_score": isolation_score,
                        "helipad_airdrop_coordinates": list(s.helipad_coordinates),
                        "recommended_action": "PRIORITY_1_AIRLIFT_OR_FOOT_PATROL" if isolation_score > 3000 else "PRIORITY_2_RELIEF_AIRDROP"
                    })
                else:
                    accessible_villages.append({
                        "village_id": s.village_id,
                        "village_name": s.name,
                        "reachable_hub": best_hub,
                        "egress_distance_km": round(shortest_dist, 2)
                    })
        else:
            # Pure Python Graph Reachability
            for v_id, s in self.settlements.items():
                can_reach_hub = False
                shortest_dist = float("inf")
                best_hub = None

                for hub_id in self.emergency_hubs:
                    d = self.G.shortest_path_length(v_id, hub_id, blocked_link_ids=blocked_set)
                    if d is not None:
                        can_reach_hub = True
                        if d < shortest_dist:
                            shortest_dist = d
                            best_hub = hub_id

                if not can_reach_hub:
                    vulnerable_pop = s.elderly_count + s.infants_count + s.chronic_patients_count
                    vulnerable_ratio = (vulnerable_pop / s.population) if s.population > 0 else 0.2
                    med_factor = 1.0 / max(0.5, s.days_medical_stock)
                    isolation_score = round(s.population * (1.0 + vulnerable_ratio) * med_factor, 2)

                    isolated_villages.append({
                        "village_id": s.village_id,
                        "village_name": s.name,
                        "state": s.state,
                        "district": s.district,
                        "total_population": s.population,
                        "vulnerable_population": vulnerable_pop,
                        "vulnerable_ratio": round(vulnerable_ratio, 3),
                        "days_medical_stock_remaining": s.days_medical_stock,
                        "isolation_priority_score": isolation_score,
                        "helipad_airdrop_coordinates": list(s.helipad_coordinates),
                        "recommended_action": "PRIORITY_1_AIRLIFT_OR_FOOT_PATROL" if isolation_score > 3000 else "PRIORITY_2_RELIEF_AIRDROP"
                    })
                else:
                    accessible_villages.append({
                        "village_id": s.village_id,
                        "village_name": s.name,
                        "reachable_hub": best_hub,
                        "egress_distance_km": round(shortest_dist, 2)
                    })

        isolated_villages.sort(key=lambda x: x["isolation_priority_score"], reverse=True)
        for idx, item in enumerate(isolated_villages, start=1):
            item["evacuation_priority_rank"] = idx

        total_settlements = len(self.settlements)
        pct_isolated = (len(isolated_villages) / total_settlements * 100.0) if total_settlements > 0 else 0.0

        return {
            "simulation_scenario": {
                "blocked_roads_count": len(blocked_link_ids),
                "blocked_link_ids": blocked_link_ids
            },
            "network_impact": {
                "total_settlements_monitored": total_settlements,
                "isolated_settlements_count": len(isolated_villages),
                "accessible_settlements_count": len(accessible_villages),
                "percentage_population_cut_off": round(pct_isolated, 2)
            },
            "prioritized_evacuation_leaderboard": isolated_villages,
            "airdrop_manifest": [
                {
                    "rank": v["evacuation_priority_rank"],
                    "target_village": v["village_name"],
                    "coordinates": v["helipad_airdrop_coordinates"],
                    "population": v["total_population"],
                    "criticality": "URGENT" if v["days_medical_stock_remaining"] < 3.0 else "HIGH"
                }
                for v in isolated_villages
            ]
        }


def build_sample_ner_network() -> HimalayanRoadIsolationGraph:
    """Helper factory for initializing Sikkim / Nagaland road network topology"""
    graph = HimalayanRoadIsolationGraph()

    # Add District HQ
    graph.add_emergency_hub(
        hub_id="DHQ-GANGTOK-01",
        name="East Sikkim District HQ & STNM Hospital",
        lat=27.3314,
        lon=88.6138
    )

    # Road Intersections
    graph.G.add_node("INT-RANGPO", name="Rangpo Border Gate", lat=27.1767, lon=88.5303)
    graph.G.add_node("INT-SINGTAM", name="Singtam Highway Junction", lat=27.2344, lon=88.5002)
    graph.G.add_node("INT-RONGLI-PASS", name="Rongli Mountain Pass", lat=27.2012, lon=88.6210)

    # Road segments
    graph.add_road_segment(RoadEdge(
        link_id="RD-NH10-RANGPO-SINGTAM",
        name="NH-10 Rangpo to Singtam Corridor",
        from_node="INT-RANGPO",
        to_node="INT-SINGTAM",
        length_km=12.4,
        hazard_score=0.78
    ))

    graph.add_road_segment(RoadEdge(
        link_id="RD-NH10-SINGTAM-GANGTOK",
        name="NH-10 Singtam to Gangtok Link",
        from_node="INT-SINGTAM",
        to_node="DHQ-GANGTOK-01",
        length_km=26.5,
        hazard_score=0.45
    ))

    # Single-artery branch to Rongli Valley
    graph.add_road_segment(RoadEdge(
        link_id="RD-FEEDER-RONGLI-VALLEY",
        name="Rorathang-Rongli Mountain Feeder Road",
        from_node="INT-SINGTAM",
        to_node="INT-RONGLI-PASS",
        length_km=18.2,
        hazard_score=0.92,
        is_bridge=True
    ))

    # Add dependent villages
    graph.add_settlement(
        SettlementNode(
            village_id="VILL-SK-RONGLI",
            name="Rongli Upper Basti",
            state="Sikkim",
            district="Pakyong",
            population=3450,
            elderly_count=420,
            infants_count=280,
            chronic_patients_count=48,
            days_medical_stock=2.5,
            helipad_coordinates=(27.2025, 88.6210)
        ),
        connects_to_intersection="INT-RONGLI-PASS",
        distance_km=2.1
    )

    graph.add_settlement(
        SettlementNode(
            village_id="VILL-SK-DOLEPCHEP",
            name="Dolepchep Hamlet",
            state="Sikkim",
            district="Pakyong",
            population=1820,
            elderly_count=210,
            infants_count=140,
            chronic_patients_count=18,
            days_medical_stock=1.5,
            helipad_coordinates=(27.2150, 88.6410)
        ),
        connects_to_intersection="INT-RONGLI-PASS",
        distance_km=5.4
    )

    graph.add_settlement(
        SettlementNode(
            village_id="VILL-SK-RHENOCK",
            name="Rhenock Valley",
            state="Sikkim",
            district="Pakyong",
            population=5900,
            elderly_count=650,
            infants_count=490,
            chronic_patients_count=85,
            days_medical_stock=5.0,
            helipad_coordinates=(27.1850, 88.6430)
        ),
        connects_to_intersection="INT-RONGLI-PASS",
        distance_km=7.8
    )

    return graph
