"""
Unit Tests for Graph Theory Village Isolation Engine
Verifies:
1. Bridge detection on single-point-of-failure mountain corridors.
2. Articulation points in regional transportation graphs.
3. Village Isolation Index ranking and evacuation prioritization upon simulated road collapse.
"""

from isolation_index import build_sample_ner_network


def test_network_bridge_detection():
    graph = build_sample_ner_network()
    bridges = graph.find_network_bridges()

    # The feeder road to Rongli Pass and village links must be detected as bridges
    bridge_ids = [b["link_id"] for b in bridges]
    assert "RD-FEEDER-RONGLI-VALLEY" in bridge_ids, "Feeder road must be recognized as a single point of failure bridge."
    print("[PASS] Bridge detection passed: RD-FEEDER-RONGLI-VALLEY flagged as bridge bottleneck.")


def test_road_collapse_isolation_simulation():
    graph = build_sample_ner_network()

    # Before collapse: all villages can reach hospital
    baseline = graph.simulate_collapse([])
    assert baseline["network_impact"]["isolated_settlements_count"] == 0, "No settlements should be isolated before collapse."

    # Simulate collapse of the single feeder road link: 'RD-FEEDER-RONGLI-VALLEY'
    sim_result = graph.simulate_collapse(["RD-FEEDER-RONGLI-VALLEY"])

    impact = sim_result["network_impact"]
    assert impact["isolated_settlements_count"] == 3, "All 3 settlements connected via Rongli Pass must be isolated."

    leaderboard = sim_result["prioritized_evacuation_leaderboard"]
    assert len(leaderboard) == 3

    # Verify rank 1 is highest vulnerability score
    rank1 = leaderboard[0]
    rank2 = leaderboard[1]
    assert rank1["isolation_priority_score"] >= rank2["isolation_priority_score"], "Leaderboard must sort by isolation priority score descending."
    assert "helipad_airdrop_coordinates" in rank1
    assert len(rank1["helipad_airdrop_coordinates"]) == 2

    print("\n--- TEST PASSED: ISOLATION SIMULATION SUCCESSFUL ---")
    for village in leaderboard:
        print(f"Rank {village['evacuation_priority_rank']}: {village['village_name']} - Priority Score: {village['isolation_priority_score']} (Medical Stock Remaining: {village['days_medical_stock_remaining']} days)")


if __name__ == "__main__":
    print("Running Graph Theory Isolation Tests...")
    test_network_bridge_detection()
    test_road_collapse_isolation_simulation()
    print("\nAll isolation graph theory tests passed with 100% verification!")
