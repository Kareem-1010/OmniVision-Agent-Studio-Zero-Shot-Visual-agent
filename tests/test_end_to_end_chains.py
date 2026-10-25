"""
Tests for Phase 10: End-to-End Visual Reasoning Chains.
Verifies all 8 canonical multimodal reasoning chains:
1. "Count the red cars"
2. "Is there a safety hazard?"
3. "Read the sign next to the door"
4. "How many unique people entered and how long did each stay?"
5. "Which object is closest to the camera and how far is it?"
6. "What is the person holding the umbrella standing next to?"
7. "When did the red truck arrive, and who got out?"
8. "Could the cart fit through the gap between the shelves?"
"""

from __future__ import annotations

import pytest
from src.agent.chains import VisualReasoningChainRunner, ChainExecutionResult


@pytest.fixture
def chain_runner() -> VisualReasoningChainRunner:
    return VisualReasoningChainRunner()


def test_chain_1_count_red_cars(chain_runner: VisualReasoningChainRunner):
    """Chain 1: Ensemble detection + color attributes + aggregation + verification."""
    res = chain_runner.run_count_red_cars_chain("mock_parking_lot.jpg", target_color="red", target_category="car")
    assert isinstance(res, ChainExecutionResult)
    assert res.chain_id == "chain_1"
    assert res.quantitative_data["count"] >= 3
    assert "red cars" in res.answer
    assert res.confidence >= 0.90
    assert "detector_ensemble" in res.tools_invoked
    assert "visual_attributes" in res.tools_invoked
    assert res.is_verified is True


def test_chain_2_safety_hazard_audit(chain_runner: VisualReasoningChainRunner):
    """Chain 2: PPE compliance audit + 3D machinery proximity risk."""
    res = chain_runner.run_safety_hazard_chain("mock_factory_bay.jpg")
    assert res.chain_id == "chain_2"
    assert "safety hazards detected" in res.answer.lower()
    assert res.quantitative_data["risk_level"] in ("HIGH", "CRITICAL")
    assert res.quantitative_data["compliance_rate"] < 100.0  # W-102 missing hard hat (50%)
    assert res.quantitative_data["min_clearance_m"] > 0.0
    assert len(res.extracted_entities) >= 3


def test_chain_3_read_sign_next_to_door(chain_runner: VisualReasoningChainRunner):
    """Chain 3: Reference object grounding + spatial relation crop + scene OCR."""
    res = chain_runner.run_read_sign_next_to_door_chain("mock_corridor.jpg")
    assert res.chain_id == "chain_3"
    assert "AUTHORIZED PERSONNEL ONLY" in res.answer
    assert res.quantitative_data["spatial_separation"] < 0.50
    assert "scene_ocr" in res.tools_invoked


def test_chain_4_unique_people_dwell(chain_runner: VisualReasoningChainRunner):
    """Chain 4: Multi-frame tracking + entrance tripwire + dwell time accumulation."""
    res = chain_runner.run_unique_people_dwell_chain("mock_retail_entrance.mp4")
    assert res.chain_id == "chain_4"
    assert res.quantitative_data["unique_entered_count"] == 3
    dwell_records = res.quantitative_data["dwell_records"]
    assert len(dwell_records) == 3
    assert dwell_records[0]["dwell_time_sec"] == 39.5
    assert "Person_1" in res.answer
    assert res.confidence >= 0.90


def test_chain_5_closest_object_distance(chain_runner: VisualReasoningChainRunner):
    """Chain 5: Metric depth map + 3D backprojection + Euclidean camera distance ranking."""
    res = chain_runner.run_closest_object_distance_chain("mock_interior.jpg")
    assert res.chain_id == "chain_5"
    assert res.quantitative_data["closest_object"] == "chair"
    assert res.quantitative_data["distance_meters"] < 2.0
    assert "chair is closest to the camera" in res.answer
    assert "position_3d" in res.quantitative_data["all_ranked"][0]


def test_chain_6_person_umbrella_relation(chain_runner: VisualReasoningChainRunner):
    """Chain 6: Pose estimation + hand contact containment + 4D Scene Graph neighbor search."""
    res = chain_runner.run_person_umbrella_relation_chain("mock_park.jpg")
    assert res.chain_id == "chain_6"
    assert res.quantitative_data["standing_next_to"] == "newsstand kiosk"
    assert "newsstand kiosk" in res.answer
    assert "persistent_scene_graph" in res.tools_invoked


def test_chain_7_vehicle_arrival_and_exit(chain_runner: VisualReasoningChainRunner):
    """Chain 7: Temporal boundary segmentation + event QA + driver exit attribution."""
    res = chain_runner.run_vehicle_arrival_and_exit_chain("mock_dock.mp4")
    assert res.chain_id == "chain_7"
    assert res.quantitative_data["arrival_timestamp_sec"] == 14.2
    assert res.quantitative_data["disembark_timestamp_sec"] == 22.0
    assert "14.2s" in res.answer
    assert "22.0s" in res.answer


def test_chain_8_cart_shelf_fit_simulation(chain_runner: VisualReasoningChainRunner):
    """Chain 8: 3D bounding geometry + gap measurement + counterfactual clearance simulation."""
    # 1. Feasible clearance case
    res_fit = chain_runner.run_cart_shelf_fit_simulation_chain(
        "mock_warehouse.jpg",
        cart_dimensions_3d=(0.75, 1.10, 0.95),
        shelf_gap_m=1.15,
    )
    assert res_fit.chain_id == "chain_8"
    assert res_fit.quantitative_data["fits"] is True
    assert "can fit" in res_fit.answer
    assert res_fit.quantitative_data["clearance_margin_m"] == 0.40

    # 2. Infeasible clearance case (cart wider than gap)
    res_nofit = chain_runner.run_cart_shelf_fit_simulation_chain(
        "mock_warehouse.jpg",
        cart_dimensions_3d=(1.30, 1.10, 0.95),
        shelf_gap_m=1.15,
    )
    assert res_nofit.quantitative_data["fits"] is False
    assert "cannot fit" in res_nofit.answer
    assert res_nofit.quantitative_data["clearance_margin_m"] < 0.0


def test_dispatch_chain_central_router(chain_runner: VisualReasoningChainRunner):
    """Verify central router resolves all chains by name or numeric ID."""
    for chain_id in range(1, 9):
        res = chain_runner.dispatch_chain(chain_id, f"media_sample_{chain_id}.jpg")
        assert res.chain_id == f"chain_{chain_id}"
        assert res.confidence > 0.80
        assert res.execution_time_ms >= 0.0
