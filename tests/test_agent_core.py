"""
Comprehensive Unit and Integration Tests for Phase 4 Agent Core (LangGraph).
Tests: Query Analyst, Tool Router, Vision Planner (with Tree Search), Active Perception,
Critic / Verifier, Hallucination Guard, Synthesizer, and the end-to-end OmniVisionAgent.
"""

import pytest
from src.agent.active_perception import ActivePerceptionManager
from src.agent.critic import CriticVerifier
from src.agent.hallucination_guard import HallucinationGuard
from src.agent.orchestrator import OmniVisionAgent, ToolDispatcher
from src.agent.planner import VisionPlanner
from src.agent.query_analyst import QueryAnalyst
from src.agent.state import OmniVisionState, PlanStep, QueryAnalysis
from src.agent.synthesizer import GroundedSynthesizer
from src.agent.tool_router import ToolRouter


def test_query_analyst_multimodal_decomposition():
    analyst = QueryAnalyst()

    # Query 1: 3D and distance reasoning
    res_3d = analyst.analyze("What is the distance between the closest car and the camera?", "test.jpg", "image")
    assert "3d_depth" in res_3d.reasoning_type
    assert res_3d.requires_3d is True
    assert "car" in res_3d.key_targets
    assert len(res_3d.sub_goals) >= 2

    # Query 2: Scene OCR text reading
    res_ocr = analyst.analyze("Can you read the text on the street sign?", "test.jpg", "image")
    assert "ocr" in res_ocr.reasoning_type
    assert any("text" in sg.lower() for sg in res_ocr.sub_goals)

    # Query 3: Pose, safety & PPE
    res_pose = analyst.analyze("Is the worker wearing a hardhat and standing safely?", "test.jpg", "image")
    assert "action_pose" in res_pose.reasoning_type
    assert "worker" in res_pose.key_targets

    # Query 4: Video motion
    res_video = analyst.analyze("Did the vehicle cross the line while moving?", "test.mp4", "video")
    assert "temporal" in res_video.reasoning_type
    assert res_video.temporal_span_required is True


def test_tool_router():
    router = ToolRouter()

    # Standard COCO object
    d1 = router.route_subgoal("Detect instances", target_entities=["car", "person"])
    assert d1.selected_tool == "detector_ensemble"

    # Rare / Open vocabulary object
    d2 = router.route_subgoal("Detect instances", target_entities=["high-visibility orange reflective cone"])
    assert d2.selected_tool == "open_vocab"

    # 3D Depth
    d3 = router.route_subgoal("Compute metric depth and 3D distances")
    assert d3.selected_tool == "metric_depth"

    # OCR text
    d4 = router.route_subgoal("Extract text from license plate")
    assert d4.selected_tool == "scene_ocr"

    # Pose and Action
    d5 = router.route_subgoal("Classify human posture and check PPE")
    assert d5.selected_tool == "pose_action"


def test_vision_planner_and_tree_search():
    planner = VisionPlanner()
    analyst = QueryAnalyst()

    analysis = analyst.analyze(
        "Find the red car, estimate its distance, and read its license plate",
        "sample.jpg",
        "image",
    )

    # Deterministic plan
    plan = planner.synthesize_plan(
        analysis,
        "sample.jpg",
        enable_tree_search=False,
    )
    tools = [s.tool_name for s in plan.steps]
    assert "detector_ensemble" in tools or "open_vocab" in tools
    assert "metric_depth" in tools
    assert "scene_ocr" in tools
    assert plan.estimated_total_cost_usd > 0.0

    # Tree-Search evaluation
    ts_plan = planner.synthesize_plan(
        analysis,
        "sample.jpg",
        enable_tree_search=True,
        num_candidates=3,
    )
    assert len(ts_plan.steps) >= len(plan.steps)

    # Replanning on feedback
    replanned = planner.replan_on_feedback(
        current_plan=plan,
        feedback="Missing depth",
        missing_evidence=["Missing 3D metric depth estimation evidence"],
        media_uri="sample.jpg",
    )
    assert len(replanned.steps) >= len(plan.steps)


def test_active_perception():
    apm = ActivePerceptionManager(uncertainty_threshold=0.65)

    # High uncertainty detection should trigger zoom crop
    observations = {
        "obs_001": {
            "tool_name": "detector_ensemble",
            "uncertainty": {"confidence": 0.40, "is_ambiguous": True},
            "data": {
                "detections": [
                    {"label": "person", "box_2d": [100.0, 100.0, 150.0, 180.0], "confidence": 0.40}
                ]
            },
        }
    }

    action = apm.evaluate_observations_for_active_perception(observations, "test.jpg")
    assert action is not None
    assert action.action_type == "zoom_crop"
    assert action.target_box is not None
    # Zoom box should have expanded margin around [100, 100, 150, 180]
    assert action.target_box[0] < 100.0
    assert action.target_box[2] > 150.0

    # Clean high confidence observation should not trigger active perception
    clean_obs = {
        "obs_002": {
            "tool_name": "detector_ensemble",
            "uncertainty": {"confidence": 0.95, "is_ambiguous": False},
            "data": {
                "detections": [
                    {"label": "car", "box_2d": [50.0, 50.0, 200.0, 200.0], "confidence": 0.95}
                ]
            },
        }
    }
    assert apm.evaluate_observations_for_active_perception(clean_obs, "test.jpg") is None


def test_critic_verifier():
    critic = CriticVerifier(passing_score_threshold=0.70)
    analyst = QueryAnalyst()
    analysis = analyst.analyze("Read the text and estimate distance to the car", "test.jpg")

    steps = [
        PlanStep(step_id="step_01", sub_goal="Detect car", tool_name="detector_ensemble", status="completed"),
        PlanStep(step_id="step_02", sub_goal="Estimate depth", tool_name="metric_depth", status="completed"),
    ]

    # Incomplete observations: missing OCR text evidence
    incomplete_obs = {
        "obs_det": {
            "tool_name": "detector_ensemble",
            "data": {"detections": [{"label": "car", "box_2d": [10, 10, 50, 50]}]},
        },
        "obs_depth": {
            "tool_name": "metric_depth",
            "data": {"instances_3d": [{"label": "car", "centroid_3d": [0, 0, 3.5]}]},
        },
    }

    verdict_fail = critic.evaluate(analysis, steps, incomplete_obs, iteration_count=1)
    assert verdict_fail.passed is False
    assert any("ocr" in m.lower() for m in verdict_fail.missing_evidence)
    assert verdict_fail.should_replan is True

    # Complete observations: includes OCR
    complete_obs = dict(incomplete_obs)
    complete_obs["obs_ocr"] = {
        "tool_name": "scene_ocr",
        "data": {"recognized_texts": [{"text": "STOP"}]},
    }
    verdict_pass = critic.evaluate(analysis, steps, complete_obs, iteration_count=1)
    assert verdict_pass.passed is True
    assert verdict_pass.score >= 0.70


def test_hallucination_guard():
    guard = HallucinationGuard()

    observations = {
        "obs_car_1": {
            "tool_name": "detector_ensemble",
            "data": {"detections": [{"label": "car", "box_2d": [50.0, 50.0, 200.0, 200.0], "confidence": 0.92}]},
        },
        "obs_text_1": {
            "tool_name": "scene_ocr",
            "data": {"recognized_texts": [{"text": "TAXI", "box_2d": [80.0, 60.0, 120.0, 80.0]}]},
        },
    }

    # Draft with one grounded claim and one completely unsupported hallucination
    draft = "A car is clearly visible in the street. The vehicle has a sign saying TAXI. A unicorn is grazing on the roof."

    result = guard.verify_draft_answer(draft, observations)
    assert len(result.verified_claims) >= 2
    assert len(result.unsupported_claims) == 1
    assert "unicorn" in result.unsupported_claims[0].statement.lower()
    assert result.hallucination_rate > 0.0
    assert "[UNVERIFIED CLAIM:" in result.remediated_answer_text
    assert "obs_car_1" in result.remediated_answer_text or "obs_text_1" in result.remediated_answer_text


def test_grounded_synthesizer():
    synthesizer = GroundedSynthesizer()
    analyst = QueryAnalyst()
    analysis = analyst.analyze("Is there a person and a car?", "test.jpg")

    observations = {
        "obs_01": {
            "tool_name": "detector_ensemble",
            "data": {
                "detections": [
                    {"label": "person", "box_2d": [10, 10, 50, 100], "confidence": 0.94},
                    {"label": "car", "box_2d": [100, 100, 300, 250], "confidence": 0.91},
                ]
            },
        }
    }

    synthesis = synthesizer.synthesize(
        analysis=analysis,
        observations=observations,
        scene_graph_dict={"nodes": [], "edges": []},
        accumulated_cost_usd=0.001,
        accumulated_latency_ms=45.0,
    )

    assert "obs_01" in synthesis.citations
    assert synthesis.calibrated_confidence > 0.80
    assert len(synthesis.key_findings) > 0
    assert synthesis.confidence_interval[0] < synthesis.confidence_interval[1]


def test_omnivision_agent_end_to_end():
    # Instantiate full agent orchestrator
    agent = OmniVisionAgent(
        max_iterations=6,
        budget_limit_usd=0.50,
        enable_tree_search=True,
        enable_active_perception=True,
    )

    # Run complex multimodal question
    synthesis = agent.run(
        query="Detect objects in the scene, measure distance to the closest object, and summarize findings",
        media_uri="sample_image.jpg",
        media_type="image",
    )

    assert synthesis is not None
    assert len(synthesis.citations) > 0
    assert len(synthesis.key_findings) > 0
    assert synthesis.calibrated_confidence > 0.0
    assert synthesis.total_latency_ms > 0.0
    assert len(synthesis.answer) > 0
