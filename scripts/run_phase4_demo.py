"""
Phase 4 Multi-Agent Core (LangGraph) End-to-End Demonstration Script.
Demonstrates:
1. Query decomposition via QueryAnalyst.
2. Capability and budget-conscious routing via ToolRouter.
3. Tree-search program synthesis via VisionPlanner.
4. Active perception loop triggering targeted uncertainty minimization.
5. Adversarial self-reflection via CriticVerifier.
6. Formal claim-evidence attribution via HallucinationGuard.
7. Calibrated conformal answer synthesis via GroundedSynthesizer.
8. Complete end-to-end multi-agent LangGraph workflow via OmniVisionAgent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import (
    ActivePerceptionManager,
    CriticVerifier,
    GroundedSynthesizer,
    HallucinationGuard,
    OmniVisionAgent,
    QueryAnalyst,
    ToolRouter,
    VisionPlanner,
)


def run_phase4_demo():
    print("=" * 75)
    print(" OmniVision Agent - Phase 4 Multi-Agent Core (LangGraph) Demonstration")
    print("=" * 75)

    sample_query = (
        "Find the worker and forklift in the warehouse, determine distance to the closest object, "
        "verify worker posture and PPE compliance, and provide a verified grounded report."
    )
    media_uri = "warehouse_scene.jpg"

    # ---------------------------------------------------------
    # 1. Query Analyst Decomposition
    # ---------------------------------------------------------
    print("\n[1] Query Analyst Node:")
    analyst = QueryAnalyst()
    analysis = analyst.analyze(sample_query, media_uri=media_uri, media_type="image")
    print(f"  Query: '{analysis.original_query}'")
    print(f"  -> Modality: {analysis.modality}")
    print(f"  -> Reasoning Facets: {analysis.reasoning_type}")
    print(f"  -> Key Targets: {analysis.key_targets}")
    print(f"  -> Requires 3D: {analysis.requires_3d}")
    print(f"  -> Complexity: {analysis.estimated_complexity}")
    print(f"  -> Sub-Goals ({len(analysis.sub_goals)}):")
    for i, sg in enumerate(analysis.sub_goals, 1):
        print(f"     {i}. {sg}")

    # ---------------------------------------------------------
    # 2. Tool Router Capability Mapping
    # ---------------------------------------------------------
    print("\n[2] Learned Tool Router Node:")
    router = ToolRouter()
    for sg in analysis.sub_goals[:4]:
        decision = router.route_subgoal(sg, target_entities=analysis.key_targets)
        print(f"  Sub-goal: '{sg}'")
        print(f"  -> Routed Tool: '{decision.selected_tool}' (Accuracy: {decision.expected_accuracy*100:.0f}%, Cost: ${decision.estimated_cost_usd:.4f}, Latency: {decision.estimated_latency_ms:.1f}ms)")
        print(f"  -> Rationale: {decision.rationale}")

    # ---------------------------------------------------------
    # 3. Vision Planner with Tree Search
    # ---------------------------------------------------------
    print("\n[3] Vision Planner & Tree-Search Program Synthesizer:")
    planner = VisionPlanner(tool_router=router)
    plan = planner.synthesize_plan(
        analysis=analysis,
        media_uri=media_uri,
        budget_limit_usd=0.50,
        enable_tree_search=True,
        num_candidates=3,
    )
    print(f"  Plan ID: {plan.plan_id}")
    print(f"  Strategy: {plan.strategy_rationale}")
    print(f"  Total Estimated Cost: ${plan.estimated_total_cost_usd:.4f}")
    print(f"  Total Estimated Latency: {plan.estimated_total_latency_ms:.1f}ms")
    print(f"  Execution Steps ({len(plan.steps)}):")
    for s in plan.steps:
        deps = f" [depends on {', '.join(s.dependencies)}]" if s.dependencies else ""
        print(f"    - [{s.step_id}] {s.tool_name} -> '{s.sub_goal}'{deps}")

    # ---------------------------------------------------------
    # 4. Active Perception Loop
    # ---------------------------------------------------------
    print("\n[4] Active Perception Loop (Uncertainty Minimization):")
    apm = ActivePerceptionManager(uncertainty_threshold=0.65)
    simulated_observations = {
        "obs_det_01": {
            "tool_name": "detector_ensemble",
            "uncertainty": {"confidence": 0.52, "is_ambiguous": True},
            "data": {
                "detections": [
                    {"label": "worker", "box_2d": [120.0, 80.0, 240.0, 360.0], "confidence": 0.52}
                ]
            },
        }
    }
    action = apm.evaluate_observations_for_active_perception(simulated_observations, media_uri)
    if action:
        print(f"  Uncertainty detected in observation 'obs_det_01' (conf < 0.65)")
        print(f"  -> Triggered Active Action: {action.action_type}")
        print(f"  -> Target Bounding Box: {action.target_box}")
        print(f"  -> Remediation Tool: {action.remediation_tool}")
        print(f"  -> Rationale: {action.rationale}")

    # ---------------------------------------------------------
    # 5. Critic & Self-Reflection Verifier
    # ---------------------------------------------------------
    print("\n[5] Critic / Verifier Node:")
    critic = CriticVerifier()
    simulated_complete_obs = {
        "obs_det": {
            "tool_name": "detector_ensemble",
            "data": {"detections": [{"label": "worker", "box": {"x1": 120, "y1": 80, "x2": 240, "y2": 360}, "confidence": 0.94}]},
        },
        "obs_depth": {
            "tool_name": "metric_depth",
            "data": {"instances_3d": [{"label": "worker", "centroid_3d": [0.2, 0.0, 2.8], "distance_to_camera_meters": 2.8}]},
        },
        "obs_pose": {
            "tool_name": "pose_action",
            "data": {"persons": [{"posture": "standing", "is_fall_detected": False, "ppe_compliance": {"hardhat": True, "high_vis_vest": True}}]},
        },
    }
    verdict = critic.evaluate(analysis, plan.steps, simulated_complete_obs)
    print(f"  Critique Verdict: {'PASSED' if verdict.passed else 'REPLAN NEEDED'}")
    print(f"  -> Completeness Score: {verdict.score*100:.1f}%")
    print(f"  -> Missing Evidence: {verdict.missing_evidence or 'None'}")
    print(f"  -> Feedback: {verdict.feedback}")

    # ---------------------------------------------------------
    # 6. Hallucination Guard (Claim Attribution)
    # ---------------------------------------------------------
    print("\n[6] Hallucination Guard (Claim-Evidence Attribution):")
    guard = HallucinationGuard()
    draft_answer = (
        "Visual analysis confirms a worker is standing near the forklift. "
        "The worker is located at depth 2.8m. "
        "A stray tiger is sleeping beside the pallets."
    )
    guard_result = guard.verify_draft_answer(draft_answer, simulated_complete_obs)
    print(f"  Draft Answer: '{draft_answer}'")
    print(f"  -> All Claims Verified: {guard_result.all_verified}")
    print(f"  -> Hallucination Rate: {guard_result.hallucination_rate*100:.1f}%")
    print(f"  -> Verified Claims: {len(guard_result.verified_claims)}")
    for vc in guard_result.verified_claims:
        print(f"     [OK] '{vc.statement}' (Obs: {vc.observation_ids or 'meta'})")
    print(f"  -> Unsupported Claims: {len(guard_result.unsupported_claims)}")
    for uc in guard_result.unsupported_claims:
        print(f"     [REJECTED] '{uc.statement}' (Reason: {uc.verifier_notes})")
    print(f"  -> Remediated Answer: '{guard_result.remediated_answer_text}'")

    # ---------------------------------------------------------
    # 7. Grounded Synthesizer
    # ---------------------------------------------------------
    print("\n[7] Grounded Synthesizer Node:")
    synthesizer = GroundedSynthesizer()
    synthesis = synthesizer.synthesize(
        analysis=analysis,
        observations=simulated_complete_obs,
        scene_graph_dict={"nodes": [], "edges": []},
        claims_verification=guard_result,
        accumulated_cost_usd=0.0035,
        accumulated_latency_ms=145.0,
    )
    print(f"  Calibrated Confidence: {synthesis.calibrated_confidence*100:.1f}% (Conformal Interval: [{synthesis.confidence_interval[0]}, {synthesis.confidence_interval[1]}])")
    print(f"  Evidence Citations ({len(synthesis.citations)}): {synthesis.citations}")
    print(f"  Key Findings:")
    for kf in synthesis.key_findings:
        print(f"    * {kf}")

    # ---------------------------------------------------------
    # 8. End-to-End OmniVisionAgent Execution
    # ---------------------------------------------------------
    print("\n[8] End-to-End OmniVisionAgent LangGraph Execution:")
    agent = OmniVisionAgent(
        max_iterations=8,
        budget_limit_usd=0.50,
        enable_tree_search=True,
        enable_active_perception=True,
    )
    final_result = agent.run(
        query=sample_query,
        media_uri=media_uri,
        media_type="image",
    )
    print(f"  Query: '{final_result.query}'")
    print(f"  Final Grounded Answer:\n{final_result.answer}")
    print(f"  Calibrated Confidence: {final_result.calibrated_confidence*100:.1f}%")
    print(f"  Citations: {final_result.citations}")
    print(f"  Accumulated Cost: ${final_result.total_cost_usd:.5f}")
    print(f"  Accumulated Latency: {final_result.total_latency_ms:.1f}ms")

    print("\n" + "=" * 75)
    print(" [OK] Phase 4 Multi-Agent Core Demonstration Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_phase4_demo()
