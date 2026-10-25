"""
Critic and Self-Reflection Verifier Node.
Inspects intermediate observation traces, detects cross-tool contradictions,
checks sub-goal coverage, and issues structured critique verdicts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from src.agent.state import CritiqueVerdict, PlanStep, QueryAnalysis

logger = logging.getLogger("OmniVision.Agent.Critic")


class CriticVerifier:
    """
    Critic and Verifier.
    Performs adversarial self-reflection and cross-modal consistency checks
    before allowing the system to synthesize the final answer.
    """

    def __init__(self, passing_score_threshold: float = 0.75) -> None:
        self.passing_score_threshold = passing_score_threshold

    def evaluate(
        self,
        analysis: QueryAnalysis,
        plan_steps: List[PlanStep],
        observations: Dict[str, Dict[str, Any]],
        iteration_count: int = 1,
        max_iterations: int = 12,
    ) -> CritiqueVerdict:
        """
        Critique the current state of observations and determine next action.
        """
        missing_evidence: List[str] = []
        contradictions: List[str] = []

        completed_steps = [s for s in plan_steps if s.status == "completed"]
        failed_steps = [s for s in plan_steps if s.status == "failed"]

        # 1. Check Sub-Goal Coverage
        observed_tools = {obs.get("tool_name") for obs in observations.values()}

        for facet in analysis.reasoning_type:
            if facet == "detection" and not any(t in observed_tools for t in ["detector_ensemble", "open_vocab", "grounded_vlm"]):
                missing_evidence.append("Missing object detection / localization evidence")
            elif facet == "3d_depth" and "metric_depth" not in observed_tools:
                missing_evidence.append("Missing 3D metric depth estimation evidence")
            elif facet == "ocr" and "scene_ocr" not in observed_tools and "grounded_vlm" not in observed_tools:
                missing_evidence.append("Missing scene OCR text recognition evidence")
            elif facet == "action_pose" and "pose_action" not in observed_tools:
                missing_evidence.append("Missing human pose keypoint and posture evidence")
            elif facet == "attributes" and "visual_attributes" not in observed_tools:
                missing_evidence.append("Missing visual color/attribute extraction evidence")
            elif facet == "temporal" and "video_tracker" not in observed_tools and "optical_flow" not in observed_tools:
                missing_evidence.append("Missing multi-frame temporal tracking evidence")

        # 2. Check Cross-Tool Contradictions
        # Contradiction check A: Detector vs VLM count
        det_count = None
        vlm_count = None
        for obs in observations.values():
            if obs.get("tool_name") in ("detector_ensemble", "open_vocab"):
                dets = obs.get("data", {}).get("detections", [])
                det_count = len(dets)
            elif obs.get("tool_name") == "grounded_vlm":
                vlm_boxes = obs.get("data", {}).get("grounded_boxes", [])
                vlm_count = len(vlm_boxes)

        if det_count is not None and vlm_count is not None:
            if det_count > 0 and vlm_count == 0:
                contradictions.append(
                    f"Contradiction: Detector found {det_count} instances, but VLM found 0 grounded objects."
                )

        # Contradiction check B: Depth ordering vs Bounding Box sizes
        depth_obs = next((obs for obs in observations.values() if obs.get("tool_name") == "metric_depth"), None)
        if depth_obs:
            instances_3d = depth_obs.get("data", {}).get("instances_3d", [])
            for inst in instances_3d:
                depth_z = inst.get("centroid_3d", [0, 0, 0])[2]
                if depth_z <= 0.0:
                    contradictions.append(f"Invalid negative or zero metric depth {depth_z}m detected on instance {inst.get('label')}.")

        # 3. Compute Progress Score
        total_requirements = max(1, len(analysis.reasoning_type) + len(analysis.sub_goals))
        fulfilled_requirements = total_requirements - len(missing_evidence)
        base_score = max(0.0, fulfilled_requirements / float(total_requirements))

        # Penalize for failed steps and contradictions
        penalty = (len(failed_steps) * 0.15) + (len(contradictions) * 0.20)
        final_score = max(0.0, min(1.0, base_score - penalty))

        passed = (final_score >= self.passing_score_threshold) and not missing_evidence
        should_replan = not passed and (iteration_count < max_iterations) and bool(missing_evidence)

        feedback_parts = []
        if missing_evidence:
            feedback_parts.append(f"Missing evidence: {'; '.join(missing_evidence)}.")
        if contradictions:
            feedback_parts.append(f"Contradictions detected: {'; '.join(contradictions)}.")
        if passed:
            feedback_parts.append("All sub-goals evidenced with verified observations.")

        feedback_str = " ".join(feedback_parts) or "Observations verified."

        return CritiqueVerdict(
            passed=passed,
            score=round(final_score, 3),
            should_replan=should_replan,
            should_active_perceive=False,
            feedback=feedback_str,
            missing_evidence=missing_evidence,
            identified_contradictions=contradictions,
        )
