"""
Vision Program Planner and Tree-Search Synthesizer.
Synthesizes executable DAG vision programs with candidate evaluation,
dependency resolution, cost/latency budget estimation, and replanning capabilities.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional
from src.agent.state import PlanStep, QueryAnalysis, VisionProgramPlan
from src.agent.tool_router import ToolRouter

logger = logging.getLogger("OmniVision.Agent.Planner")


class VisionPlanner:
    """
    Vision Program Synthesizer and Tree-Search Planner.
    Generates structured execution DAGs targeting user visual queries.
    """

    def __init__(self, tool_router: Optional[ToolRouter] = None) -> None:
        self.router = tool_router or ToolRouter()

    def synthesize_plan(
        self,
        analysis: QueryAnalysis,
        media_uri: str,
        budget_limit_usd: float = 0.50,
        enable_tree_search: bool = True,
        num_candidates: int = 3,
    ) -> VisionProgramPlan:
        """
        Synthesize an optimal executable vision plan.
        If tree search is enabled, evaluates multiple plan candidate branches and selects the Pareto-optimal plan.
        """
        if not enable_tree_search or num_candidates <= 1:
            return self._build_deterministic_plan(analysis, media_uri)

        # Tree Search: Generate candidate variations (e.g., standard vs open-vocab, deep vs fast)
        candidates: List[VisionProgramPlan] = []

        # Candidate 1: Balanced High-Accuracy Plan
        candidates.append(self._build_deterministic_plan(analysis, media_uri, prefer_speed=False))

        # Candidate 2: Fast Low-Latency Plan
        candidates.append(self._build_deterministic_plan(analysis, media_uri, prefer_speed=True))

        # Candidate 3: Multi-Oracle Plan (with VLM validation step)
        c3 = self._build_deterministic_plan(analysis, media_uri, prefer_speed=False)
        # Add grounded VLM oracle check
        vlm_step = PlanStep(
            step_id=f"step_{len(c3.steps)+1:02d}",
            sub_goal="Cross-examine findings with grounded VLM oracle",
            tool_name="grounded_vlm",
            arguments={"image_or_video_uri": media_uri, "prompt": analysis.original_query, "mode": "oracle_check"},
            dependencies=[s.step_id for s in c3.steps if s.tool_name != "synthesizer"],
            cost_estimate_usd=0.008,
            latency_estimate_ms=220.0,
        )
        c3.steps.append(vlm_step)
        c3.estimated_total_cost_usd += 0.008
        c3.estimated_total_latency_ms += 220.0
        candidates.append(c3)

        # Score candidates
        scored_candidates = [
            (self._score_plan(cand, analysis, budget_limit_usd), cand)
            for cand in candidates
        ]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_plan = scored_candidates[0]
        logger.info(f"Tree-search selected plan {best_plan.plan_id} with score {best_score:.3f}")
        return best_plan

    def _build_deterministic_plan(
        self,
        analysis: QueryAnalysis,
        media_uri: str,
        prefer_speed: bool = False,
    ) -> VisionProgramPlan:
        """Construct a sequential, dependency-resolved vision plan DAG."""
        steps: List[PlanStep] = []
        step_counter = 1
        detection_step_id: Optional[str] = None

        # Step 1: Primary Detection / Localization (if required or default)
        if "detection" in analysis.reasoning_type or "counting" in analysis.reasoning_type or "spatial_relation" in analysis.reasoning_type:
            decision = self.router.route_subgoal(
                "Detect target instances",
                target_entities=analysis.key_targets,
                prefer_speed=prefer_speed,
            )
            detection_step_id = f"step_{step_counter:02d}"
            step_counter += 1

            arg_dict: Dict[str, Any] = {"image_uri": media_uri}
            if decision.selected_tool == "open_vocab":
                queries = []
                for t in (analysis.key_targets or ["object"]):
                    queries.append(t)
                    parts = t.split()
                    if len(parts) > 1 and parts[-1] not in queries:
                        queries.append(parts[-1])
                arg_dict["text_queries"] = queries
            elif decision.selected_tool == "detector_ensemble" and analysis.key_targets:
                arg_dict["target_classes"] = analysis.key_targets

            steps.append(
                PlanStep(
                    step_id=detection_step_id,
                    sub_goal=f"Detect instances of {analysis.key_targets or 'objects'}",
                    tool_name=decision.selected_tool,
                    arguments=arg_dict,
                    dependencies=[],
                    cost_estimate_usd=decision.estimated_cost_usd,
                    latency_estimate_ms=decision.estimated_latency_ms,
                )
            )

        # Step 2: Temporal Video Tracking (if video/temporal)
        if analysis.temporal_span_required or analysis.modality == "video":
            track_step_id = f"step_{step_counter:02d}"
            step_counter += 1
            decision = self.router.route_subgoal("temporal tracking", prefer_speed=prefer_speed)
            steps.append(
                PlanStep(
                    step_id=track_step_id,
                    sub_goal="Track instances across video frames",
                    tool_name=decision.selected_tool,
                    arguments={"video_uri": media_uri},
                    dependencies=[detection_step_id] if detection_step_id else [],
                    cost_estimate_usd=decision.estimated_cost_usd,
                    latency_estimate_ms=decision.estimated_latency_ms,
                )
            )

        # Step 3: Metric Depth & 3D Back-Projection (if 3D / distance / spatial)
        depth_step_id: Optional[str] = None
        if analysis.requires_3d or "3d_depth" in analysis.reasoning_type or "spatial_relation" in analysis.reasoning_type:
            depth_step_id = f"step_{step_counter:02d}"
            step_counter += 1
            decision = self.router.route_subgoal("metric depth 3d", prefer_speed=prefer_speed)
            steps.append(
                PlanStep(
                    step_id=depth_step_id,
                    sub_goal="Compute metric depth map and 3D camera-frame bounding boxes",
                    tool_name=decision.selected_tool,
                    arguments={"image_uri": media_uri},
                    dependencies=[detection_step_id] if detection_step_id else [],
                    cost_estimate_usd=decision.estimated_cost_usd,
                    latency_estimate_ms=decision.estimated_latency_ms,
                )
            )

        # Step 4: Pose / Action Recognition (if pose facet)
        if "action_pose" in analysis.reasoning_type:
            pose_step_id = f"step_{step_counter:02d}"
            step_counter += 1
            decision = self.router.route_subgoal("pose action keypoints", prefer_speed=prefer_speed)
            steps.append(
                PlanStep(
                    step_id=pose_step_id,
                    sub_goal="Estimate 17 human keypoints, classify postures, and evaluate PPE compliance",
                    tool_name=decision.selected_tool,
                    arguments={"image_uri": media_uri, "detect_falls": True, "check_ppe": True},
                    dependencies=[detection_step_id] if detection_step_id else [],
                    cost_estimate_usd=decision.estimated_cost_usd,
                    latency_estimate_ms=decision.estimated_latency_ms,
                )
            )

        # Step 5: Scene OCR (if text facet)
        if "ocr" in analysis.reasoning_type:
            ocr_step_id = f"step_{step_counter:02d}"
            step_counter += 1
            decision = self.router.route_subgoal("scene ocr text", prefer_speed=prefer_speed)
            steps.append(
                PlanStep(
                    step_id=ocr_step_id,
                    sub_goal="Extract scene text and bind to visual instances",
                    tool_name=decision.selected_tool,
                    arguments={"image_uri": media_uri},
                    dependencies=[detection_step_id] if detection_step_id else [],
                    cost_estimate_usd=decision.estimated_cost_usd,
                    latency_estimate_ms=decision.estimated_latency_ms,
                )
            )

        # Step 6: Visual Attributes (if attributes/color facet)
        if "attributes" in analysis.reasoning_type:
            attr_step_id = f"step_{step_counter:02d}"
            step_counter += 1
            decision = self.router.route_subgoal("visual attributes color", prefer_speed=prefer_speed)
            steps.append(
                PlanStep(
                    step_id=attr_step_id,
                    sub_goal="Extract dominant CIELAB colors and material textures",
                    tool_name=decision.selected_tool,
                    arguments={"image_uri": media_uri},
                    dependencies=[detection_step_id] if detection_step_id else [],
                    cost_estimate_usd=decision.estimated_cost_usd,
                    latency_estimate_ms=decision.estimated_latency_ms,
                )
            )

        # Step: Grounded VLM & CLIP Scene Understanding (always included for scene grounding & layout)
        has_vlm = any(s.tool_name == "grounded_vlm" for s in steps)
        if not has_vlm:
            vlm_step_id = f"step_{step_counter:02d}"
            step_counter += 1
            steps.append(
                PlanStep(
                    step_id=vlm_step_id,
                    sub_goal="Analyze visual scene, detect multi-panel layouts, and ground query via CLIP",
                    tool_name="grounded_vlm",
                    arguments={"image_or_video_uri": media_uri, "prompt": analysis.original_query},
                    dependencies=[detection_step_id] if detection_step_id else [],
                    cost_estimate_usd=0.002,
                    latency_estimate_ms=80.0,
                )
            )

        total_cost = sum(s.cost_estimate_usd for s in steps)
        total_latency = sum(s.latency_estimate_ms for s in steps)

        return VisionProgramPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            steps=steps,
            strategy_rationale=f"Plan synthesized for {len(analysis.reasoning_type)} facets with {len(steps)} steps.",
            estimated_total_cost_usd=round(total_cost, 5),
            estimated_total_latency_ms=round(total_latency, 2),
        )

    def _score_plan(
        self,
        plan: VisionProgramPlan,
        analysis: QueryAnalysis,
        budget_limit_usd: float,
    ) -> float:
        """
        Evaluate candidate plan quality:
        Score = Accuracy * 0.5 + Coverage * 0.3 - (Cost / Budget) * 0.1 - (Latency / 1000) * 0.1
        """
        tools_used = [s.tool_name for s in plan.steps]

        # Accuracy estimate based on tools
        acc = sum(self.router.TOOL_PROFILES.get(t, {}).get("accuracy", 0.85) for t in tools_used) / max(1, len(tools_used))

        # Coverage: does plan cover required facets?
        coverage_hits = 0
        total_facets = len(analysis.reasoning_type)
        if "detection" in analysis.reasoning_type and any(t in tools_used for t in ["detector_ensemble", "open_vocab"]):
            coverage_hits += 1
        if "3d_depth" in analysis.reasoning_type and "metric_depth" in tools_used:
            coverage_hits += 1
        if "ocr" in analysis.reasoning_type and "scene_ocr" in tools_used:
            coverage_hits += 1
        if "action_pose" in analysis.reasoning_type and "pose_action" in tools_used:
            coverage_hits += 1
        if "attributes" in analysis.reasoning_type and "visual_attributes" in tools_used:
            coverage_hits += 1
        coverage = coverage_hits / max(1, total_facets)

        cost_penalty = min(1.0, plan.estimated_total_cost_usd / max(0.001, budget_limit_usd))
        latency_penalty = min(1.0, plan.estimated_total_latency_ms / 1000.0)

        score = (acc * 0.5) + (coverage * 0.3) - (cost_penalty * 0.1) - (latency_penalty * 0.1)
        return round(float(score), 4)

    def replan_on_feedback(
        self,
        current_plan: VisionProgramPlan,
        feedback: str,
        missing_evidence: List[str],
        media_uri: str,
    ) -> VisionProgramPlan:
        """
        Replan when Critic flags missing evidence or failed steps.
        Injects remediation steps or alternative tools.
        """
        new_steps = list(current_plan.steps)
        existing_step_ids = {s.step_id for s in new_steps}
        existing_tools = {s.tool_name for s in new_steps}

        for item in missing_evidence:
            item_lower = item.lower()
            if "text" in item_lower and "scene_ocr" not in existing_tools:
                new_steps.append(
                    PlanStep(
                        step_id=f"step_replan_{len(new_steps)+1:02d}",
                        sub_goal=f"Recover missing text evidence for: {item}",
                        tool_name="scene_ocr",
                        arguments={"image_uri": media_uri},
                        dependencies=[s.step_id for s in new_steps if s.status == "completed"],
                        cost_estimate_usd=0.001,
                        latency_estimate_ms=40.0,
                    )
                )
            elif ("depth" in item_lower or "distance" in item_lower) and "metric_depth" not in existing_tools:
                new_steps.append(
                    PlanStep(
                        step_id=f"step_replan_{len(new_steps)+1:02d}",
                        sub_goal=f"Recover missing depth evidence for: {item}",
                        tool_name="metric_depth",
                        arguments={"image_uri": media_uri},
                        dependencies=[s.step_id for s in new_steps if s.status == "completed"],
                        cost_estimate_usd=0.0012,
                        latency_estimate_ms=50.0,
                    )
                )
            elif "vlm" not in existing_tools and "grounded_vlm" not in existing_tools:
                new_steps.append(
                    PlanStep(
                        step_id=f"step_replan_{len(new_steps)+1:02d}",
                        sub_goal=f"Consult grounded VLM for ungrounded claim: {item}",
                        tool_name="grounded_vlm",
                        arguments={"image_or_video_uri": media_uri, "prompt": feedback},
                        dependencies=[s.step_id for s in new_steps if s.status == "completed"],
                        cost_estimate_usd=0.008,
                        latency_estimate_ms=220.0,
                    )
                )

        total_cost = sum(s.cost_estimate_usd for s in new_steps)
        total_latency = sum(s.latency_estimate_ms for s in new_steps)

        return VisionProgramPlan(
            plan_id=f"plan_replan_{uuid.uuid4().hex[:6]}",
            steps=new_steps,
            strategy_rationale=f"Replanned after critique feedback: {feedback}",
            estimated_total_cost_usd=round(total_cost, 5),
            estimated_total_latency_ms=round(total_latency, 2),
        )
