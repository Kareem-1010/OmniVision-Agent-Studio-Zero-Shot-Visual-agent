"""
OmniVision LangGraph Multi-Agent Orchestrator.
Compiles and coordinates the specialized agent nodes (Analyst, Planner, Executor,
Active Perception, Critic, Replanner, Hallucination Guard, Synthesizer) into an
executable stateful computation graph with conditional loops and telemetry.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Union

# Load .env early so all os.getenv() calls pick up the API keys
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(dotenv_path=_env_path, override=False)
except Exception:
    pass

from langgraph.graph import END, START, StateGraph

from src.agent.active_perception import ActivePerceptionManager
from src.agent.critic import CriticVerifier
from src.agent.hallucination_guard import HallucinationGuard
from src.agent.planner import VisionPlanner
from src.agent.query_analyst import QueryAnalyst
from src.agent.state import (
    GroundedSynthesis,
    OmniVisionState,
    PlanStep,
    TraceEvent,
)
from src.agent.synthesizer import GroundedSynthesizer
from src.agent.tool_router import ToolRouter
from src.geometry.relations import GeometryEngine
from src.memory.scene_graph import Persistent4DSceneGraph, SceneGraphEdge, SceneGraphNode
from src.tools import (
    DenseFeatureTool,
    DetectorEnsembleTool,
    GroundedSegmenterTool,
    GroundedVLMTool,
    MetricDepthTool,
    OpenVocabDetectorTool,
    OpticalFlowTool,
    PoseActionTool,
    SceneOCRTool,
    VideoTrackerTool,
    VisualAttributeTool,
    VisualizerTool,
    ZeroShotClassifierTool,
)

logger = logging.getLogger("OmniVision.Agent.Orchestrator")


class ToolDispatcher:
    """Instantiates and executes registered vision tools on demand."""

    def __init__(self) -> None:
        self._tools: Dict[str, Any] = {
            "detector_ensemble": DetectorEnsembleTool(),
            "open_vocab": OpenVocabDetectorTool(),
            "grounded_segmenter": GroundedSegmenterTool(),
            "video_tracker": VideoTrackerTool(),
            "classifier": ZeroShotClassifierTool(),
            "dense_features": DenseFeatureTool(),
            "metric_depth": MetricDepthTool(),
            "pose_action": PoseActionTool(),
            "scene_ocr": SceneOCRTool(),
            "grounded_vlm": GroundedVLMTool(),
            "optical_flow": OpticalFlowTool(),
            "visual_attributes": VisualAttributeTool(),
            "visualizer": VisualizerTool(),
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute named tool with given arguments."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Vision tool '{tool_name}' is not registered in ToolDispatcher.")
        return tool.execute(arguments)

    # Convenience alias
    dispatch = execute



def _extract_box_list(box_obj: Any) -> Optional[List[float]]:
    """Safely convert bounding box objects, models, or dicts into [x1, y1, x2, y2]."""
    if not box_obj:
        return None
    if isinstance(box_obj, (list, tuple)):
        return [float(v) for v in box_obj]
    if hasattr(box_obj, "model_dump"):
        box_obj = box_obj.model_dump()
    if isinstance(box_obj, dict):
        if "x1" in box_obj and "y1" in box_obj:
            return [float(box_obj["x1"]), float(box_obj["y1"]), float(box_obj["x2"]), float(box_obj["y2"])]
    return None


class OmniVisionAgent:
    """
    OmniVision Multi-Agent Visual Reasoning System.
    Compiles and executes a LangGraph state machine coordinating specialized agent nodes.
    """

    def __init__(
        self,
        max_iterations: int = 12,
        budget_limit_usd: float = 0.50,
        enable_tree_search: bool = True,
        enable_active_perception: bool = True,
    ) -> None:
        self.max_iterations = max_iterations
        self.budget_limit_usd = budget_limit_usd
        self.enable_tree_search = enable_tree_search
        self.enable_active_perception = enable_active_perception

        # Auto-select LLM provider from environment
        from src.utils.llm_factory import LLMFactory
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key and not groq_key.startswith("gsk_xxxx"):
            try:
                groq_model = os.getenv("OMNIVISION_LLM_MODEL", "openai/gpt-oss-120b")
                shared_llm = LLMFactory.create(
                    provider="groq",
                    model=groq_model,
                    api_key=groq_key,
                    temperature=0.0,
                    max_tokens=1024,
                )
                logger.info(f"OmniVisionAgent: using Groq {groq_model} for NL reasoning")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}. Using mock LLM.")
                shared_llm = LLMFactory.create(provider="mock")
        else:
            shared_llm = LLMFactory.create(provider="mock")

        # Initialize Nodes & Sub-Agents
        self.analyst = QueryAnalyst(llm_client=shared_llm)
        self.tool_router = ToolRouter()
        self.planner = VisionPlanner(tool_router=self.tool_router)
        self.active_perception = ActivePerceptionManager()
        self.critic = CriticVerifier()
        self.hallucination_guard = HallucinationGuard()
        self.synthesizer = GroundedSynthesizer(llm_client=shared_llm)
        self.dispatcher = ToolDispatcher()

        # Build & Compile LangGraph
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct StateGraph and compile execution edges."""
        workflow = StateGraph(OmniVisionState)

        # Register Agent Nodes
        workflow.add_node("analyst_node", self._analyst_node)
        workflow.add_node("planner_node", self._planner_node)
        workflow.add_node("executor_node", self._executor_node)
        workflow.add_node("active_perception_node", self._active_perception_node)
        workflow.add_node("critic_node", self._critic_node)
        workflow.add_node("replanner_node", self._replanner_node)
        workflow.add_node("hallucination_guard_node", self._hallucination_guard_node)
        workflow.add_node("synthesizer_node", self._synthesizer_node)

        # Linear Initial Sequence
        workflow.add_edge(START, "analyst_node")
        workflow.add_edge("analyst_node", "planner_node")
        workflow.add_edge("planner_node", "executor_node")

        # Routing after step execution
        workflow.add_conditional_edges(
            "executor_node",
            self._route_after_execution,
            {
                "continue_execution": "executor_node",
                "active_perceive": "active_perception_node",
                "critique": "critic_node",
            },
        )

        # Active perception rejoins the evaluation flow at critic
        workflow.add_edge("active_perception_node", "critic_node")

        # Routing after critic review
        workflow.add_conditional_edges(
            "critic_node",
            self._route_after_critique,
            {
                "replan": "replanner_node",
                "guard": "hallucination_guard_node",
            },
        )

        workflow.add_edge("replanner_node", "executor_node")
        workflow.add_edge("hallucination_guard_node", "synthesizer_node")
        workflow.add_edge("synthesizer_node", END)

        return workflow.compile()

    # -------------------------------------------------------------
    # Node Implementations
    # -------------------------------------------------------------

    def _analyst_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Query Analyst Node: breaks query into multimodal requirements."""
        t0 = time.time()
        analysis = self.analyst.analyze(state.query, state.media_uri, state.media_type)
        latency = (time.time() - t0) * 1000.0

        trace = TraceEvent(
            node_name="QueryAnalyst",
            action="analyze_query",
            details={"reasoning_type": analysis.reasoning_type, "sub_goals": analysis.sub_goals},
            latency_ms=latency,
        )
        return {
            "query_analysis": analysis,
            "status": "planning",
            "trace_log": state.trace_log + [trace],
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    def _planner_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Planner Node: synthesizes executable vision DAG."""
        t0 = time.time()
        plan = self.planner.synthesize_plan(
            analysis=state.query_analysis,
            media_uri=state.media_uri,
            budget_limit_usd=state.budget_limit_usd,
            enable_tree_search=self.enable_tree_search,
        )
        latency = (time.time() - t0) * 1000.0

        trace = TraceEvent(
            node_name="VisionPlanner",
            action="synthesize_plan",
            details={"plan_id": plan.plan_id, "num_steps": len(plan.steps)},
            latency_ms=latency,
        )
        return {
            "plan": plan,
            "current_step_index": 0,
            "status": "executing",
            "trace_log": state.trace_log + [trace],
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    def _executor_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Executor Node: executes pending plan step and registers observations."""
        plan = state.plan
        if not plan or state.current_step_index >= len(plan.steps):
            return {"status": "verifying"}

        step = plan.steps[state.current_step_index]
        t0 = time.time()
        logger.info(f"Executing step {step.step_id} ({step.tool_name})...")

        # Inject context from prior steps if needed
        args = dict(step.arguments)
        prior_boxes: List[List[float]] = []
        for prev_obs in state.observations.values():
            obs_data = prev_obs.get("data")
            if not obs_data or not isinstance(obs_data, dict):
                continue
            for d in obs_data.get("detections", []):
                b = _extract_box_list(d.get("box") or d.get("box_2d"))
                if b:
                    prior_boxes.append(b)
            for inst in obs_data.get("instances", []):
                b = _extract_box_list(inst.get("box"))
                if b:
                    prior_boxes.append(b)
            if prior_boxes:
                break

        if prior_boxes:
            if step.tool_name == "pose_action" and "person_boxes" not in args:
                args["person_boxes"] = prior_boxes
            elif step.tool_name == "visual_attributes" and "roi_box" not in args:
                args["roi_box"] = prior_boxes[0]
            elif step.tool_name == "scene_ocr" and "target_object_boxes" not in args:
                args["target_object_boxes"] = prior_boxes
            elif step.tool_name == "metric_depth" and "target_boxes" not in args:
                args["target_boxes"] = prior_boxes

        tool_result = self.dispatcher.execute(step.tool_name, args)
        step_latency = (time.time() - t0) * 1000.0

        # Update step status
        step.status = "completed" if tool_result.success else "failed"
        step.result_observation_id = tool_result.observation_id
        if not tool_result.success:
            step.error_message = tool_result.error_message

        # Record observation in state
        new_obs = dict(state.observations)
        new_obs[tool_result.observation_id] = tool_result.model_dump()

        # Update Scene Graph if instances or 3D info extracted
        scene_graph = Persistent4DSceneGraph.from_dict(state.scene_graph)
        self._update_scene_graph(scene_graph, step.tool_name, tool_result.data)

        trace = TraceEvent(
            node_name="ToolExecutor",
            action=f"execute_{step.tool_name}",
            details={"step_id": step.step_id, "observation_id": tool_result.observation_id, "success": tool_result.success},
            cost_usd=tool_result.cost_usd,
            latency_ms=step_latency,
        )

        return {
            "plan": plan,
            "current_step_index": state.current_step_index + 1,
            "observations": new_obs,
            "scene_graph": scene_graph.to_dict(),
            "trace_log": state.trace_log + [trace],
            "accumulated_cost_usd": state.accumulated_cost_usd + tool_result.cost_usd,
            "accumulated_latency_ms": state.accumulated_latency_ms + step_latency,
        }

    def _active_perception_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Active Perception Node: executes targeted zoom/consensus action."""
        action = self.active_perception.evaluate_observations_for_active_perception(
            observations=state.observations,
            media_uri=state.media_uri,
        )
        if not action:
            return {"status": "verifying"}

        t0 = time.time()
        tool_result = self.dispatcher.execute(action.remediation_tool, action.arguments)
        latency = (time.time() - t0) * 1000.0

        new_obs = dict(state.observations)
        new_obs[tool_result.observation_id] = tool_result.model_dump()

        trace = TraceEvent(
            node_name="ActivePerception",
            action=f"active_{action.action_type}",
            details={"remediation_tool": action.remediation_tool, "observation_id": tool_result.observation_id},
            cost_usd=tool_result.cost_usd,
            latency_ms=latency,
        )
        return {
            "observations": new_obs,
            "status": "verifying",
            "trace_log": state.trace_log + [trace],
            "accumulated_cost_usd": state.accumulated_cost_usd + tool_result.cost_usd,
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    def _critic_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Critic Node: self-reflection and cross-tool consistency check."""
        t0 = time.time()
        verdict = self.critic.evaluate(
            analysis=state.query_analysis,
            plan_steps=state.plan.steps if state.plan else [],
            observations=state.observations,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
        )
        latency = (time.time() - t0) * 1000.0

        trace = TraceEvent(
            node_name="CriticVerifier",
            action="evaluate_observations",
            details={"passed": verdict.passed, "score": verdict.score, "should_replan": verdict.should_replan},
            latency_ms=latency,
        )
        return {
            "critique": verdict,
            "status": "replanning" if verdict.should_replan else "verifying",
            "trace_log": state.trace_log + [trace],
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    def _replanner_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Replanner Node: modifies plan in response to missing evidence."""
        t0 = time.time()
        new_plan = self.planner.replan_on_feedback(
            current_plan=state.plan,
            feedback=state.critique.feedback if state.critique else "Missing evidence",
            missing_evidence=state.critique.missing_evidence if state.critique else [],
            media_uri=state.media_uri,
        )
        latency = (time.time() - t0) * 1000.0

        trace = TraceEvent(
            node_name="Replanner",
            action="replan_plan",
            details={"new_steps": len(new_plan.steps)},
            latency_ms=latency,
        )
        return {
            "plan": new_plan,
            "iteration_count": state.iteration_count + 1,
            "status": "executing",
            "trace_log": state.trace_log + [trace],
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    def _hallucination_guard_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Hallucination Guard Node: verifies claim attribution against observation index."""
        t0 = time.time()
        # Formulate initial draft summary from findings
        draft_text = f"Visual analysis for query: {state.query}. "
        counts: Dict[str, int] = {}
        for obs in state.observations.values():
            data = obs.get("data", {})
            for d in data.get("detections", []):
                lbl = d.get("label", "object")
                counts[lbl] = counts.get(lbl, 0) + 1
            for inst in data.get("instances", []):
                lbl = inst.get("phrase", "object")
                counts[lbl] = counts.get(lbl, 0) + 1
            for inst in data.get("instances_3d", []):
                draft_text += f"The {inst.get('label', 'object')} is located at depth {inst.get('centroid_3d', [0,0,0])[2]:.1f}m. "
            for t in data.get("recognized_texts", []):
                draft_text += f"Detected text reads '{t.get('text', '')}'. "

        if counts:
            count_items = [f"{cnt} {lbl}(s)" for lbl, cnt in counts.items()]
            draft_text += f"Identified {', '.join(count_items)} in the scene. "
            if state.query_analysis and ("summary" in state.query_analysis.reasoning_type or any(w in state.query.lower() for w in ["summary", "summarize", "describe", "overview"])):
                draft_text += f"The visual scene shows {', '.join(count_items)} distributed across the frame. "
        else:
            draft_text += "No target instances were detected in the scene. "

        verification = self.hallucination_guard.verify_draft_answer(
            draft_text=draft_text,
            observations=state.observations,
            scene_graph_dict=state.scene_graph,
        )
        latency = (time.time() - t0) * 1000.0

        trace = TraceEvent(
            node_name="HallucinationGuard",
            action="verify_claims",
            details={"all_verified": verification.all_verified, "hallucination_rate": verification.hallucination_rate},
            latency_ms=latency,
        )
        return {
            "claims_verification": verification,
            "status": "synthesizing",
            "trace_log": state.trace_log + [trace],
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    def _synthesizer_node(self, state: OmniVisionState) -> Dict[str, Any]:
        """Synthesizer Node: generates grounded final answer with citations."""
        t0 = time.time()
        synthesis = self.synthesizer.synthesize(
            analysis=state.query_analysis,
            observations=state.observations,
            scene_graph_dict=state.scene_graph,
            claims_verification=state.claims_verification,
            accumulated_cost_usd=state.accumulated_cost_usd,
            accumulated_latency_ms=state.accumulated_latency_ms,
            media_uri=state.media_uri,
        )
        latency = (time.time() - t0) * 1000.0

        trace = TraceEvent(
            node_name="Synthesizer",
            action="grounded_synthesis",
            details={"citations": synthesis.citations, "confidence": synthesis.calibrated_confidence},
            latency_ms=latency,
        )
        return {
            "final_synthesis": synthesis,
            "status": "completed",
            "trace_log": state.trace_log + [trace],
            "accumulated_latency_ms": state.accumulated_latency_ms + latency,
        }

    # -------------------------------------------------------------
    # Conditional Routing Logic
    # -------------------------------------------------------------

    def _route_after_execution(self, state: OmniVisionState) -> str:
        """Route to next execution step, active perception, or critic."""
        plan = state.plan
        if plan and state.current_step_index < len(plan.steps):
            return "continue_execution"

        # Check if active perception is needed for elevated uncertainty
        if self.enable_active_perception:
            action = self.active_perception.evaluate_observations_for_active_perception(
                observations=state.observations,
                media_uri=state.media_uri,
            )
            if action:
                return "active_perceive"

        return "critique"

    def _route_after_critique(self, state: OmniVisionState) -> str:
        """Route to replanner if critique failed and iterations remain, else to guard."""
        critique = state.critique
        if critique and critique.should_replan and state.iteration_count < state.max_iterations:
            return "replan"
        return "guard"

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _update_scene_graph(self, sg: Persistent4DSceneGraph, tool_name: str, data: Any) -> None:
        """Incrementally insert observation instances into persistent scene graph."""
        if not data:
            return
        data_dict = data.model_dump() if hasattr(data, "model_dump") else data
        if not isinstance(data_dict, dict):
            return

        if tool_name == "detector_ensemble":
            for i, d in enumerate(data_dict.get("detections", [])):
                lbl = d.get("label", "object")
                b = _extract_box_list(d.get("box") or d.get("box_2d"))
                nid = f"inst_{lbl}_{i}"
                if not sg.get_node(nid):
                    sg.add_node(
                        SceneGraphNode(
                            node_id=nid,
                            label=lbl,
                            bbox_2d=b,
                            attributes={"confidence": d.get("confidence", 0.9)},
                        )
                    )

        elif tool_name == "open_vocab":
            for i, inst in enumerate(data_dict.get("instances", [])):
                lbl = inst.get("phrase", "object")
                b = _extract_box_list(inst.get("box"))
                nid = f"inst_open_{lbl}_{i}"
                if not sg.get_node(nid):
                    sg.add_node(
                        SceneGraphNode(
                            node_id=nid,
                            label=lbl,
                            bbox_2d=b,
                            attributes={"confidence": inst.get("confidence", 0.9)},
                        )
                    )

        elif tool_name == "metric_depth":
            for i, inst in enumerate(data_dict.get("instances_3d", [])):
                lbl = inst.get("label", "object")
                nid = f"inst_3d_{lbl}_{i}"
                sg.add_node(
                    SceneGraphNode(
                        node_id=nid,
                        label=lbl,
                        centroid_3d=inst.get("centroid_3d"),
                        dimensions_3d=inst.get("dimensions_3d_meters"),
                        attributes={"distance": inst.get("distance_to_camera_meters")},
                    )
                )

        elif tool_name == "pose_action":
            for i, p in enumerate(data_dict.get("persons", [])):
                nid = f"person_pose_{i}"
                b = _extract_box_list(p.get("bbox"))
                sg.add_node(
                    SceneGraphNode(
                        node_id=nid,
                        label="person",
                        bbox_2d=b,
                        attributes={
                            "posture": p.get("posture"),
                            "is_fall_detected": p.get("is_fall_detected"),
                            "ppe_compliance": p.get("ppe_compliance"),
                        },
                    )
                )

    # -------------------------------------------------------------
    # Public Execution Runner
    # -------------------------------------------------------------

    def run(
        self,
        query: str,
        media_uri: str,
        media_type: Literal["image", "video"] = "image",
    ) -> GroundedSynthesis:
        """
        Execute full multi-agent pipeline and return grounded answer.
        """
        initial_state = OmniVisionState(
            query=query,
            media_uri=media_uri,
            media_type=media_type,
            max_iterations=self.max_iterations,
            budget_limit_usd=self.budget_limit_usd,
        )

        final_state = self.graph.invoke(initial_state)

        if isinstance(final_state, dict):
            synthesis = final_state.get("final_synthesis")
            if isinstance(synthesis, GroundedSynthesis):
                return synthesis
            elif isinstance(synthesis, dict):
                return GroundedSynthesis.model_validate(synthesis)

        raise RuntimeError("OmniVision pipeline did not produce a valid GroundedSynthesis result.")
