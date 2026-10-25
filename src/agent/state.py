"""
OmniVision Agent State and Typed Contracts.
Defines Pydantic v2 models and LangGraph AgentState representing the shared
reasoning state, plan steps, observations, critique verdicts, and citations.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field

from src.memory.scene_graph import Persistent4DSceneGraph
from src.tools.base import ObservationID, ToolResult, UncertaintyEstimate


class TraceEvent(BaseModel):
    """Event-sourced trace record capturing every action and state transition."""
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    node_name: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class QueryAnalysis(BaseModel):
    """Structured breakdown and decomposition of the user's visual query."""
    original_query: str
    modality: Literal["image", "video"] = "image"
    reasoning_type: List[str] = Field(
        default_factory=lambda: ["detection"],
        description="Reasoning facets: detection, spatial_relation, 3d_depth, temporal, counting, ocr, action, attributes",
    )
    sub_goals: List[str] = Field(default_factory=list)
    key_targets: List[str] = Field(default_factory=list, description="Specific object classes or targets to observe")
    temporal_span_required: bool = False
    requires_3d: bool = False
    ambiguities: List[str] = Field(default_factory=list)
    estimated_complexity: Literal["low", "medium", "high"] = "medium"


class PlanStep(BaseModel):
    """Executable atomic step in a vision program plan."""
    step_id: str
    sub_goal: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result_observation_id: Optional[str] = None
    cost_estimate_usd: float = 0.001
    latency_estimate_ms: float = 50.0
    error_message: Optional[str] = None


class VisionProgramPlan(BaseModel):
    """Complete structured execution plan synthesized by the Planner."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    steps: List[PlanStep] = Field(default_factory=list)
    strategy_rationale: str = ""
    estimated_total_cost_usd: float = 0.0
    estimated_total_latency_ms: float = 0.0


class CritiqueVerdict(BaseModel):
    """Evaluative judgment by the Critic on observation quality and plan progress."""
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    should_replan: bool = False
    should_active_perceive: bool = False
    active_perception_target: Optional[Dict[str, Any]] = None
    feedback: str = ""
    missing_evidence: List[str] = Field(default_factory=list)
    identified_contradictions: List[str] = Field(default_factory=list)


class ClaimAttribution(BaseModel):
    """Factual atomic claim verified against visual observation evidence."""
    claim_id: str = Field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:6]}")
    statement: str
    supported: bool
    observation_ids: List[str] = Field(default_factory=list)
    grounded_box: Optional[List[float]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verifier_notes: str = ""


class ClaimsVerificationResult(BaseModel):
    """Result of the Hallucination Guard verification run."""
    all_verified: bool
    verified_claims: List[ClaimAttribution] = Field(default_factory=list)
    unsupported_claims: List[ClaimAttribution] = Field(default_factory=list)
    hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    remediated_answer_text: Optional[str] = None


class GroundedSynthesis(BaseModel):
    """Final calibrated, citation-backed answer generated for the user."""
    query: str
    answer: str
    confidence_interval: Tuple[float, float] = (0.85, 0.95)
    calibrated_confidence: float = 0.90
    citations: List[str] = Field(default_factory=list, description="Observation IDs cited in the answer")
    key_findings: List[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    visual_evidence_summary: Dict[str, Any] = Field(default_factory=dict)


class OmniVisionState(BaseModel):
    """
    Typed State for the OmniVision LangGraph multi-agent execution pipeline.
    Maintains complete provenance, event-sourced traces, and persistent scene graph.
    """
    query: str
    media_uri: str
    media_type: Literal["image", "video"] = "image"
    query_analysis: Optional[QueryAnalysis] = None
    plan: Optional[VisionProgramPlan] = None
    current_step_index: int = 0
    observations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    scene_graph: Dict[str, Any] = Field(default_factory=lambda: Persistent4DSceneGraph().to_dict())
    trace_log: List[TraceEvent] = Field(default_factory=list)
    critique: Optional[CritiqueVerdict] = None
    claims_verification: Optional[ClaimsVerificationResult] = None
    final_synthesis: Optional[GroundedSynthesis] = None
    iteration_count: int = 0
    max_iterations: int = 12
    budget_limit_usd: float = 0.50
    accumulated_cost_usd: float = 0.0
    accumulated_latency_ms: float = 0.0
    status: Literal[
        "idle",
        "analyzing",
        "planning",
        "executing",
        "active_perceiving",
        "verifying",
        "replanning",
        "synthesizing",
        "completed",
        "failed",
    ] = "idle"
    error: Optional[str] = None
