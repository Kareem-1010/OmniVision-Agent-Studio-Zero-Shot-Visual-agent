"""
Trace Logger to SFT and DPO Dataset Builder.
Extracts empirical agent execution traces from LangGraph states, scores trajectory quality,
and generates Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) datasets.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.agent.state import OmniVisionState

logger = logging.getLogger("OmniVision.Training.DatasetBuilder")


class AgentTrajectory(BaseModel):
    """Complete episodic trajectory of an agent execution run."""
    trajectory_id: str = Field(default_factory=lambda: f"traj_{uuid.uuid4().hex[:8]}")
    query: str
    media_uri: str
    query_analysis: Dict[str, Any] = Field(default_factory=dict)
    plan: Dict[str, Any] = Field(default_factory=dict)
    trace_events: List[Dict[str, Any]] = Field(default_factory=list)
    critique: Dict[str, Any] = Field(default_factory=dict)
    claims_verification: Dict[str, Any] = Field(default_factory=dict)
    final_synthesis: Dict[str, Any] = Field(default_factory=dict)
    hallucination_rate: float = 0.0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    reward_score: float = 0.0  # Scalar quality score in [0.0, 1.0]


class SFTSample(BaseModel):
    """Standardized prompt-response pair for Supervised Fine-Tuning."""
    sample_id: str = Field(default_factory=lambda: f"sft_{uuid.uuid4().hex[:6]}")
    system_prompt: str = "You are OmniVision Planner, synthesizing optimal vision programs."
    instruction: str
    input_context: str
    output_response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DPOPreferencePair(BaseModel):
    """Preference pair for Direct Preference Optimization (Chosen vs Rejected)."""
    pair_id: str = Field(default_factory=lambda: f"dpo_{uuid.uuid4().hex[:6]}")
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    reward_margin: float


class DatasetBuilder:
    """
    Automated SFT and DPO Dataset Synthesis Pipeline.
    Filters high-reward trajectories for SFT and pairs contrasting runs for DPO.
    """

    def __init__(self, reward_pass_threshold: float = 0.75) -> None:
        self.reward_threshold = reward_pass_threshold
        self._harvested_trajectories: List[AgentTrajectory] = []

    def harvest_from_state(
        self,
        state: OmniVisionState,
        explicit_reward: Optional[float] = None,
    ) -> AgentTrajectory:
        """
        Extract structured trajectory from completed OmniVisionState and calculate quality reward.
        """
        # Calculate reward score if not explicitly given
        reward = explicit_reward
        if reward is None:
            critique_score = state.critique.score if state.critique else 0.5
            halluc_penalty = (state.claims_verification.hallucination_rate if state.claims_verification else 0.0) * 0.4
            cost_penalty = min(0.2, state.accumulated_cost_usd / max(0.001, state.budget_limit_usd) * 0.1)
            reward = max(0.0, min(1.0, critique_score - halluc_penalty - cost_penalty))

        traj = AgentTrajectory(
            query=state.query,
            media_uri=state.media_uri,
            query_analysis=state.query_analysis.model_dump() if state.query_analysis else {},
            plan=state.plan.model_dump() if state.plan else {},
            trace_events=[e.model_dump() for e in state.trace_log],
            critique=state.critique.model_dump() if state.critique else {},
            claims_verification=state.claims_verification.model_dump() if state.claims_verification else {},
            final_synthesis=state.final_synthesis.model_dump() if state.final_synthesis else {},
            hallucination_rate=state.claims_verification.hallucination_rate if state.claims_verification else 0.0,
            total_cost_usd=state.accumulated_cost_usd,
            total_latency_ms=state.accumulated_latency_ms,
            reward_score=round(reward, 4),
        )
        self._harvested_trajectories.append(traj)
        return traj

    def build_sft_dataset(
        self,
        trajectories: Optional[List[AgentTrajectory]] = None,
        min_reward: Optional[float] = None,
    ) -> List[SFTSample]:
        """
        Generate SFT samples from high-performing trajectories (reward >= min_reward).
        """
        pool = trajectories if trajectories is not None else self._harvested_trajectories
        threshold = min_reward if min_reward is not None else self.reward_threshold

        sft_samples: List[SFTSample] = []
        for t in pool:
            if t.reward_score >= threshold and t.plan:
                # 1. Planner SFT: Synthesizing plan from query
                plan_json = json.dumps(t.plan, indent=2)
                sample = SFTSample(
                    system_prompt="You are OmniVision Planner, an expert visual program synthesizer.",
                    instruction=f"Synthesize an optimal executable vision program for the user query.",
                    input_context=f"Query: {t.query}\nMedia: {t.media_uri}\nAnalysis: {json.dumps(t.query_analysis)}",
                    output_response=plan_json,
                    metadata={"reward": t.reward_score, "trajectory_id": t.trajectory_id},
                )
                sft_samples.append(sample)

        logger.info(f"Synthesized {len(sft_samples)} SFT training samples from {len(pool)} trajectories.")
        return sft_samples

    def build_dpo_pairs(
        self,
        trajectories: Optional[List[AgentTrajectory]] = None,
        min_margin: float = 0.20,
    ) -> List[DPOPreferencePair]:
        """
        Generate DPO chosen/rejected preference pairs by matching trajectories with shared query semantics.
        """
        pool = trajectories if trajectories is not None else self._harvested_trajectories
        dpo_pairs: List[DPOPreferencePair] = []

        # Group trajectories by query
        query_groups: Dict[str, List[AgentTrajectory]] = {}
        for t in pool:
            query_groups.setdefault(t.query.strip().lower(), []).append(t)

        for q_key, group in query_groups.items():
            if len(group) >= 2:
                # Sort descending by reward
                sorted_group = sorted(group, key=lambda x: x.reward_score, reverse=True)
                winner = sorted_group[0]
                loser = sorted_group[-1]

                margin = winner.reward_score - loser.reward_score
                if margin >= min_margin:
                    pair = DPOPreferencePair(
                        prompt=f"Task: Synthesize vision plan and answer query: '{winner.query}'. Media: {winner.media_uri}",
                        chosen=json.dumps({"plan": winner.plan, "answer": winner.final_synthesis.get("answer", "")}, indent=2),
                        rejected=json.dumps({"plan": loser.plan, "answer": loser.final_synthesis.get("answer", "")}, indent=2),
                        chosen_reward=winner.reward_score,
                        rejected_reward=loser.reward_score,
                        reward_margin=round(margin, 4),
                    )
                    dpo_pairs.append(pair)

        logger.info(f"Generated {len(dpo_pairs)} DPO preference pairs.")
        return dpo_pairs

    @staticmethod
    def export_to_jsonl(
        samples: List[Union[SFTSample, DPOPreferencePair]],
        filepath: str,
    ) -> int:
        """Export samples to JSONL format."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item.model_dump()) + "\n")
                count += 1
        logger.info(f"Exported {count} records to {filepath}")
        return count
