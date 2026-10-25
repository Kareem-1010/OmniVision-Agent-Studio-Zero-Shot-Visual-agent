"""
Active Perception Loop and Uncertainty Minimization Engine.
Dynamically triggers visual actions (high-resolution zoom crops, multi-prompt consensus,
and temporal neighborhood inspection) when uncertainty or conformal interval entropy is elevated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.tools.base import ObservationID, ToolResult, UncertaintyEstimate

logger = logging.getLogger("OmniVision.Agent.ActivePerception")


class ActivePerceptionAction(BaseModel):
    """Specific active visual action to execute for uncertainty reduction."""
    action_type: str  # 'zoom_crop', 'multi_prompt_consensus', 'temporal_neighbor_sampling'
    target_box: Optional[List[float]] = None  # [x1, y1, x2, y2]
    context_margin: float = 0.20
    remediation_tool: str = "open_vocab"
    arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ActivePerceptionManager:
    """
    Active Perception Manager.
    Evaluates observation uncertainty and synthesizes targeted active actions to resolve ambiguities.
    """

    def __init__(self, uncertainty_threshold: float = 0.65, zoom_box_area_threshold: float = 0.05) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.zoom_box_area_threshold = zoom_box_area_threshold

    def evaluate_observations_for_active_perception(
        self,
        observations: Dict[str, Dict[str, Any]],
        media_uri: str,
        image_dims: Tuple[int, int] = (640, 480),
    ) -> Optional[ActivePerceptionAction]:
        """
        Scan completed tool observations for high uncertainty or small ambiguous detections.
        Returns an ActivePerceptionAction if action is warranted, else None.
        """
        img_w, img_h = image_dims
        frame_area = float(img_w * img_h)

        for obs_id, obs_dict in observations.items():
            uncertainty = obs_dict.get("uncertainty", {})
            confidence = uncertainty.get("confidence", 1.0)
            is_ambiguous = uncertainty.get("is_ambiguous", False)
            tool_name = obs_dict.get("tool_name", "")
            data = obs_dict.get("data", {})

            # 1. Check for ambiguous low-confidence detections
            if confidence < self.uncertainty_threshold or is_ambiguous:
                # Find candidate target box to zoom in on
                target_box = None
                if tool_name in ("detector_ensemble", "open_vocab"):
                    dets = data.get("detections", [])
                    if dets:
                        # Pick detection with lowest confidence
                        sorted_dets = sorted(dets, key=lambda d: d.get("confidence", 1.0))
                        lowest = sorted_dets[0]
                        target_box = lowest.get("box_2d")

                if target_box:
                    # Calculate box area relative to frame
                    bw = max(0.0, target_box[2] - target_box[0])
                    bh = max(0.0, target_box[3] - target_box[1])
                    box_area = (bw * bh) / max(1.0, frame_area)

                    # Expand box with 20% context margin
                    margin_x = bw * 0.20
                    margin_y = bh * 0.20
                    expanded_box = [
                        max(0.0, target_box[0] - margin_x),
                        max(0.0, target_box[1] - margin_y),
                        min(float(img_w), target_box[2] + margin_x),
                        min(float(img_h), target_box[3] + margin_y),
                    ]

                    logger.info(f"Active Perception: triggering high-res zoom on {obs_id} (conf: {confidence:.2f}, area: {box_area:.3f})")
                    return ActivePerceptionAction(
                        action_type="zoom_crop",
                        target_box=expanded_box,
                        context_margin=0.20,
                        remediation_tool="open_vocab",
                        arguments={
                            "image_uri": media_uri,
                            "text_queries": ["object"],
                            "box_threshold": 0.20,
                        },
                        rationale=f"Observation {obs_id} had elevated uncertainty ({confidence:.2f}); active zoom crop triggered.",
                    )

            # 2. Check for OCR fallback ambiguity
            if tool_name == "scene_ocr" and data.get("total_text_blocks", 0) == 0 and is_ambiguous:
                return ActivePerceptionAction(
                    action_type="multi_prompt_consensus",
                    remediation_tool="grounded_vlm",
                    arguments={
                        "image_or_video_uri": media_uri,
                        "prompt": "Read and transcribe all visible text in the scene carefully with bounding coordinates.",
                        "mode": "grounded_vqa",
                    },
                    rationale=f"OCR returned zero text with high uncertainty on {obs_id}; triggering grounded VLM re-examination.",
                )

        return None
