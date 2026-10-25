"""
Learned and Heuristic Tool Router.
Selects optimal perception tools based on subtask semantics, target object taxonomy,
accuracy requirements, and latency/cost budget constraints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Agent.ToolRouter")


class ToolRoutingDecision(BaseModel):
    """Routing choice with selected tool and cost-accuracy trade-off justification."""
    selected_tool: str
    rationale: str
    estimated_cost_usd: float
    estimated_latency_ms: float
    expected_accuracy: float
    fallback_tool: Optional[str] = None


class ToolRouter:
    """
    Learned and Semantic Tool Router.
    Maps high-level sub-goals and query analysis specifications to concrete vision tools.
    """

    # Catalog of available perception tools and performance profiles
    TOOL_PROFILES: Dict[str, Dict[str, Any]] = {
        "detector_ensemble": {
            "name": "detector_ensemble",
            "cost_usd": 0.0005,
            "latency_ms": 35.0,
            "accuracy": 0.91,
            "capabilities": ["detection", "bounding_box", "coco_classes", "counting"],
        },
        "open_vocab": {
            "name": "open_vocab",
            "cost_usd": 0.0020,
            "latency_ms": 85.0,
            "accuracy": 0.89,
            "capabilities": ["open_vocab_detection", "rare_classes", "complex_queries", "detection"],
        },
        "grounded_segmenter": {
            "name": "grounded_segmenter",
            "cost_usd": 0.0035,
            "latency_ms": 110.0,
            "accuracy": 0.93,
            "capabilities": ["segmentation", "polygons", "masks", "boundaries"],
        },
        "video_tracker": {
            "name": "video_tracker",
            "cost_usd": 0.0015,
            "latency_ms": 45.0,
            "accuracy": 0.88,
            "capabilities": ["temporal_tracking", "trajectories", "reid", "movement"],
        },
        "metric_depth": {
            "name": "metric_depth",
            "cost_usd": 0.0012,
            "latency_ms": 50.0,
            "accuracy": 0.87,
            "capabilities": ["3d_depth", "distance", "camera_space", "ground_plane"],
        },
        "pose_action": {
            "name": "pose_action",
            "cost_usd": 0.0018,
            "latency_ms": 60.0,
            "accuracy": 0.90,
            "capabilities": ["action_pose", "fall_detection", "ppe", "posture", "keypoints"],
        },
        "scene_ocr": {
            "name": "scene_ocr",
            "cost_usd": 0.0010,
            "latency_ms": 40.0,
            "accuracy": 0.92,
            "capabilities": ["ocr", "text_extraction", "reading", "signs", "labels"],
        },
        "visual_attributes": {
            "name": "visual_attributes",
            "cost_usd": 0.0004,
            "latency_ms": 20.0,
            "accuracy": 0.89,
            "capabilities": ["attributes", "color", "texture", "material", "cielab"],
        },
        "dense_features": {
            "name": "dense_features",
            "cost_usd": 0.0015,
            "latency_ms": 55.0,
            "accuracy": 0.88,
            "capabilities": ["dense_features", "patch_matching", "similarity", "retrieval"],
        },
        "optical_flow": {
            "name": "optical_flow",
            "cost_usd": 0.0016,
            "latency_ms": 48.0,
            "accuracy": 0.86,
            "capabilities": ["optical_flow", "motion_clusters", "camera_pan"],
        },
        "grounded_vlm": {
            "name": "grounded_vlm",
            "cost_usd": 0.0080,
            "latency_ms": 220.0,
            "accuracy": 0.94,
            "capabilities": ["vqa", "oracle_verification", "complex_reasoning", "grounded_vqa"],
        },
        "visualizer": {
            "name": "visualizer",
            "cost_usd": 0.0002,
            "latency_ms": 15.0,
            "accuracy": 1.0,
            "capabilities": ["visualizer", "render_boxes", "annotation"],
        },
    }

    # Standard COCO 80 classes
    COCO_VOCAB = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    }

    def route_subgoal(
        self,
        subgoal: str,
        target_entities: Optional[List[str]] = None,
        budget_remaining_usd: float = 0.50,
        prefer_speed: bool = False,
    ) -> ToolRoutingDecision:
        """
        Determine the optimal tool to satisfy a specific sub-goal.
        """
        sg_lower = subgoal.lower()
        targets = [t.lower() for t in (target_entities or [])]

        # 1. OCR / Text reading
        if any(w in sg_lower for w in ["read", "text", "ocr", "sign", "words", "label"]):
            profile = self.TOOL_PROFILES["scene_ocr"]
            return ToolRoutingDecision(
                selected_tool="scene_ocr",
                rationale="Sub-goal requires text detection and recognition.",
                estimated_cost_usd=profile["cost_usd"],
                estimated_latency_ms=profile["latency_ms"],
                expected_accuracy=profile["accuracy"],
                fallback_tool="grounded_vlm",
            )

        # 2. Pose / Action / Safety / Fall
        if any(w in sg_lower for w in ["pose", "fall", "posture", "keypoint", "ppe", "safety"]):
            profile = self.TOOL_PROFILES["pose_action"]
            return ToolRoutingDecision(
                selected_tool="pose_action",
                rationale="Sub-goal requires human pose keypoint estimation or action safety analysis.",
                estimated_cost_usd=profile["cost_usd"],
                estimated_latency_ms=profile["latency_ms"],
                expected_accuracy=profile["accuracy"],
                fallback_tool="grounded_vlm",
            )

        # 3. Depth & 3D
        if any(w in sg_lower for w in ["depth", "3d", "distance", "meters", "ground plane"]):
            profile = self.TOOL_PROFILES["metric_depth"]
            return ToolRoutingDecision(
                selected_tool="metric_depth",
                rationale="Sub-goal requires metric depth estimation and 3D back-projection.",
                estimated_cost_usd=profile["cost_usd"],
                estimated_latency_ms=profile["latency_ms"],
                expected_accuracy=profile["accuracy"],
                fallback_tool=None,
            )

        # 4. Temporal Tracking & Motion
        if any(w in sg_lower for w in ["track", "temporal", "motion vector", "trajectory", "line crossing"]):
            profile = self.TOOL_PROFILES["video_tracker"]
            return ToolRoutingDecision(
                selected_tool="video_tracker",
                rationale="Sub-goal requires multi-frame tracking and trajectory estimation.",
                estimated_cost_usd=profile["cost_usd"],
                estimated_latency_ms=profile["latency_ms"],
                expected_accuracy=profile["accuracy"],
                fallback_tool="optical_flow",
            )

        # 5. Visual Attributes (Color, Texture)
        if any(w in sg_lower for w in ["color", "texture", "material", "attribute"]):
            profile = self.TOOL_PROFILES["visual_attributes"]
            return ToolRoutingDecision(
                selected_tool="visual_attributes",
                rationale="Sub-goal requires colorimetric or textural attribute extraction.",
                estimated_cost_usd=profile["cost_usd"],
                estimated_latency_ms=profile["latency_ms"],
                expected_accuracy=profile["accuracy"],
                fallback_tool="grounded_vlm",
            )

        # 6. Segmentation
        if any(w in sg_lower for w in ["segment", "mask", "polygon", "boundary"]):
            profile = self.TOOL_PROFILES["grounded_segmenter"]
            return ToolRoutingDecision(
                selected_tool="grounded_segmenter",
                rationale="Sub-goal requires fine pixel-level mask segmentation.",
                estimated_cost_usd=profile["cost_usd"],
                estimated_latency_ms=profile["latency_ms"],
                expected_accuracy=profile["accuracy"],
                fallback_tool="detector_ensemble",
            )

        # 7. Detection: Standard vs Open-Vocabulary
        if any(w in sg_lower for w in ["detect", "localize", "find", "identify", "count"]):
            # Check if all target classes are standard COCO classes (handling plurals)
            canon_targets = [t.rstrip("s") if (t.endswith("s") and t.rstrip("s") in self.COCO_VOCAB) else t for t in targets]
            all_coco = bool(canon_targets) and all(t in self.COCO_VOCAB for t in canon_targets)
            if all_coco and not prefer_speed:
                profile = self.TOOL_PROFILES["detector_ensemble"]
                return ToolRoutingDecision(
                    selected_tool="detector_ensemble",
                    rationale=f"Targets {targets} reside in standard vocabulary; using fast ensemble detector.",
                    estimated_cost_usd=profile["cost_usd"],
                    estimated_latency_ms=profile["latency_ms"],
                    expected_accuracy=profile["accuracy"],
                    fallback_tool="open_vocab",
                )
            else:
                profile = self.TOOL_PROFILES["open_vocab"]
                return ToolRoutingDecision(
                    selected_tool="open_vocab",
                    rationale=f"Targets {targets or 'open vocabulary'} require open-vocabulary grounding (Grounding DINO/OWLv2).",
                    estimated_cost_usd=profile["cost_usd"],
                    estimated_latency_ms=profile["latency_ms"],
                    expected_accuracy=profile["accuracy"],
                    fallback_tool="detector_ensemble",
                )

        # 8. High-level VQA / Complex Reasoning
        profile = self.TOOL_PROFILES["grounded_vlm"]
        return ToolRoutingDecision(
            selected_tool="grounded_vlm",
            rationale="Complex or open-ended visual question answering requiring multimodal reasoning.",
            estimated_cost_usd=profile["cost_usd"],
            estimated_latency_ms=profile["latency_ms"],
            expected_accuracy=profile["accuracy"],
            fallback_tool=None,
        )
