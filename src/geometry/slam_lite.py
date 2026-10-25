"""
SLAM-Lite Multi-View & Multi-Frame 3D Instance Fusion Engine.
Transforms local camera-coordinate detections into a persistent world-coordinate frame using
visual odometry SE(3) poses and clusters multi-view sightings into unified persistent instances.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("OmniVision.Geometry.SLAMLite")


class CameraPose(BaseModel):
    """Rigid SE(3) transformation representing camera pose in world frame: T_c2w = [R | t]."""
    rotation_matrix: List[List[float]] = Field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    frame_index: int = 0
    timestamp: float = 0.0

    def transform_camera_to_world(self, pt_cam: List[float]) -> List[float]:
        """Transform 3D point from camera coordinate frame to world coordinate frame."""
        r = np.array(self.rotation_matrix, dtype=np.float64)
        t = np.array(self.translation_vector, dtype=np.float64)
        p_c = np.array(pt_cam, dtype=np.float64)
        p_w = r @ p_c + t
        return [round(float(v), 3) for v in p_w]

    def transform_world_to_camera(self, pt_world: List[float]) -> List[float]:
        """Transform 3D point from world coordinate frame back to camera coordinates."""
        r = np.array(self.rotation_matrix, dtype=np.float64)
        t = np.array(self.translation_vector, dtype=np.float64)
        p_w = np.array(pt_world, dtype=np.float64)
        p_c = r.T @ (p_w - t)
        return [round(float(v), 3) for v in p_c]


class WorldInstance(BaseModel):
    """Canonical instance residing in unified world coordinates fused across camera views."""
    global_instance_id: str
    label: str
    world_centroid_3d: List[float] = Field(description="[X, Y, Z] in world coordinate frame (metres)")
    world_dimensions_3d: List[float] = Field(description="[width, height, depth] in metres")
    confidence: float = Field(ge=0.0, le=1.0)
    observation_count: int = Field(default=1, ge=1)
    frames_observed: List[int] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


def compute_world_distance(p1: List[float], p2: List[float]) -> float:
    """Euclidean distance in metres between two points in world coordinates."""
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2))


class SLAMVisualOdometryLite:
    """
    Multi-view visual odometry and instance fusion engine.
    Maintains persistent world instances and performs spatial-appearance deduplication.
    """

    def __init__(self, spatial_merge_threshold_meters: float = 0.65) -> None:
        self.merge_threshold = spatial_merge_threshold_meters
        self._world_instances: Dict[str, WorldInstance] = {}
        self._camera_trajectory: List[CameraPose] = []

    def register_camera_pose(self, pose: CameraPose) -> None:
        """Record camera pose in the trajectory history."""
        self._camera_trajectory.append(pose)

    def fuse_frame_observations(
        self,
        pose: CameraPose,
        detections: List[Dict[str, Any]],
    ) -> List[WorldInstance]:
        """
        Project camera-frame 3D detections into world coordinates and merge with existing instances:
        - Transforms centroid: P_world = R * P_cam + t
        - Matches against registered instances using spatial Euclidean distance and label equality
        - Updates running average of centroid and dimensions
        - Integrates observation evidence
        """
        self.register_camera_pose(pose)
        updated_instances: List[WorldInstance] = []

        for det in detections:
            label = det.get("label", "object")
            cam_pos = det.get("centroid_3d", [0.0, 0.0, 2.0])
            cam_dims = det.get("dimensions_3d_meters", [0.5, 0.5, 0.5])
            conf = det.get("confidence", 0.85)
            embedding = det.get("embedding")
            attrs = det.get("attributes", {})

            # 1. Transform to world coordinates
            world_pos = pose.transform_camera_to_world(cam_pos)

            # 2. Match with existing world instances
            best_match: Optional[WorldInstance] = None
            min_dist = float("inf")

            for existing in self._world_instances.values():
                if existing.label.lower() == label.lower():
                    dist = compute_world_distance(world_pos, existing.world_centroid_3d)
                    if dist < self.merge_threshold and dist < min_dist:
                        min_dist = dist
                        best_match = existing

            if best_match is not None:
                # 3. Fuse observation via running average
                n = best_match.observation_count
                new_n = n + 1
                fused_centroid = [
                    round((best_match.world_centroid_3d[i] * n + world_pos[i]) / new_n, 3)
                    for i in range(3)
                ]
                fused_dims = [
                    round((best_match.world_dimensions_3d[i] * n + cam_dims[i]) / new_n, 3)
                    for i in range(3)
                ]
                best_match.world_centroid_3d = fused_centroid
                best_match.world_dimensions_3d = fused_dims
                best_match.observation_count = new_n
                best_match.confidence = min(0.99, round(best_match.confidence + 0.02, 3))
                if pose.frame_index not in best_match.frames_observed:
                    best_match.frames_observed.append(pose.frame_index)
                best_match.attributes.update(attrs)
                updated_instances.append(best_match)

            else:
                # 4. Instantiate new canonical world instance
                new_id = f"world_inst_{len(self._world_instances) + 1:03d}_{uuid.uuid4().hex[:4]}"
                new_inst = WorldInstance(
                    global_instance_id=new_id,
                    label=label,
                    world_centroid_3d=world_pos,
                    world_dimensions_3d=cam_dims,
                    confidence=conf,
                    observation_count=1,
                    frames_observed=[pose.frame_index],
                    embedding=embedding,
                    attributes=attrs,
                )
                self._world_instances[new_id] = new_inst
                updated_instances.append(new_inst)

        return list(self._world_instances.values())

    def get_world_instances(self) -> List[WorldInstance]:
        """Fetch all persistent world instances."""
        return list(self._world_instances.values())

    def clear(self) -> None:
        """Clear all world instances and poses."""
        self._world_instances.clear()
        self._camera_trajectory.clear()
