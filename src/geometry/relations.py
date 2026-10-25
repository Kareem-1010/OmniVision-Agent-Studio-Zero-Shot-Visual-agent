"""
2D and 3D Spatial Geometry Engine.
Computes metric distances, qualitative spatial relations (supports, inside, touching, left/right/above/behind),
egocentric clock directions, ray-based occlusion reasoning, ROI polygon containment, and tripwire line crossing.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("OmniVision.Geometry.Relations")


class SpatialRelation(BaseModel):
    """Pairwise spatial or functional relation between two instances."""
    relation_type: str  # e.g., 'left_of', 'right_of', 'above', 'behind', 'supports', 'inside', 'touching', 'nearest'
    source_id: str
    target_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    metric_distance_meters: float = Field(ge=0.0)
    is_symmetric: bool = False
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EgocentricDirection(BaseModel):
    """Observer-centric directional orientation in the horizontal camera plane."""
    azimuth_degrees: float = Field(ge=-180.0, le=180.0)
    clock_position: str  # e.g., '12 o'clock', '2 o'clock', '9 o'clock'
    qualitative_direction: str  # e.g., 'straight_ahead', 'front_right', 'left', 'rear'
    distance_meters: float = Field(ge=0.0)


class OcclusionPair(BaseModel):
    """Occlusion relationship between foreground and background instances along visual rays."""
    occluder_id: str
    occluded_id: str
    occlusion_ratio: float = Field(ge=0.0, le=1.0)
    depth_separation_meters: float = Field(ge=0.0)


class LineSegment(BaseModel):
    """2D virtual boundary or tripwire defined in normalized coordinates."""
    name: str = "line"
    p1: Tuple[float, float]
    p2: Tuple[float, float]

    @model_validator(mode="before")
    @classmethod
    def normalize_points(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ("p1", "p2"):
                v = data.get(k)
                if hasattr(v, "x") and hasattr(v, "y"):
                    data[k] = (float(v.x), float(v.y))
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    data[k] = (float(v[0]), float(v[1]))
        return data


class LineCrossingEvent(BaseModel):
    """Discrete boundary crossing event detected for a moving instance."""
    model_config = ConfigDict(extra="allow")

    track_id: str
    line_name: str
    frame_index: int
    crossing_direction: str  # 'entering', 'exiting', or 'bidirectional'
    intersection_point: Tuple[float, float]
    direction: Optional[str] = None


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Counter-clockwise test for orientation of three 2D points."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    """Check if line segment AB intersects line segment CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def compute_line_intersection(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Compute (x, y) coordinates of intersection between segments AB and CD."""
    denom = (B[0] - A[0]) * (D[1] - C[1]) - (B[1] - A[1]) * (D[0] - C[0])
    if abs(denom) < 1e-9:
        return None  # Parallel lines
    t = ((C[0] - A[0]) * (D[1] - C[1]) - (C[1] - A[1]) * (D[0] - C[0])) / denom
    ix = A[0] + t * (B[0] - A[0])
    iy = A[1] + t * (B[1] - A[1])
    return (round(ix, 4), round(iy, 4))


class GeometryEngine:
    """
    Mathematical and computational geometry reasoning engine.
    Executes metric spatial computations, qualitative logic, and line crossings.
    """

    @staticmethod
    def euclidean_distance_3d(p1: List[float], p2: List[float]) -> float:
        """Euclidean 3D distance between two points in metres."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2))

    @staticmethod
    def horizontal_distance(p1: List[float], p2: List[float]) -> float:
        """Ground plane distance in X-Z coordinates in metres."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2))

    @classmethod
    def egocentric_direction(cls, centroid_3d: List[float]) -> EgocentricDirection:
        """
        Compute observer-centric azimuth angle and clock direction from camera frame centroid [X, Y, Z].
        Camera convention: X right, Y down, Z forward.
        """
        x, _, z = centroid_3d
        dist = float(np.sqrt(x**2 + z**2))

        # Azimuth in degrees relative to camera optical axis (straight ahead = 0 deg)
        azimuth_rad = math.atan2(x, z)
        azimuth_deg = math.degrees(azimuth_rad)

        # Map azimuth (-180 to 180) to clock positions (1 to 12)
        # 0 deg is 12 o'clock, 90 deg is 3 o'clock, -90 deg is 9 o'clock
        clock_angle = (azimuth_deg + 360.0) % 360.0
        clock_hour = int(round(clock_angle / 30.0)) % 12
        if clock_hour == 0:
            clock_hour = 12
        clock_str = f"{clock_hour} o'clock"

        # Qualitative semantic direction
        if abs(azimuth_deg) < 15.0:
            qual = "straight_ahead"
        elif 15.0 <= azimuth_deg < 60.0:
            qual = "front_right"
        elif 60.0 <= azimuth_deg < 120.0:
            qual = "to_the_right"
        elif -60.0 < azimuth_deg <= -15.0:
            qual = "front_left"
        elif -120.0 < azimuth_deg <= -60.0:
            qual = "to_the_left"
        else:
            qual = "behind_observer"

        return EgocentricDirection(
            azimuth_degrees=round(azimuth_deg, 2),
            clock_position=clock_str,
            qualitative_direction=qual,
            distance_meters=round(dist, 3),
        )

    @classmethod
    def evaluate_pairwise_relations(
        cls,
        inst_a: Dict[str, Any],
        inst_b: Dict[str, Any],
    ) -> List[SpatialRelation]:
        """
        Deduce directional and physical relations between instance A and instance B:
        - Directional: left_of, right_of, above, below, behind, in_front_of
        - Topological & Physical: inside, supports, touching, nearest
        """
        relations: List[SpatialRelation] = []
        id_a = inst_a["id"]
        id_b = inst_b["id"]

        box_a = inst_a.get("box_2d", [0.0, 0.0, 0.0, 0.0])
        box_b = inst_b.get("box_2d", [0.0, 0.0, 0.0, 0.0])

        pos_a = inst_a.get("centroid_3d", [0.0, 0.0, 2.0])
        pos_b = inst_b.get("centroid_3d", [0.0, 0.0, 2.0])

        dist_3d = cls.euclidean_distance_3d(pos_a, pos_b)
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]  # Y is down in camera coords
        dz = pos_b[2] - pos_a[2]  # Z is depth away from camera

        # 1. Lateral: left_of / right_of
        if dx > 0.3:
            relations.append(
                SpatialRelation(
                    relation_type="left_of",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.95,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )
        elif dx < -0.3:
            relations.append(
                SpatialRelation(
                    relation_type="right_of",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.95,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )

        # 2. Depth: behind / in_front_of
        if dz > 0.4:
            relations.append(
                SpatialRelation(
                    relation_type="in_front_of",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.92,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )
        elif dz < -0.4:
            relations.append(
                SpatialRelation(
                    relation_type="behind",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.92,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )

        # 3. Vertical: above / below
        if dy > 0.3:  # B has larger Y (lower visually) -> A is above B
            relations.append(
                SpatialRelation(
                    relation_type="above",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.90,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )
        elif dy < -0.3:
            relations.append(
                SpatialRelation(
                    relation_type="below",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.90,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )

        # 4. Physical Support: A supports B if B is directly above A and in contact
        # In camera coordinates, B above A means Y_b < Y_a (dy < -0.2) and horizontal proximity < 0.5m
        horiz_dist = cls.horizontal_distance(pos_a, pos_b)
        if dy < -0.15 and horiz_dist < 0.6 and dist_3d < 1.2:
            relations.append(
                SpatialRelation(
                    relation_type="supports",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.88,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )

        # 5. Containment: A contains B if 2D/3D bounds of B are inside A
        if (
            box_a[0] <= box_b[0]
            and box_a[1] <= box_b[1]
            and box_a[2] >= box_b[2]
            and box_a[3] >= box_b[3]
        ):
            relations.append(
                SpatialRelation(
                    relation_type="contains",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.93,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )
            relations.append(
                SpatialRelation(
                    relation_type="inside",
                    source_id=id_b,
                    target_id=id_a,
                    confidence=0.93,
                    metric_distance_meters=round(dist_3d, 3),
                )
            )

        # 6. Touching / Proximity
        if dist_3d < 0.35:
            relations.append(
                SpatialRelation(
                    relation_type="touching",
                    source_id=id_a,
                    target_id=id_b,
                    confidence=0.85,
                    metric_distance_meters=round(dist_3d, 3),
                    is_symmetric=True,
                )
            )

        return relations

    @classmethod
    def evaluate_occlusions(cls, instances: List[Dict[str, Any]]) -> List[OcclusionPair]:
        """
        Evaluate line-of-sight occlusion between instances based on 2D overlap and depth ordering.
        If object A has smaller depth Z than object B and their 2D boxes intersect, A occludes B.
        """
        occlusions: List[OcclusionPair] = []
        n = len(instances)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                inst_a = instances[i]
                inst_b = instances[j]

                box_a = inst_a["box_2d"]
                box_b = inst_b["box_2d"]
                z_a = inst_a["centroid_3d"][2]
                z_b = inst_b["centroid_3d"][2]

                # A must be closer to camera than B (z_a < z_b)
                if z_a < z_b - 0.2:
                    # Compute 2D intersection area over area of background object B
                    x_left = max(box_a[0], box_b[0])
                    y_top = max(box_a[1], box_b[1])
                    x_right = min(box_a[2], box_b[2])
                    y_bottom = min(box_a[3], box_b[3])

                    if x_right > x_left and y_bottom > y_top:
                        inter_area = (x_right - x_left) * (y_bottom - y_top)
                        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                        ratio = inter_area / area_b if area_b > 0.0 else 0.0
                        if ratio > 0.10:
                            occlusions.append(
                                OcclusionPair(
                                    occluder_id=inst_a["id"],
                                    occluded_id=inst_b["id"],
                                    occlusion_ratio=round(float(ratio), 3),
                                    depth_separation_meters=round(float(z_b - z_a), 3),
                                )
                            )

        return occlusions

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon_pts: List[Tuple[float, float]]) -> bool:
        """Determine if 2D point (x, y) resides inside an arbitrary polygon."""
        if len(polygon_pts) < 3:
            return False
        contour = np.array(polygon_pts, dtype=np.float32)
        # cv2.pointPolygonTest returns positive if inside, 0 on edge, negative if outside
        res = cv2.pointPolygonTest(contour, point, measureDist=False)
        return res >= 0

    @classmethod
    def check_line_crossing(
        cls,
        track_id: str,
        prev_pt: Tuple[float, float],
        curr_pt: Tuple[float, float],
        line: LineSegment,
        frame_idx: int = 0,
    ) -> Optional[LineCrossingEvent]:
        """
        Check if movement vector from prev_pt to curr_pt crossed the virtual line.
        Computes crossing direction using the cross-product orientation.
        """
        if not line_segments_intersect(prev_pt, curr_pt, line.p1, line.p2):
            return None

        # Cross product vector from line.p1 -> line.p2 with vector to curr_pt
        lx = line.p2[0] - line.p1[0]
        ly = line.p2[1] - line.p1[1]
        px = curr_pt[0] - line.p1[0]
        py = curr_pt[1] - line.p1[1]

        cross_val = lx * py - ly * px
        direction = "entering" if cross_val > 0 else "exiting"
        ix_pt = compute_line_intersection(prev_pt, curr_pt, line.p1, line.p2) or curr_pt

        return LineCrossingEvent(
            track_id=track_id,
            line_name=line.name,
            frame_index=frame_idx,
            crossing_direction=direction,
            intersection_point=ix_pt,
        )
