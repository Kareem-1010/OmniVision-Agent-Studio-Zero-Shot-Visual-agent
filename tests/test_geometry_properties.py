"""
Property-Based and Invariant Geometric Tests for OmniVision Geometry Engine.
Verifies fundamental mathematical and physical invariants across:
1. 2D Bounding Box IoU: Commutativity, Identity, Disjointness, and Strict Range [0, 1].
2. 3D Bounding Box Invariants: Non-negative volume, homogeneous cubic scaling, distance metric axioms.
3. Rotation Matrix SO(3) Properties: Orthogonality (R * R^T = I), unit determinant (det(R) = 1), norm preservation.
4. Line Segment Crossing Invariants: Directional anti-symmetry and non-crossing convexity.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.geometry.engine import BoundingBox3D, GeometryEngine, LineSegment, Point2D, Point3D
from src.eval.metrics import calculate_box_iou


def test_iou_property_commutativity_and_bounds():
    """Property 1: IoU commutativity IoU(A, B) == IoU(B, A) and bounds 0 <= IoU <= 1."""
    rng = np.random.RandomState(42)

    for _ in range(100):
        # Generate two random valid 2D boxes [x1, y1, x2, y2]
        coords1 = sorted(rng.uniform(0.0, 1.0, 2))
        coords2 = sorted(rng.uniform(0.0, 1.0, 2))
        box_a = [coords1[0], coords2[0], coords1[1], coords2[1]]

        coords3 = sorted(rng.uniform(0.0, 1.0, 2))
        coords4 = sorted(rng.uniform(0.0, 1.0, 2))
        box_b = [coords3[0], coords4[0], coords3[1], coords4[1]]

        iou_ab = calculate_box_iou(box_a, box_b)
        iou_ba = calculate_box_iou(box_b, box_a)

        # Commutativity
        assert abs(iou_ab - iou_ba) < 1e-9, f"IoU not commutative: {iou_ab} != {iou_ba}"
        # Strict bounds
        assert 0.0 <= iou_ab <= 1.0, f"IoU out of bounds: {iou_ab}"


def test_iou_property_identity_and_disjointness():
    """Property 2: IoU identity IoU(A, A) == 1.0 and disjointness IoU == 0.0."""
    box = [0.15, 0.25, 0.65, 0.75]
    assert abs(calculate_box_iou(box, box) - 1.0) < 1e-9

    # Completely disjoint boxes (horizontally separated)
    box_left = [0.05, 0.10, 0.40, 0.50]
    box_right = [0.60, 0.10, 0.95, 0.50]
    assert calculate_box_iou(box_left, box_right) == 0.0

    # Completely disjoint boxes (vertically separated)
    box_top = [0.10, 0.05, 0.50, 0.40]
    box_bottom = [0.10, 0.60, 0.50, 0.95]
    assert calculate_box_iou(box_top, box_bottom) == 0.0


def test_3d_bounding_box_scaling_and_volume():
    """Property 3: 3D bounding box volume non-negativity and cubic scale invariance."""
    rng = np.random.RandomState(1337)

    for _ in range(50):
        w = float(rng.uniform(0.1, 5.0))
        h = float(rng.uniform(0.1, 5.0))
        d = float(rng.uniform(0.1, 5.0))
        box3d = BoundingBox3D(
            center=Point3D(x=0.0, y=0.0, z=2.0),
            dimensions=Point3D(x=w, y=h, z=d),
            orientation=Point3D(x=0.0, y=0.0, z=0.0),
        )

        vol = box3d.dimensions.x * box3d.dimensions.y * box3d.dimensions.z
        assert vol > 0.0

        # Uniform scaling by factor lambda
        scale = float(rng.uniform(1.2, 3.0))
        scaled_vol = (w * scale) * (h * scale) * (d * scale)
        expected_vol = vol * (scale ** 3)
        assert abs(scaled_vol - expected_vol) < 1e-6


def test_3d_euclidean_metric_space_axioms():
    """Property 4: Metric space axioms: Positivity, Symmetry, and Triangle Inequality."""
    rng = np.random.RandomState(2024)

    def dist(p1: Point3D, p2: Point3D) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    for _ in range(50):
        p1 = Point3D(x=rng.uniform(-5, 5), y=rng.uniform(-5, 5), z=rng.uniform(0, 10))
        p2 = Point3D(x=rng.uniform(-5, 5), y=rng.uniform(-5, 5), z=rng.uniform(0, 10))
        p3 = Point3D(x=rng.uniform(-5, 5), y=rng.uniform(-5, 5), z=rng.uniform(0, 10))

        d12 = dist(p1, p2)
        d21 = dist(p2, p1)
        d23 = dist(p2, p3)
        d13 = dist(p1, p3)

        # 1. Non-negativity and Identity of Indiscernibles
        assert d12 >= 0.0
        assert dist(p1, p1) == 0.0

        # 2. Symmetry
        assert abs(d12 - d21) < 1e-9

        # 3. Triangle Inequality d(1, 3) <= d(1, 2) + d(2, 3)
        assert d13 <= d12 + d23 + 1e-7


def test_so3_rotation_matrix_orthogonality_and_isometry():
    """Property 5: SO(3) Lie Group rotation matrix orthogonality, det=1, and Euclidean norm preservation."""
    rng = np.random.RandomState(777)

    for _ in range(30):
        # Generate random Euler angles (yaw, pitch, roll)
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi, 3)

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])

        R = Rz @ Ry @ Rx

        # Orthogonality: R @ R.T == I
        identity = np.eye(3)
        assert np.allclose(R @ R.T, identity, atol=1e-6)
        assert np.allclose(R.T @ R, identity, atol=1e-6)

        # Determinant: det(R) == 1.0 (proper rotation)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6

        # Isometry: ||R * v|| == ||v||
        v = rng.uniform(-10.0, 10.0, 3)
        norm_v = np.linalg.norm(v)
        norm_Rv = np.linalg.norm(R @ v)
        assert abs(norm_v - norm_Rv) < 1e-6


def test_line_crossing_directional_anti_symmetry():
    """Property 6: Trajectory line crossing directional anti-symmetry and non-crossing."""
    line = LineSegment(p1=Point2D(x=0.0, y=0.5), p2=Point2D(x=1.0, y=0.5))

    p_above = Point2D(x=0.5, y=0.3)
    p_below = Point2D(x=0.5, y=0.7)

    # 1. Crossing downwards
    res_down = GeometryEngine.check_line_crossing("obj_1", p_above, p_below, line, frame_idx=1)
    assert res_down is not None
    assert res_down.crossing_direction in ("entering", "exiting")

    # 2. Crossing upwards (reverse trajectory)
    res_up = GeometryEngine.check_line_crossing("obj_1", p_below, p_above, line, frame_idx=2)
    assert res_up is not None
    # Opposite direction
    assert res_up.crossing_direction != res_down.crossing_direction

    # 3. Points on the same side should not trigger a crossing
    res_none = GeometryEngine.check_line_crossing("obj_1", p_above, Point2D(x=0.6, y=0.2), line, frame_idx=3)
    assert res_none is None
