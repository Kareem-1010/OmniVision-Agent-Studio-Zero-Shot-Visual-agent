"""
Tests for OmniVision Geometry Engine and SLAM-Lite multi-view fusion.
"""

import math
import pytest
from src.geometry.relations import (
    GeometryEngine,
    SpatialRelation,
    EgocentricDirection,
    OcclusionPair,
    LineSegment,
    LineCrossingEvent,
    ccw,
    line_segments_intersect,
    compute_line_intersection,
)
from src.geometry.slam_lite import (
    SLAMVisualOdometryLite,
    CameraPose,
    WorldInstance,
    compute_world_distance,
)


def test_euclidean_and_horizontal_distance():
    p1 = [0.0, 0.0, 0.0]
    p2 = [3.0, 4.0, 0.0]
    assert pytest.approx(GeometryEngine.euclidean_distance_3d(p1, p2), 0.001) == 5.0
    assert pytest.approx(GeometryEngine.horizontal_distance(p1, p2), 0.001) == 3.0

    p3 = [1.0, 2.0, 3.0]
    p4 = [4.0, 6.0, 3.0]
    # horizontal distance is sqrt( (4-1)^2 + (3-3)^2 ) = 3.0
    assert pytest.approx(GeometryEngine.horizontal_distance(p3, p4), 0.001) == 3.0


def test_egocentric_direction():
    # Straight ahead along optical axis (Z forward)
    res_front = GeometryEngine.egocentric_direction([0.0, 0.0, 5.0])
    assert res_front.clock_position == "12 o'clock"
    assert res_front.qualitative_direction == "straight_ahead"
    assert pytest.approx(res_front.azimuth_degrees, 0.1) == 0.0
    assert pytest.approx(res_front.distance_meters, 0.1) == 5.0

    # 45 degrees to the right (X=3, Z=3)
    res_right45 = GeometryEngine.egocentric_direction([3.0, 0.0, 3.0])
    assert pytest.approx(res_right45.azimuth_degrees, 0.1) == 45.0
    assert res_right45.clock_position in ("1 o'clock", "2 o'clock")
    assert res_right45.qualitative_direction == "front_right"

    # Direct right (90 deg)
    res_right90 = GeometryEngine.egocentric_direction([5.0, 0.0, 0.0])
    assert pytest.approx(res_right90.azimuth_degrees, 0.1) == 90.0
    assert res_right90.clock_position == "3 o'clock"
    assert res_right90.qualitative_direction == "to_the_right"

    # Direct left (-90 deg)
    res_left90 = GeometryEngine.egocentric_direction([-5.0, 0.0, 0.0])
    assert pytest.approx(res_left90.azimuth_degrees, 0.1) == -90.0
    assert res_left90.clock_position == "9 o'clock"
    assert res_left90.qualitative_direction == "to_the_left"

    # Behind observer (Z negative)
    res_behind = GeometryEngine.egocentric_direction([0.0, 0.0, -4.0])
    assert res_behind.clock_position == "6 o'clock"
    assert res_behind.qualitative_direction == "behind_observer"


def test_evaluate_pairwise_relations():
    # A is to the left of B (A has smaller X, dx = B.x - A.x > 0.3)
    inst_a = {
        "id": "obj_a",
        "box_2d": [10.0, 10.0, 50.0, 50.0],
        "centroid_3d": [-1.0, 0.0, 2.0],
    }
    inst_b = {
        "id": "obj_b",
        "box_2d": [70.0, 10.0, 120.0, 50.0],
        "centroid_3d": [1.0, 0.0, 2.0],
    }

    rels = GeometryEngine.evaluate_pairwise_relations(inst_a, inst_b)
    rel_types = [r.relation_type for r in rels]
    assert "left_of" in rel_types

    # Invert to test right_of
    rels_inv = GeometryEngine.evaluate_pairwise_relations(inst_b, inst_a)
    assert "right_of" in [r.relation_type for r in rels_inv]


def test_depth_and_vertical_relations():
    # A is in front of B (A has smaller depth Z, dz = B.z - A.z > 0.4)
    inst_near = {
        "id": "near",
        "box_2d": [100.0, 100.0, 200.0, 200.0],
        "centroid_3d": [0.0, 0.0, 1.0],
    }
    inst_far = {
        "id": "far",
        "box_2d": [100.0, 100.0, 200.0, 200.0],
        "centroid_3d": [0.0, 0.0, 3.0],
    }

    rels = GeometryEngine.evaluate_pairwise_relations(inst_near, inst_far)
    assert "in_front_of" in [r.relation_type for r in rels]

    rels_far = GeometryEngine.evaluate_pairwise_relations(inst_far, inst_near)
    assert "behind" in [r.relation_type for r in rels_far]


def test_supports_and_containment_and_touching():
    # Table supports cup: cup is above table (Y_cup < Y_table, so dy < -0.15)
    # in camera coordinates Y is down
    table = {
        "id": "table_1",
        "box_2d": [50.0, 200.0, 300.0, 400.0],
        "centroid_3d": [0.0, 0.5, 2.0],
    }
    cup = {
        "id": "cup_1",
        "box_2d": [120.0, 120.0, 180.0, 195.0],
        "centroid_3d": [0.0, 0.2, 2.0],
    }

    rels = GeometryEngine.evaluate_pairwise_relations(table, cup)
    rel_types = [r.relation_type for r in rels]
    assert "supports" in rel_types
    assert "touching" in rel_types

    # Containment
    container = {
        "id": "box_container",
        "box_2d": [50.0, 50.0, 350.0, 350.0],
        "centroid_3d": [0.0, 0.0, 2.0],
    }
    content = {
        "id": "item_inside",
        "box_2d": [100.0, 100.0, 200.0, 200.0],
        "centroid_3d": [0.0, 0.0, 2.0],
    }

    rels_cont = GeometryEngine.evaluate_pairwise_relations(container, content)
    rel_types_cont = [r.relation_type for r in rels_cont]
    assert "contains" in rel_types_cont
    assert "inside" in rel_types_cont


def test_occlusion_reasoning():
    instances = [
        {
            "id": "person_front",
            "box_2d": [100.0, 100.0, 250.0, 400.0],
            "centroid_3d": [0.0, 0.0, 1.5],
        },
        {
            "id": "car_back",
            "box_2d": [150.0, 150.0, 450.0, 450.0],
            "centroid_3d": [0.5, 0.0, 3.5],
        },
    ]

    occlusions = GeometryEngine.evaluate_occlusions(instances)
    assert len(occlusions) == 1
    occ = occlusions[0]
    assert occ.occluder_id == "person_front"
    assert occ.occluded_id == "car_back"
    assert occ.occlusion_ratio > 0.0
    assert occ.depth_separation_meters == pytest.approx(2.0, 0.01)


def test_point_in_polygon():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert GeometryEngine.point_in_polygon((5.0, 5.0), square) is True
    assert GeometryEngine.point_in_polygon((15.0, 5.0), square) is False


def test_check_line_crossing():
    tripwire = LineSegment(
        name="store_entrance",
        p1=(100.0, 200.0),
        p2=(300.0, 200.0),
    )

    # Cross from (200, 150) to (200, 250)
    event = GeometryEngine.check_line_crossing(
        track_id="trk_42",
        prev_pt=(200.0, 150.0),
        curr_pt=(200.0, 250.0),
        line=tripwire,
        frame_idx=15,
    )

    assert event is not None
    assert event.track_id == "trk_42"
    assert event.line_name == "store_entrance"
    assert event.frame_index == 15
    assert event.crossing_direction in ("entering", "exiting")
    assert event.intersection_point == (200.0, 200.0)

    # No crossing (stays on one side)
    no_event = GeometryEngine.check_line_crossing(
        track_id="trk_42",
        prev_pt=(200.0, 100.0),
        curr_pt=(250.0, 150.0),
        line=tripwire,
        frame_idx=16,
    )
    assert no_event is None


def test_camera_pose_and_slam_fusion():
    pose1 = CameraPose(
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.0, 0.0, 0.0],
        frame_index=0,
    )
    # World to camera roundtrip
    p_cam = [1.0, 2.0, 3.0]
    p_world = pose1.transform_camera_to_world(p_cam)
    assert p_world == p_cam
    p_cam_back = pose1.transform_world_to_camera(p_world)
    assert p_cam_back == p_cam

    slam = SLAMVisualOdometryLite(spatial_merge_threshold_meters=0.65)

    # Frame 0 detection
    dets_f0 = [
        {
            "label": "chair",
            "centroid_3d": [1.0, 0.0, 2.0],
            "dimensions_3d_meters": [0.6, 0.8, 0.6],
            "confidence": 0.88,
        }
    ]
    instances_f0 = slam.fuse_frame_observations(pose1, dets_f0)
    assert len(instances_f0) == 1
    assert instances_f0[0].label == "chair"
    assert instances_f0[0].observation_count == 1
    assert instances_f0[0].world_centroid_3d == [1.0, 0.0, 2.0]

    # Frame 1: camera has translated slightly: t=[0.2, 0.0, 0.0]
    # Local detection in camera frame: [0.85, 0.0, 2.05]
    # In world frame: [0.85 + 0.2, 0.0, 2.05] = [1.05, 0.0, 2.05] (very close to [1.0, 0.0, 2.0])
    pose2 = CameraPose(
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.2, 0.0, 0.0],
        frame_index=1,
    )
    dets_f1 = [
        {
            "label": "chair",
            "centroid_3d": [0.85, 0.0, 2.05],
            "dimensions_3d_meters": [0.6, 0.8, 0.6],
            "confidence": 0.90,
        }
    ]
    instances_f1 = slam.fuse_frame_observations(pose2, dets_f1)
    # Should fuse into single persistent instance
    assert len(instances_f1) == 1
    fused_chair = instances_f1[0]
    assert fused_chair.observation_count == 2
    assert fused_chair.frames_observed == [0, 1]
    # Running average centroid: (1.0 + 1.05)/2 = 1.025
    assert pytest.approx(fused_chair.world_centroid_3d[0], 0.01) == 1.025

    # New distinct object far away
    dets_f2 = [
        {
            "label": "lamp",
            "centroid_3d": [-3.0, 0.5, 4.0],
            "dimensions_3d_meters": [0.3, 1.2, 0.3],
            "confidence": 0.92,
        }
    ]
    instances_f2 = slam.fuse_frame_observations(pose2, dets_f2)
    assert len(instances_f2) == 2
