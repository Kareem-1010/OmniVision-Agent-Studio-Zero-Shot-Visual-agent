"""
Phase 3 Geometry and 4D Scene Graph End-to-End Demonstration Script.
Demonstrates:
1. Metric 3D distances & egocentric clock directions.
2. Pairwise directional and physical relation extraction (supports, contains, left_of).
3. Ray-based occlusion reasoning.
4. Virtual tripwire line-crossing event detection.
5. SLAM-Lite multi-view SE(3) camera pose projection and running-average instance fusion.
6. Persistent 4D Scene Graph construction with temporal slicing.
7. Declarative Query DSL execution (e.g. FIND ?x, ?y WHERE ...).
8. Graph-diff event detection across timestamps (appearance, disappearance, attribute change, relation change).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geometry.relations import (
    GeometryEngine,
    LineSegment,
)
from src.geometry.slam_lite import (
    SLAMVisualOdometryLite,
    CameraPose,
)
from src.memory.scene_graph import (
    Persistent4DSceneGraph,
    SceneGraphNode,
    SceneGraphEdge,
    compute_graph_diff,
)


def run_phase3_demo():
    print("=" * 75)
    print(" OmniVision Agent - Phase 3 Geometry & 4D Scene Graph Demonstration")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. 3D Metric & Egocentric Direction Reasoning
    # ---------------------------------------------------------
    print("\n[1] 3D Metric & Egocentric Observer Direction:")
    p_cam = [1.5, 0.2, 3.0]  # X right, Y down, Z forward
    ego_dir = GeometryEngine.egocentric_direction(p_cam)
    print(f"  Target 3D position (camera frame): {p_cam}")
    print(f"  -> Azimuth angle: {ego_dir.azimuth_degrees} deg")
    print(f"  -> Clock direction: {ego_dir.clock_position}")
    print(f"  -> Qualitative direction: {ego_dir.qualitative_direction}")
    print(f"  -> Radial distance: {ego_dir.distance_meters} m")

    # ---------------------------------------------------------
    # 2. Pairwise Spatial & Physical Relation Extraction
    # ---------------------------------------------------------
    print("\n[2] Pairwise Spatial & Physical Relation Reasoning:")
    table_inst = {
        "id": "table_01",
        "box_2d": [100.0, 300.0, 500.0, 600.0],
        "centroid_3d": [0.0, 0.6, 2.5],
    }
    mug_inst = {
        "id": "mug_01",
        "box_2d": [250.0, 200.0, 320.0, 290.0],
        "centroid_3d": [0.0, 0.25, 2.5],
    }
    relations = GeometryEngine.evaluate_pairwise_relations(table_inst, mug_inst)
    for r in relations:
        print(f"  Relation detected: '{r.source_id}' --[{r.relation_type}]--> '{r.target_id}' (conf: {r.confidence}, dist: {r.metric_distance_meters}m)")

    # ---------------------------------------------------------
    # 3. Ray-Based Occlusion Reasoning
    # ---------------------------------------------------------
    print("\n[3] Visual Ray Occlusion Reasoning:")
    person_front = {
        "id": "person_front",
        "box_2d": [120.0, 100.0, 280.0, 450.0],
        "centroid_3d": [0.0, 0.0, 1.8],
    }
    car_behind = {
        "id": "car_behind",
        "box_2d": [160.0, 150.0, 500.0, 500.0],
        "centroid_3d": [0.3, 0.0, 4.2],
    }
    occlusions = GeometryEngine.evaluate_occlusions([person_front, car_behind])
    for occ in occlusions:
        print(f"  Occlusion: '{occ.occluder_id}' occludes '{occ.occluded_id}' (ratio: {occ.occlusion_ratio*100:.1f}%, depth delta: {occ.depth_separation_meters}m)")

    # ---------------------------------------------------------
    # 4. Virtual Tripwire Boundary Crossing Detection
    # ---------------------------------------------------------
    print("\n[4] Virtual Tripwire Boundary Crossing:")
    tripwire = LineSegment(name="security_perimeter", p1=(100.0, 300.0), p2=(500.0, 300.0))
    crossing_event = GeometryEngine.check_line_crossing(
        track_id="pedestrian_7",
        prev_pt=(300.0, 250.0),
        curr_pt=(300.0, 350.0),
        line=tripwire,
        frame_idx=42,
    )
    if crossing_event:
        print(f"  Tripwire Event: track '{crossing_event.track_id}' crossed '{crossing_event.line_name}' at frame {crossing_event.frame_index}")
        print(f"  -> Direction: {crossing_event.crossing_direction}, intersection point: {crossing_event.intersection_point}")

    # ---------------------------------------------------------
    # 5. SLAM-Lite Multi-View 3D Instance Fusion
    # ---------------------------------------------------------
    print("\n[5] SLAM-Lite Multi-View Fusion:")
    slam = SLAMVisualOdometryLite(spatial_merge_threshold_meters=0.65)

    # Frame 0: Camera at origin
    pose_f0 = CameraPose(
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.0, 0.0, 0.0],
        frame_index=0,
    )
    f0_dets = [
        {"label": "chair", "centroid_3d": [0.95, 0.0, 2.1], "dimensions_3d_meters": [0.6, 0.8, 0.6], "confidence": 0.86}
    ]
    instances_f0 = slam.fuse_frame_observations(pose_f0, f0_dets)
    chair_id = instances_f0[0].global_instance_id
    print(f"  Frame 0: Registered new world instance '{chair_id}' [label: {instances_f0[0].label}] at world pos {instances_f0[0].world_centroid_3d}")

    # Frame 1: Camera translated by +0.3m along X axis
    pose_f1 = CameraPose(
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.3, 0.0, 0.0],
        frame_index=1,
    )
    # Local detection in camera frame is now ~[0.7, 0.0, 2.05] (since camera moved +0.3 right)
    # In world frame: [0.7 + 0.3, 0.0, 2.05] = [1.0, 0.0, 2.05]
    f1_dets = [
        {"label": "chair", "centroid_3d": [0.7, 0.0, 2.05], "dimensions_3d_meters": [0.6, 0.8, 0.6], "confidence": 0.89}
    ]
    instances_f1 = slam.fuse_frame_observations(pose_f1, f1_dets)
    fused_chair = [i for i in instances_f1 if i.global_instance_id == chair_id][0]
    print(f"  Frame 1: Fused sighting into persistent instance '{chair_id}'")
    print(f"  -> Observation count: {fused_chair.observation_count}, updated world centroid: {fused_chair.world_centroid_3d}")

    # ---------------------------------------------------------
    # 6. Persistent 4D Scene Graph & Query DSL
    # ---------------------------------------------------------
    print("\n[6] Persistent 4D Scene Graph & Declarative Query DSL:")
    sg = Persistent4DSceneGraph()

    worker = SceneGraphNode(
        node_id="worker_01",
        label="person",
        track_id=101,
        time_span=(0, 100),
        centroid_3d=[0.5, 0.0, 2.2],
        attributes={"vest": "high_vis_yellow", "hardhat": "white", "posture": "standing"},
    )
    forklift = SceneGraphNode(
        node_id="forklift_01",
        label="forklift",
        track_id=202,
        time_span=(0, 100),
        centroid_3d=[2.5, 0.0, 4.0],
        attributes={"status": "operational", "color": "orange"},
    )
    box = SceneGraphNode(
        node_id="pallet_01",
        label="wooden_pallet",
        track_id=303,
        time_span=(20, 100),
        centroid_3d=[2.5, 0.4, 3.8],
        attributes={"material": "wood"},
    )

    sg.add_node(worker)
    sg.add_node(forklift)
    sg.add_node(box)

    sg.add_edge(SceneGraphEdge(source_id="forklift_01", target_id="pallet_01", relation_type="supports", time_span=(20, 100)))

    # Execute DSL queries
    q1 = "FIND ?x WHERE ?x.label == 'person' AND ?x.posture == 'standing'"
    results_q1 = sg.execute_query(q1)
    print(f"  Query: '{q1}'")
    print(f"  -> Matches ({len(results_q1)}): {[r['x']['node_id'] for r in results_q1]}")

    q2 = "FIND ?x, ?y WHERE ?x.label == 'forklift' AND ?y.label == 'wooden_pallet' AND ?x.supports == ?y"
    results_q2 = sg.execute_query(q2)
    print(f"  Query: '{q2}'")
    print(f"  -> Relational matches ({len(results_q2)}): {[(r['x']['node_id'], r['y']['node_id']) for r in results_q2]}")

    # ---------------------------------------------------------
    # 7. Graph-Diff Event Detection Across Timestamps
    # ---------------------------------------------------------
    print("\n[7] Graph-Diff Temporal Event Detection:")
    sg_t1 = sg.get_active_subgraph_at_frame(10)  # pallet not active at frame 10
    sg_t2 = sg.get_active_subgraph_at_frame(30)  # pallet is active and supported at frame 30

    diff_events = compute_graph_diff(sg_t1, sg_t2, frame_index=30)
    for ev in diff_events:
        obj_str = f" -> '{ev.object_id}'" if ev.object_id else ""
        print(f"  Event: [{ev.event_type}] subject '{ev.subject_id}'{obj_str} at frame {ev.frame_index} (details: {ev.details})")

    print("\n" + "=" * 75)
    print(" [OK] Phase 3 Demonstration Completed Successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_phase3_demo()
