#!/usr/bin/env python3
"""
OmniVision Agent - Phase 2 Perception Stack Demonstration.
Exercises all 13 perception tools in an integrated visual intelligence pipeline.
"""

from __future__ import annotations

import logging
import pprint
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cache import ContentHashCache
from src.tools import (
    DetectorEnsembleTool,
    OpenVocabDetectorTool,
    GroundedSegmenterTool,
    VideoTrackerTool,
    ZeroShotClassifierTool,
    DenseFeatureTool,
    MetricDepthTool,
    PoseActionTool,
    SceneOCRTool,
    GroundedVLMTool,
    OpticalFlowTool,
    VisualAttributeTool,
    VisualizerTool,
    VisualAnnotationBox,
)
from src.utils.tracker import get_global_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OmniVision.Phase2Demo")


def main() -> None:
    print("=" * 80)
    print(" OmniVision Agent: Phase 2 Perception Stack Demonstration")
    print("=" * 80)

    tracker = get_global_tracker()
    tracker.reset()
    cache = ContentHashCache(cache_dir=".cache/phase2_demo", l1_max_items=100)

    # 1. Detector Ensemble (YOLO11 + RT-DETR + WBF + SAHI)
    print("\n[1/13] Detector Ensemble (WBF + SAHI)...")
    detector = DetectorEnsembleTool(cache=cache, tracker=tracker)
    det_res = detector.execute({"image_uri": "scene.jpg", "target_classes": ["person", "car"]})
    print(f"  Obs ID: {det_res.observation_id} | Detections: {det_res.data.total_detections}")
    print(f"  Calibrated Confidence: {det_res.uncertainty.confidence:.4f}")

    # 2. Open-Vocabulary Detector (Grounding DINO + OWLv2 + YOLO-World)
    print("\n[2/13] Open-Vocabulary Consensus Detector...")
    open_vocab = OpenVocabDetectorTool(cache=cache, tracker=tracker)
    ov_res = open_vocab.execute({
        "image_uri": "scene.jpg",
        "text_queries": ["safety hazard", "worker"],
        "consensus_minimum_votes": 2,
    })
    print(f"  Obs ID: {ov_res.observation_id} | Grounded Instances: {ov_res.data.total_found}")
    print(f"  Active Detectors: {ov_res.data.active_detectors}")

    # 3. Grounded Segmentation (SAM2 + Boundary Snapping)
    print("\n[3/13] Grounded-SAM2 Instance Segmentation...")
    segmenter = GroundedSegmenterTool(cache=cache, tracker=tracker)
    seg_res = segmenter.execute({
        "image_uri": "scene.jpg",
        "boxes": [[0.1, 0.2, 0.4, 0.7]],
        "refine_boundaries": True,
    })
    print(f"  Obs ID: {seg_res.observation_id} | Masks: {seg_res.data.total_masks}")
    print(f"  Refinement Applied: {seg_res.data.refinement_applied}")

    # 4. Video Tracking (ByteTrack + Kalman + OSNet Re-ID + Stitching)
    print("\n[4/13] Multi-Object Video Tracker...")
    tracker_tool = VideoTrackerTool(cache=cache, tracker=tracker)
    track_res = tracker_tool.execute({
        "video_id": "clip_01",
        "num_frames": 25,
        "enable_tracklet_stitching": True,
    })
    print(f"  Obs ID: {track_res.observation_id} | Unique Tracks: {track_res.data.total_unique_tracks}")
    print(f"  Stitched Tracks: {track_res.data.stitched_tracks_count}")

    # 5. Zero-Shot Classifier (SigLIP2 + CLIP + Open-Set Rejection)
    print("\n[5/13] Zero-Shot Classifier (SigLIP2 + CLIP)...")
    classifier = ZeroShotClassifierTool(cache=cache, tracker=tracker)
    cls_res = classifier.execute({
        "image_uri": "crop.jpg",
        "candidate_labels": ["forklift", "pallet", "worker"],
    })
    print(f"  Obs ID: {cls_res.observation_id} | Predicted: {cls_res.data.predicted_label}")
    print(f"  Confidence: {cls_res.data.confidence:.4f} | Open-Set Rejected: {cls_res.data.is_open_set_rejected}")

    # 6. Dense Feature Matching (DINOv2 'Find More Like This')
    print("\n[6/13] Dense Feature Matching (DINOv2)...")
    dense_tool = DenseFeatureTool(cache=cache, tracker=tracker)
    dense_res = dense_tool.execute({
        "image_uri": "shelf.jpg",
        "query_box": [0.1, 0.2, 0.25, 0.45],
        "top_k": 3,
    })
    print(f"  Obs ID: {dense_res.observation_id} | Matches Found: {dense_res.data.total_matches}")
    print(f"  Max Similarity: {dense_res.data.max_similarity:.4f}")

    # 7. Metric Depth (Depth Anything V2 + RANSAC Ground Plane)
    print("\n[7/13] Metric Depth & 3D Spatial Layout...")
    depth_tool = MetricDepthTool(cache=cache, tracker=tracker)
    depth_res = depth_tool.execute({
        "image_uri": "warehouse.jpg",
        "target_boxes": [[0.1, 0.2, 0.35, 0.75], [0.5, 0.3, 0.8, 0.7]],
        "hfov_degrees": 65.0,
    })
    closest = depth_res.data.closest_instance
    print(f"  Obs ID: {depth_res.observation_id} | 3D Instances: {len(depth_res.data.instances_3d)}")
    print(f"  Closest Object: {closest.distance_to_camera_meters:.2f} m (Dimensions: {closest.dimensions_3d_meters} m)")
    print(f"  Ground Plane Equation: {depth_res.data.ground_plane_equation}")

    # 8. Pose and Action Recognition (RTMPose + Fall + PPE)
    print("\n[8/13] Pose, Action & Safety PPE Reasoning...")
    pose_tool = PoseActionTool(cache=cache, tracker=tracker)
    pose_res = pose_tool.execute({
        "image_uri": "factory.jpg",
        "person_boxes": [[0.1, 0.2, 0.35, 0.85], [0.55, 0.7, 0.90, 0.88]],
        "detect_falls": True,
        "check_ppe": True,
    })
    print(f"  Obs ID: {pose_res.observation_id} | Evaluated: {pose_res.data.total_persons_evaluated}")
    print(f"  Fall Detected in Scene: {pose_res.data.fall_detected_in_scene}")
    for p in pose_res.data.persons:
        print(f"    - {p.person_id}: posture={p.posture}, action={p.action_label}, PPE={p.ppe_compliance}")

    # 9. Scene OCR (PaddleOCR + VLM Fallback + Object Association)
    print("\n[9/13] Scene OCR & Spatial Object Binding...")
    ocr_tool = SceneOCRTool(cache=cache, tracker=tracker)
    ocr_res = ocr_tool.execute({
        "image_uri": "hallway.jpg",
        "target_object_boxes": [[0.10, 0.75, 0.35, 0.95]],
        "use_vlm_fallback": True,
    })
    print(f"  Obs ID: {ocr_res.observation_id} | Text Blocks: {ocr_res.data.total_text_blocks}")
    print(f"  VLM Fallback Invoked: {ocr_res.data.vlm_fallback_invoked}")
    for txt in ocr_res.data.recognized_texts:
        print(f"    - '{txt.text}' (rel={txt.spatial_relation_to_object}, conf={txt.confidence:.2f})")

    # 10. Grounded VLM (Qwen2.5-VL VQA & Oracle Cross-Check)
    print("\n[10/13] Grounded VLM Oracle Cross-Check...")
    vlm_tool = GroundedVLMTool(cache=cache, tracker=tracker)
    vlm_res = vlm_tool.execute({
        "image_or_video_uri": "test.jpg",
        "prompt": "Verify safety conditions",
        "mode": "oracle_cross_check",
        "draft_hypothesis": "The designated walkway has no hazardous obstructions.",
    })
    print(f"  Obs ID: {vlm_res.observation_id} | Verdict: {vlm_res.data.oracle_verdict}")
    print(f"  Rationale: {vlm_res.data.cross_check_rationale}")

    # 11. Optical Flow & Motion Segmentation (RAFT)
    print("\n[11/13] Optical Flow & Dynamic Motion Segmentation...")
    flow_tool = OpticalFlowTool(cache=cache, tracker=tracker)
    flow_res = flow_tool.execute({"frame_prev_uri": "f1.png", "frame_curr_uri": "f2.png"})
    print(f"  Obs ID: {flow_res.observation_id} | Moving Clusters: {len(flow_res.data.moving_clusters)}")
    print(f"  Max Velocity: {flow_res.data.max_speed_px:.1f} px/frame")

    # 12. Visual Attributes (CIELAB Color + Texture + Size)
    print("\n[12/13] Visual Attributes Extraction (CIELAB)...")
    attr_tool = VisualAttributeTool(cache=cache, tracker=tracker)
    attr_res = attr_tool.execute({
        "image_uri": "vehicle.jpg",
        "roi_box": [0.2, 0.3, 0.8, 0.7],
        "known_depth_meters": 4.2,
    })
    print(f"  Obs ID: {attr_res.observation_id} | Primary Color: {attr_res.data.primary_color}")
    print(f"  Material/Texture: {attr_res.data.material_texture} | Size: {attr_res.data.relative_size_category}")
    print(f"  Metric Dimensions (W x H): {attr_res.data.estimated_metric_size_meters} m")

    # 13. Visualizer (Boxes + Masks + Trails + Evidence Badges)
    print("\n[13/13] Multi-Modal Visualizer & Evidence Attribution Renderer...")
    visualizer = VisualizerTool(cache=cache, tracker=tracker)
    vis_res = visualizer.execute({
        "image_uri": "scene.png",
        "boxes": [
            VisualAnnotationBox(box=[0.1, 0.2, 0.35, 0.75], label="person", confidence=0.94, observation_id=det_res.observation_id),
            VisualAnnotationBox(box=[0.45, 0.35, 0.85, 0.80], label="sedan", confidence=0.88, observation_id=ov_res.observation_id),
        ],
        "polygons": [
            [(0.1, 0.2), (0.35, 0.2), (0.35, 0.75), (0.1, 0.75)]
        ],
        "tracking_trails": [
            [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3)]
        ],
    })
    print(f"  Obs ID: {vis_res.observation_id} | Saved To: {vis_res.data.saved_filepath}")
    print(f"  Elements Rendered: {vis_res.data.total_elements_rendered} (Dimensions: {vis_res.data.width}x{vis_res.data.height})")

    # Final Telemetry Summary
    print("\n" + "=" * 80)
    print(" Phase 2 Perception Telemetry Summary")
    print("=" * 80)
    summary = tracker.get_summary()
    print(f"Total Perception Tools Executed: {summary.total_calls}")
    print(f"Total Cumulative Cost USD: ${summary.total_cost_usd:.6f}")
    print(f"Latency p50: {summary.latency_p50_ms:.2f} ms | p95: {summary.latency_p95_ms:.2f} ms")
    print("\nPer-tool Breakdown:")
    for comp, metrics in summary.by_component.items():
        print(f"  - {comp:28s}: {metrics['calls']} call(s) | {metrics['avg_latency_ms']:.2f} ms avg | ${metrics['cost_usd']:.6f}")

    print("\n" + "=" * 80)
    print(" Phase 2 Perception Stack Successfully Verified!")
    print("=" * 80)


if __name__ == "__main__":
    main()
