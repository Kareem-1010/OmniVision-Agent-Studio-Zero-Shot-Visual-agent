"""
Comprehensive Unit Test Suite for Phase 2 Perception Stack.
Tests all 13 perception tools, schema validations, uncertainty quantification, and observation IDs.
"""

import numpy as np
import pytest

from src.models.cache import ContentHashCache
from src.tools.detector_ensemble import (
    DetectorEnsembleTool,
    weighted_boxes_fusion,
    compute_iou,
)
from src.tools.open_vocab import OpenVocabDetectorTool, NormalizedBox, box_iou
from src.tools.segmentation import GroundedSegmenterTool, refine_mask_boundaries
from src.tools.video_tracker import VideoTrackerTool, SimpleKalmanBoxTracker
from src.tools.classifier import ZeroShotClassifierTool, softmax_with_temperature
from src.tools.dense_features import DenseFeatureTool
from src.tools.depth import MetricDepthTool, fit_ground_plane_ransac
from src.tools.pose_action import PoseActionTool
from src.tools.ocr import SceneOCRTool, compute_box_center_distance
from src.tools.vlm import GroundedVLMTool, parse_vlm_coordinate_tags
from src.tools.optical_flow import OpticalFlowTool
from src.tools.attributes import VisualAttributeTool, match_lab_to_color_name
from src.tools.visualizer import VisualizerTool, VisualAnnotationBox
from src.utils.tracker import MetricsTracker


# ---------------------------------------------------------------------------
# 1. Detector Ensemble Tests
# ---------------------------------------------------------------------------
def test_iou_and_wbf():
    box1 = [0.1, 0.1, 0.5, 0.5]
    box2 = [0.12, 0.11, 0.48, 0.52]
    iou = compute_iou(box1, box2)
    assert iou > 0.80

    fused_b, fused_s, fused_l = weighted_boxes_fusion(
        boxes_list=[box1, box2],
        scores_list=[0.9, 0.85],
        labels_list=["person", "person"],
        weights=[1.0, 1.0],
        iou_threshold=0.5,
    )
    assert len(fused_b) == 1
    assert fused_l[0] == "person"
    assert fused_s[0] >= 0.90


def test_detector_ensemble_tool(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = DetectorEnsembleTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({"image_uri": "test_scene.png", "target_classes": ["person", "car"]})
    assert res.success is True
    assert res.observation_id.startswith("obs_")
    assert res.data is not None
    assert res.data.total_detections >= 2
    assert res.data.detections[0].calibrated_score > 0.0
    assert res.uncertainty.confidence > 0.0


# ---------------------------------------------------------------------------
# 2. Open-Vocabulary Tool Tests
# ---------------------------------------------------------------------------
def test_open_vocab_consensus(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = OpenVocabDetectorTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "safety_scene.jpg",
        "text_queries": ["safety hazard", "worker"],
        "consensus_minimum_votes": 2,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.total_found >= 1
    # Check that consensus voting required multiple model votes
    for inst in res.data.instances:
        assert len(inst.models_agreed) >= 2
        assert inst.consensus_score >= 0.60


# ---------------------------------------------------------------------------
# 3. Grounded Segmentation Tests
# ---------------------------------------------------------------------------
def test_grounded_segmenter(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = GroundedSegmenterTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "scene.png",
        "boxes": [[0.1, 0.2, 0.4, 0.7]],
        "refine_boundaries": True,
    })
    assert res.success is True
    assert res.data is not None
    assert len(res.data.masks) == 1
    mask = res.data.masks[0]
    assert len(mask.polygon) >= 3
    assert mask.area_fraction > 0.0
    assert mask.stability_score >= 0.90


# ---------------------------------------------------------------------------
# 4. Video Tracker Tests
# ---------------------------------------------------------------------------
def test_kalman_and_video_tracker(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    # Test Kalman Box Tracker
    kalman = SimpleKalmanBoxTracker([0.1, 0.2, 0.3, 0.5])
    pred = kalman.predict()
    assert pred[0] > 0.1

    # Test Video Tracker Tool
    tool = VideoTrackerTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "video_id": "surveillance_clip_01",
        "num_frames": 20,
        "enable_tracklet_stitching": True,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.total_unique_tracks == 2
    assert res.data.tracks[0].average_confidence > 0.80
    assert len(res.data.tracks[0].trajectory) > 0


# ---------------------------------------------------------------------------
# 5. Zero-Shot Classifier Tests
# ---------------------------------------------------------------------------
def test_classifier_temperature_and_open_set(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax_with_temperature(logits, temperature=0.1)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > 0.99  # Sharp distribution under low temp

    tool = ZeroShotClassifierTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "crop.png",
        "candidate_labels": ["forklift", "pallet", "worker"],
        "open_set_threshold": 0.20,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.predicted_label in ["forklift", "pallet", "worker", "unknown"]
    assert res.data.confidence >= 0.0


# ---------------------------------------------------------------------------
# 6. Dense Feature Matching Tests
# ---------------------------------------------------------------------------
def test_dense_feature_tool(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = DenseFeatureTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "shelf.jpg",
        "query_box": [0.1, 0.2, 0.25, 0.45],
        "similarity_threshold": 0.65,
        "top_k": 3,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.total_matches >= 1
    assert res.data.max_similarity > 0.90
    assert res.data.query_feature_dim == 768


# ---------------------------------------------------------------------------
# 7. Metric Depth Tests
# ---------------------------------------------------------------------------
def test_metric_depth_and_ransac(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    # Test RANSAC plane fitting
    synthetic_ground = np.array([
        [0.0, 1.5, 2.0],
        [1.0, 1.5, 3.0],
        [-1.0, 1.5, 4.0],
        [0.5, 1.5, 2.5],
    ])
    plane, inliers = fit_ground_plane_ransac(synthetic_ground)
    assert len(plane) == 4

    tool = MetricDepthTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "warehouse_cam.jpg",
        "target_boxes": [[0.1, 0.2, 0.35, 0.75], [0.5, 0.3, 0.8, 0.7]],
        "hfov_degrees": 65.0,
    })
    assert res.success is True
    assert res.data is not None
    assert len(res.data.instances_3d) == 2
    assert res.data.closest_instance is not None
    assert res.data.closest_instance.distance_to_camera_meters > 0.0
    assert len(res.data.instances_3d[0].dimensions_3d_meters) == 3


# ---------------------------------------------------------------------------
# 8. Pose and Action Tests
# ---------------------------------------------------------------------------
def test_pose_action_and_fall_detection(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = PoseActionTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "factory_floor.jpg",
        "person_boxes": [
            [0.1, 0.2, 0.35, 0.85],  # Standing person (narrow & tall)
            [0.5, 0.7, 0.90, 0.88],  # Fallen person (wide & flat aspect ratio)
        ],
        "detect_falls": True,
        "check_ppe": True,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.total_persons_evaluated == 2
    assert res.data.fall_detected_in_scene is True
    # Verify Person 2 is classified as fallen
    assert res.data.persons[1].is_fallen is True
    assert res.data.persons[1].posture == "lying_horizontal"
    # Verify Person 1 has PPE compliance
    assert res.data.persons[0].ppe_compliance["hard_hat"] is True


# ---------------------------------------------------------------------------
# 9. Scene OCR Tests
# ---------------------------------------------------------------------------
def test_scene_ocr_and_object_binding(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = SceneOCRTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "hallway.jpg",
        "target_object_boxes": [[0.10, 0.75, 0.35, 0.95]],  # Extinguisher cabinet box
        "use_vlm_fallback": True,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.total_text_blocks >= 1
    assert res.data.vlm_fallback_invoked is True
    # Find associated extinguisher text
    found_binding = any(t.associated_object_index == 0 for t in res.data.recognized_texts)
    assert found_binding is True


# ---------------------------------------------------------------------------
# 10. Grounded VLM Tests
# ---------------------------------------------------------------------------
def test_grounded_vlm_vqa_and_oracle(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = GroundedVLMTool(cache=isolated_cache, tracker=isolated_tracker)

    # Test grounded VQA mode
    res_vqa = tool.execute({
        "image_or_video_uri": "test.jpg",
        "prompt": "Where is the red truck?",
        "mode": "grounded_vqa",
    })
    assert res_vqa.success is True
    assert len(res_vqa.data.grounded_boxes) >= 1

    # Test independent oracle verification mode
    res_oracle = tool.execute({
        "image_or_video_uri": "test.jpg",
        "prompt": "Verify claims",
        "mode": "oracle_cross_check",
        "draft_hypothesis": "The pedestrian pathway is clear with no hazard.",
    })
    assert res_oracle.success is True
    assert res_oracle.data.oracle_verdict == "CONFIRMED"
    assert res_oracle.data.cross_check_rationale is not None


# ---------------------------------------------------------------------------
# 11. Optical Flow Tests
# ---------------------------------------------------------------------------
def test_optical_flow_tool(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = OpticalFlowTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "frame_prev_uri": "frame_001.png",
        "frame_curr_uri": "frame_002.png",
    })
    assert res.success is True
    assert res.data is not None
    assert len(res.data.moving_clusters) >= 1
    assert res.data.max_speed_px > 0.0
    for cluster in res.data.moving_clusters:
        assert cluster.speed_px_per_frame > 0.0
        assert 0.0 <= cluster.motion_angle_degrees <= 360.0


# ---------------------------------------------------------------------------
# 12. Visual Attributes Tests
# ---------------------------------------------------------------------------
def test_visual_attributes(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    assert match_lab_to_color_name((53.0, 80.0, 67.0)) == "red"
    assert match_lab_to_color_name((100.0, 0.0, 0.0)) == "white"

    tool = VisualAttributeTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "sedan.png",
        "roi_box": [0.2, 0.3, 0.8, 0.7],
        "known_depth_meters": 3.5,
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.primary_color == "red"
    assert res.data.relative_size_category in ["small", "medium", "large"]
    assert res.data.estimated_metric_size_meters is not None
    assert res.data.estimated_metric_size_meters[0] > 0.0


# ---------------------------------------------------------------------------
# 13. Visualizer Tool Tests
# ---------------------------------------------------------------------------
def test_visualizer_tool(isolated_cache: ContentHashCache, isolated_tracker: MetricsTracker):
    tool = VisualizerTool(cache=isolated_cache, tracker=isolated_tracker)
    res = tool.execute({
        "image_uri": "mock.png",
        "boxes": [
            VisualAnnotationBox(box=[0.1, 0.2, 0.4, 0.7], label="person", confidence=0.95, observation_id="obs_test123"),
            VisualAnnotationBox(box=[0.5, 0.3, 0.8, 0.6], label="car", confidence=0.88),
        ],
        "polygons": [
            [(0.1, 0.2), (0.4, 0.2), (0.4, 0.7), (0.1, 0.7)]
        ],
        "tracking_trails": [
            [(0.1, 0.2), (0.12, 0.22), (0.15, 0.25)]
        ],
    })
    assert res.success is True
    assert res.data is not None
    assert res.data.total_elements_rendered == 4
    assert res.data.width == 640
    assert res.data.height == 480
