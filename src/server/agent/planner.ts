/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { PlanDAG, PlanStep, QueryIntent } from '../../types/index.ts';
import { QueryAnalysisOutput } from './queryAnalyst.ts';

export async function synthesizePlan(
  query: string,
  analysis: QueryAnalysisOutput
): Promise<PlanDAG> {
  const steps: PlanStep[] = [];
  let pythonCode = '';

  switch (analysis.intent) {
    case 'count':
      steps.push({
        id: 'step_1',
        tool: 'detector_ensemble',
        description: `Run YOLO11 + RT-DETR ensemble with SAHI tiling for target "${analysis.targetEntities.join(', ')}"`,
        inputs: { classes: analysis.targetEntities, conf_threshold: 0.35, sahi: true },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_2',
        tool: 'mask_segmentation',
        description: 'Compute SAM2 masks and crop instances for colorLAB & SigLIP2 filtering',
        inputs: { source_step: 'step_1', refine_crf: true },
        dependencies: ['step_1'],
        status: 'pending',
      });
      steps.push({
        id: 'step_3',
        tool: 'clip_color_verifier',
        description: 'Verify attribute filter (e.g. red color / model match) using SigLIP2 + LAB k-means',
        inputs: { source_step: 'step_2', target_attribute: 'red' },
        dependencies: ['step_2'],
        status: 'pending',
      });
      steps.push({
        id: 'step_4',
        tool: 'counting_reconciliation',
        description: 'Class-agnostic NMS de-duplication, Poisson count reconciliation with 95% Confidence Interval',
        inputs: { source_step: 'step_3', iou_thresh: 0.5 },
        dependencies: ['step_3'],
        status: 'pending',
      });
      steps.push({
        id: 'step_5',
        tool: 'adversarial_verifier',
        description: 'Verify counting consensus against raw image and cross-model oracle',
        inputs: { source_step: 'step_4' },
        dependencies: ['step_4'],
        status: 'pending',
      });

      pythonCode = `# Auto-synthesized ViperGPT Program: Counting with calibration
def execute_vision_program(image):
    detections = detector_ensemble(image, classes=${JSON.stringify(analysis.targetEntities)}, sahi_tiling=True)
    masks = sam2_segment(image, boxes=detections.boxes, crf_refine=True)
    filtered = [m for m in masks if siglip2_match(m.crop, "red ${analysis.targetEntities[0] || 'car'}") > 0.65]
    deduped = class_agnostic_nms(filtered, iou_thresh=0.5)
    ci95 = compute_poisson_ci(len(deduped))
    verify_result = adversarial_verifier(image, count=len(deduped), ci=ci95)
    return {"count": len(deduped), "ci95": ci95, "instances": deduped}
`;
      break;

    case 'safety_hazard':
      steps.push({
        id: 'step_1',
        tool: 'detector_ensemble',
        description: 'Detect workers, vehicles, safety gear, and industrial equipment',
        inputs: { classes: ['worker', 'forklift', 'machinery', 'ladder', 'box', 'spill'], conf_threshold: 0.4 },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_2',
        tool: 'open_vocab_grounding',
        description: 'Grounding DINO 1.5 + OWLv2 search for hazard cues: PPE missing, liquid spill, blocked exit',
        inputs: { text_queries: ['hard hat', 'high-vis vest', 'liquid spill on floor', 'blocked emergency exit'] },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_3',
        tool: 'metric_depth',
        description: 'Depth Anything V2 metric depth estimation & ground-plane RANSAC back-projection',
        inputs: { return_3d_pointcloud: true },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_4',
        tool: 'pose_action_reasoner',
        description: 'RTMPose skeleton extraction & worker safety posture inspection (bending, fall risk)',
        inputs: { source_step: 'step_1' },
        dependencies: ['step_1'],
        status: 'pending',
      });
      steps.push({
        id: 'step_5',
        tool: 'spatial_hazard_matrix',
        description: '3D proximity analysis between workers and active machinery, compute risk severity score',
        inputs: { depth_step: 'step_3', detect_step: 'step_1', openvocab_step: 'step_2', pose_step: 'step_4' },
        dependencies: ['step_1', 'step_2', 'step_3', 'step_4'],
        status: 'pending',
      });

      pythonCode = `# Auto-synthesized ViperGPT Program: 3D Spatial Hazard Analysis
def execute_vision_program(image):
    workers = detector_ensemble(image, classes=["worker", "forklift", "equipment"])
    hazards = open_vocab_ground(image, queries=["spill", "unsecured load", "missing ppe"])
    depth_cloud = depth_anything_v2(image, metric=True)
    poses = rtmpose_extract(image, boxes=workers)
    risk_report = compute_3d_hazard_matrix(workers, hazards, depth_cloud, poses)
    return risk_report
`;
      break;

    case 'ocr':
      steps.push({
        id: 'step_1',
        tool: 'open_vocab_grounding',
        description: `Ground target spatial anchor: "${analysis.spatialQualifiers.join(' ') || 'sign near door'}"`,
        inputs: { query: 'sign, notice, billboard, door' },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_2',
        tool: 'active_perception_crop',
        description: 'Active perception loop: crop high-resolution sign bounding box and apply super-resolution',
        inputs: { source_step: 'step_1', padding: 0.15 },
        dependencies: ['step_1'],
        status: 'pending',
      });
      steps.push({
        id: 'step_3',
        tool: 'paddleocr_engine',
        description: 'Dual-engine OCR (PaddleOCR text detector + recognizer with orientation fix)',
        inputs: { source_step: 'step_2' },
        dependencies: ['step_2'],
        status: 'pending',
      });
      steps.push({
        id: 'step_4',
        tool: 'vlm_cross_check',
        description: 'VLM reading oracle cross-check for small or blurred characters',
        inputs: { source_step: 'step_3' },
        dependencies: ['step_3'],
        status: 'pending',
      });

      pythonCode = `# Auto-synthesized ViperGPT Program: Grounded High-Resolution OCR
def execute_vision_program(image):
    sign_boxes = grounding_dino(image, "sign next to door")
    cropped = active_perception_zoom(image, sign_boxes[0], super_res=True)
    ocr_tokens = paddleocr_read(cropped)
    verified_text = vlm_oracle_read(cropped, fallback=ocr_tokens)
    return {"text": verified_text, "confidence": 0.96}
`;
      break;

    case 'temporal':
      steps.push({
        id: 'step_1',
        tool: 'video_sam2_tracking',
        description: 'SAM2 Video Predictor with memory bank & BoT-SORT re-ID across video frames',
        inputs: { targets: analysis.targetEntities },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_2',
        tool: 'roi_line_crossing',
        description: 'Calibrate entry/exit threshold lines and calculate dwell times per tracklet',
        inputs: { source_step: 'step_1' },
        dependencies: ['step_1'],
        status: 'pending',
      });
      steps.push({
        id: 'step_3',
        tool: 'temporal_grounding',
        description: 'Episodic memory timestamp grounding for target events (e.g. arrival/departure)',
        inputs: { source_step: 'step_2', event_query: query },
        dependencies: ['step_2'],
        status: 'pending',
      });

      pythonCode = `# Auto-synthesized ViperGPT Program: Long-Video SAM2 Tracking & Dwell Times
def execute_vision_program(video):
    tracks = sam2_video_tracker(video, memory_bank_size=128)
    events = compute_dwell_and_line_crossing(tracks, entry_line=((100, 400), (700, 400)))
    timestamps = temporal_grounding(video, event_query="${query}")
    return {"tracks": tracks, "events": events, "timestamps": timestamps}
`;
      break;

    case 'counterfactual':
      steps.push({
        id: 'step_1',
        tool: 'detector_ensemble',
        description: 'Detect object and bounding obstacles (e.g. cart and shelves)',
        inputs: { classes: ['cart', 'shelf', 'furniture', 'doorway'] },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_2',
        tool: 'metric_depth',
        description: 'Depth Anything V2 metric point cloud generation with camera intrinsics calibration',
        inputs: { metric_meters: true },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_3',
        tool: 'geometry_engine_3d',
        description: 'Compute 3D oriented bounding boxes (OBBs), calculate aisle gap width vs cart width in meters',
        inputs: { detect_step: 'step_1', depth_step: 'step_2' },
        dependencies: ['step_1', 'step_2'],
        status: 'pending',
      });
      steps.push({
        id: 'step_4',
        tool: 'counterfactual_collision_check',
        description: 'Simulate virtual path clearance with 10cm safety margin',
        inputs: { source_step: 'step_3' },
        dependencies: ['step_3'],
        status: 'pending',
      });

      pythonCode = `# Auto-synthesized ViperGPT Program: Counterfactual 3D Metric Clearance
def execute_vision_program(image):
    objects = detector_ensemble(image, classes=["cart", "shelf"])
    depth_map = depth_anything_v2(image, metric=True)
    pcd = backproject_to_3d(depth_map, intrinsics=calibrate_camera(image))
    cart_obb = compute_oriented_bbox_3d(pcd, objects["cart"])
    gap_width = compute_surface_distance_3d(pcd, objects["shelf_left"], objects["shelf_right"])
    fits = gap_width.meters > (cart_obb.width_meters + 0.10)
    return {"fits": fits, "gap_m": gap_width.meters, "cart_width_m": cart_obb.width_meters}
`;
      break;

    default: // multi_hop, relation, etc.
      steps.push({
        id: 'step_1',
        tool: 'detector_ensemble',
        description: `Detect all candidate foreground objects and relations for "${query}"`,
        inputs: { classes: analysis.targetEntities },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_2',
        tool: 'metric_depth',
        description: 'Compute per-pixel depth and 3D centroid coordinates',
        inputs: { metric_meters: true },
        dependencies: [],
        status: 'pending',
      });
      steps.push({
        id: 'step_3',
        tool: 'scene_graph_builder',
        description: 'Construct 4D scene graph with 3D spatial relations (near, left_of, behind, touching)',
        inputs: { source_detect: 'step_1', source_depth: 'step_2' },
        dependencies: ['step_1', 'step_2'],
        status: 'pending',
      });
      steps.push({
        id: 'step_4',
        tool: 'multi_hop_graph_query',
        description: `Execute graph query DSL: find instance matching "${query}"`,
        inputs: { source_step: 'step_3', dsl_query: query },
        dependencies: ['step_3'],
        status: 'pending',
      });
      steps.push({
        id: 'step_5',
        tool: 'adversarial_verifier',
        description: 'Cross-check multi-hop reference chain against VLM oracle',
        inputs: { source_step: 'step_4' },
        dependencies: ['step_4'],
        status: 'pending',
      });

      pythonCode = `# Auto-synthesized ViperGPT Program: 3D Scene Graph Multi-Hop Query
def execute_vision_program(image):
    instances = detector_ensemble(image)
    depth = depth_anything_v2(image, metric=True)
    graph = build_4d_scene_graph(instances, depth)
    result = graph.query_dsl("${query}")
    return verify_with_vlm(image, result)
`;
      break;
  }

  // Tree candidates for test-time compute scaling
  const treeCandidates = [
    {
      planId: 'cand_dag_primary',
      description: 'Standard multi-specialist DAG with adversarial verification (Recommended)',
      predictedCost: 0.0018,
      predictedAccuracy: 0.94,
      selected: true,
    },
    {
      planId: 'cand_dag_fast',
      description: 'Low-latency direct VLM perception without depth RANSAC (Fast path)',
      predictedCost: 0.0006,
      predictedAccuracy: 0.81,
      selected: false,
    },
    {
      planId: 'cand_dag_exhaustive',
      description: 'Exhaustive active perception with 3x multi-crop super-resolution and triple consensus',
      predictedCost: 0.0042,
      predictedAccuracy: 0.97,
      selected: false,
    },
  ];

  return {
    query,
    intent: analysis.intent,
    targetEntities: analysis.targetEntities,
    steps,
    synthesizedPythonProgram: pythonCode,
    treeCandidates,
  };
}
