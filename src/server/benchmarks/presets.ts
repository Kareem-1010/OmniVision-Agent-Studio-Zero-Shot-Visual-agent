/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DemoPreset } from '../../types/index.ts';

export const REQUIRED_BENCHMARK_CHAINS: DemoPreset[] = [
  {
    id: 'chain_count_cars',
    title: '1. Count the Red Cars',
    category: 'count',
    query: 'Count the red cars',
    description: 'Ensemble detection (YOLO11 + RT-DETR) -> SAM2 mask crops -> LAB color + SigLIP2 filter -> NMS de-duplication -> Poisson 95% Confidence Interval counting.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Parking lot / Multi-car traffic',
  },
  {
    id: 'chain_safety_hazard',
    title: '2. Is There a Safety Hazard?',
    category: 'safety',
    query: 'Is there a safety hazard?',
    description: 'Open-vocab hazard detection + Depth Anything V2 metric distance + RTMPose body keypoints -> 3D proximity risk scoring matrix.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Industrial Warehouse Logistics',
  },
  {
    id: 'chain_ocr_sign',
    title: '3. Read the Sign Next to the Door',
    category: 'ocr',
    query: 'Read the sign next to the door',
    description: 'Grounded detection -> spatial anchor filtering -> perspective-rectified crop -> PaddleOCR + VLM oracle cross-check.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1541888946425-d0fbb186f5f7?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Laboratory Access Corridor',
  },
  {
    id: 'chain_temporal_dwell',
    title: '4. People Entered & Dwell Times',
    category: 'temporal',
    query: 'How many unique people entered and how long did each stay?',
    description: 'SAM2 video predictor with memory bank + BoT-SORT re-ID -> entry/exit line cordon events -> dwell time telemetry.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1528698827591-e19ccd7bc23d?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Retail Store Lobby Video Surveillance',
  },
  {
    id: 'chain_closest_object',
    title: '5. Closest Object to Camera',
    category: 'spatial',
    query: 'Which object is closest to the camera and how far is it?',
    description: 'Metric Depth Anything V2 -> camera intrinsics back-projection -> 3D centroid ranking with ground-plane RANSAC.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1508873696983-2df57046475a?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Urban Sidewalk Street Scene',
  },
  {
    id: 'chain_multihop_umbrella',
    title: '6. Multi-Hop Spatial Query',
    category: 'multihop',
    query: 'What is the person holding the umbrella standing next to?',
    description: '4D Scene Graph traversal -> node attribute filtering (holding=umbrella) -> adjacent spatial edge lookup in 3D.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Rainy City Walkway',
  },
  {
    id: 'chain_video_truck',
    title: '7. Long-Video Event Retrieval',
    category: 'video',
    query: 'When did the red truck arrive, and who got out?',
    description: 'Episodic video memory vector search -> hierarchical clip indexing -> temporal event grounding with exact timestamps.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1586191582056-a6021f15be3c?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Loading Dock Video Stream',
  },
  {
    id: 'chain_counterfactual_clearance',
    title: '8. Counterfactual Metric Clearance',
    category: 'counterfactual',
    query: 'Could the cart fit through the gap between the shelves?',
    description: '3D oriented bounding boxes (OBBs) -> aisle gap width surface calculation vs cart dimensions with 10cm safety threshold.',
    sampleImageUrl: 'https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=1000&q=80',
    syntheticSceneType: 'Retail Hardware Aisle',
  },
];
