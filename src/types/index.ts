/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface BoundingBox2D {
  ymin: number; // 0..1000 scale
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface BoundingBox3D {
  center: [number, number, number]; // [x, y, z] in meters
  dimensions: [number, number, number]; // [width, height, depth] in meters
  yaw: number; // radians
}

export interface DepthEstimate {
  metricDistanceM: number;
  confidence: number;
  depthRange: [number, number];
  planeEquation?: [number, number, number, number];
}

export interface InstanceNode {
  id: string;
  label: string;
  confidence: number;
  uncertainty: number;
  bbox2d: BoundingBox2D;
  bbox3d?: BoundingBox3D;
  depth: DepthEstimate;
  colorLAB?: {
    l: number;
    a: number;
    b: number;
    name: string;
  };
  polygon2d?: Array<[number, number]>; // Normalized [y, x] or [x, y] polygon vertices (0..1000)
  clipScore?: number; // CLIP / SigLIP open-vocabulary text-image similarity (0..1)
  yoloClass?: string; // YOLO-World open-vocabulary class name
  yoloConfidence?: number; // YOLO detector confidence
  colorPurity?: {
    primary: string;
    purityScore: number;
    rgbSample?: [number, number, number];
  };
  trackId?: number;
  timeSpan?: [number, number];
  attributes: Record<string, string | number | boolean>;
  embedding?: number[];
}

export type CVEngineMode = 'omnivision' | 'yolo_world' | 'sam2' | 'classic_cv';

export type RelationType =
  | 'left_of'
  | 'right_of'
  | 'above'
  | 'below'
  | 'in_front_of'
  | 'behind'
  | 'inside'
  | 'supports'
  | 'touching'
  | 'near'
  | 'occludes'
  | 'approaching';

export interface RelationEdge {
  id: string;
  sourceId: string;
  targetId: string;
  relationType: RelationType;
  distanceMeters: number;
  confidence: number;
  timeRange?: [number, number];
}

export interface SceneGraph4D {
  timestamp: number;
  frameIndex: number;
  instances: Record<string, InstanceNode>;
  edges: RelationEdge[];
  groundPlane?: [number, number, number, number];
  cameraIntrinsics: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
  };
}

export type QueryIntent =
  | 'count'
  | 'existence'
  | 'attribute'
  | 'relation'
  | 'referring_expression'
  | 'ocr'
  | 'temporal'
  | 'safety_hazard'
  | 'counterfactual'
  | 'multi_hop'
  | 'comparative';

export interface PlanStep {
  id: string;
  tool: string;
  description: string;
  inputs: Record<string, any>;
  dependencies: string[];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  result?: any;
  observationId?: string;
  executionTimeMs?: number;
}

export interface PlanDAG {
  query: string;
  intent: QueryIntent;
  targetEntities: string[];
  steps: PlanStep[];
  synthesizedPythonProgram: string;
  treeCandidates?: Array<{
    planId: string;
    description: string;
    predictedCost: number;
    predictedAccuracy: number;
    selected: boolean;
  }>;
}

export interface ClaimAttribution {
  claimId: string;
  claimText: string;
  observationIds: string[];
  entailmentScore: number;
  verified: boolean;
  notes: string;
}

export interface VerificationResult {
  consensusScore: number;
  contradictionsFound: string[];
  falsificationAttempt: {
    hypothesis: string;
    refuted: boolean;
    evidence: string;
  };
  countingInterval?: {
    count: number;
    ci95: [number, number];
    rawDetections: number;
    deduplicated: number;
  };
  claimAttributions: ClaimAttribution[];
  calibratedConfidence: number;
}

export interface AgentResponse {
  query: string;
  answer: string;
  calibratedConfidence: number;
  intent: QueryIntent;
  plan: PlanDAG;
  sceneGraph: SceneGraph4D;
  verification: VerificationResult;
  evidenceTrail: Array<{
    step: string;
    tool: string;
    observationId: string;
    summary: string;
    metric?: string;
  }>;
  visualOverlays: {
    detections: Array<{
      id: string;
      label: string;
      bbox: BoundingBox2D;
      confidence: number;
      trackId?: number;
      colorHex?: string;
      distanceMeters?: number;
    }>;
    depthMapSummary?: string;
    opticalFlowMagnitude?: number;
    annotatedKeyframes?: Array<{
      timestamp: number;
      label: string;
      url: string;
    }>;
  };
  uncertaintyOrFailureNote?: string;
  metaLearningTraceId: string;
  latencyTotalMs: number;
  cvMetrics?: {
    cvEngine: string;
    nmsSuppressedCount: number;
    meanClipScore: number;
    sam2PolygonsGenerated: number;
    activeIouThreshold: number;
  };
}

export interface AgentTraceLog {
  id: string;
  timestamp: string;
  query: string;
  intent: QueryIntent;
  planStepsCount: number;
  toolsUsed: string[];
  latencyMs: number;
  tokenCount: number;
  costDollars: number;
  verifierScore: number;
  calibratedConfidence: number;
  humanFeedback?: 'accepted' | 'rejected' | 'corrected';
  dpoPreferenceScore?: number; // 0..1
}

export interface DemoPreset {
  id: string;
  title: string;
  category: 'count' | 'safety' | 'ocr' | 'temporal' | 'spatial' | 'multihop' | 'video' | 'counterfactual';
  query: string;
  description: string;
  sampleImageUrl: string;
  syntheticSceneType: string;
}
