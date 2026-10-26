# OmniVision Agent - System Interfaces & Architecture Contracts (v3)

## System Overview
OmniVision Agent is a research-grade, multi-agent, self-improving vision-reasoning system answering open-ended queries on images and long-form video. The system couples multi-agent program synthesis with real-time perception models (detection ensemble, open-vocabulary grounding, metric depth estimation, OCR, pose/action) and a persistent 4D (3D spatial + time) Scene Graph.

---

## Core Data Models & Schemas

### 1. Spatial & Geometric Types
```typescript
export interface BoundingBox2D {
  ymin: number; // normalized [0, 1000] or [0, 1]
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface BoundingBox3D {
  center: [number, number, number]; // [x, y, z] in meters (camera frame)
  dimensions: [number, number, number]; // [dx: width, dy: height, dz: depth] in meters
  yaw: number; // rotation in radians around Y
}

export interface DepthEstimate {
  metricDistanceM: number;
  confidence: number;
  depthRange: [number, number]; // min and max depth in ROI in meters
  planeEquation?: [number, number, number, number]; // ax + by + cz + d = 0 (RANSAC ground plane)
}
```

### 2. Scene Graph 4D (Nodes & Edges)
```typescript
export interface InstanceNode {
  id: string; // e.g. "inst_01"
  label: string; // "red_car", "worker_with_helmet"
  confidence: number; // [0, 1] calibrated
  uncertainty: number; // [0, 1]
  bbox2d: BoundingBox2D;
  bbox3d?: BoundingBox3D;
  depth: DepthEstimate;
  colorLAB?: { l: number; a: number; b: number; name: string };
  trackId?: number; // temporal tracker identifier
  timeSpan?: [number, number]; // [start_sec, end_sec] in video
  attributes: Record<string, string | number | boolean>;
  embedding?: number[]; // normalized visual representation (e.g. 512-dim)
}

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
  cameraIntrinsics: { fx: number; fy: number; cx: number; cy: number };
}
```

### 3. Agent Execution & Program Synthesis Types
```typescript
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
  id: string; // e.g. "step_1"
  tool: string; // "detect_ensemble" | "metric_depth" | "verify_clip" | "crop_ocr" | "geometry_query" | "temporal_track"
  description: string;
  inputs: Record<string, any>;
  dependencies: string[]; // step IDs that must finish first
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
  treeCandidates?: Array<{ plan: PlanDAG; predictedCost: number; predictedAccuracy: number }>;
}
```

### 4. Adversarial Verification & Verification Table
```typescript
export interface ClaimAttribution {
  claimId: string;
  claimText: string;
  observationIds: string[];
  entailmentScore: number; // [0, 1] NLI entailment
  verified: boolean;
  notes: string;
}

export interface VerificationResult {
  consensusScore: number; // 0..1 cross-model score
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
  calibratedConfidence: number; // Isotonic regression calibrated
}
```

### 5. Final Synthesized Output Schema
```typescript
export interface AgentResponse {
  query: string;
  answer: string;
  calibratedConfidence: number; // e.g. 0.94
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
    detections: Array<{ label: string; bbox: BoundingBox2D; confidence: number; trackId?: number }>;
    depthColorMapUrl?: string;
    annotatedKeyframes?: Array<{ timestamp: number; label: string; url: string }>;
  };
  uncertaintyOrFailureNote?: string;
  metaLearningTraceId: string;
  latencyTotalMs: number;
}
```

---

## Public Agent & API Endpoints

1. `POST /api/agent/run`
   - Input: `{ query: string, imageBase64?: string, videoFrames?: string[], presetChainId?: string }`
   - Output: `AgentResponse`

2. `POST /api/perception/detect`
   - Input: `{ imageBase64: string, vocabulary?: string[], confidenceThreshold?: number }`
   - Output: `{ instances: InstanceNode[] }`

3. `POST /api/perception/depth`
   - Input: `{ imageBase64: string, cameraFovDeg?: number }`
   - Output: `{ depthMapData: number[][], metricStats: Record<string, number>, pointCloudPreview: any[] }`

4. `POST /api/agent/meta-learn`
   - Input: `{ action: 'get_traces' | 'export_dpo' | 'update_router_policy' }`
   - Output: Trace logs, DPO pairs, contextual bandit router weights.

5. `GET /api/health`
   - Health check with active model status and GPU/CPU fallback flags.
