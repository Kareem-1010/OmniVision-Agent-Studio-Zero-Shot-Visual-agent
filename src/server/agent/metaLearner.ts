/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgentResponse, AgentTraceLog } from '../../types/index.ts';

// In-memory trace store for self-improvement
const traceMemory: AgentTraceLog[] = [
  {
    id: 'trace_seed_01',
    timestamp: new Date(Date.now() - 3600000 * 4).toISOString(),
    query: 'Count the red cars',
    intent: 'count',
    planStepsCount: 5,
    toolsUsed: ['detector_ensemble', 'mask_segmentation', 'clip_color_verifier', 'counting_reconciliation', 'adversarial_verifier'],
    latencyMs: 840,
    tokenCount: 420,
    costDollars: 0.0016,
    verifierScore: 0.97,
    calibratedConfidence: 0.96,
    humanFeedback: 'accepted',
    dpoPreferenceScore: 0.98,
  },
  {
    id: 'trace_seed_02',
    timestamp: new Date(Date.now() - 3600000 * 2).toISOString(),
    query: 'Is there a safety hazard?',
    intent: 'safety_hazard',
    planStepsCount: 5,
    toolsUsed: ['detector_ensemble', 'open_vocab_grounding', 'metric_depth', 'pose_action_reasoner', 'spatial_hazard_matrix'],
    latencyMs: 1120,
    tokenCount: 680,
    costDollars: 0.0028,
    verifierScore: 0.95,
    calibratedConfidence: 0.95,
    humanFeedback: 'accepted',
    dpoPreferenceScore: 0.96,
  },
  {
    id: 'trace_seed_03',
    timestamp: new Date(Date.now() - 1800000).toISOString(),
    query: 'Could the cart fit through the gap between the shelves?',
    intent: 'counterfactual',
    planStepsCount: 4,
    toolsUsed: ['detector_ensemble', 'metric_depth', 'geometry_engine_3d', 'counterfactual_collision_check'],
    latencyMs: 910,
    tokenCount: 510,
    costDollars: 0.0019,
    verifierScore: 0.96,
    calibratedConfidence: 0.95,
    humanFeedback: 'accepted',
    dpoPreferenceScore: 0.97,
  },
];

export function logAgentExecutionTrace(response: AgentResponse): AgentTraceLog {
  const trace: AgentTraceLog = {
    id: response.metaLearningTraceId,
    timestamp: new Date().toISOString(),
    query: response.query,
    intent: response.intent,
    planStepsCount: response.plan.steps.length,
    toolsUsed: response.plan.steps.map((s) => s.tool),
    latencyMs: response.latencyTotalMs,
    tokenCount: 450 + response.plan.steps.length * 60,
    costDollars: 0.0012 + response.plan.steps.length * 0.0003,
    verifierScore: response.verification.consensusScore,
    calibratedConfidence: response.calibratedConfidence,
    humanFeedback: 'accepted',
    dpoPreferenceScore: response.calibratedConfidence * 0.98,
  };

  traceMemory.unshift(trace);
  if (traceMemory.length > 100) {
    traceMemory.pop();
  }
  return trace;
}

export function getTraces(): AgentTraceLog[] {
  return [...traceMemory];
}

export function generateDPODataset(): Array<{ prompt: string; chosen: string; rejected: string; margin: number }> {
  return traceMemory.map((t) => ({
    prompt: `[INST] Query: "${t.query}" - Synthesize an executable vision program with verification steps. [/INST]`,
    chosen: `Plan: [${t.toolsUsed.join(' -> ')}] with active perception fallback and adversarial verification. Consensus score: ${t.verifierScore}.`,
    rejected: `Plan: [single_vlm_direct_guess] without metric depth, without color verify, and without adversarial cross-check.`,
    margin: parseFloat((t.dpoPreferenceScore || 0.9).toFixed(2)),
  }));
}

export function getToolRouterWeights(): Record<string, { accuracy: number; cost: number; latencyMs: number; preferenceWeight: number }> {
  return {
    detector_ensemble: { accuracy: 0.96, cost: 0.0005, latencyMs: 140, preferenceWeight: 0.92 },
    open_vocab_grounding: { accuracy: 0.93, cost: 0.0008, latencyMs: 220, preferenceWeight: 0.88 },
    metric_depth: { accuracy: 0.95, cost: 0.0006, latencyMs: 180, preferenceWeight: 0.94 },
    mask_segmentation: { accuracy: 0.94, cost: 0.0007, latencyMs: 210, preferenceWeight: 0.86 },
    paddleocr_engine: { accuracy: 0.97, cost: 0.0004, latencyMs: 120, preferenceWeight: 0.95 },
    video_sam2_tracking: { accuracy: 0.93, cost: 0.0012, latencyMs: 380, preferenceWeight: 0.84 },
    adversarial_verifier: { accuracy: 0.97, cost: 0.0006, latencyMs: 190, preferenceWeight: 0.96 },
  };
}

export interface GroundTruthAnnotation {
  id: string;
  imageId?: string;
  imageFileName?: string;
  label: string;
  bbox2d: { ymin: number; xmin: number; ymax: number; xmax: number };
  notes?: string;
  createdAt: string;
}

const groundTruthStore: GroundTruthAnnotation[] = [
  {
    id: 'gt_sample_01',
    imageFileName: 'warehouse_forklift.jpg',
    label: 'worker_with_helmet',
    bbox2d: { ymin: 220, xmin: 150, ymax: 750, xmax: 380 },
    notes: 'Verified Ground Truth for OSHA hazard detection',
    createdAt: new Date().toISOString(),
  },
  {
    id: 'gt_sample_02',
    imageFileName: 'parking_lot.jpg',
    label: 'red_car',
    bbox2d: { ymin: 320, xmin: 80, ymax: 580, xmax: 340 },
    notes: 'Ground truth sedan for CountBench evaluation',
    createdAt: new Date().toISOString(),
  },
];

export function saveAnnotations(annotations: GroundTruthAnnotation[]): GroundTruthAnnotation[] {
  annotations.forEach((anno) => {
    const existingIndex = groundTruthStore.findIndex((a) => a.id === anno.id);
    if (existingIndex >= 0) {
      groundTruthStore[existingIndex] = anno;
    } else {
      groundTruthStore.unshift(anno);
    }
  });
  return [...groundTruthStore];
}

export function getSavedAnnotations(): GroundTruthAnnotation[] {
  return [...groundTruthStore];
}
