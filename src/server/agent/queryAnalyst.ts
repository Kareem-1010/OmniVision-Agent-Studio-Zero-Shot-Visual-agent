/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { QueryIntent } from '../../types/index.ts';
import { getGeminiClient, PRIMARY_VISION_MODEL } from '../geminiClient.ts';

export interface QueryAnalysisOutput {
  intent: QueryIntent;
  targetEntities: string[];
  spatialQualifiers: string[];
  temporalBounds?: {
    isTemporal: boolean;
    referenceEvent?: string;
  };
  metricRequirements: boolean;
  ambiguityScore: number; // 0..1
  recommendedTools: string[];
  clarificationPrompt?: string;
}

export async function analyzeQuery(query: string): Promise<QueryAnalysisOutput> {
  const normalizedQuery = query.toLowerCase();

  // Rule-based fast-path intent classifier & entity extractor
  let intent: QueryIntent = 'existence';
  let metricRequirements = false;
  const targetEntities: string[] = [];
  const spatialQualifiers: string[] = [];
  let isTemporal = false;

  if (normalizedQuery.includes('count') || normalizedQuery.includes('how many')) {
    intent = 'count';
  } else if (normalizedQuery.includes('hazard') || normalizedQuery.includes('safety') || normalizedQuery.includes('ppe') || normalizedQuery.includes('spill') || normalizedQuery.includes('risk')) {
    intent = 'safety_hazard';
    metricRequirements = true;
  } else if (normalizedQuery.includes('read') || normalizedQuery.includes('text') || normalizedQuery.includes('sign') || normalizedQuery.includes('words')) {
    intent = 'ocr';
  } else if (normalizedQuery.includes('closest') || normalizedQuery.includes('how far') || normalizedQuery.includes('distance') || normalizedQuery.includes('distance to')) {
    intent = 'relation';
    metricRequirements = true;
  } else if (normalizedQuery.includes('fit') || normalizedQuery.includes('could') || normalizedQuery.includes('clearance') || normalizedQuery.includes('gap')) {
    intent = 'counterfactual';
    metricRequirements = true;
  } else if (normalizedQuery.includes('when') || normalizedQuery.includes('how long') || normalizedQuery.includes('arrive') || normalizedQuery.includes('stay') || normalizedQuery.includes('dwell')) {
    intent = 'temporal';
    isTemporal = true;
  } else if (normalizedQuery.includes('holding') || normalizedQuery.includes('standing next to') || normalizedQuery.includes('which then') || normalizedQuery.includes('what is the')) {
    intent = 'multi_hop';
  } else if (normalizedQuery.includes('color') || normalizedQuery.includes('size') || normalizedQuery.includes('material')) {
    intent = 'attribute';
  }

  // Common entity extractions
  const words = normalizedQuery.split(/\s+/);
  for (const w of ['car', 'cars', 'truck', 'person', 'people', 'worker', 'helmet', 'vest', 'sign', 'door', 'umbrella', 'cart', 'shelf', 'shelves', 'box', 'forklift', 'hazard']) {
    if (normalizedQuery.includes(w)) {
      targetEntities.push(w);
    }
  }

  // LLM refinement when available
  if (process.env.GEMINI_API_KEY) {
    try {
      const ai = getGeminiClient();
      const prompt = `You are the Query Analyst agent in OmniVision Agent v3. Analyze this user query: "${query}".
Output JSON matching this exact structure:
{
  "intent": "count" | "existence" | "attribute" | "relation" | "referring_expression" | "ocr" | "temporal" | "safety_hazard" | "counterfactual" | "multi_hop" | "comparative",
  "targetEntities": ["list", "of", "entities"],
  "spatialQualifiers": ["e.g. next to the door", "between shelves"],
  "metricRequirements": boolean,
  "ambiguityScore": number between 0 and 1,
  "recommendedTools": ["detect_ensemble", "metric_depth", "grounded_sam2", "crop_ocr", "temporal_tracker", "geometry_engine"]
}`;

      const response = await ai.models.generateContent({
        model: PRIMARY_VISION_MODEL,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
      });

      const text = response.text;
      if (text) {
        const parsed = JSON.parse(text);
        return {
          intent: parsed.intent || intent,
          targetEntities: parsed.targetEntities && parsed.targetEntities.length > 0 ? parsed.targetEntities : (targetEntities.length > 0 ? targetEntities : ['target_object']),
          spatialQualifiers: parsed.spatialQualifiers || [],
          temporalBounds: {
            isTemporal: parsed.intent === 'temporal' || isTemporal,
            referenceEvent: query,
          },
          metricRequirements: parsed.metricRequirements ?? metricRequirements,
          ambiguityScore: typeof parsed.ambiguityScore === 'number' ? parsed.ambiguityScore : 0.05,
          recommendedTools: parsed.recommendedTools || ['detect_ensemble', 'metric_depth', 'verifier_adversarial'],
        };
      }
    } catch (err) {
      console.warn('[QueryAnalyst] Gemini parsing error fallback:', err);
    }
  }

  return {
    intent,
    targetEntities: targetEntities.length > 0 ? targetEntities : ['object'],
    spatialQualifiers,
    temporalBounds: {
      isTemporal,
      referenceEvent: query,
    },
    metricRequirements,
    ambiguityScore: 0.05,
    recommendedTools: ['detect_ensemble', 'metric_depth', 'geometry_specialist', 'verifier_adversarial'],
  };
}
