/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { runOmniVisionAgent } from './src/server/agent/orchestrator.ts';
import { runPerceptionSpecialist } from './src/server/agent/perception.ts';
import { getTraces, generateDPODataset, getToolRouterWeights, saveAnnotations, getSavedAnnotations } from './src/server/agent/metaLearner.ts';
import { REQUIRED_BENCHMARK_CHAINS } from './src/server/benchmarks/presets.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  // AI Studio requires the dev server to run strictly on port 3000
  const PORT = 3000;

  // CORS headers for iframe embedding
  app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    if (req.method === 'OPTIONS') {
      res.sendStatus(200);
      return;
    }
    next();
  });

  // JSON parser with high payload ceiling for base64 camera frames
  app.use(express.json({ limit: '60mb' }));
  app.use(express.urlencoded({ extended: true, limit: '60mb' }));

  // API Routes
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'online',
      system: 'OmniVision Agent v3',
      geminiApiKeyConfigured: !!process.env.GEMINI_API_KEY,
      sotaModels: [
        'gemini-3.8-flash',
        'yolo11-rtdetr-ensemble',
        'grounding-dino-1.5',
        'sam2-video-predictor',
        'depth-anything-v2-metric',
        'paddleocr-v4',
        'siglip2-clip-ensemble',
        'rtmpose-action-reasoner',
      ],
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/benchmarks', (_req, res) => {
    res.json({ benchmarks: REQUIRED_BENCHMARK_CHAINS });
  });

  app.post('/api/agent/run', async (req, res) => {
    try {
      const { query, imageBase64, presetChainId, cvEngine, nmsThreshold } = req.body;
      if (!query || typeof query !== 'string') {
        res.status(400).json({ error: 'Query parameter is required' });
        return;
      }
      const response = await runOmniVisionAgent({
        query,
        imageBase64,
        presetChainId,
        cvEngine,
        nmsThreshold,
      });
      res.json(response);
    } catch (err: any) {
      console.error('[API /api/agent/run Error]', err);
      res.status(500).json({ error: err.message || 'Agent pipeline execution failed' });
    }
  });

  app.post('/api/perception/detect', async (req, res) => {
    try {
      const { imageBase64, queryHint, vocabulary, presetType, cvEngine, nmsThreshold } = req.body;
      const result = await runPerceptionSpecialist({
        imageBase64,
        queryHint,
        vocabulary,
        presetType,
        cvEngine,
        nmsThreshold,
      });
      res.json(result);
    } catch (err: any) {
      console.error('[API /api/perception/detect Error]', err);
      res.status(500).json({ error: err.message || 'Perception failed' });
    }
  });

  app.get('/api/agent/meta-learn/traces', (_req, res) => {
    res.json({ traces: getTraces() });
  });

  app.get('/api/agent/meta-learn/dpo', (_req, res) => {
    res.json({ dpoPairs: generateDPODataset() });
  });

  app.get('/api/agent/meta-learn/router', (_req, res) => {
    res.json({ routerWeights: getToolRouterWeights() });
  });

  app.get('/api/agent/meta-learn/annotations', (_req, res) => {
    res.json({ annotations: getSavedAnnotations() });
  });

  app.post('/api/agent/meta-learn/annotations', (req, res) => {
    try {
      const { annotations } = req.body;
      if (!Array.isArray(annotations)) {
        res.status(400).json({ error: 'annotations must be an array' });
        return;
      }
      const saved = saveAnnotations(annotations);
      res.json({ success: true, count: saved.length, annotations: saved });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to save annotations' });
    }
  });

  // Setup Vite middlewares for development or serve dist in production
  const isProduction = process.env.NODE_ENV === 'production';
  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true, port: PORT, host: '0.0.0.0' },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[OmniVision Server] Running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('[Server Start Fatal Error]', err);
  process.exit(1);
});
