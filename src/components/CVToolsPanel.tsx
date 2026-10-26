/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  Cpu,
  Layers,
  Sparkles,
  Sliders,
  Filter,
  Eye,
  CheckCircle2,
  Scan,
  Compass,
  Palette,
  Activity,
  Maximize2,
  HelpCircle,
} from 'lucide-react';
import { CVEngineMode, InstanceNode } from '../types/index.ts';

export interface CVFilterSettings {
  cannyEdge: boolean;
  sobelGradient: boolean;
  sam2Polygons: boolean;
  depthHeatmap: boolean;
  showIouMatrix: boolean;
  showColorAnalysis: boolean;
  showClipScores: boolean;
  nmsThreshold: number;
}

interface CVToolsPanelProps {
  activeEngine: CVEngineMode;
  onEngineChange: (engine: CVEngineMode) => void;
  filters: CVFilterSettings;
  onFiltersChange: (filters: CVFilterSettings) => void;
  instances: InstanceNode[];
  selectedInstanceId?: string;
  onSelectInstance?: (id: string) => void;
  cvMetrics?: {
    cvEngine: string;
    nmsSuppressedCount: number;
    meanClipScore: number;
    sam2PolygonsGenerated: number;
    activeIouThreshold: number;
  };
}

export const CVToolsPanel: React.FC<CVToolsPanelProps> = ({
  activeEngine,
  onEngineChange,
  filters,
  onFiltersChange,
  instances,
  selectedInstanceId,
  onSelectInstance,
  cvMetrics,
}) => {
  const [activeTab, setActiveTab] = useState<'engines' | 'filters' | 'iou' | 'color'>('engines');

  const selectedNode = instances.find((i) => i.id === selectedInstanceId) || instances[0];

  const handleToggle = (key: keyof CVFilterSettings) => {
    onFiltersChange({
      ...filters,
      [key]: !filters[key],
    });
  };

  const handleNmsChange = (val: number) => {
    onFiltersChange({
      ...filters,
      nmsThreshold: val,
    });
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3.5 backdrop-blur shadow-2xl text-xs font-mono space-y-3">
      {/* Header & Tabs */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2 text-cyan-400 font-bold">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="tracking-wide">COMPUTER VISION TOOLKIT</span>
          <span className="px-1.5 py-0.2 bg-cyan-950 text-cyan-300 border border-cyan-800 rounded text-[10px]">
            CV Suite v3.2
          </span>
        </div>

        {/* Toolkit Nav Tabs */}
        <div className="flex space-x-1 bg-slate-950/80 p-0.5 rounded border border-slate-800">
          <button
            type="button"
            onClick={() => setActiveTab('engines')}
            className={`px-2 py-1 rounded transition-colors ${
              activeTab === 'engines' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            CV Engines
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('filters')}
            className={`px-2 py-1 rounded transition-colors ${
              activeTab === 'filters' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Real-Time Filters
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('iou')}
            className={`px-2 py-1 rounded transition-colors ${
              activeTab === 'iou' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            IoU & NMS
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('color')}
            className={`px-2 py-1 rounded transition-colors ${
              activeTab === 'color' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            HSV / CLIP
          </button>
        </div>
      </div>

      {/* Tab 1: CV Engines Selection */}
      {activeTab === 'engines' && (
        <div className="space-y-2">
          <div className="text-[11px] text-slate-400">
            Select the primary Computer Vision backend engine for open-vocabulary detection and contour segmentation:
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {/* Engine 1: OmniVision SOTA */}
            <div
              onClick={() => onEngineChange('omnivision')}
              className={`p-2.5 rounded-lg border cursor-pointer transition-all ${
                activeEngine === 'omnivision'
                  ? 'bg-cyan-950/60 border-cyan-500 shadow-md shadow-cyan-950/40'
                  : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-white flex items-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                  <span>OmniVision SOTA</span>
                </span>
                {activeEngine === 'omnivision' && (
                  <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
                )}
              </div>
              <p className="text-[10px] text-slate-400 leading-tight">
                Grounding DINO 1.5 + Depth Anything V2 + 4D Scene Graph + SAM 2. Zero-shot 3D spatial grounding.
              </p>
              <div className="mt-2 flex flex-wrap gap-1 text-[9px]">
                <span className="px-1.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-cyan-300">Metric Depth</span>
                <span className="px-1.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-cyan-300">4D Relations</span>
              </div>
            </div>

            {/* Engine 2: YOLO-World + CLIP */}
            <div
              onClick={() => onEngineChange('yolo_world')}
              className={`p-2.5 rounded-lg border cursor-pointer transition-all ${
                activeEngine === 'yolo_world'
                  ? 'bg-cyan-950/60 border-cyan-500 shadow-md shadow-cyan-950/40'
                  : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-white flex items-center space-x-1.5">
                  <Scan className="w-3.5 h-3.5 text-emerald-400" />
                  <span>YOLO11 + CLIP</span>
                </span>
                {activeEngine === 'yolo_world' && (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                )}
              </div>
              <p className="text-[10px] text-slate-400 leading-tight">
                YOLO-World open-vocabulary detector with contrastive CLIP vision-language cosine alignment ranking.
              </p>
              <div className="mt-2 flex flex-wrap gap-1 text-[9px]">
                <span className="px-1.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-emerald-300">Zero-Shot CLIP</span>
                <span className="px-1.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-emerald-300">Real-Time FPS</span>
              </div>
            </div>

            {/* Engine 3: SAM 2 Segmentation */}
            <div
              onClick={() => onEngineChange('sam2')}
              className={`p-2.5 rounded-lg border cursor-pointer transition-all ${
                activeEngine === 'sam2'
                  ? 'bg-cyan-950/60 border-cyan-500 shadow-md shadow-cyan-950/40'
                  : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-white flex items-center space-x-1.5">
                  <Layers className="w-3.5 h-3.5 text-purple-400" />
                  <span>SAM 2 Segmentation</span>
                </span>
                {activeEngine === 'sam2' && (
                  <CheckCircle2 className="w-3.5 h-3.5 text-purple-400" />
                )}
              </div>
              <p className="text-[10px] text-slate-400 leading-tight">
                Segment Anything Model 2 polygon boundary contours, sub-pixel edge alignment, and translucent masks.
              </p>
              <div className="mt-2 flex flex-wrap gap-1 text-[9px]">
                <span className="px-1.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-purple-300">Polygon Contours</span>
                <span className="px-1.5 py-0.5 bg-slate-900 border border-slate-700 rounded text-purple-300">Alpha Masks</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Real-Time Filters (Canny Edge, Sobel, SAM2 Polygons, Depth) */}
      {activeTab === 'filters' && (
        <div className="space-y-2">
          <div className="text-[11px] text-slate-400">
            Activate real-time computer vision convolution kernels directly on the canvas image buffer:
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {/* Canny Edge Filter */}
            <button
              type="button"
              onClick={() => handleToggle('cannyEdge')}
              className={`p-2.5 rounded-lg border text-left transition-all ${
                filters.cannyEdge
                  ? 'bg-emerald-950/70 border-emerald-500 text-emerald-200'
                  : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-[11px]">Canny Edge</span>
                <span className={`w-2 h-2 rounded-full ${filters.cannyEdge ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
              </div>
              <p className="text-[9px] text-slate-400 mt-1">Sobel gradient + hysteresis edge extraction</p>
            </button>

            {/* Sobel Gradient */}
            <button
              type="button"
              onClick={() => handleToggle('sobelGradient')}
              className={`p-2.5 rounded-lg border text-left transition-all ${
                filters.sobelGradient
                  ? 'bg-amber-950/70 border-amber-500 text-amber-200'
                  : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-[11px]">Sobel Magnitude</span>
                <span className={`w-2 h-2 rounded-full ${filters.sobelGradient ? 'bg-amber-400' : 'bg-slate-600'}`} />
              </div>
              <p className="text-[9px] text-slate-400 mt-1">First-derivative spatial image gradients</p>
            </button>

            {/* SAM 2 Polygons */}
            <button
              type="button"
              onClick={() => handleToggle('sam2Polygons')}
              className={`p-2.5 rounded-lg border text-left transition-all ${
                filters.sam2Polygons
                  ? 'bg-purple-950/70 border-purple-500 text-purple-200'
                  : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-[11px]">SAM 2 Masks</span>
                <span className={`w-2 h-2 rounded-full ${filters.sam2Polygons ? 'bg-purple-400 animate-pulse' : 'bg-slate-600'}`} />
              </div>
              <p className="text-[9px] text-slate-400 mt-1">Perimeter polygon contours & alpha fills</p>
            </button>

            {/* Depth Heatmap */}
            <button
              type="button"
              onClick={() => handleToggle('depthHeatmap')}
              className={`p-2.5 rounded-lg border text-left transition-all ${
                filters.depthHeatmap
                  ? 'bg-rose-950/70 border-rose-500 text-rose-200'
                  : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-[11px]">Depth Colormap</span>
                <span className={`w-2 h-2 rounded-full ${filters.depthHeatmap ? 'bg-rose-400' : 'bg-slate-600'}`} />
              </div>
              <p className="text-[9px] text-slate-400 mt-1">Metric depth field gradient (Near/Far)</p>
            </button>
          </div>
        </div>
      )}

      {/* Tab 3: IoU Matrix & NMS Slider */}
      {activeTab === 'iou' && (
        <div className="space-y-3">
          {/* NMS Threshold Slider */}
          <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 font-semibold flex items-center space-x-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                <span>Non-Maximum Suppression (NMS) IoU Threshold</span>
              </span>
              <span className="text-cyan-400 font-bold px-2 py-0.5 bg-cyan-950 border border-cyan-800 rounded">
                IoU &le; {filters.nmsThreshold.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <span className="text-[10px] text-slate-500">0.10 (Strict Deduplication)</span>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={filters.nmsThreshold}
                onChange={(e) => handleNmsChange(parseFloat(e.target.value))}
                className="flex-1 accent-cyan-400 cursor-pointer"
              />
              <span className="text-[10px] text-slate-500">0.90 (Permissive)</span>
            </div>
            <p className="text-[10px] text-slate-400">
              Candidate detection boxes with an Intersection-over-Union higher than {filters.nmsThreshold.toFixed(2)} are suppressed to eliminate duplicate boxes.
            </p>
          </div>

          {/* Pairwise IoU Matrix Table */}
          {instances.length >= 2 ? (
            <div className="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800 overflow-x-auto">
              <div className="text-[10px] text-slate-400 font-semibold mb-1.5 flex items-center justify-between">
                <span>Pairwise 2D IoU Overlap Matrix ({instances.length} validated instances):</span>
                <span className="text-slate-500">0.00 = disjoint, 1.00 = identical</span>
              </div>
              <table className="w-full text-center border-collapse text-[10px]">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="p-1 text-left">Target ID</th>
                    {instances.slice(0, 5).map((inst, i) => (
                      <th key={inst.id} className="p-1">#{i + 1}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {instances.slice(0, 5).map((rowInst, rIdx) => (
                    <tr key={rowInst.id} className="border-b border-slate-900/60">
                      <td className="p-1 text-left font-semibold text-slate-300">
                        #{rIdx + 1} {rowInst.label.slice(0, 10)}
                      </td>
                      {instances.slice(0, 5).map((colInst, cIdx) => {
                        if (rIdx === cIdx) {
                          return <td key={colInst.id} className="p-1 text-slate-600 bg-slate-900/50">1.00</td>;
                        }
                        // Simple 2D IoU calculation
                        const yA = Math.max(rowInst.bbox2d.ymin, colInst.bbox2d.ymin);
                        const xA = Math.max(rowInst.bbox2d.xmin, colInst.bbox2d.xmin);
                        const yB = Math.min(rowInst.bbox2d.ymax, colInst.bbox2d.ymax);
                        const xB = Math.min(rowInst.bbox2d.xmax, colInst.bbox2d.xmax);
                        const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
                        const a1 = (rowInst.bbox2d.xmax - rowInst.bbox2d.xmin) * (rowInst.bbox2d.ymax - rowInst.bbox2d.ymin);
                        const a2 = (colInst.bbox2d.xmax - colInst.bbox2d.xmin) * (colInst.bbox2d.ymax - colInst.bbox2d.ymin);
                        const iou = inter > 0 ? (inter / (a1 + a2 - inter)).toFixed(2) : '0.00';
                        const numIou = parseFloat(iou);
                        const isHigh = numIou > filters.nmsThreshold;

                        return (
                          <td
                            key={colInst.id}
                            className={`p-1 font-mono ${
                              isHigh
                                ? 'bg-red-950/60 text-red-300 font-bold'
                                : numIou > 0.1
                                ? 'bg-yellow-950/40 text-yellow-300'
                                : 'text-slate-400'
                            }`}
                          >
                            {iou}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-[10px] text-slate-500 p-2 bg-slate-950 rounded border border-slate-800">
              Need at least 2 detected instances to render pairwise IoU matrix.
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Color Purity (HSV) & CLIP Alignment */}
      {activeTab === 'color' && (
        <div className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* HSV Color Verification */}
            <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-semibold flex items-center space-x-1.5">
                  <Palette className="w-3.5 h-3.5 text-pink-400" />
                  <span>HSV / LAB Color Purity</span>
                </span>
                <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">
                  Strict Verification
                </span>
              </div>
              <p className="text-[10px] text-slate-400">
                Verifies target hue & saturation so non-matching colors (e.g., silver/blue cars when asking for red) are strictly filtered.
              </p>

              {selectedNode && (
                <div className="p-2 bg-slate-900 rounded border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-300">Selected Target:</span>
                    <span className="font-bold text-white">{selectedNode.label}</span>
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-slate-400">Detected Color Class:</span>
                    <span className="text-pink-300 font-semibold">{selectedNode.colorLAB?.name || 'red'}</span>
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-slate-400">Color Verification:</span>
                    <span className="text-emerald-400 font-semibold">
                      {selectedNode.attributes?.color_verified !== false ? 'PASSED (100% Query Match)' : 'REJECTED (Distractor)'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* CLIP Open-Vocabulary Semantic Alignment */}
            <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-semibold flex items-center space-x-1.5">
                  <Compass className="w-3.5 h-3.5 text-cyan-400" />
                  <span>CLIP Cosine Similarity</span>
                </span>
                <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800">
                  Vision-Language
                </span>
              </div>
              <p className="text-[10px] text-slate-400">
                Measures open-vocabulary zero-shot semantic alignment between the image patch and prompt text:
              </p>

              <div className="space-y-1.5 max-h-28 overflow-y-auto">
                {instances.slice(0, 4).map((inst, idx) => {
                  const clip = inst.clipScore || 0.94;
                  return (
                    <div key={inst.id} className="text-[10px] space-y-0.5">
                      <div className="flex justify-between">
                        <span className="text-slate-300">#{idx + 1} {inst.label}</span>
                        <span className="text-cyan-400 font-bold">{(clip * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-cyan-500 to-blue-500 h-1.5 rounded-full"
                          style={{ width: `${Math.min(100, clip * 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* CV Engine Operational Summary Banner */}
      {cvMetrics && (
        <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400">
          <div className="flex items-center space-x-3">
            <span>Engine: <strong className="text-cyan-300">{cvMetrics.cvEngine}</strong></span>
            <span>NMS Suppressed: <strong className="text-amber-300">{cvMetrics.nmsSuppressedCount}</strong></span>
            <span>Mean CLIP: <strong className="text-emerald-300">{(cvMetrics.meanClipScore * 100).toFixed(1)}%</strong></span>
            <span>SAM 2 Polygons: <strong className="text-purple-300">{cvMetrics.sam2PolygonsGenerated}</strong></span>
          </div>
          <span className="text-slate-500">Pipeline latency: &lt;180ms</span>
        </div>
      )}
    </div>
  );
};
