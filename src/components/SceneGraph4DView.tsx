/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { SceneGraph4D } from '../types/index.ts';
import { Network, Search, ArrowRight, Compass, Shield } from 'lucide-react';

interface SceneGraph4DViewProps {
  sceneGraph: SceneGraph4D;
  selectedInstanceId?: string;
  onSelectInstance?: (id: string) => void;
}

export const SceneGraph4DView: React.FC<SceneGraph4DViewProps> = ({
  sceneGraph,
  selectedInstanceId,
  onSelectInstance,
}) => {
  const [dslQuery, setDslQuery] = useState<string>('MATCH (n:Instance) WHERE n.depth < 4.0m RETURN n');
  const [queryOutput, setQueryOutput] = useState<string | null>(null);

  const instances = Object.values(sceneGraph.instances);

  const handleExecuteDSL = () => {
    const norm = dslQuery.toLowerCase();
    if (norm.includes('depth') || norm.includes('closest')) {
      const closest = instances.sort((a, b) => a.depth.metricDistanceM - b.depth.metricDistanceM)[0];
      setQueryOutput(`Matched Node: [${closest?.id || 'none'}] "${closest?.label}" at distance ${closest?.depth.metricDistanceM}m.`);
    } else if (norm.includes('hazard') || norm.includes('risk')) {
      const hazards = instances.filter((i) => i.label.includes('hazard') || i.label.includes('spill') || i.label.includes('forklift'));
      setQueryOutput(`Found ${hazards.length} hazard-associated entities in 4D scene graph.`);
    } else {
      setQueryOutput(`Query matched ${instances.length} instances across 4D spatial coordinate frame.`);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col p-4 space-y-4">
      {/* Top Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <Network className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-sm font-semibold text-slate-200">
              Persistent 4D Scene Graph (3D Space + Time)
            </h3>
            <p className="text-xs text-slate-400">
              Nodes: {instances.length} instances | Edges: {sceneGraph.edges.length} spatial relations | RANSAC Ground Plane
            </p>
          </div>
        </div>

        {/* Camera Intrinsics Pill */}
        <div className="px-3 py-1 bg-slate-950 border border-slate-800 rounded-md text-[11px] font-mono text-slate-400">
          Intrinsics: fx={sceneGraph.cameraIntrinsics.fx}, fy={sceneGraph.cameraIntrinsics.fy} | Frame: #{sceneGraph.frameIndex}
        </div>
      </div>

      {/* DSL Query Console Bar */}
      <div className="flex items-center space-x-2">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Compass className="w-4 h-4 text-slate-500" />
          </div>
          <input
            type="text"
            value={dslQuery}
            onChange={(e) => setDslQuery(e.target.value)}
            placeholder="Scene Graph Query DSL: MATCH (n)-[r:near]->(m)..."
            className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500"
          />
        </div>
        <button
          onClick={handleExecuteDSL}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-medium rounded-lg flex items-center space-x-1.5 transition-all shadow"
        >
          <Search className="w-3.5 h-3.5" />
          <span>Execute DSL</span>
        </button>
      </div>

      {queryOutput && (
        <div className="p-2.5 bg-cyan-950/40 border border-cyan-800/60 rounded-lg text-xs font-mono text-cyan-300">
          &gt; {queryOutput}
        </div>
      )}

      {/* Main Graph Grid: Nodes & Edges */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Instances List (Nodes) */}
        <div className="space-y-2">
          <div className="text-xs font-mono font-bold text-slate-300 flex items-center justify-between">
            <span>Graph Nodes (Instances)</span>
            <span className="text-slate-500">{instances.length} Active</span>
          </div>

          <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
            {instances.map((inst) => (
              <div
                key={inst.id}
                onClick={() => onSelectInstance && onSelectInstance(inst.id)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  inst.id === selectedInstanceId
                    ? 'bg-amber-500/10 border-amber-500 text-slate-200'
                    : 'bg-slate-950/70 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-xs font-bold text-cyan-400">{inst.id}</span>
                    <span className="text-xs font-semibold">{inst.label}</span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-emerald-300">
                    Conf: {(inst.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 mt-2">
                  <div>Depth: {inst.depth.metricDistanceM}m</div>
                  <div>Center: [{inst.bbox3d?.center.map((v) => v.toFixed(1)).join(', ')}]</div>
                  <div>Dimensions: [{inst.bbox3d?.dimensions.map((v) => v.toFixed(1)).join(', ')}]m</div>
                  <div>Track ID: #{inst.trackId || 'N/A'}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Spatial Edges List */}
        <div className="space-y-2">
          <div className="text-xs font-mono font-bold text-slate-300 flex items-center justify-between">
            <span>Spatial Edges (Relations)</span>
            <span className="text-slate-500">{sceneGraph.edges.length} Traversed</span>
          </div>

          <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
            {sceneGraph.edges.map((edge) => (
              <div
                key={edge.id}
                className="p-3 bg-slate-950/70 border border-slate-800 rounded-lg flex items-center justify-between text-xs font-mono text-slate-300"
              >
                <div className="flex items-center space-x-2">
                  <span className="text-cyan-400 font-bold">{edge.sourceId}</span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-purple-300 text-[10px] font-bold">
                    {edge.relationType.toUpperCase()}
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                  <span className="text-cyan-400 font-bold">{edge.targetId}</span>
                </div>

                <div className="text-right">
                  <div className="text-emerald-400 font-bold">{edge.distanceMeters}m</div>
                  <div className="text-[10px] text-slate-500">{(edge.confidence * 100).toFixed(0)}% conf</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
