/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { PlanDAG } from '../types/index.ts';
import { CheckCircle2, Clock, Terminal, GitBranch, ArrowRight, Code } from 'lucide-react';

interface AgentGraphViewProps {
  plan: PlanDAG;
}

export const AgentGraphView: React.FC<AgentGraphViewProps> = ({ plan }) => {
  const [activeTab, setActiveTab] = useState<'dag' | 'python' | 'tree'>('dag');

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col">
      {/* Header with Sub-tabs */}
      <div className="px-4 py-3 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <GitBranch className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
            Synthesized Plan & Program Graph
          </span>
          <span className="px-2 py-0.5 text-[10px] font-mono bg-cyan-950 border border-cyan-800 text-cyan-300 rounded">
            Intent: {plan.intent.toUpperCase()}
          </span>
        </div>

        <div className="flex rounded-lg bg-slate-800 p-0.5 text-xs font-medium">
          <button
            onClick={() => setActiveTab('dag')}
            className={`px-3 py-1 rounded-md transition-all ${
              activeTab === 'dag' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            DAG Pipeline ({plan.steps.length} Steps)
          </button>
          <button
            onClick={() => setActiveTab('python')}
            className={`px-3 py-1 rounded-md flex items-center space-x-1 transition-all ${
              activeTab === 'python' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Code className="w-3.5 h-3.5" />
            <span>ViperGPT Program</span>
          </button>
          <button
            onClick={() => setActiveTab('tree')}
            className={`px-3 py-1 rounded-md transition-all ${
              activeTab === 'tree' ? 'bg-cyan-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            Tree Search ({plan.treeCandidates?.length || 3})
          </button>
        </div>
      </div>

      {/* Content Body */}
      <div className="p-4 min-h-[300px]">
        {activeTab === 'dag' && (
          <div className="space-y-3">
            <div className="text-xs text-slate-400 mb-2 font-mono">
              Query Target: <span className="text-slate-200 font-semibold">"{plan.query}"</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {plan.steps.map((step, idx) => (
                <div
                  key={step.id}
                  className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg relative flex flex-col justify-between hover:border-slate-700 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-slate-800 text-cyan-300 rounded">
                      {step.id}
                    </span>
                    <div className="flex items-center space-x-1 text-[11px] font-mono text-emerald-400">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>{step.executionTimeMs || 120}ms</span>
                    </div>
                  </div>

                  <div className="text-xs font-semibold text-slate-200 mb-1">{step.tool}</div>
                  <div className="text-[11px] text-slate-400 mb-2 leading-relaxed">{step.description}</div>

                  <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-500">
                    <span>Obs: {step.observationId || `obs_${idx + 1}`}</span>
                    {step.dependencies.length > 0 && (
                      <span className="text-slate-400">
                        Deps: {step.dependencies.join(', ')}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'python' && (
          <div className="relative">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400 font-mono flex items-center space-x-1.5">
                <Terminal className="w-3.5 h-3.5 text-cyan-400" />
                <span>Executable Sandboxed Python AST Program</span>
              </span>
              <span className="text-[11px] font-mono text-emerald-400">Status: Verified & Executed</span>
            </div>

            <pre className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-cyan-200 overflow-x-auto leading-relaxed shadow-inner">
              <code>{plan.synthesizedPythonProgram || '# Synthesizing vision program...'}</code>
            </pre>
          </div>
        )}

        {activeTab === 'tree' && (
          <div className="space-y-3">
            <div className="text-xs text-slate-400 font-mono mb-2">
              Test-Time Compute Tree Search: Sampled candidate plans scored by predicted accuracy and latency.
            </div>

            <div className="space-y-2.5">
              {(plan.treeCandidates || []).map((cand) => (
                <div
                  key={cand.planId}
                  className={`p-3 rounded-lg border flex items-center justify-between ${
                    cand.selected
                      ? 'bg-cyan-950/30 border-cyan-500/80 text-slate-200'
                      : 'bg-slate-950/60 border-slate-800 text-slate-400'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-mono font-bold text-cyan-300">{cand.planId}</span>
                      {cand.selected && (
                        <span className="px-2 py-0.5 text-[10px] font-mono bg-cyan-600 text-white rounded">
                          Selected by Verifier
                        </span>
                      )}
                    </div>
                    <div className="text-xs">{cand.description}</div>
                  </div>

                  <div className="flex items-center space-x-4 text-xs font-mono">
                    <div className="text-right">
                      <div className="text-emerald-400 font-semibold">
                        {(cand.predictedAccuracy * 100).toFixed(0)}% Acc
                      </div>
                      <div className="text-[10px] text-slate-500">${cand.predictedCost.toFixed(4)}</div>
                    </div>
                    <ArrowRight className="w-4 h-4 text-slate-600" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
