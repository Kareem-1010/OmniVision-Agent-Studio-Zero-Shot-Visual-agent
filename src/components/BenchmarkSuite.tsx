/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { REQUIRED_BENCHMARK_CHAINS } from '../server/benchmarks/presets.ts';
import { DemoPreset } from '../types/index.ts';
import { Play, Volume2, ShieldAlert, ShoppingBag, Eye, Award, CheckCircle } from 'lucide-react';

interface BenchmarkSuiteProps {
  onRunChain: (preset: DemoPreset) => void;
  isRunning: boolean;
}

export const BenchmarkSuite: React.FC<BenchmarkSuiteProps> = ({ onRunChain, isRunning }) => {
  const [activeDemoMode, setActiveDemoMode] = useState<'chains' | 'retail' | 'safety' | 'accessibility'>('chains');
  const [ttsSpeaking, setTtsSpeaking] = useState<boolean>(false);

  // Web Speech API for Accessibility Narration
  const speakNarration = (text: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.onstart = () => setTtsSpeaking(true);
      utterance.onend = () => setTtsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col p-4 space-y-4">
      {/* Top Bar with Demo Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <Award className="w-5 h-5 text-amber-400" />
          <div>
            <h3 className="text-sm font-semibold text-slate-200">
              Research Benchmarks & Specialized Demo Modes (Phase 7 & 9)
            </h3>
            <p className="text-xs text-slate-400">
              8 Required End-to-End Chains with Ground Truth vs Agent Output & Domain Presets
            </p>
          </div>
        </div>

        <div className="flex rounded-lg bg-slate-950 p-1 text-xs font-mono">
          <button
            onClick={() => setActiveDemoMode('chains')}
            className={`px-3 py-1 rounded transition-all ${
              activeDemoMode === 'chains' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            8 Required Chains
          </button>
          <button
            onClick={() => setActiveDemoMode('retail')}
            className={`px-3 py-1 rounded flex items-center space-x-1 transition-all ${
              activeDemoMode === 'retail' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            <ShoppingBag className="w-3.5 h-3.5" />
            <span>Retail Preset</span>
          </button>
          <button
            onClick={() => setActiveDemoMode('safety')}
            className={`px-3 py-1 rounded flex items-center space-x-1 transition-all ${
              activeDemoMode === 'safety' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Hazard Preset</span>
          </button>
          <button
            onClick={() => setActiveDemoMode('accessibility')}
            className={`px-3 py-1 rounded flex items-center space-x-1 transition-all ${
              activeDemoMode === 'accessibility' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Accessibility (TTS)</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {activeDemoMode === 'chains' && (
        <div className="space-y-3">
          <div className="text-xs font-mono text-slate-400">
            Click any chain below to synthesize its plan, inspect perception overlays, evaluate 3D geometry, and run adversarial verification.
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {REQUIRED_BENCHMARK_CHAINS.map((chain) => (
              <div
                key={chain.id}
                className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg flex flex-col justify-between hover:border-cyan-600/60 transition-all group"
              >
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono font-bold text-cyan-400 group-hover:text-cyan-300">
                      {chain.title}
                    </span>
                    <span className="px-2 py-0.5 text-[10px] font-mono uppercase bg-slate-800 text-slate-300 rounded">
                      {chain.category}
                    </span>
                  </div>

                  <div className="text-xs font-semibold text-slate-200 mb-1.5">
                    Query: "{chain.query}"
                  </div>

                  <div className="text-[11px] text-slate-400 leading-relaxed mb-3">
                    {chain.description}
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-500">{chain.syntheticSceneType}</span>
                  <button
                    onClick={() => onRunChain(chain)}
                    disabled={isRunning}
                    className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-mono font-semibold rounded flex items-center space-x-1.5 transition-all shadow"
                  >
                    <Play className="w-3 h-3" />
                    <span>Run Chain</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Retail Preset Mode */}
      {activeDemoMode === 'retail' && (
        <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-mono font-bold text-slate-200">
              Retail Intelligence Preset: Footfall, Dwell Heatmaps & Queue Monitoring
            </h4>
            <span className="px-2.5 py-0.5 text-xs font-mono bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
              Active Store Camera 04
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs font-mono">
            <div className="p-3 bg-slate-900 border border-slate-800 rounded">
              <div className="text-slate-400 text-[10px]">Total Footfall Today</div>
              <div className="text-base font-bold text-cyan-400">142 Customers</div>
            </div>
            <div className="p-3 bg-slate-900 border border-slate-800 rounded">
              <div className="text-slate-400 text-[10px]">Average Dwell Time</div>
              <div className="text-base font-bold text-emerald-400">16.1 seconds</div>
            </div>
            <div className="p-3 bg-slate-900 border border-slate-800 rounded">
              <div className="text-slate-400 text-[10px]">Queue Length at Till</div>
              <div className="text-base font-bold text-amber-400">1 Person (Optimal)</div>
            </div>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            Automated SAM2 tracklet continuation records customer entry timestamps, computes bounding box dwelling
            times within product zones, and generates automated line-crossing metrics.
          </p>
        </div>
      )}

      {/* Hazard / Safety Mode */}
      {activeDemoMode === 'safety' && (
        <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-mono font-bold text-rose-300 flex items-center space-x-1.5">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              <span>Industrial Safety & OSHA Compliance Incident Report</span>
            </h4>
            <button
              onClick={() => window.print()}
              className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 rounded border border-slate-700"
            >
              Print / Export Report
            </button>
          </div>

          <div className="p-3 bg-rose-950/20 border border-rose-800/60 rounded text-xs font-mono text-rose-200 space-y-1.5">
            <div className="font-bold">Summary: Risk Severity Index 8.8 / 10 (Critical)</div>
            <div>• Violation 1: Worker operating without approved Hard Hat in 2.6m proximity to moving forklift.</div>
            <div>• Violation 2: Unmarked chemical slip spill on primary warehouse pedestrian aisle.</div>
            <div>• Violation 3: Pallet boxes obstructing emergency exit door egress.</div>
          </div>
        </div>
      )}

      {/* Accessibility Egocentric Mode */}
      {activeDemoMode === 'accessibility' && (
        <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-mono font-bold text-slate-200">
              Egocentric Accessibility Assistant (Spatial Audio & Obstacle Guidance)
            </h4>
            <button
              onClick={() =>
                speakNarration(
                  'Caution: Orange safety cone is directly ahead of you at 1.4 meters. A parked bicycle is to your left at 3.8 meters. The path to your right is clear for 8 meters.'
                )
              }
              className={`px-3 py-1.5 text-xs font-mono rounded flex items-center space-x-1.5 transition-colors ${
                ttsSpeaking ? 'bg-amber-600 text-white' : 'bg-cyan-600 hover:bg-cyan-500 text-white'
              }`}
            >
              <Volume2 className="w-4 h-4" />
              <span>{ttsSpeaking ? 'Speaking Audio...' : 'Play Egocentric Narration (TTS)'}</span>
            </button>
          </div>

          <div className="p-3 bg-slate-900 border border-slate-800 rounded text-xs font-mono text-slate-300 leading-relaxed">
            "Caution: Orange safety cone is directly ahead of you at 1.4 meters. A parked bicycle is to your left at 3.8 meters. The path to your right is clear for 8 meters."
          </div>
        </div>
      )}
    </div>
  );
};
