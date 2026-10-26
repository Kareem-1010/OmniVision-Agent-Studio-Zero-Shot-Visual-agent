/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Eye, Network, ShieldCheck, Brain, Award, Sparkles, Search, Layers, Play, Square } from 'lucide-react';

export type ActiveTab = 'vision' | 'annotation' | 'reasoning' | 'verification' | 'scenegraph' | 'metalearner' | 'benchmarks';

interface NavigationProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  onSubmitQuery: (query: string) => void;
  isProcessing: boolean;
  onUploadClick?: () => void;
  hasUploadedImage?: boolean;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  onTabChange,
  onSubmitQuery,
  isProcessing,
  onUploadClick,
  hasUploadedImage,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onSubmitQuery(searchQuery.trim());
    }
  };

  return (
    <header className="bg-slate-950 border-b border-slate-800 sticky top-0 z-50">
      {/* Top Brand & Status Line */}
      <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-900/30">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-base font-bold text-white tracking-tight">OmniVision Agent</span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800 rounded">
                v3.0 RESEARCH
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Zero-Shot Visual Agent & 4D Scene Reasoner</p>
          </div>
        </div>

        {/* Global Query Search Bar */}
        <form onSubmit={handleFormSubmit} className="flex-1 max-w-xl mx-2">
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Ask any open-ended question: 'Count the red cars', 'Is there a safety hazard?'..."
              className="w-full pl-9 pr-24 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
            />
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="w-4 h-4 text-slate-500" />
            </div>
            <button
              type="submit"
              disabled={isProcessing}
              className="absolute right-1.5 top-1.5 bottom-1.5 px-3 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded text-xs font-mono font-semibold flex items-center space-x-1 transition-colors"
            >
              {isProcessing ? (
                <span>Reasoning...</span>
              ) : (
                <>
                  <Play className="w-3 h-3 fill-current" />
                  <span>Execute</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Upload Image Quick Action Button */}
        {onUploadClick && (
          <button
            type="button"
            onClick={onUploadClick}
            className={`px-3 py-1.5 rounded-lg border text-xs font-mono font-medium flex items-center space-x-1.5 transition-all shadow-sm ${
              hasUploadedImage
                ? 'bg-cyan-950 border-cyan-500 text-cyan-300'
                : 'bg-slate-900 border-slate-700 hover:border-cyan-500 text-slate-200 hover:text-white'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span>{hasUploadedImage ? 'Uploaded Image Active' : 'Upload Image'}</span>
          </button>
        )}

        {/* System Model Status Badge */}
        <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-slate-300">SOTA Models Online</span>
          </div>
          <span className="px-2 py-0.5 bg-slate-900 border border-slate-800 rounded text-[10px]">
            Gemini 3.8 + YOLO11 + SAM2
          </span>
        </div>
      </div>

      {/* Main Tab Bar */}
      <div className="max-w-7xl mx-auto px-4 flex items-center space-x-1 overflow-x-auto border-t border-slate-900 py-1">
        <button
          onClick={() => onTabChange('vision')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'vision'
              ? 'bg-slate-800 text-cyan-300 border border-slate-700'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Live Vision & 3D Spatial</span>
        </button>

        <button
          onClick={() => onTabChange('annotation')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'annotation'
              ? 'bg-slate-800 text-amber-300 border border-amber-600/60'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Square className="w-4 h-4 text-amber-400" />
          <span>Annotate & Ground Truth</span>
        </button>

        <button
          onClick={() => onTabChange('reasoning')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'reasoning'
              ? 'bg-slate-800 text-cyan-300 border border-slate-700'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          <span>Multi-Agent Synthesizer</span>
        </button>

        <button
          onClick={() => onTabChange('verification')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'verification'
              ? 'bg-slate-800 text-cyan-300 border border-slate-700'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <ShieldCheck className="w-4 h-4" />
          <span>Adversarial Verifier</span>
        </button>

        <button
          onClick={() => onTabChange('scenegraph')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'scenegraph'
              ? 'bg-slate-800 text-cyan-300 border border-slate-700'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Network className="w-4 h-4" />
          <span>4D Scene Graph</span>
        </button>

        <button
          onClick={() => onTabChange('metalearner')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'metalearner'
              ? 'bg-slate-800 text-cyan-300 border border-slate-700'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Brain className="w-4 h-4" />
          <span>Meta-Learner (SFT/DPO)</span>
        </button>

        <button
          onClick={() => onTabChange('benchmarks')}
          className={`px-3.5 py-2 text-xs font-mono font-medium rounded-lg flex items-center space-x-2 transition-all whitespace-nowrap ${
            activeTab === 'benchmarks'
              ? 'bg-slate-800 text-cyan-300 border border-slate-700'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Award className="w-4 h-4" />
          <span>8 Research Chains</span>
        </button>
      </div>
    </header>
  );
};
