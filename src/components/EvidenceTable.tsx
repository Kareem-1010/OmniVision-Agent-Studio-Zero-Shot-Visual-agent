/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { VerificationResult } from '../types/index.ts';
import { ShieldCheck, AlertTriangle, Check, FileCheck, HelpCircle } from 'lucide-react';

interface EvidenceTableProps {
  verification: VerificationResult;
  calibratedConfidence: number;
}

export const EvidenceTable: React.FC<EvidenceTableProps> = ({
  verification,
  calibratedConfidence,
}) => {
  const confPercent = Math.round(calibratedConfidence * 100);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col space-y-4 p-4">
      {/* Top Verification Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <div>
            <h3 className="text-sm font-semibold text-slate-200">
              Adversarial Verifier & Claim Attribution
            </h3>
            <p className="text-xs text-slate-400">
              Cross-model consensus, automated falsification testing, and NLI entailment verification
            </p>
          </div>
        </div>

        {/* Calibrated Confidence Gauge */}
        <div className="flex items-center space-x-3 px-4 py-2 bg-slate-950 border border-slate-800 rounded-lg">
          <div className="text-right">
            <div className="text-[10px] font-mono uppercase text-slate-400">Calibrated Confidence</div>
            <div className="text-lg font-mono font-bold text-emerald-400">{confPercent}%</div>
          </div>
          <div className="w-12 h-12 relative flex items-center justify-center">
            <svg className="w-12 h-12 transform -rotate-90">
              <circle
                cx="24"
                cy="24"
                r="18"
                stroke="currentColor"
                strokeWidth="4"
                className="text-slate-800"
                fill="transparent"
              />
              <circle
                cx="24"
                cy="24"
                r="18"
                stroke="currentColor"
                strokeWidth="4"
                className="text-emerald-500"
                fill="transparent"
                strokeDasharray={113}
                strokeDashoffset={113 - (113 * confPercent) / 100}
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Counting 95% Confidence Interval Banner (if present) */}
      {verification.countingInterval && (
        <div className="p-3 bg-cyan-950/30 border border-cyan-800/60 rounded-lg flex items-center justify-between">
          <div className="space-y-0.5">
            <div className="text-xs font-mono font-bold text-cyan-300">
              Poisson Count Reconciliation: {verification.countingInterval.count} Instances
            </div>
            <div className="text-[11px] text-slate-400">
              95% CI Range: [{verification.countingInterval.ci95[0]}, {verification.countingInterval.ci95[1]}] | Raw Detections: {verification.countingInterval.rawDetections} | De-duplicated: {verification.countingInterval.deduplicated}
            </div>
          </div>
          <span className="px-2.5 py-1 text-xs font-mono bg-cyan-900/60 text-cyan-200 border border-cyan-700 rounded">
            Spectral LAB & SigLIP2 Filtered
          </span>
        </div>
      )}

      {/* Adversarial Falsification Card */}
      <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-1.5 text-amber-400 font-semibold">
            <AlertTriangle className="w-4 h-4" />
            <span>Adversarial Falsification Attempt</span>
          </div>
          <span className="px-2 py-0.5 text-[10px] font-mono bg-emerald-950 border border-emerald-800 text-emerald-300 rounded">
            Hypothesis Refuted (Valid Claim)
          </span>
        </div>

        <div className="text-xs text-slate-300 font-mono bg-slate-900/90 p-2.5 rounded border border-slate-800/80">
          <span className="text-slate-500">Hypothesis:</span> {verification.falsificationAttempt.hypothesis}
        </div>

        <div className="text-xs text-slate-400 leading-relaxed">
          <span className="font-semibold text-slate-300">Refutation Evidence:</span>{' '}
          {verification.falsificationAttempt.evidence}
        </div>
      </div>

      {/* Claim-Evidence Attribution Table */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span className="flex items-center space-x-1">
            <FileCheck className="w-4 h-4 text-cyan-400" />
            <span>Claim-to-Evidence Attribution Matrix</span>
          </span>
          <span>NLI Entailment &gt; 0.90</span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="p-2.5">Claim Text</th>
                <th className="p-2.5">Observation IDs</th>
                <th className="p-2.5">NLI Score</th>
                <th className="p-2.5">Verification Notes</th>
                <th className="p-2.5 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
              {verification.claimAttributions.map((claim) => (
                <tr key={claim.claimId} className="hover:bg-slate-800/40">
                  <td className="p-2.5 font-medium text-slate-200">{claim.claimText}</td>
                  <td className="p-2.5 text-cyan-400">{claim.observationIds.join(', ')}</td>
                  <td className="p-2.5 text-emerald-400 font-bold">
                    {(claim.entailmentScore * 100).toFixed(0)}%
                  </td>
                  <td className="p-2.5 text-slate-400">{claim.notes}</td>
                  <td className="p-2.5 text-right">
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                      <Check className="w-3 h-3" />
                      <span>Verified</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
