import React, { useState } from 'react';
import { FileText, Sparkles, Info } from 'lucide-react';
import type { TextDetails, TextSentence } from '../api/types';

interface TextResultProps {
  details?: TextDetails | Record<string, unknown>;
  rawText?: string;
}

export const TextResult: React.FC<TextResultProps> = ({ details, rawText }) => {
  const [hoveredSentence, setHoveredSentence] = useState<TextSentence | null>(null);
  const [selectedSentence, setSelectedSentence] = useState<TextSentence | null>(null);

  const textDetails = details as TextDetails | undefined;
  const sentences: TextSentence[] = textDetails?.sentences || [];

  const getHighlightStyle = (prob: number) => {
    // Accessible color scale (subtle tints)
    if (prob > 0.75) {
      return 'bg-rose-100 dark:bg-rose-950/70 text-rose-950 dark:text-rose-100 border-b-2 border-rose-400';
    }
    if (prob > 0.45) {
      return 'bg-amber-100 dark:bg-amber-950/60 text-amber-950 dark:text-amber-100 border-b-2 border-amber-400';
    }
    return 'bg-teal-50 dark:bg-teal-950/40 text-teal-950 dark:text-teal-100 border-b-2 border-teal-300';
  };

  const getScoreBadgeColor = (prob: number) => {
    if (prob > 0.75) return 'text-rose-600 bg-rose-50 dark:bg-rose-950/60 border-rose-200 dark:border-rose-800';
    if (prob > 0.45) return 'text-amber-600 bg-amber-50 dark:bg-amber-950/60 border-amber-200 dark:border-amber-800';
    return 'text-teal-600 bg-teal-50 dark:bg-teal-950/60 border-teal-200 dark:border-teal-800';
  };

  const activeSentence = hoveredSentence || selectedSentence;

  return (
    <div className="space-y-6">
      {/* Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60">
          <span className="text-[11px] font-semibold text-slate-500">Total words</span>
          <div className="text-lg font-bold font-mono text-slate-800 dark:text-slate-200 mt-0.5">
            {textDetails?.word_count || 0}
          </div>
        </div>
        <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60">
          <span className="text-[11px] font-semibold text-slate-500">Sentences</span>
          <div className="text-lg font-bold font-mono text-slate-800 dark:text-slate-200 mt-0.5">
            {sentences.length}
          </div>
        </div>
        <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60">
          <span className="text-[11px] font-semibold text-slate-500">Max AI probability</span>
          <div className="text-lg font-bold font-mono text-rose-600 dark:text-rose-400 mt-0.5">
            {sentences.length > 0
              ? `${(Math.max(...sentences.map((s) => s.probability)) * 100).toFixed(0)}%`
              : 'N/A'}
          </div>
        </div>
        <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60">
          <span className="text-[11px] font-semibold text-slate-500">Min AI probability</span>
          <div className="text-lg font-bold font-mono text-teal-600 dark:text-teal-400 mt-0.5">
            {sentences.length > 0
              ? `${(Math.min(...sentences.map((s) => s.probability)) * 100).toFixed(0)}%`
              : 'N/A'}
          </div>
        </div>
      </div>

      {/* Main Text Breakdown Box */}
      <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
          <h4 className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
            <FileText className="w-4 h-4 text-brand-500" />
            Sentence-Level Attribution
          </h4>
          <div className="flex items-center gap-3 text-[11px] text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-teal-200 dark:bg-teal-900" /> Human / Organic
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-amber-200 dark:bg-amber-900" /> Mixed / Moderate
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-rose-200 dark:bg-rose-900" /> High AI Likelihood
            </span>
          </div>
        </div>

        {/* Highlighted Prose Body */}
        {sentences.length > 0 ? (
          <div className="leading-relaxed text-sm text-slate-800 dark:text-slate-200 space-y-1 font-serif">
            {sentences.map((s, idx) => (
              <span
                key={idx}
                onMouseEnter={() => setHoveredSentence(s)}
                onMouseLeave={() => setHoveredSentence(null)}
                onClick={() => setSelectedSentence(s)}
                className={`inline cursor-pointer px-1 py-0.5 rounded transition-colors mr-1.5 ${getHighlightStyle(
                  s.probability
                )} ${
                  activeSentence?.start === s.start
                    ? 'ring-2 ring-brand-500 dark:ring-brand-400 font-medium'
                    : ''
                }`}
              >
                {s.text}{' '}
              </span>
            ))}
          </div>
        ) : (
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 text-slate-600 dark:text-slate-400 text-sm font-mono whitespace-pre-wrap">
            {rawText || 'No sentence breakdown available.'}
          </div>
        )}

        {/* Active Inspection Inspector Banner */}
        {activeSentence && (
          <div className="mt-4 p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/80 flex items-start justify-between gap-4 animate-in fade-in duration-200">
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5 text-brand-500" />
                <span className="font-semibold text-slate-700 dark:text-slate-200">
                  Inspected Sentence
                </span>
                <span
                  className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${getScoreBadgeColor(
                    activeSentence.probability
                  )}`}
                >
                  {(activeSentence.probability * 100).toFixed(0)}% AI Probability
                </span>
              </div>
              <p className="text-slate-600 dark:text-slate-300 italic font-serif">
                &ldquo;{activeSentence.text}&rdquo;
              </p>
            </div>
            <button
              onClick={() => {
                setHoveredSentence(null);
                setSelectedSentence(null);
              }}
              className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            >
              Close
            </button>
          </div>
        )}
      </div>

      {/* Forensic Explanatory Note */}
      <div className="p-4 rounded-xl bg-slate-100 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 flex items-start gap-3 text-xs text-slate-600 dark:text-slate-300">
        <Info className="w-4 h-4 text-brand-500 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-medium text-slate-800 dark:text-slate-200">
            Stylometric & Perplexity Feature Breakdown:
          </p>
          <p>
            Our meta-classifier fuses n-gram perplexity, sentence burstiness, vocabulary entropy, and semantic token distributions. AI language models typically exhibit lower perplexity variance (more uniform phrasing) across consecutive sentences compared to human prose.
          </p>
        </div>
      </div>
    </div>
  );
};
