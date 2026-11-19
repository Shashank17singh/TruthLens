import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ShieldCheck,
  Clock,
  Download,
  Bookmark,
  MessageSquare,
  ArrowLeft,
  AlertTriangle,
  Loader2,
  Cpu,
  Copy,
  Check,
} from 'lucide-react';
import { useAnalysis, useSaveAnalysis, useRequestReport, useReport } from '../api/hooks';
import { useAuth } from '../context/AuthContext';
import { getStoredToken } from '../api/client';
import type { ImageDetails, AudioDetails, TextDetails } from '../api/types';
import { Gauge } from '../components/Gauge';
import { ImageResult } from '../components/ImageResult';
import { AudioResult } from '../components/AudioResult';
import { TextResult } from '../components/TextResult';
import { FeedbackModal } from '../components/FeedbackModal';

export const ResultPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const { user, loginAsDemo } = useAuth();
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);
  const [reportId, setReportId] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [copiedId, setCopiedId] = useState(false);

  useEffect(() => {
    if (!user && !getStoredToken()) {
      loginAsDemo();
    }
  }, [user, loginAsDemo]);

  // Poll every 1.5s if in queued or processing state
  const { data: analysis, isLoading, error } = useAnalysis(
    id,
    true,
    1500
  );
  const saveMutation = useSaveAnalysis();
  const requestReportMutation = useRequestReport();
  const reportQuery = useReport(reportId || undefined, Boolean(reportId));

  const copyAnalysisId = () => {
    if (analysis?.id) {
      navigator.clipboard.writeText(analysis.id);
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-[50vh] flex flex-col items-center justify-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-brand-50 dark:bg-brand-950/60 flex items-center justify-center text-brand-600">
          <Loader2 className="w-6 h-6 animate-spin" />
        </div>
        <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
          Loading forensic record #<span className="font-mono">{id?.slice(0, 8)}</span>…
        </p>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="max-w-xl mx-auto py-16 px-4 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-rose-50 dark:bg-rose-950/50 text-rose-600 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <h2 className="text-xl font-bold text-slate-900 dark:text-white">Analysis Not Found</h2>
        <p className="text-xs text-slate-500">
          {(error as Error)?.message || 'The requested analysis record does not exist or may have expired.'}
        </p>
        <Link
          to="/analyze"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-600 text-white text-xs font-semibold shadow-sm hover:bg-brand-700 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Analyze
        </Link>
      </div>
    );
  }

  // Pipeline in progress state
  if (analysis.status === 'queued' || analysis.status === 'processing') {
    return (
      <div className="max-w-2xl mx-auto py-16 px-4 space-y-6">
        <div className="p-8 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xl space-y-6 text-center relative overflow-hidden">
          <div className="w-16 h-16 rounded-2xl bg-brand-50 dark:bg-brand-950/60 border border-brand-200 dark:border-brand-800 text-brand-600 dark:text-brand-400 mx-auto flex items-center justify-center relative">
            <Loader2 className="w-8 h-8 animate-spin" />
          </div>

          <div className="space-y-2">
            <h2 className="text-2xl font-extrabold text-slate-900 dark:text-white capitalize">
              Evaluating {analysis.modality} forensic attribution
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto">
              Running spatial feature extraction, frequency spectrum analysis, and calibrated inference thresholds…
            </p>
          </div>

          <div className="space-y-2 max-w-md mx-auto pt-2">
            <div className="flex justify-between text-xs text-slate-500">
              <span className="capitalize">{analysis.status}…</span>
              <span className="font-mono font-bold">{analysis.progress || 50}%</span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-brand-600 to-emerald-400 rounded-full origin-left transition-transform duration-300"
                style={{ transform: `scaleX(${Math.max(20, analysis.progress || 50) / 100})`, width: '100%' }}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Pipeline failure state
  if (analysis.status === 'failed') {
    return (
      <div className="max-w-xl mx-auto py-16 px-4 text-center space-y-5">
        <div className="w-14 h-14 rounded-2xl bg-rose-50 dark:bg-rose-950/60 border border-rose-200 dark:border-rose-900 text-rose-600 mx-auto flex items-center justify-center">
          <AlertTriangle className="w-7 h-7" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Pipeline Execution Error</h2>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          {analysis.error?.message || 'An error occurred during multi-modal inference execution.'}
        </p>
        <Link
          to="/analyze"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold shadow-md transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Try Another Analysis
        </Link>
      </div>
    );
  }

  const result = analysis.result;
  const isRetained = analysis.retain_media;

  const handleSaveToggle = async () => {
    if (!analysis.id) return;
    try {
      await saveMutation.mutateAsync(analysis.id);
    } catch {
      // handled
    }
  };

  const handleDownloadReport = async () => {
    if (!analysis.id) return;
    setGeneratingReport(true);
    try {
      const rep = await requestReportMutation.mutateAsync({ analysisId: analysis.id, format: 'pdf' });
      setReportId(rep.report_id);
      if (rep.status === 'ready' && rep.pdf_url) {
        window.open(rep.pdf_url, '_blank');
        setGeneratingReport(false);
      }
    } catch {
      setGeneratingReport(false);
    }
  };

  // If report was polling and is now ready
  if (reportQuery.data?.status === 'ready' && generatingReport) {
    window.open(reportQuery.data.pdf_url, '_blank');
    setGeneratingReport(false);
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Link
          to="/history"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Analyses History
        </Link>

        <div className="flex flex-wrap items-center gap-2">
          {/* Copy Analysis ID */}
          <button
            onClick={copyAnalysisId}
            className="px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 flex items-center gap-1.5 shadow-sm transition-colors"
            title="Copy Analysis ID"
          >
            {copiedId ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-500" />
                <span className="text-emerald-500">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" />
                <span>Copy ID</span>
              </>
            )}
          </button>

          {/* Save / Retain Media Button */}
          <button
            onClick={handleSaveToggle}
            disabled={saveMutation.isPending || isRetained}
            className={`px-3 py-1.5 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition-colors ${
              isRetained
                ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 border-slate-200 dark:border-slate-700 cursor-default'
                : 'bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
            }`}
          >
            <Bookmark className={`w-3.5 h-3.5 ${isRetained ? 'fill-current' : ''}`} />
            {isRetained ? 'Media Retained' : 'Retain Media'}
          </button>

          {/* Feedback Trigger */}
          <button
            onClick={() => setFeedbackModalOpen(true)}
            className="px-3 py-1.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:outline-none active:scale-[0.98] text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 shadow-sm transition-colors"
          >
            <MessageSquare className="w-3.5 h-3.5 text-brand-500" />
            Was this correct?
          </button>

          {/* PDF Report Trigger */}
          <button
            onClick={handleDownloadReport}
            disabled={generatingReport || requestReportMutation.isPending}
            className="px-3.5 py-1.5 rounded-xl bg-brand-600 hover:bg-brand-700 focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:outline-none active:scale-[0.98] disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-brand-500/20 transition-colors motion-safe:transition-transform motion-safe:hover:scale-[1.01]"
          >
            {generatingReport ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            Download PDF Report
          </button>
        </div>
      </div>

      {/* Main Results Header Banner */}
      <div className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm p-6 sm:p-8 space-y-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-8 border-b border-slate-100 dark:border-slate-800 pb-6">
          {/* Metadata & Verdict Summary */}
          <div className="space-y-3 flex-1 text-center md:text-left">
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-2">
              <span className="px-3 py-1 rounded-full bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400 text-xs font-semibold border border-brand-200 dark:border-brand-800 capitalize">
                {analysis.modality} forensics
              </span>
              <span className="text-xs text-slate-400">
                ID: <span className="font-mono">{analysis.id.slice(0, 18)}…</span>
              </span>
            </div>

            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
              {result?.verdict_label || 'Forensic Assessment Completed'}
            </h1>

            <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl">
              {result?.explanation || 'Statistical feature evaluation performed on input payload.'}
            </p>

            <div className="flex flex-wrap items-center justify-center md:justify-start gap-4 text-xs text-slate-500 pt-1">
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5" />
                {analysis.created_at ? new Date(analysis.created_at).toLocaleString() : 'N/A'}
              </span>
              {result?.model_name && (
                <span className="flex items-center gap-1">
                  <Cpu className="w-3.5 h-3.5 text-brand-500" />
                  <span className="font-mono">{result.model_name}</span> (v<span className="font-mono">{result.model_version}</span>)
                </span>
              )}
              {result?.latency_ms && (
                <span className="bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-[11px]">
                  <span className="font-mono">{result.latency_ms}</span> ms
                </span>
              )}
            </div>
          </div>

          {/* Probability Gauge */}
          {result && (
            <div className="flex-shrink-0">
              <Gauge
                probability={result.probability}
                confidenceBand={result.confidence_band}
                verdict={result.verdict}
                verdictLabel={result.verdict_label}
                size="md"
              />
            </div>
          )}
        </div>

        {/* 24-Hour Privacy Purge Notice */}
        {!isRetained && analysis.media_expires_at && (
          <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs text-slate-500">
            <span className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-amber-500 flex-shrink-0" />
              <span>
                Raw media scheduled for automatic cryptographic purge on{' '}
                <strong className="text-slate-700 dark:text-slate-300">
                  {new Date(analysis.media_expires_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </strong>{' '}
                under TruthLens 24-hour privacy policy.
              </span>
            </span>
            <button
              onClick={handleSaveToggle}
              className="text-brand-600 dark:text-brand-400 font-semibold hover:underline"
            >
              Keep Forever
            </button>
          </div>
        )}
      </div>

      {/* Modality Specific Explainability Visualization */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-brand-600" />
          Explainability & Attribution Evidence
        </h3>

        {analysis.modality === 'image' && (
          <ImageResult
            analysisId={analysis.id}
            details={result?.details as ImageDetails | undefined}
          />
        )}

        {analysis.modality === 'audio' && (
          <AudioResult
            analysisId={analysis.id}
            details={result?.details as AudioDetails | undefined}
          />
        )}

        {analysis.modality === 'text' && (
          <TextResult
            details={result?.details as TextDetails | undefined}
          />
        )}
      </div>

      {/* Disclaimer Card */}
      <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 text-xs text-slate-500 dark:text-slate-400 space-y-1">
        <span className="font-bold text-slate-700 dark:text-slate-300 block">
          Forensic Integrity & Human Review Disclaimer
        </span>
        <p>
          {result?.disclaimer ||
            'Results are probabilistic machine learning inferences and should not be used in isolation for legal decisions or sanctions without corroborating contextual evidence.'}
        </p>
      </div>

      {/* Feedback Modal */}
      <FeedbackModal
        analysisId={analysis.id}
        isOpen={feedbackModalOpen}
        onClose={() => setFeedbackModalOpen(false)}
      />
    </div>
  );
};
