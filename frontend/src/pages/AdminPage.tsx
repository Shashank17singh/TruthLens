import React, { useState } from 'react';
import {
  Layers,
  CheckCircle,
  Clock,
  Cpu,
  BarChart3,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  AlertTriangle,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useAdminStats, useAdminFeedback, useModels } from '../api/hooks';

export const AdminPage: React.FC = () => {
  const [page, setPage] = useState(1);
  const { data: stats, isLoading: statsLoading, error: statsError, refetch: refetchStats } = useAdminStats();
  const { data: feedbackData, isLoading: fbLoading, error: fbError } = useAdminFeedback(page, 15);
  const { data: modelsData } = useModels();

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Title */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Layers className="w-6 h-6 text-brand-600" />
            Forensic Telemetry & Operations Dashboard
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Real-time inference statistics, feedback accuracy, and model calibration registry.
          </p>
        </div>

        <button
          onClick={() => refetchStats()}
          className="px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5 hover:bg-slate-50 shadow-sm"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh Stats
        </button>
      </div>

      {statsLoading ? (
        <div className="p-12 flex flex-col items-center justify-center space-y-3">
          <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
          <p className="text-xs text-slate-500">Loading operations telemetry...</p>
        </div>
      ) : statsError ? (
        <div className="p-6 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-800 dark:text-amber-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>Admin privileges required to view telemetry. Sign in with an administrator role.</span>
        </div>
      ) : (
        <>
          {/* KPI Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                Total Analyses
              </span>
              <div className="text-2xl font-extrabold text-slate-900 dark:text-white font-mono">
                {stats?.total_analyses ?? 0}
              </div>
              <span className="text-[10px] text-slate-400">
                {stats?.completed_analyses ?? 0} successfully completed
              </span>
            </div>

            <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                Feedback Accuracy
              </span>
              <div className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400 font-mono flex items-center gap-1.5">
                <CheckCircle className="w-5 h-5" />
                {((stats?.feedback_accuracy ?? 1) * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-slate-400">
                From {stats?.total_feedback ?? 0} verified user feedback reports
              </span>
            </div>

            <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                Average Latency
              </span>
              <div className="text-2xl font-extrabold text-brand-600 dark:text-brand-400 font-mono flex items-center gap-1.5">
                <Clock className="w-5 h-5" />
                {(stats?.average_latency_ms ?? 0).toFixed(0)} ms
              </div>
              <span className="text-[10px] text-slate-400">CPU ONNX execution time</span>
            </div>

            <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                Modality Distribution
              </span>
              <div className="text-xs text-slate-600 dark:text-slate-300 space-y-1 pt-1 font-mono">
                <div className="flex justify-between">
                  <span>Image:</span>
                  <span className="font-bold">{stats?.modality_breakdown?.image ?? 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Audio:</span>
                  <span className="font-bold">{stats?.modality_breakdown?.audio ?? 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Text:</span>
                  <span className="font-bold">{stats?.modality_breakdown?.text ?? 0}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Registry Catalog */}
          {modelsData && modelsData.models.length > 0 && (
            <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-brand-600" />
                <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                  Active Model Inference Registry
                </h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {modelsData.models.map((m, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40 space-y-2 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900 dark:text-white capitalize">
                        {m.modality}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-brand-100 dark:bg-brand-950 text-brand-700 dark:text-brand-300 font-mono text-[10px]">
                        v{m.version}
                      </span>
                    </div>
                    <div className="text-slate-600 dark:text-slate-300 font-mono text-[11px]">
                      {m.name}
                    </div>
                    <div className="text-slate-500 text-[11px] flex justify-between pt-1 border-t border-slate-200 dark:border-slate-700">
                      <span>Threshold:</span>
                      <span className="font-mono font-bold">{m.decision_threshold}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* User Feedback Audit Log */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
            <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-brand-600" />
                Recent User Verification Feedback
              </h3>
            </div>

            {fbLoading ? (
              <div className="p-8 text-center text-xs text-slate-500">Loading feedback log...</div>
            ) : fbError ? (
              <div className="p-8 text-center text-xs text-rose-500">Failed to load feedback records.</div>
            ) : feedbackData?.items.length === 0 ? (
              <div className="p-8 text-center text-xs text-slate-400">No feedback submitted yet.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 dark:bg-slate-800/60 border-b border-slate-200 dark:border-slate-800 text-slate-500 font-semibold uppercase tracking-wider">
                    <tr>
                      <th className="py-3 px-4">Analysis ID</th>
                      <th className="py-3 px-4">Accuracy</th>
                      <th className="py-3 px-4">Reported Label</th>
                      <th className="py-3 px-4">User Notes</th>
                      <th className="py-3 px-4">Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-700 dark:text-slate-300">
                    {feedbackData?.items.map((fb) => (
                      <tr key={fb.id} className="hover:bg-slate-50/60 dark:hover:bg-slate-800/40">
                        <td className="py-3 px-4 font-mono text-[11px] text-brand-600 dark:text-brand-400">
                          {fb.analysis_id.slice(0, 16)}...
                        </td>
                        <td className="py-3 px-4">
                          {fb.is_correct ? (
                            <span className="inline-flex items-center gap-1 text-emerald-600 font-semibold">
                              <ThumbsUp className="w-3.5 h-3.5" /> Correct
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-rose-600 font-semibold">
                              <ThumbsDown className="w-3.5 h-3.5" /> Inaccurate
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 capitalize">
                          {fb.reported_label || 'Confirmed by user'}
                        </td>
                        <td className="py-3 px-4 max-w-xs truncate text-slate-500">
                          {fb.comment || '—'}
                        </td>
                        <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                          {new Date(fb.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {feedbackData && feedbackData.total > 15 && (
              <div className="flex items-center justify-between p-3 border-t border-slate-100 dark:border-slate-800 text-xs text-slate-500">
                <span>Page {page} of {Math.ceil(feedbackData.total / 15)}</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page <= 1}
                    className="p-1 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setPage((p) => p + 1)}
                    disabled={page >= Math.ceil(feedbackData.total / 15)}
                    className="p-1 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
