import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Clock,
  Search,
  Trash2,
  Bookmark,
  ExternalLink,
  ImageIcon,
  Music,
  FileText,
  AlertCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useAnalyses, useDeleteAnalysis } from '../api/hooks';

export const HistoryPage: React.FC = () => {
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedModality, setSelectedModality] = useState<string>('');
  const [selectedStatus, setSelectedStatus] = useState<string>('');

  const { data, isLoading, error } = useAnalyses({
    page,
    page_size: 15,
    modality: selectedModality || undefined,
    status: selectedStatus || undefined,
    q: searchQuery.trim() || undefined,
  });

  const deleteMutation = useDeleteAnalysis();

  const getModalityIcon = (mod: string | null) => {
    switch (mod) {
      case 'image':
        return <ImageIcon className="w-4 h-4 text-emerald-500" />;
      case 'audio':
        return <Music className="w-4 h-4 text-indigo-500" />;
      case 'text':
        return <FileText className="w-4 h-4 text-violet-500" />;
      default:
        return <Clock className="w-4 h-4 text-slate-400" />;
    }
  };

  const getVerdictBadge = (verdict?: string, label?: string) => {
    if (verdict === 'likely_real') {
      return (
        <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-teal-50 dark:bg-teal-950/60 text-teal-700 dark:text-teal-400 border border-teal-200 dark:border-teal-850">
          {label || 'Likely Real'}
        </span>
      );
    }
    if (verdict === 'likely_fake') {
      return (
        <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-400 border border-rose-200 dark:border-rose-850">
          {label || 'Likely AI'}
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-850">
        {label || 'Uncertain'}
      </span>
    );
  };

  const handleDelete = async (id: string) => {
    if (window.confirm('Are you sure you want to delete this analysis record and all associated artefacts?')) {
      await deleteMutation.mutateAsync(id);
    }
  };

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Analysis History</h1>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Review previous forensic scans, retention statuses, and generated reports.
        </p>
      </div>

      {/* Filters & Search Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="relative flex-1 min-w-[220px]">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by file name or ID..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setPage(1);
            }}
            className="w-full pl-9 pr-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-xs text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500"
          />
        </div>

        <div className="flex items-center gap-2">
          {/* Modality Filter */}
          <select
            value={selectedModality}
            onChange={(e) => {
              setSelectedModality(e.target.value);
              setPage(1);
            }}
            className="px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-xs text-slate-700 dark:text-slate-300"
          >
            <option value="">All Modalities</option>
            <option value="image">Image</option>
            <option value="audio">Audio</option>
            <option value="text">Text</option>
          </select>

          {/* Status Filter */}
          <select
            value={selectedStatus}
            onChange={(e) => {
              setSelectedStatus(e.target.value);
              setPage(1);
            }}
            className="px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-xs text-slate-700 dark:text-slate-300"
          >
            <option value="">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="queued">Queued</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Table Container */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-12 flex flex-col items-center justify-center space-y-3">
            <Loader2 className="w-6 h-6 animate-spin text-brand-600" />
            <span className="text-xs text-slate-500">Loading analysis records…</span>
          </div>
        ) : error ? (
          <div className="p-12 text-center text-xs text-rose-600 space-y-1">
            <AlertCircle className="w-5 h-5 mx-auto" />
            <p>Failed to load analyses.</p>
          </div>
        ) : data?.items.length === 0 ? (
          <div className="p-16 text-center space-y-3">
            <Clock className="w-8 h-8 mx-auto text-slate-400" />
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              No analysis records found
            </p>
            <p className="text-xs text-slate-500">
              Try adjusting search filters or launch a new analysis scan.
            </p>
            <Link
              to="/analyze"
              className="inline-block px-4 py-2 rounded-lg bg-brand-600 text-white text-xs font-semibold mt-2"
            >
              Analyze Media
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-800/60 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-semibold">
                <tr>
                  <th className="py-3 px-4">Modality</th>
                  <th className="py-3 px-4">Subject / File</th>
                  <th className="py-3 px-4">Verdict</th>
                  <th className="py-3 px-4">Probability</th>
                  <th className="py-3 px-4">Retention</th>
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-700 dark:text-slate-300">
                {data?.items.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors">
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2">
                        {getModalityIcon(item.modality)}
                        <span className="capitalize font-medium">{item.modality || 'Unknown'}</span>
                      </div>
                    </td>

                    <td className="py-3.5 px-4 font-medium max-w-[200px] truncate">
                      {item.file_name || item.id.slice(0, 16)}
                    </td>

                    <td className="py-3.5 px-4">
                      {item.result ? (
                        getVerdictBadge(item.result.verdict, item.result.verdict_label)
                      ) : (
                        <span className="text-slate-400 capitalize">{item.status}</span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 font-mono">
                      {item.result ? `${(item.result.probability * 100).toFixed(0)}%` : '-'}
                    </td>

                    <td className="py-3.5 px-4">
                      {item.retain_media ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-brand-600 dark:text-brand-400">
                          <Bookmark className="w-3 h-3 fill-current" /> Retained
                        </span>
                      ) : (
                        <span className="text-[11px] text-slate-400">24h Purge</span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 text-slate-500 font-mono text-[11px]">
                      {item.created_at ? new Date(item.created_at).toLocaleDateString() : 'N/A'}
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      <div className="inline-flex items-center gap-1">
                        <Link
                          to={`/results/${item.id}`}
                          className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 text-brand-600 dark:text-brand-400"
                          title="View Result"
                        >
                          <ExternalLink className="w-4 h-4" />
                        </Link>
                        <button
                          onClick={() => handleDelete(item.id)}
                          className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-rose-600"
                          title="Delete Analysis"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Bar */}
        {data && totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-slate-100 dark:border-slate-800 text-xs text-slate-500">
            <span>
              Showing page {page} of {totalPages} ({data.total} records)
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="p-1.5 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="p-1.5 rounded border border-slate-200 dark:border-slate-700 disabled:opacity-40"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
