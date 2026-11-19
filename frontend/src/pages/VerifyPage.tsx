import React, { useState } from 'react';
import {
  FileCheck2,
  AlertTriangle,
  UploadCloud,
  Loader2,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { useVerifyReport } from '../api/hooks';
import type { VerifyResponse } from '../api/types';

export const VerifyPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [reportId, setReportId] = useState('');
  const [sha256, setSha256] = useState('');
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null);

  const verifyMutation = useVerifyReport();

  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file && !reportId.trim() && !sha256.trim()) return;

    try {
      const res = await verifyMutation.mutateAsync({
        file: file || undefined,
        reportId: reportId.trim() || undefined,
        sha256: sha256.trim() || undefined,
      });
      setVerifyResult(res);
    } catch {
      // handled
    }
  };

  const handleReset = () => {
    setFile(null);
    setReportId('');
    setSha256('');
    setVerifyResult(null);
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-8">
      {/* Title */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-brand-50 dark:bg-brand-950/60 border border-brand-200 dark:border-brand-800 text-brand-600 dark:text-brand-400 text-xs font-semibold">
          <FileCheck2 className="w-4 h-4" />
          Cryptographic Audit Verification
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
          Verify TruthLens Forensic PDF
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 max-w-lg mx-auto">
          Confirm authenticity of generated PDF reports using HMAC-SHA256 signature verification and tamper detection.
        </p>
      </div>

      {/* Main Verification Card */}
      <div className="p-6 sm:p-8 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xl space-y-6">
        <form onSubmit={handleVerify} className="space-y-5">
          {/* File Upload Box */}
          <div className="space-y-2">
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
              Option 1: Upload Forensic PDF Report
            </label>
            <div className="relative border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-2xl p-6 flex flex-col items-center justify-center text-center hover:bg-slate-50 dark:hover:bg-slate-800/40 cursor-pointer transition-colors">
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setFile(e.target.files[0]);
                  }
                }}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              <UploadCloud className="w-8 h-8 text-brand-500 mb-2" />
              {file ? (
                <div className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                  Selected: <span className="text-brand-600">{file.name}</span> ({(file.size / 1024).toFixed(0)} KB)
                </div>
              ) : (
                <>
                  <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                    Click or drag & drop TruthLens PDF report
                  </p>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Validates cryptographic footer signature and embedded QR checksum
                  </p>
                </>
              )}
            </div>
          </div>

          <div className="relative flex py-2 items-center">
            <div className="flex-grow border-t border-slate-200 dark:border-slate-800" />
            <span className="flex-shrink mx-4 text-xs font-medium text-slate-400">OR</span>
            <div className="flex-grow border-t border-slate-200 dark:border-slate-800" />
          </div>

          {/* Manual Input Fields */}
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Option 2: Enter Report ID
              </label>
              <input
                type="text"
                placeholder="rep_019318b8-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                value={reportId}
                onChange={(e) => setReportId(e.target.value)}
                className="w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-mono text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Document SHA-256 Hash (Optional)
              </label>
              <input
                type="text"
                placeholder="64-character hexadecimal SHA-256 checksum..."
                value={sha256}
                onChange={(e) => setSha256(e.target.value)}
                className="w-full px-3 py-2 rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-mono text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500"
              />
            </div>
          </div>

          {verifyMutation.isError && (
            <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900 text-xs text-rose-700 dark:text-rose-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{(verifyMutation.error as Error)?.message || 'Verification failed.'}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-3 pt-2">
            {(file || reportId || sha256 || verifyResult) && (
              <button
                type="button"
                onClick={handleReset}
                className="px-4 py-2.5 rounded-xl text-xs font-medium text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
              >
                Clear
              </button>
            )}
            <button
              type="submit"
              disabled={verifyMutation.isPending || (!file && !reportId.trim() && !sha256.trim())}
              className="px-6 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white font-bold text-xs shadow flex items-center gap-2 transition-colors whitespace-nowrap"
            >
              {verifyMutation.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
              Verify Cryptographic Signature
            </button>
          </div>
        </form>

        {/* Verification Result Seal */}
        {verifyResult && (
          <div
            className={`p-6 rounded-2xl border transition-colors motion-safe:animate-in motion-safe:zoom-in-95 duration-200 ${
              verifyResult.genuine
                ? 'bg-emerald-50/60 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-800'
                : 'bg-rose-50/60 dark:bg-rose-950/30 border-rose-300 dark:border-rose-800'
            }`}
          >
            <div className="flex items-start gap-4">
              <div
                className={`w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0 shadow ${
                  verifyResult.genuine
                    ? 'bg-emerald-600 text-white'
                    : 'bg-rose-600 text-white'
                }`}
              >
                {verifyResult.genuine ? (
                  <CheckCircle2 className="w-7 h-7" />
                ) : (
                  <XCircle className="w-7 h-7" />
                )}
              </div>

              <div className="space-y-2 flex-1">
                <div className="flex items-center justify-between">
                  <h3
                    className={`text-base font-extrabold ${
                      verifyResult.genuine
                        ? 'text-emerald-900 dark:text-emerald-200'
                        : 'text-rose-900 dark:text-rose-200'
                    }`}
                  >
                    {verifyResult.genuine
                      ? 'Authentic Genuine TruthLens Report'
                      : 'Cryptographic Verification Failed'}
                  </h3>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      verifyResult.genuine
                        ? 'bg-emerald-200/80 dark:bg-emerald-900 text-emerald-900 dark:text-emerald-100'
                        : 'bg-rose-200/80 dark:bg-rose-900 text-rose-900 dark:text-rose-100'
                    }`}
                  >
                    {verifyResult.genuine ? 'Valid Signature' : 'Tampered / Unrecognized'}
                  </span>
                </div>

                <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
                  {verifyResult.message ||
                    (verifyResult.genuine
                      ? 'This document was verified against the TruthLens master signing registry. The content hash, metadata, and timestamps are unaltered.'
                      : 'The document checksum or HMAC signature does not match any record issued by TruthLens.')}
                </p>

                {verifyResult.report_id && (
                  <div className="pt-2 border-t border-slate-200/60 dark:border-slate-800/60 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                    <div className="text-slate-600 dark:text-slate-400">
                      <span className="font-semibold block text-[10px] text-slate-400">
                        Report ID:
                      </span>
                      <span className="font-mono">{verifyResult.report_id}</span>
                    </div>
                    {verifyResult.issued_at && (
                      <div className="text-slate-600 dark:text-slate-400">
                        <span className="font-semibold block text-[10px] text-slate-400">
                          Issued At:
                        </span>
                        <span className="font-mono">{new Date(verifyResult.issued_at).toLocaleString()}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
