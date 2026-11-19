import React, { useState } from 'react';
import {
  KeyRound,
  Plus,
  Trash2,
  Copy,
  Check,
  AlertTriangle,
  Code,
  Shield,
  Loader2,
} from 'lucide-react';
import { useApiKeys, useCreateApiKey, useRevokeApiKey } from '../api/hooks';
import type { NewApiKeyResponse } from '../api/types';

export const ApiKeysPage: React.FC = () => {
  const { data: keys, isLoading, error } = useApiKeys();
  const createMutation = useCreateApiKey();
  const revokeMutation = useRevokeApiKey();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [keyName, setKeyName] = useState('');
  const [rateLimit, setRateLimit] = useState(120);
  const [createdKeyData, setCreatedKeyData] = useState<NewApiKeyResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!keyName.trim()) return;

    try {
      const res = await createMutation.mutateAsync({
        name: keyName.trim(),
        rateLimit,
      });
      setCreatedKeyData(res);
      setKeyName('');
    } catch {
      // handled by mutation state
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRevoke = async (id: string, name: string) => {
    if (window.confirm(`Are you sure you want to revoke API key "${name}"? This cannot be undone.`)) {
      await revokeMutation.mutateAsync(id);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Title & Action */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <KeyRound className="w-6 h-6 text-brand-600" />
            API Keys & Developer Access
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Programmatically query TruthLens multi-modal inference via REST API endpoints.
          </p>
        </div>

        <button
          onClick={() => {
            setCreatedKeyData(null);
            setIsModalOpen(true);
          }}
          className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold shadow flex items-center gap-2 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Create API Key
        </button>
      </div>

      {/* Keys Table Card */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-12 flex flex-col items-center justify-center space-y-3">
            <Loader2 className="w-6 h-6 animate-spin text-brand-600" />
            <span className="text-xs text-slate-500">Loading API keys…</span>
          </div>
        ) : error ? (
          <div className="p-8 text-center text-xs text-rose-600 space-y-1">
            <AlertTriangle className="w-5 h-5 mx-auto" />
            <p>Failed to retrieve API keys. Please sign in.</p>
          </div>
        ) : keys?.length === 0 ? (
          <div className="p-16 text-center space-y-3">
            <KeyRound className="w-8 h-8 mx-auto text-slate-400" />
            <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
              No active API keys
            </p>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Create an API key to authenticate Python scripts, Chrome extensions, or CI/CD automated test pipelines.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-800/60 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-semibold">
                <tr>
                  <th className="py-3 px-4">Key Name</th>
                  <th className="py-3 px-4">Prefix</th>
                  <th className="py-3 px-4">Rate Limit</th>
                  <th className="py-3 px-4">Created Date</th>
                  <th className="py-3 px-4">Last Used</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-700 dark:text-slate-300">
                {keys?.map((k) => (
                  <tr key={k.id} className="hover:bg-slate-50/60 dark:hover:bg-slate-800/40 transition-colors">
                    <td className="py-3.5 px-4 font-semibold text-slate-900 dark:text-white">
                      {k.name}
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-600 dark:text-slate-400">
                      {k.key_prefix}••••••••
                    </td>
                    <td className="py-3.5 px-4 text-slate-600 dark:text-slate-400">
                      <span className="font-mono">{k.rate_limit_per_min}</span> req/min
                    </td>
                    <td className="py-3.5 px-4 text-slate-500 font-mono text-[11px]">
                      {new Date(k.created_at).toLocaleDateString()}
                    </td>
                    <td className="py-3.5 px-4 text-slate-500 font-mono text-[11px]">
                      {k.last_used_at ? new Date(k.last_used_at).toLocaleString() : 'Never'}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={() => handleRevoke(k.id, k.name)}
                        className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-rose-600 transition-colors"
                        title="Revoke Key"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Developer Integration Documentation */}
      <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
        <div className="flex items-center gap-2">
          <Code className="w-5 h-5 text-brand-600" />
          <h3 className="text-sm font-bold text-slate-900 dark:text-white">
            Developer Integration Example
          </h3>
        </div>
        <p className="text-xs text-slate-600 dark:text-slate-400">
          Pass your API key in the <code className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-brand-600 font-mono text-[11px]">X-API-Key</code> request header:
        </p>
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-200 overflow-x-auto leading-relaxed">
          <span className="text-slate-500"># Analyze image via cURL:</span>
          <br />
          curl -X POST &quot;https://truthlens.ai/api/v1/analyze/image&quot; \
          <br />
          &nbsp;&nbsp;-H &quot;X-API-Key: tl_live_xxxxxxxxxxxxxxxx&quot; \
          <br />
          &nbsp;&nbsp;-F &quot;file=@evidence.png&quot;
        </div>
      </div>

      {/* Create Key Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="relative w-full max-w-md rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xl p-6 space-y-5">
            {createdKeyData ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-emerald-600 font-bold text-base">
                  <Shield className="w-5 h-5" />
                  API Key Created Successfully
                </div>

                <div className="p-3 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-xs text-amber-800 dark:text-amber-300">
                  <AlertTriangle className="w-4 h-4 inline mr-1 text-amber-600" />
                  <strong>Important:</strong> Copy this key now. For security, TruthLens stores only an SHA-256 hash and you will never be able to view this full key again.
                </div>

                <div className="space-y-1">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Your New Secret Key:
                  </label>
                  <div className="flex items-center gap-2 p-2.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-800">
                    <span className="font-mono text-xs text-brand-600 dark:text-brand-400 select-all flex-1 truncate">
                      {createdKeyData.key}
                    </span>
                    <button
                      onClick={() => handleCopy(createdKeyData.key)}
                      className="px-2.5 py-1 rounded bg-brand-600 hover:bg-brand-700 text-white text-xs font-medium flex items-center gap-1 shadow-sm"
                    >
                      {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                </div>

                <button
                  onClick={() => setIsModalOpen(false)}
                  className="w-full py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 text-slate-800 dark:text-slate-200 font-semibold text-xs transition-colors"
                >
                  Done
                </button>
              </div>
            ) : (
              <form onSubmit={handleCreate} className="space-y-4">
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    Generate New API Key
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    API keys prefix starts with <code className="font-mono text-brand-600">tl_</code> and authenticate all v1 endpoints.
                  </p>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Key Identifier / Name
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g., CI Pipeline, Browser Extension, Research Script"
                    value={keyName}
                    onChange={(e) => setKeyName(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Rate Limit (requests per minute)
                  </label>
                  <input
                    type="number"
                    min={10}
                    max={600}
                    value={rateLimit}
                    onChange={(e) => setRateLimit(Number(e.target.value))}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500 font-mono"
                  />
                </div>

                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setIsModalOpen(false)}
                    className="px-4 py-2 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={createMutation.isPending || !keyName.trim()}
                    className="px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-1.5 shadow"
                  >
                    {createMutation.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                    Generate Key
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
