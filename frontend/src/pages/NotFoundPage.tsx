import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldAlert, ArrowLeft, Search } from 'lucide-react';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="max-w-xl mx-auto px-4 py-20 text-center space-y-6">
      <div className="w-16 h-16 rounded-2xl bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400 border border-brand-200 dark:border-brand-800 flex items-center justify-center mx-auto shadow-sm">
        <ShieldAlert className="w-8 h-8" />
      </div>
      <div>
        <p className="text-xs uppercase tracking-widest font-mono text-brand-600 dark:text-brand-400 font-semibold">
          Error 404
        </p>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white mt-2">
          Page Not Found
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-2 max-w-md mx-auto">
          The forensic report, analysis session, or endpoint you are trying to access does not exist or has expired.
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-brand-600 hover:bg-brand-700 text-white text-sm font-semibold transition-colors shadow-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>
        <Link
          to="/analyze"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-200 text-sm font-medium transition-colors"
        >
          <Search className="w-4 h-4" />
          Analyze Media
        </Link>
      </div>
    </div>
  );
};
