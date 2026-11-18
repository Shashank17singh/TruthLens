import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, X, Check, Loader2 } from 'lucide-react';
import { useSubmitFeedback } from '../api/hooks';

interface FeedbackModalProps {
  analysisId: string;
  isOpen: boolean;
  onClose: () => void;
}

export const FeedbackModal: React.FC<FeedbackModalProps> = ({
  analysisId,
  isOpen,
  onClose,
}) => {
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [reportedLabel, setReportedLabel] = useState<'real' | 'fake' | 'unsure'>('real');
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const mutation = useSubmitFeedback(analysisId);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isCorrect === null) return;

    try {
      await mutation.mutateAsync({
        is_correct: isCorrect,
        reported_label: isCorrect ? undefined : reportedLabel,
        comment: comment.trim() || undefined,
      });
      setSubmitted(true);
      setTimeout(() => {
        onClose();
        setSubmitted(false);
        setIsCorrect(null);
        setComment('');
      }, 1400);
    } catch {
      // Error handled by mutation state
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-md rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xl p-6 space-y-5">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
        >
          <X className="w-5 h-5" />
        </button>

        {submitted ? (
          <div className="py-8 flex flex-col items-center justify-center text-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-600 flex items-center justify-center">
              <Check className="w-6 h-6" />
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Thank You for Your Feedback!
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-xs">
              Your feedback is securely recorded to continuously evaluate and calibrate our detection confidence intervals.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Was this verdict accurate?
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Help calibrate model thresholds by confirming or correcting the forensic classification.
              </p>
            </div>

            {/* Thumbs Selection */}
            <div className="grid grid-cols-2 gap-3 pt-2">
              <button
                type="button"
                onClick={() => setIsCorrect(true)}
                className={`flex items-center justify-center gap-2 p-3 rounded-xl border text-sm font-semibold transition-all ${
                  isCorrect === true
                    ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400 ring-2 ring-emerald-500/30'
                    : 'border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/60 text-slate-700 dark:text-slate-300'
                }`}
              >
                <ThumbsUp className="w-4 h-4 text-emerald-600" />
                Yes, Accurate
              </button>

              <button
                type="button"
                onClick={() => setIsCorrect(false)}
                className={`flex items-center justify-center gap-2 p-3 rounded-xl border text-sm font-semibold transition-all ${
                  isCorrect === false
                    ? 'border-rose-500 bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-400 ring-2 ring-rose-500/30'
                    : 'border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/60 text-slate-700 dark:text-slate-300'
                }`}
              >
                <ThumbsDown className="w-4 h-4 text-rose-600" />
                No, Inaccurate
              </button>
            </div>

            {/* If Inaccurate, ask for corrected label */}
            {isCorrect === false && (
              <div className="space-y-2 pt-2 animate-in fade-in duration-200">
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                  What is the true ground truth label?
                </label>
                <select
                  value={reportedLabel}
                  onChange={(e) => setReportedLabel(e.target.value as 'real' | 'fake' | 'unsure')}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-brand-500"
                >
                  <option value="real">Real / Authentic (Human Created)</option>
                  <option value="fake">AI-Generated / Synthetic / Deepfake</option>
                  <option value="unsure">Unsure / Ambiguous</option>
                </select>
              </div>
            )}

            {/* Comment */}
            {isCorrect !== null && (
              <div className="space-y-1.5 pt-1">
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Additional Notes (Optional)
                </label>
                <textarea
                  rows={2}
                  maxLength={500}
                  placeholder="e.g., Known generator was Midjourney v6, or authentic voice recorded on iPhone..."
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-brand-500"
                />
              </div>
            )}

            {mutation.isError && (
              <p className="text-xs text-rose-600">
                {(mutation.error as Error)?.message || 'Failed to submit feedback.'}
              </p>
            )}

            {/* Submit Button */}
            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isCorrect === null || mutation.isPending}
                className="px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-1.5 shadow"
              >
                {mutation.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                Submit Feedback
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
