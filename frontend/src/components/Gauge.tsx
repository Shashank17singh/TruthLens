import React from 'react';
import { ShieldCheck, AlertTriangle, HelpCircle } from 'lucide-react';
import type { ConfidenceBand, Verdict } from '../api/types';

interface GaugeProps {
  probability: number; // 0.0 to 1.0
  confidenceBand?: ConfidenceBand;
  verdict?: Verdict;
  verdictLabel?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const Gauge: React.FC<GaugeProps> = ({
  probability,
  confidenceBand,
  verdict = 'uncertain',
  verdictLabel,
  size = 'md',
  className = '',
}) => {
  // Clamp to 0.0 - 1.0
  const probClamped = Math.max(0, Math.min(1, probability));
  const percent = Math.round(probClamped * 100);

  // Default confidence bounds if not provided
  const low = confidenceBand ? Math.max(0, Math.min(1, confidenceBand.low)) : Math.max(0, probClamped - 0.06);
  const high = confidenceBand ? Math.max(0, Math.min(1, confidenceBand.high)) : Math.min(1, probClamped + 0.06);
  const lowPercent = Math.round(low * 100);
  const highPercent = Math.round(high * 100);

  // Verdict style & icon (accessible: icon + color + clear label)
  const getVerdictConfig = () => {
    switch (verdict) {
      case 'likely_real':
        return {
          label: verdictLabel || 'Likely Real',
          icon: ShieldCheck,
          textColor: 'text-teal-700 dark:text-teal-400',
          bgColor: 'bg-teal-50 dark:bg-teal-950/60',
          borderColor: 'border-teal-200 dark:border-teal-850',
          arcStroke: '#0d9488', // teal-600
          bandStroke: 'rgba(13, 148, 136, 0.25)',
        };
      case 'likely_fake':
        return {
          label: verdictLabel || 'Likely AI-Generated',
          icon: AlertTriangle,
          textColor: 'text-rose-700 dark:text-rose-400',
          bgColor: 'bg-rose-50 dark:bg-rose-950/60',
          borderColor: 'border-rose-200 dark:border-rose-850',
          arcStroke: '#e11d48', // rose-600
          bandStroke: 'rgba(225, 29, 72, 0.25)',
        };
      case 'uncertain':
      default:
        return {
          label: verdictLabel || 'Uncertain / Mixed',
          icon: HelpCircle,
          textColor: 'text-amber-700 dark:text-amber-400',
          bgColor: 'bg-amber-50 dark:bg-amber-950/60',
          borderColor: 'border-amber-200 dark:border-amber-850',
          arcStroke: '#d97706', // amber-600
          bandStroke: 'rgba(217, 119, 6, 0.25)',
        };
    }
  };

  const config = getVerdictConfig();
  const IconComponent = config.icon;

  // Gauge SVG calculations
  // Semicircle arc from -180 deg (left) to 0 deg (right)
  const radius = 80;
  const strokeWidth = 14;
  const cx = 100;
  const cy = 95;
  const arcLength = Math.PI * radius; // Half circumference ~ 251.32

  // Progress offsets
  const progressOffset = arcLength * (1 - probClamped);
  const bandStartOffset = arcLength * (1 - low);
  const bandEndOffset = arcLength * (1 - high);
  const bandLength = Math.max(2, bandStartOffset - bandEndOffset);

  const dimensionClass = {
    sm: 'w-40',
    md: 'w-56',
    lg: 'w-72',
  }[size];

  return (
    <div
      className={`flex flex-col items-center justify-center ${dimensionClass} ${className}`}
      role="meter"
      aria-label="Synthetic Probability Gauge"
      aria-valuenow={percent}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuetext={`${config.label}: ${percent}% synthetic probability, 95% confidence interval [${lowPercent}%, ${highPercent}%]`}
    >
      <div className="relative w-full aspect-[200/125]">
        <svg
          viewBox="0 0 200 120"
          className="w-full h-full overflow-visible"
        >
          {/* Background Arc */}
          <path
            d="M 20 95 A 80 80 0 0 1 180 95"
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            className="text-slate-200 dark:text-slate-800"
          />

          {/* Confidence Band Segment */}
          <path
            d="M 20 95 A 80 80 0 0 1 180 95"
            fill="none"
            stroke={config.bandStroke}
            strokeWidth={strokeWidth + 6}
            strokeDasharray={`${arcLength}`}
            strokeDashoffset={bandEndOffset}
            style={{
              strokeDasharray: `${bandLength} ${arcLength}`,
              strokeDashoffset: -bandEndOffset,
            }}
            strokeLinecap="butt"
          />

          {/* Active Value Arc */}
          <path
            d="M 20 95 A 80 80 0 0 1 180 95"
            fill="none"
            stroke={config.arcStroke}
            strokeWidth={strokeWidth}
            strokeDasharray={arcLength}
            strokeDashoffset={progressOffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />

          {/* Ticks: 0%, 50%, 100% */}
          <text x="20" y="112" textAnchor="middle" className="text-[10px] font-medium fill-slate-400 dark:fill-slate-500">
            0%
          </text>
          <text x="100" y="24" textAnchor="middle" className="text-[10px] font-medium fill-slate-400 dark:fill-slate-500">
            50%
          </text>
          <text x="180" y="112" textAnchor="middle" className="text-[10px] font-medium fill-slate-400 dark:fill-slate-500">
            100%
          </text>

          {/* Center Text */}
          <text
            x={cx}
            y={cy - 12}
            textAnchor="middle"
            className="text-3xl font-extrabold fill-slate-900 dark:fill-white font-mono tracking-tight"
          >
            {percent}%
          </text>
          <text
            x={cx}
            y={cy + 4}
            textAnchor="middle"
            className="text-[10px] font-semibold tracking-wider uppercase fill-slate-500 dark:fill-slate-400"
          >
            AI Probability
          </text>
        </svg>
      </div>

      {/* Confidence Interval Badge */}
      <div className="mt-1 flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
        <span className="font-medium">95% CI:</span>
        <span className="font-mono text-slate-700 dark:text-slate-300">
          [{lowPercent}% – {highPercent}%]
        </span>
      </div>

      {/* Accessible Verdict Badge with Icon */}
      <div
        className={`mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${config.bgColor} ${config.textColor} ${config.borderColor} shadow-sm`}
      >
        <IconComponent className="w-4 h-4 flex-shrink-0" />
        <span>{config.label}</span>
      </div>
    </div>
  );
};
