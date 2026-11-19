import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, Volume2, RotateCcw, AlertTriangle } from 'lucide-react';
import type { AudioDetails, AudioWindow } from '../api/types';
import { api } from '../api/client';

interface AudioResultProps {
  analysisId: string;
  details?: AudioDetails | Record<string, unknown>;
  audioUrl?: string;
}

export const AudioResult: React.FC<AudioResultProps> = ({
  analysisId,
  details,
  audioUrl,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [isReady, setIsReady] = useState<boolean>(false);
  const [activeWindow, setActiveWindow] = useState<AudioWindow | null>(null);

  const audioDetails = details as AudioDetails | undefined;
  const windows: AudioWindow[] = audioDetails?.windows || [];
  const src = audioUrl || api.getArtefactUrl(analysisId, 'input');

  useEffect(() => {
    if (!containerRef.current) return;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#94a3b8',
      progressColor: '#3b82f6',
      cursorColor: '#2563eb',
      barWidth: 3,
      barGap: 2,
      barRadius: 2,
      height: 90,
      url: src,
    });

    wavesurferRef.current = ws;

    ws.on('ready', () => {
      setIsReady(true);
      setDuration(ws.getDuration());
    });

    ws.on('audioprocess', (time: number) => {
      setCurrentTime(time);
      // Determine active window
      const match = windows.find((w) => time >= w.start && time < w.end);
      setActiveWindow(match || null);
    });

    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    ws.on('finish', () => {
      setIsPlaying(false);
      setCurrentTime(0);
    });

    return () => {
      ws.destroy();
    };
  }, [src, windows]);

  const togglePlay = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  };

  const handleSeekWindow = (win: AudioWindow) => {
    if (wavesurferRef.current) {
      wavesurferRef.current.setTime(win.start);
      wavesurferRef.current.play();
      setActiveWindow(win);
    }
  };

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainder = Math.floor(secs % 60);
    return `${mins}:${remainder.toString().padStart(2, '0')}`;
  };

  const getWindowColor = (prob: number) => {
    if (prob > 0.6) return 'bg-rose-500/80 hover:bg-rose-600 border-rose-600';
    if (prob >= 0.4) return 'bg-amber-500/80 hover:bg-amber-600 border-amber-600';
    return 'bg-teal-500/80 hover:bg-teal-600 border-teal-600';
  };

  return (
    <div className="space-y-6">
      {/* Waveform Card */}
      <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="flex items-center justify-between mb-3 text-xs font-medium text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1.5">
            <Volume2 className="w-4 h-4 text-brand-500" />
            Acoustic Signal & LFCC Feature Timeline
          </span>
          <span className="font-mono">
            {formatTime(currentTime)} / {formatTime(duration || audioDetails?.duration_seconds || 0)}
          </span>
        </div>

        {/* Waveform Container */}
        <div ref={containerRef} className="w-full cursor-pointer py-1" />

        {/* Playback Controls */}
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlay}
              disabled={!isReady}
              className="flex items-center justify-center w-10 h-10 rounded-full bg-brand-600 hover:bg-brand-700 text-white shadow transition-colors active:scale-95 disabled:opacity-50"
              aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
            </button>
            <button
              onClick={() => {
                if (wavesurferRef.current) {
                  wavesurferRef.current.setTime(0);
                }
              }}
              disabled={!isReady}
              className="p-2 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors active:scale-95"
              title="Restart"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {activeWindow && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs">
              <span className="text-slate-400">
                Segment [<span className="font-mono">{activeWindow.start.toFixed(1)}</span>s - <span className="font-mono">{activeWindow.end.toFixed(1)}</span>s]:
              </span>
              <span
                className={`font-semibold ${
                  activeWindow.probability > 0.6
                    ? 'text-rose-600 dark:text-rose-400'
                    : activeWindow.probability >= 0.4
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-teal-600 dark:text-teal-400'
                }`}
              >
                <span className="font-mono">{(activeWindow.probability * 100).toFixed(0)}%</span> AI
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Per-Window Temporal Timeline */}
      {windows.length > 0 && (
        <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              Temporal window heatmap (4-second segments)
            </h4>
            <div className="flex items-center gap-3 text-[11px] text-slate-500">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-500" /> &lt;40% (Organic)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> 40–60% (Uncertain)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500" /> &gt;60% (Synthetic)
              </span>
            </div>
          </div>

          {/* Timeline Bar */}
          <div className="w-full flex h-8 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 p-0.5 gap-0.5">
            {windows.map((win, idx) => {
              const totalSecs = duration || audioDetails?.duration_seconds || 1;
              const winWidth = ((win.end - win.start) / totalSecs) * 100;
              const isSelected = activeWindow?.start === win.start;
              return (
                <button
                  key={idx}
                  onClick={() => handleSeekWindow(win)}
                  style={{ width: `${winWidth}%` }}
                  title={`Segment ${idx + 1}: ${win.start.toFixed(1)}s-${win.end.toFixed(1)}s (${(
                    win.probability * 100
                  ).toFixed(0)}% AI)`}
                  className={`h-full rounded transition-opacity hover:opacity-90 cursor-pointer border ${getWindowColor(
                    win.probability
                  )} ${isSelected ? 'ring-2 ring-brand-500 ring-offset-1' : ''}`}
                />
              );
            })}
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Click any segment above to jump playback to that window. The LCNN model evaluates Linear Frequency Cepstral Coefficients (LFCC) across each 4.0-second slice.
          </p>
        </div>
      )}

      {/* Aggregate Stats */}
      {audioDetails?.aggregate && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60">
            <span className="text-[11px] font-medium text-slate-500">Mean probability</span>
            <div className="text-lg font-bold font-mono text-slate-800 dark:text-slate-200 mt-0.5">
              {(audioDetails.aggregate.mean * 100).toFixed(1)}%
            </div>
          </div>
          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60">
            <span className="text-[11px] font-medium text-slate-500">Peak synthetic window</span>
            <div className="text-lg font-bold font-mono text-rose-600 dark:text-rose-400 mt-0.5 flex items-center gap-1">
              <AlertTriangle className="w-4 h-4" />
              {(audioDetails.aggregate.max * 100).toFixed(1)}%
            </div>
          </div>
          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 col-span-2 sm:col-span-1">
            <span className="text-[11px] font-medium text-slate-500">Total slices</span>
            <div className="text-lg font-bold font-mono text-slate-800 dark:text-slate-200 mt-0.5">
              {windows.length} windows
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
