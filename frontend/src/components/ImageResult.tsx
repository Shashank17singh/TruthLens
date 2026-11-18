import React, { useState } from 'react';
import { Eye, Sliders, Layers, Activity, AlertCircle } from 'lucide-react';
import type { ImageDetails } from '../api/types';
import { api } from '../api/client';

interface ImageResultProps {
  analysisId: string;
  details?: ImageDetails | Record<string, unknown>;
  originalImageUrl?: string;
}

export const ImageResult: React.FC<ImageResultProps> = ({
  analysisId,
  details,
  originalImageUrl,
}) => {
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(70);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'overlay' | 'sideBySide' | 'spectrum'>('overlay');

  const imgDetails = details as ImageDetails | undefined;
  const hasHeatmap = Boolean(imgDetails?.artefacts?.heatmap);
  const hasSpectrum = Boolean(imgDetails?.artefacts?.spectrum);
  const regions = imgDetails?.regions || [];

  const heatmapUrl = hasHeatmap ? api.getArtefactUrl(analysisId, 'heatmap') : undefined;
  const spectrumUrl = hasSpectrum ? api.getArtefactUrl(analysisId, 'spectrum') : undefined;
  const originalSrc = originalImageUrl || api.getArtefactUrl(analysisId, 'input');

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
        {/* Tab Selection */}
        <div className="flex items-center gap-1 bg-slate-200/80 dark:bg-slate-700/80 p-1 rounded-lg">
          <button
            onClick={() => setActiveTab('overlay')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'overlay'
                ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-400 shadow-sm'
                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Heatmap Overlay
          </button>
          <button
            onClick={() => setActiveTab('sideBySide')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === 'sideBySide'
                ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-400 shadow-sm'
                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            Side-by-Side
          </button>
          {hasSpectrum && (
            <button
              onClick={() => setActiveTab('spectrum')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                activeTab === 'spectrum'
                  ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-400 shadow-sm'
                  : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              Frequency Spectrum
            </button>
          )}
        </div>

        {/* Heatmap Opacity & Bounding Box Toggles */}
        <div className="flex items-center gap-4 text-xs font-medium text-slate-700 dark:text-slate-300">
          {activeTab === 'overlay' && hasHeatmap && (
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <label htmlFor="heatmap-opacity-slider" className="text-slate-500 dark:text-slate-400">
                Heatmap Opacity:
              </label>
              <input
                id="heatmap-opacity-slider"
                type="range"
                min="0"
                max="100"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                className="w-24 accent-brand-600 cursor-pointer"
              />
              <span className="w-7 text-right font-mono">{heatmapOpacity}%</span>
            </div>
          )}

          {regions.length > 0 && activeTab !== 'spectrum' && (
            <label className="flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showBoundingBoxes}
                onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                className="rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              <span>Highlight Regions ({regions.length})</span>
            </label>
          )}
        </div>
      </div>

      {/* Main Visualizer Area */}
      {activeTab === 'overlay' && (
        <div className="relative rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center min-h-[360px] max-h-[560px]">
          {/* Base image */}
          <img
            src={originalSrc}
            alt="Analyzed media"
            className="w-full h-full object-contain max-h-[540px]"
            onError={(e) => {
              // fallback placeholder if image cannot be rendered directly
              (e.target as HTMLElement).style.display = 'none';
            }}
          />

          {/* Grad-CAM Heatmap Overlay */}
          {hasHeatmap && heatmapUrl && (
            <img
              src={heatmapUrl}
              alt="Grad-CAM Saliency Heatmap"
              className="absolute inset-0 w-full h-full object-contain max-h-[540px] pointer-events-none transition-opacity duration-150 mix-blend-screen"
              style={{ opacity: heatmapOpacity / 100 }}
            />
          )}

          {/* Bounding Boxes */}
          {showBoundingBoxes &&
            regions.map((region, idx) => {
              const [left, top, right, bottom] = region.box;
              const width = right - left;
              const height = bottom - top;
              return (
                <div
                  key={idx}
                  className="absolute border-2 border-rose-500 bg-rose-500/10 pointer-events-none rounded transition-all"
                  style={{
                    left: `${left * 100}%`,
                    top: `${top * 100}%`,
                    width: `${width * 100}%`,
                    height: `${height * 100}%`,
                  }}
                >
                  <span className="absolute -top-5 left-0 px-1.5 py-0.5 rounded bg-rose-600 text-[10px] font-bold text-white shadow-sm whitespace-nowrap">
                    {region.part} ({(region.score * 100).toFixed(0)}%)
                  </span>
                </div>
              );
            })}
        </div>
      )}

      {activeTab === 'sideBySide' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex flex-col">
            <div className="px-3 py-2 bg-slate-900 border-b border-slate-800 text-xs font-semibold text-slate-300">
              Original Media
            </div>
            <div className="p-2 flex items-center justify-center min-h-[280px]">
              <img
                src={originalSrc}
                alt="Original media"
                className="w-full h-auto object-contain max-h-[400px] rounded"
              />
            </div>
          </div>

          <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex flex-col">
            <div className="px-3 py-2 bg-slate-900 border-b border-slate-800 text-xs font-semibold text-slate-300 flex items-center justify-between">
              <span>Grad-CAM Attribution Map</span>
              <span className="text-[10px] text-brand-400">Warm = High AI Contribution</span>
            </div>
            <div className="p-2 flex items-center justify-center min-h-[280px]">
              {hasHeatmap && heatmapUrl ? (
                <img
                  src={heatmapUrl}
                  alt="Grad-CAM attribution"
                  className="w-full h-auto object-contain max-h-[400px] rounded"
                />
              ) : (
                <div className="text-xs text-slate-500">Heatmap artefact not generated for this model.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'spectrum' && (
        <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-900 text-slate-100 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h4 className="text-sm font-bold flex items-center gap-2 text-white">
                <Activity className="w-4 h-4 text-brand-400" />
                2D Discrete Cosine Transform (DCT) Frequency Domain
              </h4>
              <p className="text-xs text-slate-400 mt-0.5">
                Evaluates periodic grid artifacts and high-frequency energy anomalies typical of generative diffusion and GAN upsamplers.
              </p>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center gap-6 justify-center">
            {spectrumUrl ? (
              <img
                src={spectrumUrl}
                alt="DCT Radial Frequency Spectrum"
                className="w-72 h-72 rounded-xl border border-slate-700 bg-black object-contain shadow-lg"
              />
            ) : (
              <div className="w-72 h-72 rounded-xl border border-dashed border-slate-800 flex items-center justify-center text-xs text-slate-500">
                Spectrum visualization unavailable
              </div>
            )}

            <div className="flex-1 space-y-3 text-xs text-slate-300 max-w-md">
              <div className="p-3 rounded-lg bg-slate-800/80 border border-slate-700/80">
                <span className="font-semibold text-brand-300">Spectral Analysis:</span>
                <p className="mt-1 text-slate-300">
                  Natural photography exhibits smooth <span className="font-mono">1/f²</span> power falloff from center (low frequencies) to edges. Generative pipelines frequently display checkerboard spikes or unnatural high-frequency attenuation.
                </p>
              </div>

              {imgDetails?.face_detected && (
                <div className="flex items-start gap-2 p-3 rounded-lg bg-indigo-950/40 border border-indigo-800/40 text-indigo-300">
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>
                    Focal face detected. Frequency inspection focuses on blending boundaries around hairline, jawline, and eye regions.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
