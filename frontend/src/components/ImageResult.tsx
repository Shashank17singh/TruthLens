import React, { useState, useEffect } from 'react';
import { Eye, Sliders, Layers, Activity, AlertCircle, Loader2, Image as ImageIcon, Scan } from 'lucide-react';
import type { ImageDetails } from '../api/types';
import { api } from '../api/client';

interface ImageWithFallbackProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  fallbackTitle?: string;
  fallbackSubtitle?: string;
  containerClassName?: string;
}

export const ImageWithFallback: React.FC<ImageWithFallbackProps> = ({
  src,
  alt,
  className = '',
  containerClassName = '',
  fallbackTitle = 'Forensic Asset Unavailable',
  fallbackSubtitle = 'The media file could not be loaded or may have been purged.',
  ...props
}) => {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoaded(false);
    setError(false);
  }, [src]);

  if (!src || error) {
    return (
      <div
        className={`flex flex-col items-center justify-center p-6 text-center rounded-xl bg-slate-900/80 border border-slate-800 text-slate-400 min-h-[240px] w-full ${containerClassName}`}
      >
        <div className="w-11 h-11 rounded-xl bg-slate-800/90 border border-slate-700/60 flex items-center justify-center mb-2.5 text-slate-400">
          <ImageIcon className="w-5 h-5 text-slate-500" />
        </div>
        <p className="text-xs font-semibold text-slate-300">{fallbackTitle}</p>
        <p className="text-[11px] text-slate-500 mt-1 max-w-xs">{fallbackSubtitle}</p>
      </div>
    );
  }

  return (
    <div className={`relative flex items-center justify-center overflow-hidden w-full h-full ${containerClassName}`}>
      {!loaded && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/70 backdrop-blur-sm z-10 transition-opacity">
          <Loader2 className="w-6 h-6 text-brand-500 animate-spin mb-2" />
          <span className="text-[11px] font-medium text-slate-400">Loading forensic asset…</span>
        </div>
      )}
      <img
        src={src}
        alt={alt}
        className={`transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'} ${className}`}
        onLoad={() => setLoaded(true)}
        onError={() => setError(true)}
        {...props}
      />
    </div>
  );
};

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
  const [heatmapError, setHeatmapError] = useState<boolean>(false);
  const [blobUrls, setBlobUrls] = useState<{
    input?: string;
    heatmap?: string;
    spectrum?: string;
  }>({});

  const imgDetails = details as ImageDetails | undefined;
  const hasHeatmap = Boolean(imgDetails?.artefacts?.heatmap);
  const hasSpectrum = Boolean(imgDetails?.artefacts?.spectrum);
  const regions = imgDetails?.regions || [];

  useEffect(() => {
    let isMounted = true;
    const fetchBlobs = async () => {
      const updates: { input?: string; heatmap?: string; spectrum?: string } = {};
      if (!originalImageUrl) {
        try {
          updates.input = await api.getArtefactBlobUrl(analysisId, 'input');
        } catch {
          updates.input = api.getArtefactUrl(analysisId, 'input');
        }
      }
      if (hasHeatmap) {
        try {
          updates.heatmap = await api.getArtefactBlobUrl(analysisId, 'heatmap');
        } catch {
          updates.heatmap = api.getArtefactUrl(analysisId, 'heatmap');
        }
      }
      if (hasSpectrum) {
        try {
          updates.spectrum = await api.getArtefactBlobUrl(analysisId, 'spectrum');
        } catch {
          updates.spectrum = api.getArtefactUrl(analysisId, 'spectrum');
        }
      }
      if (isMounted) {
        setBlobUrls(updates);
      }
    };
    fetchBlobs();
    return () => {
      isMounted = false;
    };
  }, [analysisId, hasHeatmap, hasSpectrum, originalImageUrl]);

  const heatmapUrl = blobUrls.heatmap || (hasHeatmap ? api.getArtefactUrl(analysisId, 'heatmap') : undefined);
  const spectrumUrl = blobUrls.spectrum || (hasSpectrum ? api.getArtefactUrl(analysisId, 'spectrum') : undefined);
  const originalSrc = originalImageUrl || blobUrls.input || api.getArtefactUrl(analysisId, 'input');

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-3 rounded-2xl bg-white dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 shadow-sm backdrop-blur">
        {/* Tab Selection */}
        <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800/80 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('overlay')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === 'overlay'
                ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-400 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Heatmap Overlay
          </button>
          <button
            onClick={() => setActiveTab('sideBySide')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === 'sideBySide'
                ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-400 shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            Side-by-Side
          </button>
          {hasSpectrum && (
            <button
              onClick={() => setActiveTab('spectrum')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                activeTab === 'spectrum'
                  ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-400 shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              Frequency Spectrum
            </button>
          )}
        </div>

        {/* Heatmap Opacity & Bounding Box Toggles */}
        <div className="flex flex-wrap items-center gap-4 text-xs font-medium text-slate-700 dark:text-slate-300">
          {activeTab === 'overlay' && hasHeatmap && (
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <label htmlFor="heatmap-opacity-slider" className="text-slate-500 dark:text-slate-400 text-xs">
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
              <span className="w-8 text-right font-mono font-semibold text-brand-600 dark:text-brand-400">
                {heatmapOpacity}%
              </span>
              <div className="flex items-center gap-1 pl-1">
                {[25, 50, 75, 100].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setHeatmapOpacity(preset)}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-colors ${
                      heatmapOpacity === preset
                        ? 'bg-brand-600 text-white font-bold'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-500 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    {preset}%
                  </button>
                ))}
              </div>
            </div>
          )}

          {regions.length > 0 && activeTab !== 'spectrum' && (
            <label className="flex items-center gap-1.5 cursor-pointer select-none bg-slate-50 dark:bg-slate-800/60 px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-700/60">
              <input
                type="checkbox"
                checked={showBoundingBoxes}
                onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                className="rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              <span className="text-xs font-semibold">Highlight Regions ({regions.length})</span>
            </label>
          )}
        </div>
      </div>

      {/* Main Visualizer Area */}
      {activeTab === 'overlay' && (
        <div className="relative rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center min-h-[380px] max-h-[580px] shadow-inner">
          {/* Subtle forensic grid in background */}
          <div className="absolute inset-0 bg-forensic-grid opacity-30 pointer-events-none" />

          {/* Base image with fallback */}
          <ImageWithFallback
            src={originalSrc}
            alt="Analyzed media"
            className="w-full h-full object-contain max-h-[560px]"
            containerClassName="max-h-[560px]"
            fallbackTitle="Original Payload Image Unavailable"
            fallbackSubtitle="Original uploaded image could not be loaded or was purged after 24h."
          />

          {/* Grad-CAM Heatmap Overlay */}
          {hasHeatmap && heatmapUrl && !heatmapError && (
            <img
              src={heatmapUrl}
              alt=""
              aria-hidden="true"
              onError={() => setHeatmapError(true)}
              className="absolute inset-0 w-full h-full object-contain max-h-[560px] pointer-events-none transition-opacity duration-150 mix-blend-screen"
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
                  className="absolute border-2 border-rose-500 bg-rose-500/10 pointer-events-none rounded transition-transform motion-safe:animate-pulse"
                  style={{
                    left: `${left * 100}%`,
                    top: `${top * 100}%`,
                    width: `${width * 100}%`,
                    height: `${height * 100}%`,
                  }}
                >
                  <span className="absolute -top-5 left-0 px-2 py-0.5 rounded bg-rose-600 text-[10px] font-bold text-white shadow-md whitespace-nowrap flex items-center gap-1">
                    <Scan className="w-2.5 h-2.5" />
                    {region.part} ({(region.score * 100).toFixed(0)}%)
                  </span>
                </div>
              );
            })}
        </div>
      )}

      {activeTab === 'sideBySide' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Left: Original Media */}
          <div className="rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex flex-col shadow-sm">
            <div className="px-4 py-2.5 bg-slate-900 border-b border-slate-800 text-xs font-semibold text-slate-300 flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <ImageIcon className="w-3.5 h-3.5 text-slate-400" />
                Original Media
              </span>
              <span className="text-[10px] text-slate-400">Pristine input</span>
            </div>
            <div className="p-3 flex items-center justify-center min-h-[300px] flex-1">
              <ImageWithFallback
                src={originalSrc}
                alt="Original media"
                className="w-full h-auto object-contain max-h-[420px] rounded-lg"
                fallbackTitle="Original Media Unavailable"
                fallbackSubtitle="File was purged or not found."
              />
            </div>
          </div>

          {/* Right: Grad-CAM Attribution Map */}
          <div className="rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex flex-col shadow-sm">
            <div className="px-4 py-2.5 bg-slate-900 border-b border-slate-800 text-xs font-semibold text-slate-300 flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Scan className="w-3.5 h-3.5 text-rose-400" />
                Grad-CAM Attribution Map
              </span>
              <span className="text-[10px] text-amber-400 font-medium">Warm = High AI Contribution</span>
            </div>
            <div className="p-3 flex items-center justify-center min-h-[300px] flex-1">
              {hasHeatmap && heatmapUrl ? (
                <ImageWithFallback
                  src={heatmapUrl}
                  alt="Grad-CAM attribution"
                  className="w-full h-auto object-contain max-h-[420px] rounded-lg"
                  fallbackTitle="Grad-CAM Heatmap Unavailable"
                  fallbackSubtitle="Attribution artefact could not be loaded."
                />
              ) : (
                <div className="text-xs text-slate-500 p-6 text-center">
                  Heatmap artefact not generated for this model.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'spectrum' && (
        <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-900/90 text-slate-100 space-y-5 shadow-lg backdrop-blur">
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
              <div className="w-72 h-72 rounded-2xl border border-slate-700 bg-black flex items-center justify-center p-2 shadow-2xl relative group">
                <ImageWithFallback
                  src={spectrumUrl}
                  alt="DCT Radial Frequency Spectrum"
                  className="w-full h-full object-contain rounded-xl"
                  fallbackTitle="Spectrum Unavailable"
                  fallbackSubtitle="Spectral artefact could not be loaded."
                />
              </div>
            ) : (
              <div className="w-72 h-72 rounded-2xl border border-dashed border-slate-800 flex items-center justify-center text-xs text-slate-500">
                Spectrum visualization unavailable
              </div>
            )}

            <div className="flex-1 space-y-3.5 text-xs text-slate-300 max-w-md">
              <div className="p-4 rounded-xl bg-slate-800/90 border border-slate-700/80 shadow-sm space-y-1.5">
                <span className="font-semibold text-brand-300 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5" />
                  Spectral Analysis
                </span>
                <p className="text-slate-300 leading-relaxed">
                  Natural photography exhibits smooth <span className="font-mono text-amber-300">1/f²</span> power falloff from center (low frequencies) to edges. Generative pipelines frequently display checkerboard spikes or unnatural high-frequency attenuation.
                </p>
              </div>

              {imgDetails?.face_detected && (
                <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-indigo-950/50 border border-indigo-800/50 text-indigo-200">
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-indigo-400" />
                  <span className="leading-relaxed">
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
