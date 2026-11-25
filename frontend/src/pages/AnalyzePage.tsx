import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ImageIcon,
  Music,
  FileText,
  Globe,
  Loader2,
  AlertCircle,
  Shield,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import { Dropzone } from '../components/Dropzone';
import {
  useSubmitImage,
  useSubmitAudio,
  useSubmitText,
  useSubmitUrl,
} from '../api/hooks';
import { useAuth } from '../context/AuthContext';
import { api, getStoredToken, retryOnUnauthorized } from '../api/client';

type TabType = 'image' | 'audio' | 'text' | 'url';

export const AnalyzePage: React.FC = () => {
  const navigate = useNavigate();
  const { user, loginAsDemo } = useAuth();

  const [activeTab, setActiveTab] = useState<TabType>('image');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);
  const [textContent, setTextContent] = useState<string>('');
  const [urlInput, setUrlInput] = useState<string>('');
  const [urlModalityHint, setUrlModalityHint] = useState<string>('');
  const [retainMedia, setRetainMedia] = useState<boolean>(false);

  // Submission & Progress state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null);
  const [progressStage, setProgressStage] = useState<string>('queued');
  const [progressPercent, setProgressPercent] = useState<number>(10);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  const submitImageMutation = useSubmitImage();
  const submitAudioMutation = useSubmitAudio();
  const submitTextMutation = useSubmitText();
  const submitUrlMutation = useSubmitUrl();

  // Clean up WS and polling timers on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const wordCount = textContent.trim() ? textContent.trim().split(/\s+/).length : 0;
  const charCount = textContent.length;

  const sampleAIText =
    'The accelerated integration of multi-modal generative intelligence models across contemporary institutional ecosystems represents a transformative paradigm shift of unprecedented magnitude. Furthermore, optimizing computational throughput across distributed inference pipelines inherently facilitates continuous knowledge distillation. Consequently, verifiable cryptographic watermarking remains paramount for sustained epistemic integrity.';

  const sampleHumanText =
    'Early this morning, the harbor fog was so dense that none of the fishing trawlers could leave the dock until well past seven. We sat in the small coffee shack by the pier, drinking black tea from paper cups and listening to the low groan of the buoy horn out in the bay while waiting for the sun to break through the cloud cover.';

  const loadSampleImage = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 320;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const grad = ctx.createLinearGradient(0, 0, 320, 320);
      grad.addColorStop(0, '#064e3b');
      grad.addColorStop(0.5, '#047857');
      grad.addColorStop(1, '#065f46');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 320, 320);

      ctx.beginPath();
      ctx.ellipse(160, 160, 80, 110, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#10b981';
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 15px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('TruthLens Forensic Sample', 160, 155);
      ctx.font = '12px monospace';
      ctx.fillText('Visual Synthetic Test', 160, 185);
    }
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], 'forensic_sample_image.png', { type: 'image/png' });
        setSelectedImage(file);
        setErrorMessage(null);
      }
    }, 'image/png');
  };

  const loadSampleAudio = () => {
    const sampleRate = 16000;
    const numSamples = sampleRate * 2;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    view.setUint32(0, 0x52494646, false);
    view.setUint32(4, 36 + numSamples * 2, true);
    view.setUint32(8, 0x57415645, false);
    view.setUint32(12, 0x666d7420, false);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    view.setUint32(36, 0x64617461, false);
    view.setUint32(40, numSamples * 2, true);

    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const sample = Math.sin(2 * Math.PI * 440 * t) * 0.4 * 32767;
      view.setInt16(44 + i * 2, sample, true);
    }
    const blob = new Blob([buffer], { type: 'audio/wav' });
    const file = new File([blob], 'forensic_sample_audio.wav', { type: 'audio/wav' });
    setSelectedAudio(file);
    setErrorMessage(null);
  };

  const ensureAuthenticated = async () => {
    if (!user || !getStoredToken()) {
      // Auto-initialize demo analyst session so users can test friction-free
      await loginAsDemo();
    }
  };

  const startProgressTracking = async (analysisId: string) => {
    setCurrentAnalysisId(analysisId);
    setProgressStage('queued');
    setProgressPercent(15);

    // Try WebSocket connection via ticket
    let wsConnected = false;
    try {
      const { ticket } = await api.createWebSocketTicket(analysisId);
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = window.location.host;
      const wsUrl = `${wsProtocol}//${wsHost}/ws/analyses/${analysisId}?ticket=${ticket}`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        wsConnected = true;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'progress') {
            setProgressStage(data.stage || 'inference');
            setProgressPercent(data.progress || 50);
          } else if (data.type === 'completed') {
            setProgressStage('completed');
            setProgressPercent(100);
            setTimeout(() => {
              navigate(`/results/${analysisId}`);
            }, 600);
          } else if (data.type === 'failed') {
            setErrorMessage(data.error?.message || 'Analysis processing failed.');
            setIsSubmitting(false);
            ws.close();
          }
        } catch {
          // ignore parsing error
        }
      };

      ws.onerror = () => {
        startPollingFallback(analysisId);
      };

      ws.onclose = () => {
        if (!wsConnected) {
          startPollingFallback(analysisId);
        }
      };
    } catch {
      // WS ticket failed, fallback to polling
      startPollingFallback(analysisId);
    }
  };

  const startPollingFallback = (analysisId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const item = await api.getAnalysis(analysisId);
        if (item.status === 'completed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setProgressStage('completed');
          setProgressPercent(100);
          setTimeout(() => {
            navigate(`/results/${analysisId}`);
          }, 600);
        } else if (item.status === 'failed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setErrorMessage(item.error?.message || 'Analysis failed.');
          setIsSubmitting(false);
        } else {
          setProgressPercent(Math.max(30, item.progress));
          setProgressStage('inference');
        }
      } catch {
        // Continue polling
      }
    }, 1200);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setIsSubmitting(true);

    try {
      await ensureAuthenticated();

      const res = await retryOnUnauthorized(async () => {
      if (activeTab === 'image') {
        if (!selectedImage) {
          throw new Error('Please select an image file to analyze.');
        }
        return submitImageMutation.mutateAsync({
          file: selectedImage,
          retainMedia,
        });
      } else if (activeTab === 'audio') {
        if (!selectedAudio) {
          throw new Error('Please select an audio file to analyze.');
        }
        return submitAudioMutation.mutateAsync({
          file: selectedAudio,
          retainMedia,
        });
      } else if (activeTab === 'text') {
        if (!textContent.trim()) {
          throw new Error('Please enter text to analyze.');
        }
        if (wordCount < 10) {
          throw new Error('Please provide at least 10 words for stylometric feature extraction.');
        }
        return submitTextMutation.mutateAsync({
          text: textContent,
          retainMedia,
        });
      } else if (activeTab === 'url') {
        if (!urlInput.trim()) {
          throw new Error('Please enter a valid URL to analyze.');
        }
        return submitUrlMutation.mutateAsync({
          url: urlInput.trim(),
          modalityHint: urlModalityHint || undefined,
          retainMedia,
        });
      }
        return undefined;
      }, loginAsDemo);

      if (res && res.analysis_id) {
        startProgressTracking(res.analysis_id);
      }
    } catch (err: unknown) {
      const e = err as { message?: string };
      setErrorMessage(e.message || 'Submission failed. Please check inputs.');
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      {/* Page Title */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
          Forensic Inspection Chamber
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 max-w-lg mx-auto">
          Upload media or paste text to perform deep multi-modal feature extraction and authenticity attribution.
        </p>
      </div>

      {/* Main Tabbed Container */}
      <div className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">
        {/* Navigation Tabs */}
        <div className="grid grid-cols-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
          <button
            type="button"
            onClick={() => {
              setActiveTab('image');
              setErrorMessage(null);
            }}
            className={`py-4 px-2 flex flex-col sm:flex-row items-center justify-center gap-2 text-xs sm:text-sm font-semibold transition-colors border-b-2 ${
              activeTab === 'image'
                ? 'border-brand-600 text-brand-600 dark:text-brand-400 bg-white dark:bg-slate-900'
                : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            <span>Image</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab('audio');
              setErrorMessage(null);
            }}
            className={`py-4 px-2 flex flex-col sm:flex-row items-center justify-center gap-2 text-xs sm:text-sm font-semibold transition-colors border-b-2 ${
              activeTab === 'audio'
                ? 'border-brand-600 text-brand-600 dark:text-brand-400 bg-white dark:bg-slate-900'
                : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Music className="w-4 h-4" />
            <span>Audio</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab('text');
              setErrorMessage(null);
            }}
            className={`py-4 px-2 flex flex-col sm:flex-row items-center justify-center gap-2 text-xs sm:text-sm font-semibold transition-colors border-b-2 ${
              activeTab === 'text'
                ? 'border-brand-600 text-brand-600 dark:text-brand-400 bg-white dark:bg-slate-900'
                : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Text</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab('url');
              setErrorMessage(null);
            }}
            className={`py-4 px-2 flex flex-col sm:flex-row items-center justify-center gap-2 text-xs sm:text-sm font-semibold transition-colors border-b-2 ${
              activeTab === 'url'
                ? 'border-brand-600 text-brand-600 dark:text-brand-400 bg-white dark:bg-slate-900'
                : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <Globe className="w-4 h-4" />
            <span>URL</span>
          </button>
        </div>

        {/* Tab Forms */}
        <form onSubmit={handleSubmit} className="p-6 sm:p-8 space-y-6">
          {activeTab === 'image' && (
            <div className="space-y-4">
              <Dropzone
                acceptType="image"
                maxSizeMB={10}
                selectedFile={selectedImage}
                onFileSelect={setSelectedImage}
                disabled={isSubmitting}
              />
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
                <span>
                  Stripping EXIF metadata automatically before pipeline processing. Supports ConvNeXt-Tiny and 2D DCT radial spectrum.
                </span>
                <button
                  type="button"
                  onClick={loadSampleImage}
                  disabled={isSubmitting}
                  className="text-brand-600 dark:text-brand-400 font-semibold hover:underline flex items-center gap-1"
                >
                  <Sparkles className="w-3 h-3" />
                  Try sample image
                </button>
              </div>
            </div>
          )}

          {activeTab === 'audio' && (
            <div className="space-y-4">
              <Dropzone
                acceptType="audio"
                maxSizeMB={25}
                selectedFile={selectedAudio}
                onFileSelect={setSelectedAudio}
                disabled={isSubmitting}
              />
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500">
                <span>
                  Evaluates LFCC coefficients across 4-second temporal sliding windows via LCNN architecture.
                </span>
                <button
                  type="button"
                  onClick={loadSampleAudio}
                  disabled={isSubmitting}
                  className="text-brand-600 dark:text-brand-400 font-semibold hover:underline flex items-center gap-1"
                >
                  <Sparkles className="w-3 h-3" />
                  Try sample audio
                </button>
              </div>
            </div>
          )}

          {activeTab === 'text' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
                <span>Enter English prose or article excerpt:</span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setTextContent(sampleAIText)}
                    className="text-brand-600 dark:text-brand-400 font-semibold hover:underline flex items-center gap-1"
                  >
                    <Sparkles className="w-3 h-3" />
                    Load AI sample
                  </button>
                  <span className="text-slate-300 dark:text-slate-700">|</span>
                  <button
                    type="button"
                    onClick={() => setTextContent(sampleHumanText)}
                    className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                  >
                    Load human sample
                  </button>
                </div>
              </div>
              <textarea
                rows={8}
                value={textContent}
                onChange={(e) => setTextContent(e.target.value)}
                disabled={isSubmitting}
                placeholder="Paste suspicious text, news article, or AI-generated essay here to analyze burstiness, perplexity variance, and vocabulary entropy..."
                className="w-full p-4 rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500 font-serif leading-relaxed"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span className={wordCount < 10 ? 'text-amber-500' : 'text-slate-400'}>
                  <span className="font-mono">{wordCount}</span> words {wordCount < 10 && '(min. 10 words required)'}
                </span>
                <span>
                  <span className="font-mono">{charCount}</span> characters
                </span>
              </div>
            </div>
          )}

          {activeTab === 'url' && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Target Media or Article URL
                </label>
                <input
                  type="url"
                  placeholder="https://example.com/suspicious_media.png"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  disabled={isSubmitting}
                  className="w-full px-4 py-2.5 rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-brand-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Modality Hint (Optional)
                </label>
                <select
                  value={urlModalityHint}
                  onChange={(e) => setUrlModalityHint(e.target.value)}
                  disabled={isSubmitting}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm text-slate-800 dark:text-slate-200"
                >
                  <option value="">Auto-Detect via Content-Type</option>
                  <option value="image">Image</option>
                  <option value="audio">Audio</option>
                  <option value="text">Text / Web Article</option>
                </select>
              </div>

              <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 text-[11px] text-slate-500 space-y-1">
                <span className="font-semibold text-slate-700 dark:text-slate-300">SSRF Protection Active:</span>
                <p>Private IP ranges (RFC 1918 / loopback / cloud metadata) are blocked. Maximum download size is 30 MB with a 15-second socket timeout.</p>
              </div>
            </div>
          )}

          {/* Retention Option */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-800">
            <label className="flex items-center gap-2 cursor-pointer select-none text-xs text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={retainMedia}
                onChange={(e) => setRetainMedia(e.target.checked)}
                disabled={isSubmitting}
                className="rounded border-slate-300 text-brand-600 focus:ring-brand-500"
              />
              <span>Retain media permanently (exempt from 24-hour privacy purge)</span>
            </label>
          </div>

          {errorMessage && (
            <div className="flex items-center gap-2 p-3.5 rounded-xl bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Submit Action */}
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full py-3.5 rounded-xl bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white font-bold text-sm shadow-md shadow-brand-500/20 flex items-center justify-center gap-2 transition-colors whitespace-nowrap"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing Forensic Pipeline…
              </>
            ) : (
              <>
                Analyze Media
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>
      </div>

      {/* Progress Dialog Overlay when Submitting */}
      {isSubmitting && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/70 backdrop-blur-md">
          <div className="w-full max-w-md rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 space-y-6 shadow-2xl motion-safe:animate-in motion-safe:zoom-in-95 duration-150">
            <div className="text-center space-y-2">
              <div className="w-12 h-12 rounded-2xl bg-brand-50 dark:bg-brand-950 text-brand-600 dark:text-brand-400 mx-auto flex items-center justify-center shadow-inner">
                <Shield className="w-6 h-6 motion-safe:animate-pulse" />
              </div>
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Forensic Pipeline in Progress
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Executing anti-spoofing classifiers on CPU runtime
                {currentAnalysisId && (
                  <span className="block text-[10px] text-slate-400 mt-0.5">
                    ID: <span className="font-mono">{currentAnalysisId.slice(0, 18)}…</span>
                  </span>
                )}
              </p>
            </div>

            {/* Progress bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-brand-600 dark:text-brand-400 font-semibold">
                  Stage: {progressStage}
                </span>
                <span className="text-slate-500 font-mono font-bold">{progressPercent}%</span>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-brand-600 to-indigo-500 origin-left transition-transform duration-300 rounded-full"
                  style={{ transform: `scaleX(${progressPercent / 100})`, width: '100%' }}
                />
              </div>
            </div>

            {/* Pipeline Step Indicators */}
            <div className="grid grid-cols-5 gap-1 text-[10px] text-center font-medium text-slate-400">
              <span className={progressPercent >= 10 ? 'text-brand-600 font-bold' : ''}>Queued</span>
              <span className={progressPercent >= 25 ? 'text-brand-600 font-bold' : ''}>Validate</span>
              <span className={progressPercent >= 60 ? 'text-brand-600 font-bold' : ''}>Inference</span>
              <span className={progressPercent >= 85 ? 'text-brand-600 font-bold' : ''}>Explain</span>
              <span className={progressPercent >= 100 ? 'text-brand-600 font-bold' : ''}>Save</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
