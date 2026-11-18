import React from 'react';
import { Link } from 'react-router-dom';
import {
  ShieldCheck,
  ImageIcon,
  Music,
  FileText,
  Lock,
  Cpu,
  ArrowRight,
  FileCheck2,
  AlertTriangle,
} from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div className="space-y-20 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="text-center space-y-6 pt-6 sm:pt-12 max-w-4xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-50 dark:bg-brand-950/60 border border-brand-200 dark:border-brand-800 text-brand-700 dark:text-brand-300 text-xs font-semibold">
          <ShieldCheck className="w-4 h-4 text-brand-600" />
          Multi-Modal Forensic Evidence Platform
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-[1.15]">
          Unmask Synthetic Media with{' '}
          <span className="bg-gradient-to-r from-brand-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent">
            Cryptographic Rigor
          </span>
        </h1>

        <p className="text-base sm:text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto leading-relaxed">
          TruthLens is a multi-modal AI detection system engineered for high-stakes intelligence, academic defense, and journalism. Detecting visual, acoustic, and text deepfakes with explainable attribution and HMAC-signed audit trails.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <Link
            to="/analyze"
            className="px-6 py-3 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-semibold text-sm shadow-lg shadow-brand-500/25 flex items-center gap-2 transition-all hover:scale-[1.02]"
          >
            Start Forensic Analysis
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/verify"
            className="px-6 py-3 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 font-semibold text-sm border border-slate-200 dark:border-slate-700 flex items-center gap-2 transition-colors"
          >
            <FileCheck2 className="w-4 h-4 text-brand-500" />
            Verify PDF Report
          </Link>
        </div>
      </section>

      {/* Tri-Modal Core Architecture */}
      <section className="space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white">
            Forensic Defense Across Three Modalities
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
            Each modality utilizes custom ONNX inference runtimes optimized for CPU execution with calibrated probability outputs.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Image */}
          <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors">
            <div className="w-12 h-12 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 flex items-center justify-center">
              <ImageIcon className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Visual Forensics
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              Combines ConvNeXt-Tiny spatial features with 2D Discrete Cosine Transform (DCT) radial power falloff analysis. Exposes generative upsampler artifacts and diffusion grid anomalies.
            </p>
            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-500 dark:text-slate-400 space-y-1">
              <div>• Grad-CAM Spatial Heatmap</div>
              <div>• High-frequency DCT Spectral Falloff</div>
              <div>• Face Bounding Box & Eye Region Scrutiny</div>
            </div>
          </div>

          {/* Audio */}
          <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors">
            <div className="w-12 h-12 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 flex items-center justify-center">
              <Music className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Acoustic Forensics
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              Light Convolutional Neural Network (LCNN) powered by Linear Frequency Cepstral Coefficients (LFCC). Uncovers vocoder phase discontinuities in cloned voices and TTS engines.
            </p>
            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-500 dark:text-slate-400 space-y-1">
              <div>• 60 LFCC Static + Delta + Delta-Delta</div>
              <div>• 4-Second Temporal Sliding Window</div>
              <div>• WaveSurfer Audio Player with Segment Seek</div>
            </div>
          </div>

          {/* Text */}
          <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors">
            <div className="w-12 h-12 rounded-xl bg-violet-50 dark:bg-violet-950/60 text-violet-600 flex items-center justify-center">
              <FileText className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">
              Stylometric & Perplexity
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              Meta-classifier evaluating sentence-level burstiness, vocabulary entropy, and compression ratios. Flags low-variance text characteristic of large language models.
            </p>
            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-500 dark:text-slate-400 space-y-1">
              <div>• Sentence-by-Sentence AI Heatmap</div>
              <div>• Burstiness & Compression Variance</div>
              <div>• Minimum 50-word Grounding Validation</div>
            </div>
          </div>
        </div>
      </section>

      {/* Honest Limitations & Biases Disclosure */}
      <section className="p-8 rounded-3xl border border-amber-200 dark:border-amber-900/60 bg-amber-50/50 dark:bg-amber-950/20 space-y-4">
        <div className="flex items-center gap-2 text-amber-800 dark:text-amber-300">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <h3 className="text-base font-bold">
            Honest Limitations, Biases, and Viva Defense Transparency
          </h3>
        </div>

        <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
          In adherence to academic integrity and ethical forensic practice, TruthLens explicitly documents known model boundaries:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs text-slate-600 dark:text-slate-300 pt-2">
          <div className="p-3.5 rounded-xl bg-white/80 dark:bg-slate-900/80 border border-amber-200/60 dark:border-amber-900/40 space-y-1">
            <span className="font-bold text-amber-900 dark:text-amber-200 block">
              1. Non-Native English Stylistics
            </span>
            <span>
              Formal non-native English essays often exhibit lower perplexity and higher repetition, which can induce false positives in text models without human review.
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-white/80 dark:bg-slate-900/80 border border-amber-200/60 dark:border-amber-900/40 space-y-1">
            <span className="font-bold text-amber-900 dark:text-amber-200 block">
              2. Lossy Compression & Social Reshares
            </span>
            <span>
              Aggressive WhatsApp or Twitter re-compression attenuates high-frequency spectral cues, shifting image/audio confidence intervals closer to the uncertain region.
            </span>
          </div>

          <div className="p-3.5 rounded-xl bg-white/80 dark:bg-slate-900/80 border border-amber-200/60 dark:border-amber-900/40 space-y-1">
            <span className="font-bold text-amber-900 dark:text-amber-200 block">
              3. Unseen Generator Architectures
            </span>
            <span>
              Zero-shot generalization against post-2025 generative architectures requires Bayesian temperature calibration; all outputs include 95% confidence intervals.
            </span>
          </div>
        </div>
      </section>

      {/* Enterprise Security & Privacy */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-center">
        <div className="space-y-2">
          <div className="w-10 h-10 rounded-xl bg-brand-100 dark:bg-brand-950 text-brand-600 mx-auto flex items-center justify-center">
            <Lock className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 dark:text-white">24-Hour Privacy Purge</h4>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Uploaded raw media is purged from MinIO storage within 24 hours unless explicit user retention is confirmed.
          </p>
        </div>

        <div className="space-y-2">
          <div className="w-10 h-10 rounded-xl bg-brand-100 dark:bg-brand-950 text-brand-600 mx-auto flex items-center justify-center">
            <FileCheck2 className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 dark:text-white">HMAC Cryptographic Reports</h4>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Each downloadable forensic report PDF is signed with an HMAC-SHA256 signature and verification QR code.
          </p>
        </div>

        <div className="space-y-2">
          <div className="w-10 h-10 rounded-xl bg-brand-100 dark:bg-brand-950 text-brand-600 mx-auto flex items-center justify-center">
            <Cpu className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 dark:text-white">Pure CPU Inference</h4>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Deployable on commodity edge nodes or standard cloud virtual machines without requiring expensive GPU clusters.
          </p>
        </div>
      </section>
    </div>
  );
};
