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
    <div className="space-y-24 py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="text-center space-y-8 pt-6 sm:pt-14 max-w-4xl mx-auto relative">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-50/90 dark:bg-brand-950/70 border border-brand-200/80 dark:border-brand-800/80 text-brand-700 dark:text-brand-300 text-xs font-semibold shadow-sm">
          <ShieldCheck className="w-4 h-4 text-brand-600" />
          Multi-Modal Forensic Evidence Platform
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-[1.12]">
          Unmask Synthetic Media with{' '}
          <span className="text-brand-600 dark:text-brand-400 font-black">
            Cryptographic Rigor
          </span>
        </h1>

        <p className="text-base sm:text-lg text-slate-600 dark:text-slate-300 max-w-xl mx-auto leading-relaxed">
          TruthLens detects multi-modal visual, acoustic, and text deepfakes with explainable cryptographic proof.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-3.5 pt-2">
          <Link
            to="/analyze"
            className="px-6 py-3.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-semibold text-sm shadow-lg shadow-brand-500/25 flex items-center gap-2 transition-colors motion-safe:transition-transform motion-safe:hover:scale-[1.01]"
          >
            Start Forensic Analysis
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/verify"
            className="px-6 py-3.5 rounded-xl bg-white hover:bg-slate-50 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 font-semibold text-sm border border-slate-200 dark:border-slate-700 flex items-center gap-2 transition-colors shadow-sm"
          >
            <FileCheck2 className="w-4 h-4 text-brand-500" />
            Verify PDF Report
          </Link>
        </div>

        {/* Forensic Highlights Badges */}
        <div className="pt-6 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-100/70 dark:bg-slate-800/60 border border-slate-200/60 dark:border-slate-700/60">
            <Cpu className="w-3.5 h-3.5 text-brand-500" /> Tri-Modal ONNX Pipeline
          </span>
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-100/70 dark:bg-slate-800/60 border border-slate-200/60 dark:border-slate-700/60">
            <Lock className="w-3.5 h-3.5 text-brand-500" /> 24h Zero-Retention Purge
          </span>
          <span className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-100/70 dark:bg-slate-800/60 border border-slate-200/60 dark:border-slate-700/60">
            <ShieldCheck className="w-3.5 h-3.5 text-brand-500" /> HMAC-SHA256 Signed PDF
          </span>
        </div>
      </section>

      {/* Bento Grid Architecture */}
      <section className="space-y-6">
        <div className="text-center space-y-1.5">
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white">
            Forensic Command Center
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
            Modular multi-modal inspection compartments powered by CPU-optimized ONNX inference engines.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Bento Item 1: Visual Forensics (Span 2 cols on md) */}
          <div className="md:col-span-2 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 flex items-center justify-center">
                  <ImageIcon className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-1 rounded-md bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 text-[11px] font-medium border border-emerald-200/60 dark:border-emerald-800/60">
                  ConvNeXt + 2D DCT
                </span>
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                Spatial and Frequency Visual Forensics
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                Evaluates generative diffusion artifacts and GAN grid boundaries. Combines deep convolutional activations with radial 2D discrete cosine transform energy falloff curves.
              </p>
            </div>
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 grid grid-cols-1 sm:grid-cols-3 gap-2">
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">Grad-CAM Saliency</div>
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">Radial Power Falloff</div>
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">Bounding Box Scrutiny</div>
            </div>
          </div>

          {/* Bento Item 2: Acoustic Forensics */}
          <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 flex items-center justify-center">
                  <Music className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-1 rounded-md bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-400 text-[11px] font-medium border border-indigo-200/60 dark:border-indigo-800/60">
                  LCNN + LFCC
                </span>
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                Acoustic Voice Forensics
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                Light Convolutional Neural Network with 60 linear frequency cepstral coefficients detects phase anomalies in cloned voices.
              </p>
            </div>
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 space-y-1.5">
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">4-Second Sliding Windows</div>
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">WaveSurfer Audio Seek</div>
            </div>
          </div>

          {/* Bento Item 3: Stylometric & Perplexity */}
          <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-violet-50 dark:bg-violet-950/60 text-violet-600 flex items-center justify-center">
                  <FileText className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-1 rounded-md bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-violet-400 text-[11px] font-medium border border-violet-200/60 dark:border-violet-800/60">
                  Perplexity Variance
                </span>
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                Text Stylometrics
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                Measures sentence burstiness, vocabulary compression entropy, and token predictability characteristic of generative LLMs.
              </p>
            </div>
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 space-y-1.5">
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">Sentence Attribution Map</div>
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">50-Word Grounding Gate</div>
            </div>
          </div>

          {/* Bento Item 4: Cryptographic Vault (Span 2 cols on md) */}
          <div className="md:col-span-2 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4 hover:border-brand-500/50 transition-colors flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 flex items-center justify-center">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-1 rounded-md bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 text-[11px] font-medium border border-amber-200/60 dark:border-amber-800/60">
                  HMAC-SHA256 Signed
                </span>
              </div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                Cryptographic Evidence Audit Vault
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                Downloadable PDF reports are cryptographically signed with tamper-evident HMAC signatures, public verification links, and verifiable payload hashes for court, press, and academic review.
              </p>
            </div>
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 grid grid-cols-1 sm:grid-cols-3 gap-2">
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">Court-Ready Audit Trails</div>
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">24h Privacy Purge Default</div>
              <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50">Public Verification Registry</div>
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
