/**
 * TruthLens Chrome Extension - In-Page Overlay (Shadow DOM isolated)
 */

interface OverlayResultData {
  id: string;
  modality: 'image' | 'audio' | 'text';
  probability: number;
  confidence_band?: { low: number; high: number };
  verdict: 'likely_real' | 'uncertain' | 'likely_fake';
  verdict_label: string;
  explanation: string;
  model_name?: string;
  model_version?: string;
  fullReportUrl: string;
}

let overlayHost: HTMLElement | null = null;
let shadowRoot: ShadowRoot | null = null;

function ensureOverlayHost(): ShadowRoot {
  if (overlayHost && shadowRoot) {
    return shadowRoot;
  }

  overlayHost = document.createElement('div');
  overlayHost.id = 'truthlens-inspector-host';
  overlayHost.style.position = 'fixed';
  overlayHost.style.top = '0';
  overlayHost.style.left = '0';
  overlayHost.style.width = '100vw';
  overlayHost.style.height = '100vh';
  overlayHost.style.pointerEvents = 'none';
  overlayHost.style.zIndex = '2147483647'; // Max browser z-index

  shadowRoot = overlayHost.attachShadow({ mode: 'open' });
  document.body.appendChild(overlayHost);

  return shadowRoot;
}

function removeOverlay() {
  if (overlayHost) {
    overlayHost.remove();
    overlayHost = null;
    shadowRoot = null;
  }
}

function renderLoading(modality: string, targetPreview: string) {
  const root = ensureOverlayHost();
  root.innerHTML = `
    <style>
      ${getOverlayStyles()}
    </style>
    <div class="tl-backdrop">
      <div class="tl-modal tl-animate-in">
        <div class="tl-header">
          <div class="tl-brand">
            <div class="tl-logo">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <span class="tl-title">TruthLens Forensic Scan</span>
          </div>
          <button class="tl-close" id="tl-close-btn" aria-label="Close">&times;</button>
        </div>

        <div class="tl-body tl-center">
          <div class="tl-spinner"></div>
          <div class="tl-status-title">Analyzing ${escapeHtml(modality)} Content...</div>
          <div class="tl-status-desc">${escapeHtml(targetPreview)}</div>
          <div class="tl-progress-bar-wrap">
            <div class="tl-progress-bar" id="tl-loading-bar" style="width: 25%"></div>
          </div>
          <div class="tl-footnote">Evaluating spatial and spectral features via ONNX CPU runtime</div>
        </div>
      </div>
    </div>
  `;

  root.getElementById('tl-close-btn')?.addEventListener('click', removeOverlay);
}

function renderError(message: string) {
  const root = ensureOverlayHost();
  root.innerHTML = `
    <style>
      ${getOverlayStyles()}
    </style>
    <div class="tl-backdrop">
      <div class="tl-modal tl-animate-in">
        <div class="tl-header">
          <div class="tl-brand">
            <div class="tl-logo tl-logo-error">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
            </div>
            <span class="tl-title">Inspection Alert</span>
          </div>
          <button class="tl-close" id="tl-close-btn" aria-label="Close">&times;</button>
        </div>

        <div class="tl-body">
          <div class="tl-error-box">
            ${escapeHtml(message)}
          </div>
          <div style="text-align: right; margin-top: 16px;">
            <button class="tl-button tl-button-secondary" id="tl-dismiss-btn">Dismiss</button>
          </div>
        </div>
      </div>
    </div>
  `;

  root.getElementById('tl-close-btn')?.addEventListener('click', removeOverlay);
  root.getElementById('tl-dismiss-btn')?.addEventListener('click', removeOverlay);
}

function renderResult(data: OverlayResultData) {
  const root = ensureOverlayHost();
  const percent = Math.round(data.probability * 100);
  const low = data.confidence_band ? Math.round(data.confidence_band.low * 100) : Math.max(0, percent - 6);
  const high = data.confidence_band ? Math.round(data.confidence_band.high * 100) : Math.min(100, percent + 6);

  let verdictColorClass = 'tl-verdict-uncertain';
  let badgeIcon = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  `;

  if (data.verdict === 'likely_real') {
    verdictColorClass = 'tl-verdict-real';
    badgeIcon = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
      </svg>
    `;
  } else if (data.verdict === 'likely_fake') {
    verdictColorClass = 'tl-verdict-fake';
    badgeIcon = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    `;
  }

  root.innerHTML = `
    <style>
      ${getOverlayStyles()}
    </style>
    <div class="tl-backdrop">
      <div class="tl-modal tl-animate-in">
        <div class="tl-header">
          <div class="tl-brand">
            <div class="tl-logo">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <span class="tl-title">TruthLens Forensic Assessment</span>
          </div>
          <button class="tl-close" id="tl-close-btn" aria-label="Close">&times;</button>
        </div>

        <div class="tl-body">
          <!-- Verdict and probability row -->
          <div class="tl-verdict-row">
            <div>
              <div class="tl-verdict-badge ${verdictColorClass}">
                ${badgeIcon}
                <span>${escapeHtml(data.verdict_label || data.verdict)}</span>
              </div>
              <div class="tl-ci-label">95% Confidence Band: [${low}% – ${high}%]</div>
            </div>
            <div class="tl-score-box">
              <div class="tl-score-number">${percent}%</div>
              <div class="tl-score-caption">AI Likelihood</div>
            </div>
          </div>

          <!-- Probability bar -->
          <div class="tl-meter-bar">
            <div class="tl-meter-fill ${verdictColorClass}" style="width: ${percent}%;"></div>
          </div>

          <!-- Explanation text -->
          <div class="tl-explanation">
            ${escapeHtml(data.explanation || 'Statistical feature analysis completed.')}
          </div>

          <!-- Metadata -->
          <div class="tl-meta-row">
            <span>Model: <strong>${escapeHtml(data.model_name || 'ONNX Runtime')}</strong> (v${escapeHtml(data.model_version || '1.0')})</span>
            <span>Modality: <strong style="text-transform: capitalize;">${escapeHtml(data.modality)}</strong></span>
          </div>

          <!-- Action buttons -->
          <div class="tl-actions">
            <button class="tl-button tl-button-secondary" id="tl-close-bottom-btn">Close</button>
            <a href="${data.fullReportUrl}" target="_blank" class="tl-button tl-button-primary">
              View Full Report & Explanations &rarr;
            </a>
          </div>
        </div>
      </div>
    </div>
  `;

  root.getElementById('tl-close-btn')?.addEventListener('click', removeOverlay);
  root.getElementById('tl-close-bottom-btn')?.addEventListener('click', removeOverlay);
}

function updateProgress(progress: number) {
  if (shadowRoot) {
    const bar = shadowRoot.getElementById('tl-loading-bar');
    if (bar) {
      bar.style.width = `${Math.min(95, Math.max(10, progress))}%`;
    }
  }
}

// Scoped CSS styles completely isolated within the Shadow DOM
function getOverlayStyles(): string {
  return `
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .tl-backdrop {
      pointer-events: auto;
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(15, 23, 42, 0.65);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }

    .tl-modal {
      width: 100%;
      max-width: 440px;
      background: #ffffff;
      color: #0f172a;
      border-radius: 18px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
      border: 1px solid #e2e8f0;
      overflow: hidden;
    }

    @media (prefers-color-scheme: dark) {
      .tl-modal {
        background: #0f172a;
        color: #f8fafc;
        border-color: #334155;
      }
    }

    .tl-animate-in {
      animation: tl-fade-in 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes tl-fade-in {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }

    .tl-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid #f1f5f9;
    }

    @media (prefers-color-scheme: dark) {
      .tl-header {
        border-bottom-color: #1e293b;
      }
    }

    .tl-brand {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .tl-logo {
      width: 28px;
      height: 28px;
      border-radius: 8px;
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .tl-logo-error {
      background: #e11d48;
    }

    .tl-title {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .tl-close {
      background: none;
      border: none;
      font-size: 22px;
      line-height: 1;
      color: #94a3b8;
      cursor: pointer;
      padding: 2px 6px;
      border-radius: 6px;
    }

    .tl-close:hover {
      background: #f1f5f9;
      color: #334155;
    }

    @media (prefers-color-scheme: dark) {
      .tl-close:hover {
        background: #1e293b;
        color: #f1f5f9;
      }
    }

    .tl-body {
      padding: 18px;
    }

    .tl-center {
      text-align: center;
      padding: 24px 18px;
    }

    .tl-spinner {
      width: 36px;
      height: 36px;
      margin: 0 auto 14px auto;
      border: 3px solid #e2e8f0;
      border-top-color: #4f46e5;
      border-radius: 50%;
      animation: tl-spin 0.8s linear infinite;
    }

    @keyframes tl-spin {
      to { transform: rotate(360deg); }
    }

    .tl-status-title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 4px;
    }

    .tl-status-desc {
      font-size: 11px;
      color: #64748b;
      margin-bottom: 16px;
      max-width: 340px;
      margin-left: auto;
      margin-right: auto;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .tl-progress-bar-wrap {
      width: 100%;
      height: 6px;
      background: #f1f5f9;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 12px;
    }

    @media (prefers-color-scheme: dark) {
      .tl-progress-bar-wrap {
        background: #1e293b;
      }
    }

    .tl-progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #4f46e5, #06b6d4);
      transition: width 0.3s ease;
    }

    .tl-footnote {
      font-size: 10px;
      color: #94a3b8;
    }

    .tl-verdict-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }

    .tl-verdict-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }

    .tl-verdict-real {
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #a7f3d0;
    }

    .tl-verdict-uncertain {
      background: #fffbeb;
      color: #b45309;
      border: 1px solid #fde68a;
    }

    .tl-verdict-fake {
      background: #fff1f2;
      color: #be123c;
      border: 1px solid #fecdd3;
    }

    @media (prefers-color-scheme: dark) {
      .tl-verdict-real { background: rgba(4, 120, 87, 0.2); color: #34d399; border-color: rgba(4, 120, 87, 0.4); }
      .tl-verdict-uncertain { background: rgba(180, 83, 9, 0.2); color: #fbbf24; border-color: rgba(180, 83, 9, 0.4); }
      .tl-verdict-fake { background: rgba(190, 18, 60, 0.2); color: #fb7185; border-color: rgba(190, 18, 60, 0.4); }
    }

    .tl-ci-label {
      font-size: 10px;
      color: #64748b;
      margin-top: 4px;
    }

    .tl-score-box {
      text-align: right;
    }

    .tl-score-number {
      font-size: 24px;
      font-weight: 800;
      font-family: monospace;
      line-height: 1;
    }

    .tl-score-caption {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #64748b;
      margin-top: 2px;
    }

    .tl-meter-bar {
      width: 100%;
      height: 8px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 14px;
    }

    @media (prefers-color-scheme: dark) {
      .tl-meter-bar { background: #334155; }
    }

    .tl-meter-fill {
      height: 100%;
      transition: width 0.6s ease-out;
    }

    .tl-meter-fill.tl-verdict-real { background: #059669; }
    .tl-meter-fill.tl-verdict-uncertain { background: #d97706; }
    .tl-meter-fill.tl-verdict-fake { background: #e11d48; }

    .tl-explanation {
      font-size: 12px;
      line-height: 1.5;
      color: #334155;
      background: #f8fafc;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #e2e8f0;
      margin-bottom: 12px;
    }

    @media (prefers-color-scheme: dark) {
      .tl-explanation {
        background: #1e293b;
        color: #cbd5e1;
        border-color: #334155;
      }
    }

    .tl-meta-row {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #64748b;
      margin-bottom: 16px;
    }

    .tl-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }

    .tl-button {
      padding: 8px 14px;
      font-size: 11px;
      font-weight: 600;
      border-radius: 8px;
      text-decoration: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      transition: background 0.15s;
    }

    .tl-button-primary {
      background: #4f46e5;
      color: #ffffff;
      border: none;
    }

    .tl-button-primary:hover {
      background: #4338ca;
    }

    .tl-button-secondary {
      background: #f1f5f9;
      color: #334155;
      border: 1px solid #cbd5e1;
    }

    .tl-button-secondary:hover {
      background: #e2e8f0;
    }

    @media (prefers-color-scheme: dark) {
      .tl-button-secondary {
        background: #1e293b;
        color: #cbd5e1;
        border-color: #334155;
      }
      .tl-button-secondary:hover {
        background: #334155;
      }
    }

    .tl-error-box {
      font-size: 12px;
      color: #b91c1c;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 10px;
      padding: 12px;
      line-height: 1.4;
    }

    @media (prefers-color-scheme: dark) {
      .tl-error-box {
        background: rgba(185, 28, 28, 0.2);
        color: #fca5a5;
        border-color: rgba(185, 28, 28, 0.4);
      }
    }
  `;
}

function escapeHtml(str: string): string {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Listen for messages from background service worker
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === 'SHOW_LOADING') {
    renderLoading(message.modality, message.target);
  } else if (message.action === 'UPDATE_PROGRESS') {
    updateProgress(message.progress);
  } else if (message.action === 'SHOW_RESULT') {
    renderResult(message.result);
  } else if (message.action === 'SHOW_ERROR') {
    renderError(message.message);
  }
});
