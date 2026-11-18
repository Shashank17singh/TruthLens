/**
 * TruthLens Chrome Extension - Popup Controller
 */

interface RecentCheck {
  id: string;
  modality: 'image' | 'audio' | 'text';
  title: string;
  probability: number;
  verdict: string;
  verdictLabel: string;
  timestamp: string;
  reportUrl: string;
}

document.addEventListener('DOMContentLoaded', async () => {
  const toggleEl = document.getElementById('tl-toggle') as HTMLInputElement | null;
  const bannerEl = document.getElementById('tl-api-banner');
  const statusEl = document.getElementById('tl-api-status');
  const configLink = document.getElementById('tl-config-link');
  const recentContainer = document.getElementById('tl-recent-container');
  const clearBtn = document.getElementById('tl-clear-btn');
  const optionsBtn = document.getElementById('tl-options-btn');

  // Load configuration
  const { enabled = true, apiKey = '', recentChecks = [] } = await chrome.storage.local.get([
    'enabled',
    'apiKey',
    'recentChecks',
  ]);

  // Set toggle
  if (toggleEl) {
    toggleEl.checked = enabled;
    toggleEl.addEventListener('change', () => {
      chrome.storage.local.set({ enabled: toggleEl.checked });
    });
  }

  // Set API Key status
  if (bannerEl && statusEl && configLink) {
    if (apiKey && apiKey.trim().startsWith('tl_')) {
      bannerEl.className = 'tl-banner tl-banner-ok';
      statusEl.textContent = 'API Key: Connected';
      configLink.textContent = 'Change';
    } else {
      bannerEl.className = 'tl-banner tl-banner-warn';
      statusEl.textContent = 'API Key: Missing';
      configLink.textContent = 'Setup';
    }

    configLink.addEventListener('click', (e) => {
      e.preventDefault();
      chrome.runtime.openOptionsPage();
    });
  }

  // Render recent checks
  renderRecentChecks(recentChecks);

  // Clear button
  clearBtn?.addEventListener('click', async () => {
    await chrome.storage.local.set({ recentChecks: [] });
    renderRecentChecks([]);
  });

  // Options button
  optionsBtn?.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });

  function renderRecentChecks(items: RecentCheck[]) {
    if (!recentContainer) return;

    if (!items || items.length === 0) {
      recentContainer.innerHTML = `
        <div class="tl-empty">
          No recent scans recorded.<br>
          Right-click any image, selection, or audio on a webpage to inspect.
        </div>
      `;
      return;
    }

    recentContainer.innerHTML = '';
    items.forEach((item) => {
      const card = document.createElement('a');
      card.className = 'tl-card';
      card.href = item.reportUrl;
      card.target = '_blank';

      const percent = Math.round(item.probability * 100);
      let badgeClass = 'tl-badge-uncertain';
      if (item.verdict === 'likely_real') badgeClass = 'tl-badge-real';
      else if (item.verdict === 'likely_fake') badgeClass = 'tl-badge-fake';

      const timeAgo = formatTimeAgo(item.timestamp);

      card.innerHTML = `
        <div class="tl-card-left">
          <div class="tl-card-title">${escapeHtml(item.title || 'Analysis ' + item.id.slice(0, 8))}</div>
          <div class="tl-card-sub">${escapeHtml(item.modality)} &bull; ${timeAgo}</div>
        </div>
        <div class="tl-card-badge ${badgeClass}">
          ${percent}% AI
        </div>
      `;

      recentContainer.appendChild(card);
    });
  }

  function formatTimeAgo(isoString: string): string {
    const diffMs = Date.now() - new Date(isoString).getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${Math.floor(diffHours / 24)}d ago`;
  }

  function escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});
