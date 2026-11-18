/**
 * TruthLens Chrome Extension - Background Service Worker (Manifest V3)
 */

interface StorageConfig {
  apiKey?: string;
  apiUrl?: string;
  enabled?: boolean;
}

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

const DEFAULT_API_URL = 'http://localhost:8000';

// Register context menus on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'truthlens-check-image',
    title: 'Check image with TruthLens',
    contexts: ['image'],
  });

  chrome.contextMenus.create({
    id: 'truthlens-check-text',
    title: 'Check selected text with TruthLens',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: 'truthlens-check-media',
    title: 'Check audio/video with TruthLens',
    contexts: ['audio', 'video'],
  });

  // Set default settings if unset
  chrome.storage.local.get(['enabled', 'apiUrl'], (res) => {
    if (res.enabled === undefined) {
      chrome.storage.local.set({ enabled: true });
    }
    if (!res.apiUrl) {
      chrome.storage.local.set({ apiUrl: DEFAULT_API_URL });
    }
  });
});

// Context menu click handler
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab || !tab.id) return;

  const storage: StorageConfig = await chrome.storage.local.get([
    'apiKey',
    'apiUrl',
    'enabled',
  ]);

  if (storage.enabled === false) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'SHOW_ERROR',
      message: 'TruthLens extension is currently disabled. Toggle it back on in the extension popup.',
    });
    return;
  }

  const apiKey = storage.apiKey?.trim();
  const apiUrl = (storage.apiUrl?.trim() || DEFAULT_API_URL).replace(/\/+$/, '');

  if (!apiKey) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'SHOW_ERROR',
      message: 'TruthLens API key required. Please open extension options to enter your API key (tl_...).',
    });
    chrome.runtime.openOptionsPage();
    return;
  }

  try {
    if (info.menuItemId === 'truthlens-check-image' && info.srcUrl) {
      await handleImageCheck(info.srcUrl, tab.id, apiUrl, apiKey);
    } else if (info.menuItemId === 'truthlens-check-text' && info.selectionText) {
      await handleTextCheck(info.selectionText, tab.id, apiUrl, apiKey);
    } else if (info.menuItemId === 'truthlens-check-media' && info.srcUrl) {
      await handleAudioCheck(info.srcUrl, tab.id, apiUrl, apiKey);
    }
  } catch (err: unknown) {
    const errorMsg = (err as Error)?.message || 'Inspection failed.';
    chrome.tabs.sendMessage(tab.id, {
      action: 'SHOW_ERROR',
      message: errorMsg,
    });
  }
});

async function handleImageCheck(
  srcUrl: string,
  tabId: number,
  apiUrl: string,
  apiKey: string
): Promise<void> {
  chrome.tabs.sendMessage(tabId, {
    action: 'SHOW_LOADING',
    modality: 'image',
    target: srcUrl.slice(0, 80),
  });

  let analysisId: string;

  // If it's a data URL, convert to Blob and upload directly
  if (srcUrl.startsWith('data:')) {
    const blob = dataUrlToBlob(srcUrl);
    const formData = new FormData();
    formData.append('file', blob, 'web_capture.png');

    const res = await fetch(`${apiUrl}/api/v1/analyze/image`, {
      method: 'POST',
      headers: {
        'X-API-Key': apiKey,
      },
      body: formData,
    });

    if (!res.ok) {
      const err = await parseApiError(res);
      throw new Error(err);
    }

    const data = await res.json();
    analysisId = data.analysis_id;
  } else {
    // Standard HTTP/HTTPS image URL - send URL to backend to bypass browser CORS
    const res = await fetch(`${apiUrl}/api/v1/analyze/url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify({
        url: srcUrl,
        modality_hint: 'image',
      }),
    });

    if (!res.ok) {
      const err = await parseApiError(res);
      throw new Error(err);
    }

    const data = await res.json();
    analysisId = data.analysis_id;
  }

  // Poll for result
  const result = await pollAnalysis(analysisId, apiUrl, apiKey, tabId);
  await saveRecentCheck('image', srcUrl.slice(0, 60), result, apiUrl);

  chrome.tabs.sendMessage(tabId, {
    action: 'SHOW_RESULT',
    result: {
      ...result,
      fullReportUrl: `${apiUrl.replace(':8000', ':3000')}/results/${analysisId}`,
    },
  });
}

async function handleTextCheck(
  text: string,
  tabId: number,
  apiUrl: string,
  apiKey: string
): Promise<void> {
  chrome.tabs.sendMessage(tabId, {
    action: 'SHOW_LOADING',
    modality: 'text',
    target: `"${text.slice(0, 60)}..."`,
  });

  const res = await fetch(`${apiUrl}/api/v1/analyze/text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey,
    },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const err = await parseApiError(res);
    throw new Error(err);
  }

  const data = await res.json();
  const analysisId = data.analysis_id;

  const result = await pollAnalysis(analysisId, apiUrl, apiKey, tabId);
  await saveRecentCheck('text', text.slice(0, 50), result, apiUrl);

  chrome.tabs.sendMessage(tabId, {
    action: 'SHOW_RESULT',
    result: {
      ...result,
      fullReportUrl: `${apiUrl.replace(':8000', ':3000')}/results/${analysisId}`,
    },
  });
}

async function handleAudioCheck(
  srcUrl: string,
  tabId: number,
  apiUrl: string,
  apiKey: string
): Promise<void> {
  chrome.tabs.sendMessage(tabId, {
    action: 'SHOW_LOADING',
    modality: 'audio',
    target: srcUrl.slice(0, 80),
  });

  const res = await fetch(`${apiUrl}/api/v1/analyze/url`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey,
    },
    body: JSON.stringify({
      url: srcUrl,
      modality_hint: 'audio',
    }),
  });

  if (!res.ok) {
    const err = await parseApiError(res);
    throw new Error(err);
  }

  const data = await res.json();
  const analysisId = data.analysis_id;

  const result = await pollAnalysis(analysisId, apiUrl, apiKey, tabId);
  await saveRecentCheck('audio', srcUrl.slice(0, 50), result, apiUrl);

  chrome.tabs.sendMessage(tabId, {
    action: 'SHOW_RESULT',
    result: {
      ...result,
      fullReportUrl: `${apiUrl.replace(':8000', ':3000')}/results/${analysisId}`,
    },
  });
}

// Poll analysis until complete
async function pollAnalysis(
  analysisId: string,
  apiUrl: string,
  apiKey: string,
  tabId: number
): Promise<any> {
  const maxAttempts = 45; // 45 seconds timeout
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await new Promise((r) => setTimeout(r, 1000));

    const res = await fetch(`${apiUrl}/api/v1/analyses/${analysisId}`, {
      headers: { 'X-API-Key': apiKey },
    });

    if (res.ok) {
      const item = await res.json();
      if (item.status === 'completed' && item.result) {
        return {
          id: item.id,
          modality: item.modality,
          probability: item.result.probability,
          confidence_band: item.result.confidence_band,
          verdict: item.result.verdict,
          verdict_label: item.result.verdict_label,
          explanation: item.result.explanation,
          model_name: item.result.model_name,
          model_version: item.result.model_version,
          disclaimer: item.result.disclaimer,
        };
      } else if (item.status === 'failed') {
        throw new Error(item.error?.message || 'Analysis processing failed.');
      }

      // Notify progress
      chrome.tabs.sendMessage(tabId, {
        action: 'UPDATE_PROGRESS',
        progress: item.progress || (attempt * 3),
      });
    }
  }

  throw new Error('Analysis timed out after 45 seconds.');
}

async function saveRecentCheck(
  modality: 'image' | 'audio' | 'text',
  title: string,
  result: any,
  apiUrl: string
): Promise<void> {
  const { recentChecks = [] } = await chrome.storage.local.get(['recentChecks']);
  const newEntry: RecentCheck = {
    id: result.id,
    modality,
    title,
    probability: result.probability,
    verdict: result.verdict,
    verdictLabel: result.verdict_label,
    timestamp: new Date().toISOString(),
    reportUrl: `${apiUrl.replace(':8000', ':3000')}/results/${result.id}`,
  };

  const updated = [newEntry, ...recentChecks].slice(0, 20);
  await chrome.storage.local.set({ recentChecks: updated });
}

function dataUrlToBlob(dataUrl: string): Blob {
  const parts = dataUrl.split(';base64,');
  const contentType = parts[0].split(':')[1];
  const raw = atob(parts[1]);
  const uInt8Array = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }
  return new Blob([uInt8Array], { type: contentType });
}

async function parseApiError(res: Response): Promise<string> {
  try {
    const err = await res.json();
    if (err.error?.message) return err.error.message;
    if (err.detail) return typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
  } catch {
    // fallback
  }
  return `Server returned error status ${res.status}`;
}
