/**
 * TruthLens Chrome Extension - Options Controller
 */

document.addEventListener('DOMContentLoaded', async () => {
  const form = document.getElementById('tl-options-form') as HTMLFormElement | null;
  const urlInput = document.getElementById('tl-api-url') as HTMLInputElement | null;
  const keyInput = document.getElementById('tl-api-key') as HTMLInputElement | null;
  const alertEl = document.getElementById('tl-alert');
  const testBtn = document.getElementById('tl-test-btn') as HTMLButtonElement | null;

  const DEFAULT_API_URL = 'http://localhost:8000';

  // Load existing values
  const { apiUrl = DEFAULT_API_URL, apiKey = '' } = await chrome.storage.local.get([
    'apiUrl',
    'apiKey',
  ]);

  if (urlInput) urlInput.value = apiUrl;
  if (keyInput) keyInput.value = apiKey;

  function showAlert(msg: string, isError = false) {
    if (!alertEl) return;
    alertEl.textContent = msg;
    alertEl.className = isError ? 'tl-alert tl-alert-error' : 'tl-alert tl-alert-success';
    alertEl.style.display = 'block';
  }

  // Save handler
  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!urlInput || !keyInput) return;

    const newUrl = urlInput.value.trim().replace(/\/+$/, '') || DEFAULT_API_URL;
    const newKey = keyInput.value.trim();

    if (!newKey.startsWith('tl_')) {
      showAlert('API key should start with "tl_". Please verify the key format.', true);
      return;
    }

    await chrome.storage.local.set({
      apiUrl: newUrl,
      apiKey: newKey,
    });

    showAlert('Settings saved successfully!');
  });

  // Test connection handler
  testBtn?.addEventListener('click', async () => {
    if (!urlInput || !keyInput) return;

    const testUrl = urlInput.value.trim().replace(/\/+$/, '') || DEFAULT_API_URL;
    const testKey = keyInput.value.trim();

    if (!testKey) {
      showAlert('Please enter an API key to test.', true);
      return;
    }

    testBtn.disabled = true;
    testBtn.textContent = 'Connecting...';

    try {
      const res = await fetch(`${testUrl}/api/v1/models`, {
        headers: {
          'X-API-Key': testKey,
        },
      });

      if (res.ok) {
        const data = await res.json();
        const count = data.models ? data.models.length : 0;
        showAlert(`Connection successful! Connected to TruthLens backend (${count} models active).`);
      } else {
        let errMessage = `Server returned status ${res.status}`;
        try {
          const errData = await res.json();
          if (errData.error?.message) errMessage = errData.error.message;
        } catch {
          // ignore
        }
        showAlert(`Connection failed: ${errMessage}`, true);
      }
    } catch (err: unknown) {
      showAlert(
        `Failed to connect to ${testUrl}. Ensure the backend server is running and accessible.`,
        true
      );
    } finally {
      testBtn.disabled = false;
      testBtn.textContent = 'Test Connection';
    }
  });
});
