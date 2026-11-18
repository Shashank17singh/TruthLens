# TruthLens Chrome Extension (Manifest V3)

The TruthLens Chrome extension brings multi-modal synthetic media and AI content detection directly into the browser workflow. Inspect suspicious images, audio streams, and text passages across any webpage with right-click context menu actions and an isolated in-page forensic overlay.

---

## Features

- **Context Menu Inspection**:
  - Right-click any image: *"Check image with TruthLens"*
  - Highlight any text excerpt: *"Check selected text with TruthLens"*
  - Right-click HTML5 audio/video elements: *"Check audio/video with TruthLens"*
- **Shadow DOM Overlay**:
  - Injects forensic results inside a closed/isolated Shadow DOM tree (`#truthlens-inspector-host`), ensuring the overlay's appearance is never corrupted or overridden by the host website's CSS styles (Bootstrap, Tailwind, WordPress themes, etc.).
- **CORS Bypass via Backend Ingestion**:
  - Browser security models block direct JavaScript `fetch` of cross-origin image bytes. When an image URL is inspected, the extension passes the URL to `/api/v1/analyze/url`, allowing the backend to fetch, validate (with SSRF protections), and analyze the media cleanly.
- **On/Off Toggle & History**:
  - Quick popup interface allows toggling active inspection and viewing the last 20 inspection scans with direct links to the full forensic report.
- **Secure Key Storage**:
  - Stores the user's API key (`tl_...`) locally in `chrome.storage.local` with zero telemetry leakage.

---

## Minimal Permissions Justification (Chrome Web Store Review)

This extension adheres strictly to Google's principle of least privilege. The declared permissions are:

| Permission | Technical Justification |
|---|---|
| `contextMenus` | Required to register right-click menu items (*"Check image with TruthLens"*, *"Check selected text"*, *"Check audio/video"*). |
| `storage` | Required to locally persist extension preferences, the user's API key, backend URL, and recent scan history in `chrome.storage.local`. |
| `activeTab` | Grants temporary execution permission to the current tab only when the user explicitly triggers a TruthLens context menu action or popup interaction. Does not allow background tracking across arbitrary tabs. |
| `scripting` | Used in combination with `activeTab` to inject the isolated Shadow DOM overlay into the target page upon user request. |
| `host_permissions` | Restricted to the TruthLens backend API server (`http://localhost:8000/*` and public endpoints) to dispatch inference requests. |

---

## Building and Installation

### 1. Build the Extension
```bash
cd extension
pnpm install
pnpm build
```

This compiles TypeScript, bundles assets with Vite, and copies `manifest.json` and icons into `extension/dist/`.

### 2. Load into Google Chrome
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** using the toggle switch in the upper-right corner.
3. Click **Load unpacked**.
4. Select the `TruthLens/extension/dist` folder.
5. TruthLens will appear in your Chrome toolbar.

### 3. Configure API Credentials
1. Click the TruthLens puzzle icon in the Chrome toolbar.
2. Click **Setup** or open the extension **Options** page.
3. Enter your TruthLens Server API URL (default: `http://localhost:8000`).
4. Enter your user API key (e.g., `tl_live_...`).
5. Click **Test Connection** to confirm connectivity, then click **Save Settings**.

### 4. Create Web Store Distribution Package
To package the extension into a compressed archive ready for Chrome Web Store submission:

```bash
pnpm package
```
This produces `truthlens-extension.zip` in the root of `extension/`.
