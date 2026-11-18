import { resolve } from 'path';
import { defineConfig } from 'vite';
import fs from 'fs';

function copyExtensionAssets() {
  return {
    name: 'copy-extension-assets',
    closeBundle() {
      const outDir = resolve(__dirname, 'dist');
      if (!fs.existsSync(outDir)) {
        fs.mkdirSync(outDir, { recursive: true });
      }

      // Copy manifest
      fs.copyFileSync(resolve(__dirname, 'manifest.json'), resolve(outDir, 'manifest.json'));

      // Copy icons
      const iconsDir = resolve(outDir, 'icons');
      if (!fs.existsSync(iconsDir)) {
        fs.mkdirSync(iconsDir, { recursive: true });
      }
      for (const icon of ['icon16.png', 'icon48.png', 'icon128.png']) {
        const src = resolve(__dirname, 'icons', icon);
        if (fs.existsSync(src)) {
          fs.copyFileSync(src, resolve(iconsDir, icon));
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [copyExtensionAssets()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        popup: resolve(__dirname, 'popup.html'),
        options: resolve(__dirname, 'options.html'),
        'service-worker': resolve(__dirname, 'src/background/service-worker.ts'),
        overlay: resolve(__dirname, 'src/content/overlay.ts'),
      },
      output: {
        entryFileNames: (chunkInfo) => {
          if (chunkInfo.name === 'service-worker') return 'service-worker.js';
          if (chunkInfo.name === 'overlay') return 'overlay.js';
          return 'assets/[name].js';
        },
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]',
      },
    },
  },
});
