/**
 * Package TruthLens Chrome Extension into a distributable .zip file.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const zipFile = path.join(rootDir, 'truthlens-extension.zip');

if (!fs.existsSync(distDir)) {
  console.error('Error: "dist" directory not found. Please run "pnpm build" first.');
  process.exit(1);
}

// Remove old zip if exists
if (fs.existsSync(zipFile)) {
  fs.unlinkSync(zipFile);
}

console.log(`Packaging TruthLens extension from ${distDir} into ${zipFile}...`);

try {
  if (process.platform === 'win32') {
    // Windows PowerShell Compress-Archive
    const psCommand = `powershell -NoProfile -Command "Compress-Archive -Path '${distDir}\\*' -DestinationPath '${zipFile}' -Force"`;
    execSync(psCommand, { stdio: 'inherit' });
  } else {
    // Linux / macOS zip utility
    execSync(`cd "${distDir}" && zip -r "${zipFile}" ./*`, { stdio: 'inherit' });
  }

  const stats = fs.statSync(zipFile);
  console.log(`Success! Created ${zipFile} (${(stats.size / 1024).toFixed(1)} KB)`);
} catch (err) {
  console.error('Failed to create zip file:', err);
  process.exit(1);
}
