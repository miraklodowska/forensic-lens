#!/usr/bin/env node
/**
 * End-to-end check in a real Chromium.
 *
 * Loads the built extension, opens a page of labelled images and reads back the
 * scores it actually produced, so what gets verified is the shipped pipeline —
 * canvas resampling, ONNX Runtime, the WebGPU/WASM backend — rather than a
 * Python approximation of it.
 *
 * Note on browsers: Chrome stopped honouring `--load-extension` in M137, so
 * this harness needs a build that still accepts it (see CHROME_BIN below).
 * Loading the same dist/ by hand via chrome://extensions works on any Chrome.
 */

import { chromium } from 'playwright';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const EXT = resolve(process.env.EXT ?? join(ROOT, 'dist'));
const PAGE = process.env.PAGE ?? 'http://localhost:8799/index.html';
const WAIT_MS = Number(process.env.WAIT_MS ?? 240000);
const OUT = process.env.OUT ?? '/tmp/fl-e2e-results.json';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const profile = await mkdtemp(join(tmpdir(), 'fl-profile-'));
  const t0 = Date.now();
  const ctx = await chromium.launchPersistentContext(profile, {
    headless: false,
    timeout: 300000,
    ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}),
    // Playwright passes --disable-extensions by default, which would silently
    // defeat the entire point of this harness.
    ignoreDefaultArgs: ['--disable-extensions'],
    args: [
      `--disable-extensions-except=${EXT}`,
      `--load-extension=${EXT}`,
      '--enable-unsafe-webgpu',
      '--no-first-run',
    ],
  });
  console.log(`browser up in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  let worker = ctx.serviceWorkers()[0];
  if (!worker) worker = await ctx.waitForEvent('serviceworker', { timeout: 60000 }).catch(() => null);
  if (!worker) {
    console.error('NO service worker — the extension did not load.');
    console.error('This browser build probably refuses --load-extension (Chrome >= 137).');
    await ctx.close();
    process.exit(2);
  }
  console.log('service worker:', worker.url());

  const page = await ctx.newPage();
  await page.goto(PAGE, { waitUntil: 'load', timeout: 60000 });
  const total = await page.evaluate('document.images.length');
  console.log(`page loaded: ${total} images; waiting for scores…`);

  const started = Date.now();
  let rows = [];
  let last = -1;
  while (Date.now() - started < WAIT_MS) {
    rows = await page.evaluate(`(() => {
      const host = document.getElementById('forensic-lens-badge-host');
      const sr = host && host.shadowRoot;
      const badges = sr ? [...sr.querySelectorAll('.badge')] : [];
      const out = [];
      for (const fig of document.querySelectorAll('figure')) {
        const img = fig.querySelector('img');
        out.push({
          family: fig.dataset.family,
          truth: Number(fig.dataset.truth),
          done: img.dataset.forensicLens === 'done',
        });
      }
      return { badges: badges.map(b => ({ text: (b.textContent||'').trim(), verdict: b.dataset.verdict })), figures: out };
    })()`);
    const done = rows.figures.filter((f) => f.done).length;
    if (done !== last) {
      last = done;
      console.log(`  ${done}/${total} scored (${((Date.now() - started) / 1000).toFixed(0)}s)`);
    }
    if (done >= total) break;
    await sleep(2000);
  }

  // Badges render in DOM order, so they line up with the figures that finished.
  const scored = rows.figures.filter((f) => f.done);
  const probs = rows.badges.map((b) => {
    const m = /(\d+)\s*%/.exec(b.text);
    return m ? Number(m[1]) / 100 : null;
  });

  let tp = 0, fn = 0, tn = 0, fp = 0;
  scored.forEach((f, i) => {
    const p = probs[i];
    if (p === null || p === undefined) return;
    const flagged = p >= 0.65;
    if (f.truth === 1) flagged ? tp++ : fn++;
    else flagged ? fp++ : tn++;
  });
  const tpr = tp + fn ? tp / (tp + fn) : 0;
  const tnr = tn + fp ? tn / (tn + fp) : 0;

  console.log(`\nscored ${scored.length}/${total} images in-browser`);
  console.log(`  AI recall      ${(tpr * 100).toFixed(1)}%  (${tp}/${tp + fn})`);
  console.log(`  real specificity ${(tnr * 100).toFixed(1)}%  (${tn}/${tn + fp})`);
  console.log(`  balanced accuracy @0.65 = ${(((tpr + tnr) / 2) * 100).toFixed(2)}%`);
  console.log('  sample badges:', JSON.stringify(rows.badges.slice(0, 6)));

  await writeFile(OUT, JSON.stringify({ ...rows, probs }, null, 2));
  await ctx.close();
  await rm(profile, { recursive: true, force: true });
  if (scored.length === 0) process.exitCode = 1;
}

main().catch((error) => {
  console.error('e2e failed:', error.message);
  process.exit(1);
});
