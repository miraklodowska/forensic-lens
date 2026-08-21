#!/usr/bin/env node
/**
 * Live end-to-end test of the packed extension in a real Chrome.
 *
 * This exercises the parts nothing else covers: the content script's image
 * discovery, the service-worker round trip, offscreen document creation, and
 * badge rendering in a page's DOM. The Python harness and tools/parity/ both
 * verify the *numbers*; only this verifies that the extension works.
 *
 * Needs a browser that still honours --load-extension, which Chrome removed in
 * M137. CHROME_BIN should point at a 136-or-earlier Chrome for Testing.
 */

import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const EXT = resolve(process.env.EXT ?? join(ROOT, 'dist'));
const PAGE = process.env.PAGE ?? 'http://localhost:8799/index.html';
const PORT = Number(process.env.CDP_PORT ?? 9455);
const CHROME = process.env.CHROME_BIN;
const WAIT_MS = Number(process.env.WAIT_MS ?? 240000);

if (!CHROME) {
  console.error('set CHROME_BIN to a Chrome <= 136 (M137 removed --load-extension)');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Minimal CDP client over the DevTools WebSocket. */
class Session {
  #ws;
  #id = 0;
  #pending = new Map();

  static async connect(url) {
    const s = new Session();
    s.#ws = new WebSocket(url);
    await new Promise((res, rej) => {
      s.#ws.addEventListener('open', res, { once: true });
      s.#ws.addEventListener('error', rej, { once: true });
    });
    s.#ws.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data);
      const p = s.#pending.get(m.id);
      if (p) {
        s.#pending.delete(m.id);
        m.error ? p.reject(new Error(JSON.stringify(m.error))) : p.resolve(m.result);
      }
    });
    return s;
  }

  send(method, params = {}) {
    const id = ++this.#id;
    this.#ws.send(JSON.stringify({ id, method, params }));
    return new Promise((res, rej) => this.#pending.set(id, { resolve: res, reject: rej }));
  }

  async evaluate(expression) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
    return r.result.value;
  }

  close() {
    this.#ws.close();
  }
}

const targets = async () => (await fetch(`http://127.0.0.1:${PORT}/json`)).json();

async function main() {
  const profile = await mkdtemp(join(tmpdir(), 'fl-live-'));
  const args = [
    `--user-data-dir=${profile}`,
    `--load-extension=${EXT}`,
    `--disable-extensions-except=${EXT}`,
    `--remote-debugging-port=${PORT}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--enable-unsafe-webgpu',
    PAGE,
  ];
  console.log(`launching Chrome 136 with ${EXT}`);
  const chrome = spawn(CHROME, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  const consoleErrors = [];
  chrome.stderr.on('data', (d) => {
    const s = String(d);
    if (/forensic-lens/i.test(s)) consoleErrors.push(s.trim());
  });

  // Wait for CDP.
  let list = [];
  for (let i = 0; i < 80 && list.length === 0; i++) {
    try {
      list = await targets();
    } catch {
      await sleep(500);
    }
  }
  if (!list.length) throw new Error('CDP never came up');

  // The extension's service worker and offscreen document are the two contexts
  // that prove the extension actually started, not merely that Chrome did.
  let sw = null;
  let off = null;
  for (let i = 0; i < 120; i++) {
    list = await targets();
    sw ??= list.find((t) => t.type === 'service_worker' && t.url.includes('service-worker.js'));
    off ??= list.find((t) => t.url.includes('offscreen/offscreen.html'));
    if (sw && off) break;
    await sleep(1000);
  }
  console.log(`service worker: ${sw ? 'FOUND' : 'MISSING'}`);
  console.log(`offscreen doc:  ${off ? 'FOUND' : 'MISSING'}`);
  if (!sw) {
    // MV3 workers are event-driven and may idle out between polls, so a missing
    // target is not proof of failure. The badges below are the real evidence.
    console.log('  (no worker target caught — judging by page badges instead)');
    console.log('  targets:', list.map((t) => `${t.type} ${t.url.slice(0, 70)}`).join('\n            '));
  }

  if (off) {
    const o = await Session.connect(off.webSocketDebuggerUrl);
    await o.send('Runtime.enable');
    const status = await o.evaluate(
      `(async () => { for (let i=0;i<180;i++){ if (globalThis.__flStatus) return globalThis.__flStatus;
         await new Promise(r=>setTimeout(r,500)); } return {error:'engine never reported'}; })()`,
    );
    console.log('engine:', JSON.stringify(status));
    o.close();
  }

  // Now the part that has never been observed: badges in a real page.
  const pageTarget = list.find((t) => t.type === 'page' && t.url.startsWith('http://localhost:8799'));
  if (!pageTarget) throw new Error('test page target missing');
  const page = await Session.connect(pageTarget.webSocketDebuggerUrl);
  await page.send('Runtime.enable');

  const total = await page.evaluate('document.images.length');
  console.log(`page: ${total} images; waiting for badges…`);

  const started = Date.now();
  let last = -1;
  let snap = { badges: 0, done: 0 };
  while (Date.now() - started < WAIT_MS) {
    snap = await page.evaluate(`(() => {
      const host = document.getElementById('forensic-lens-badge-host');
      const sr = host && host.shadowRoot;
      return {
        hostPresent: !!host,
        badges: sr ? sr.querySelectorAll('.badge').length : 0,
        done: document.querySelectorAll('img[data-forensic-lens="done"]').length,
      };
    })()`);
    if (snap.done !== last) {
      last = snap.done;
      console.log(`  ${snap.done}/${total} scored, ${snap.badges} badges (${((Date.now() - started) / 1000).toFixed(0)}s)`);
    }
    if (snap.done >= total) break;
    await sleep(2000);
  }

  // Pair each badge with its image's ground truth.
  const result = await page.evaluate(`(() => {
    const host = document.getElementById('forensic-lens-badge-host');
    const sr = host && host.shadowRoot;
    const badges = sr ? [...sr.querySelectorAll('.badge')] : [];
    const rows = [];
    let i = 0;
    for (const fig of document.querySelectorAll('figure')) {
      const img = fig.querySelector('img');
      if (img.dataset.forensicLens !== 'done') continue;
      const b = badges[i++];
      const m = b && /(\\d+)\\s*%/.exec(b.textContent || '');
      rows.push({ truth: Number(fig.dataset.truth), family: fig.dataset.family,
                  p: m ? Number(m[1]) / 100 : null, text: b ? b.textContent.trim() : null });
    }
    return rows;
  })()`);

  const scored = result.filter((r) => r.p !== null);
  let tp = 0, fn = 0, tn = 0, fp = 0;
  for (const r of scored) {
    const flagged = r.p >= 0.65;
    if (r.truth === 1) flagged ? tp++ : fn++;
    else flagged ? fp++ : tn++;
  }
  const tpr = tp + fn ? tp / (tp + fn) : 0;
  const tnr = tn + fp ? tn / (tn + fp) : 0;

  console.log(`\nbadge host in page DOM: ${snap.hostPresent}`);
  console.log(`scored ${scored.length}/${total} images end-to-end`);
  console.log(`  AI recall        ${(tpr * 100).toFixed(1)}%  (${tp}/${tp + fn})`);
  console.log(`  real specificity ${(tnr * 100).toFixed(1)}%  (${tn}/${tn + fp})`);
  console.log(`  balanced accuracy@0.65 = ${(((tpr + tnr) / 2) * 100).toFixed(2)}%`);
  console.log('  sample badges:', JSON.stringify(scored.slice(0, 5).map((r) => r.text)));
  if (consoleErrors.length) console.log('\nextension stderr:', consoleErrors.slice(0, 5));

  page.close();
  chrome.kill();
  await rm(profile, { recursive: true, force: true });
  if (scored.length === 0) {
    console.error('\nFAIL: no badges rendered');
    process.exitCode = 1;
  }
}

main().catch((e) => {
  console.error('live test failed:', e.message);
  process.exit(1);
});
