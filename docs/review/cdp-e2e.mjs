// Live-extension e2e via manual launch + connectOverCDP (tolerates version skew
// between 2026 Playwright and pre-M137 Chrome builds that still honour
// --load-extension).
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const BIN = process.env.CHROME_BIN;
const EXT = process.env.EXT ?? new URL('../../dist', import.meta.url).pathname;
const PAGE = process.env.PAGE ?? 'http://127.0.0.1:8799/index.html';
const PORT = Number(process.env.CDP_PORT ?? 9223);
const WAIT_MS = Number(process.env.WAIT_MS ?? 300000);
const OUT = process.env.OUT ?? '/private/tmp/claude-501/-Users-miraklodowska/1f1f7be6-9096-447e-89f5-471a329a8083/scratchpad/e2e-cdp.json';

const profile = await mkdtemp(join(tmpdir(), 'fl-cdp-'));
const proc = spawn(BIN, [
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  `--disable-extensions-except=${EXT}`,
  `--load-extension=${EXT}`,
  '--enable-unsafe-webgpu',
  '--no-first-run',
  '--no-default-browser-check',
  PAGE,
], { stdio: 'ignore' });
console.log('spawned pid', proc.pid);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let version = null;
for (let i = 0; i < 120; i += 1) {
  try {
    version = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json();
    break;
  } catch { await sleep(1000); }
}
if (!version) { console.error('CDP never came up'); proc.kill(); process.exit(2); }
console.log('CDP up:', version.Browser);

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const extTargets = targets.filter((t) => String(t.url).startsWith('chrome-extension://'));
console.log('extension targets:', extTargets.map((t) => `${t.type}:${t.url.split('/').slice(3).join('/')}`));

const browser = await chromium.connectOverCDP(`http://127.0.0.1:${PORT}`);
const ctx = browser.contexts()[0];
let page = ctx.pages().find((p) => p.url().startsWith(PAGE.slice(0, 24)));
if (!page) { page = await ctx.newPage(); await page.goto(PAGE, { waitUntil: 'load' }); }

const total = await page.evaluate(() => document.images.length);
console.log(`page has ${total} images; polling badges…`);

const started = Date.now();
let snapshot = { badges: [], figures: [] };
let last = -1;
const timeline = [];
while (Date.now() - started < WAIT_MS) {
  snapshot = await page.evaluate(() => {
    const host = document.getElementById('forensic-lens-badge-host');
    const sr = host && host.shadowRoot;
    const badges = sr ? [...sr.querySelectorAll('.badge')] : [];
    const out = [];
    for (const fig of document.querySelectorAll('figure')) {
      const img = fig.querySelector('img');
      out.push({ family: fig.dataset.family, truth: Number(fig.dataset.truth), done: img.dataset.forensicLens === 'done' });
    }
    return { badges: badges.map((b) => ({ text: (b.textContent || '').trim(), verdict: b.dataset.verdict })), figures: out };
  });
  const done = snapshot.figures.filter((f) => f.done).length;
  if (done !== last) {
    last = done;
    timeline.push({ t: (Date.now() - started) / 1000, done });
    console.log(`  ${done}/${total} scored at ${((Date.now() - started) / 1000).toFixed(0)}s`);
  }
  if (done >= total) break;
  await sleep(2000);
}

// Grab engine status from the offscreen document if present.
let engine = null;
try {
  const off = ctx.pages().find((p) => p.url().includes('offscreen'));
  if (off) engine = await off.evaluate(() => globalThis.__flStatus ?? null);
} catch {}

await writeFile(OUT, JSON.stringify({ ...snapshot, timeline, engine, extTargets: extTargets.map((t) => t.url) }, null, 2));
console.log('engine status:', JSON.stringify(engine));
console.log('badges sample:', JSON.stringify(snapshot.badges.slice(0, 8)));
proc.kill();
await sleep(1500);
await rm(profile, { recursive: true, force: true }).catch(() => {});
