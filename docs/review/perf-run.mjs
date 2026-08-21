// Per-image latency of the shipped pipeline in a real Chromium, measured by
// fine-grained polling of the parity harness's incremental rows array.
import { chromium } from 'playwright';

const PAGE = 'http://127.0.0.1:8802/index.html';
const browser = await chromium.launch({ headless: false, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
await page.goto(PAGE, { waitUntil: 'load' });

const t0 = Date.now();
const stamps = [];
let done = false;
while (!done && Date.now() - t0 < 600000) {
  const s = await page.evaluate('({n: (window.__parity?.rows ?? []).length, done: window.__parity?.done ?? false, backend: window.__parity?.backend})');
  while (stamps.length < s.n) stamps.push((Date.now() - t0) / 1000);
  done = s.done;
  if (done) console.log('backend:', s.backend);
  await new Promise((r) => setTimeout(r, 100));
}
const deltas = stamps.map((t, i) => (i === 0 ? t : t - stamps[i - 1]));
console.log('n =', stamps.length);
console.log('first image (incl. model load + shader compile):', deltas[0]?.toFixed(1), 's');
const rest = deltas.slice(1);
rest.sort((a, b) => a - b);
const mean = rest.reduce((a, b) => a + b, 0) / rest.length;
console.log(`steady-state per image: mean ${mean.toFixed(2)}s  median ${rest[Math.floor(rest.length / 2)].toFixed(2)}s  p90 ${rest[Math.floor(rest.length * 0.9)].toFixed(2)}s  max ${rest[rest.length - 1].toFixed(2)}s`);
console.log(`total for ${stamps.length} images: ${stamps[stamps.length - 1].toFixed(1)}s`);
await browser.close();
