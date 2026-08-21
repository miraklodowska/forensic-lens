// Drive the parity harness page and dump window.__parity when done.
import { chromium } from 'playwright';
import { writeFile } from 'node:fs/promises';

const PAGE = process.env.PAGE ?? 'http://127.0.0.1:8802/index.html';
const OUT = process.env.OUT ?? '/private/tmp/claude-501/-Users-miraklodowska/1f1f7be6-9096-447e-89f5-471a329a8083/scratchpad/degen-results.json';
const WAIT_MS = Number(process.env.WAIT_MS ?? 300000);

const browser = await chromium.launch({
  headless: false,
  args: ['--enable-unsafe-webgpu', '--no-first-run'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text().slice(0, 200)));
await page.goto(PAGE, { waitUntil: 'load', timeout: 60000 });

const started = Date.now();
let state = null;
while (Date.now() - started < WAIT_MS) {
  state = await page.evaluate('window.__parity');
  if (state?.done) break;
  await new Promise((r) => setTimeout(r, 2000));
}
await writeFile(OUT, JSON.stringify(state, null, 2));
console.log('backend:', state?.backend, 'rows:', state?.rows?.length, 'error:', state?.error ?? 'none');
await browser.close();
