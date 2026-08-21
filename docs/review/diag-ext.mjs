// Diagnose whether Chromium actually loads dist/ as an unpacked extension.
import { chromium } from 'playwright';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const EXT = process.env.EXT ?? new URL('../../dist', import.meta.url).pathname;
const BIN = process.env.CHROME_BIN;

const profile = await mkdtemp(join(tmpdir(), 'fl-diag-'));
const ctx = await chromium.launchPersistentContext(profile, {
  headless: false,
  timeout: 300000,
  ...(BIN ? { executablePath: BIN } : {}),
  ignoreDefaultArgs: ['--disable-extensions'],
  args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`, '--no-first-run'],
});
console.log('browser up; service workers now:', ctx.serviceWorkers().map((w) => w.url()));

const page = await ctx.newPage();
await page.goto('chrome://extensions-internals', { waitUntil: 'load' });
const text = await page.evaluate('document.body.innerText');
const parsed = JSON.parse(text);
console.log('installed extensions:', parsed.length);
for (const e of parsed) {
  console.log(`- ${e.name ?? '?'} id=${e.id} location=${e.location} enabled=${JSON.stringify(e.state ?? e.enabled ?? '?')}`);
}
const mine = parsed.find((e) => /forensic/i.test(String(e.name)));
if (mine) {
  console.log('FOUND. id:', mine.id);
  const sw = ctx.serviceWorkers().map((w) => w.url());
  console.log('service workers after settle:', sw);
  // Give the SW a moment, then check again and try the popup page.
  await new Promise((r) => setTimeout(r, 8000));
  console.log('service workers +8s:', ctx.serviceWorkers().map((w) => w.url()));
  const pop = await ctx.newPage();
  try {
    await pop.goto(`chrome-extension://${mine.id}/popup/popup.html`, { timeout: 15000 });
    await new Promise((r) => setTimeout(r, 3000));
    console.log('popup body:', JSON.stringify(await pop.evaluate('document.body.innerText')).slice(0, 400));
  } catch (err) {
    console.log('popup failed:', String(err).slice(0, 200));
  }
} else {
  console.log('NOT FOUND — extension refused. Dumping any load errors visible:');
  console.log(text.slice(0, 1500));
}
await ctx.close();
await rm(profile, { recursive: true, force: true });
