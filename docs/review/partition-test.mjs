// Does Chrome's cache partitioning defeat the offscreen document's
// "same URL the page already loaded, so normally a cache hit" assumption?
// Same-partition refetch vs cross-top-level-site refetch of one image URL.
import { chromium } from 'playwright';

const IMG = 'http://127.0.0.1:8807/pic.png';
const browser = await chromium.launch({ headless: false });
const ctx = await browser.newContext();

const pageA = await ctx.newPage();
await pageA.goto('http://127.0.0.1:8807/page.html', { waitUntil: 'load' });
await pageA.waitForFunction('window.__loaded');
console.log('A: page on 127.0.0.1 loaded <img> once');

// Same-partition refetch with the offscreen doc's exact options.
const r1 = await pageA.evaluate(
  `fetch(${JSON.stringify(IMG)}, { credentials: 'omit', cache: 'force-cache' }).then(r => r.status)`,
);
console.log('B: same-partition force-cache refetch status', r1);

// Different top-level site (localhost vs 127.0.0.1) — same partition situation
// as a chrome-extension:// offscreen document refetching a page's image.
const pageB = await ctx.newPage();
await pageB.goto('http://localhost:8807/blank.html', { waitUntil: 'load' });
const r2 = await pageB.evaluate(
  `fetch(${JSON.stringify(IMG)}, { credentials: 'omit', cache: 'force-cache', mode: 'no-cors' }).then(r => r.status)`,
);
console.log('C: cross-partition force-cache refetch status', r2);

await browser.close();
console.log('done — count GET /pic.png lines in the server log');
