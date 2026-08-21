/**
 * Can a content script supply pixels to the offscreen document instead of the
 * offscreen document refetching the URL? Two mechanisms to test:
 *
 * 1. Reading pixels in the page (what a content script could do): draw the
 *    <img> into a canvas and getImageData. Expectation: SecurityError for a
 *    cross-origin image without CORS, works with crossorigin="anonymous" +
 *    Access-Control-Allow-Origin.
 * 2. Shipping an ImageBitmap through chrome.runtime messaging. Extension
 *    messaging is JSON-serialised, so this cannot work by design; here we
 *    demonstrate the JSON half (what survives JSON.stringify), the live
 *    extension check is item 3's job.
 *
 * Two local HTTP servers on different ports make the image cross-origin.
 */
import http from 'node:http';
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright';

// Any small valid PNG; generated with Pillow (64x64 solid colour).
const PNG = readFileSync(process.env.TAINT_PNG ?? '/tmp/taint-test.png');

function serve(port, cors) {
  const srv = http.createServer((req, res) => {
    if (req.url.startsWith('/img.png')) {
      const headers = { 'Content-Type': 'image/png', 'Cache-Control': 'max-age=3600' };
      if (cors) headers['Access-Control-Allow-Origin'] = '*';
      res.writeHead(200, headers);
      res.end(PNG);
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><body></body>');
  });
  return new Promise((ok) => srv.listen(port, '127.0.0.1', () => ok(srv)));
}

const A = 8811, B = 8812, C = 8813; // page origin, image origin (no CORS), image origin (CORS)
const servers = await Promise.all([serve(A, false), serve(B, false), serve(C, true)]);

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${A}/`);

const result = await page.evaluate(async ({ B, C }) => {
  const out = {};
  const load = (src, cors) => new Promise((ok, err) => {
    const img = new Image();
    if (cors) img.crossOrigin = 'anonymous';
    img.onload = () => ok(img);
    img.onerror = err;
    img.src = src;
  });

  async function tryRead(img, label) {
    // Path A: plain canvas readback.
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    try {
      ctx.getImageData(0, 0, 4, 4);
      out[`${label}.getImageData`] = 'OK';
    } catch (e) {
      out[`${label}.getImageData`] = e.name;
    }
    // Path B: createImageBitmap then OffscreenCanvas readback (the bitmap
    // itself is created fine either way; the question is whether its pixels
    // are reachable).
    try {
      const bmp = await createImageBitmap(img);
      out[`${label}.createImageBitmap`] = `OK ${bmp.width}x${bmp.height}`;
      const oc = new OffscreenCanvas(bmp.width, bmp.height);
      const octx = oc.getContext('2d');
      octx.drawImage(bmp, 0, 0);
      try {
        octx.getImageData(0, 0, 4, 4);
        out[`${label}.bitmapReadback`] = 'OK';
      } catch (e) {
        out[`${label}.bitmapReadback`] = e.name;
      }
      // Path C: what would survive chrome.runtime messaging (JSON).
      out[`${label}.jsonSerialized`] = JSON.stringify(bmp);
    } catch (e) {
      out[`${label}.createImageBitmap`] = e.name;
    }
  }

  await tryRead(await load(`http://127.0.0.1:${B}/img.png`), 'crossOrigin-noCORS');
  await tryRead(await load(`http://127.0.0.1:${C}/img.png`, true), 'crossOrigin-CORS-anon');
  await tryRead(await load('/img.png'), 'sameOrigin');
  return out;
}, { B, C });

console.log(JSON.stringify(result, null, 2));
await browser.close();
for (const s of servers) s.close();
