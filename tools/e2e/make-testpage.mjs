#!/usr/bin/env node
/**
 * Builds a local page of labelled corpus images, so the extension can be checked
 * against known answers in a real browser rather than only in the Python harness.
 */

import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { homedir } from 'node:os';

const CORPUS = process.env.CORPUS ?? join(homedir(), 'aidetect-data', 'corpus');
const OUT = resolve(process.argv[2] ?? '/tmp/fl-testpage');
const PER_SOURCE = Number(process.env.PER_SOURCE ?? 3);

const labels = JSON.parse(await readFile(join(CORPUS, '_labels.json'), 'utf8'));

await rm(OUT, { recursive: true, force: true });
await mkdir(join(OUT, 'img'), { recursive: true });

const rows = [];
for (const [source, meta] of Object.entries(labels)) {
  let files;
  try {
    files = (await readdir(join(CORPUS, source))).filter((f) => f.endsWith('.png'));
  } catch {
    continue;
  }
  for (const f of files.slice(0, PER_SOURCE)) {
    const name = `${source}__${f}`;
    await cp(join(CORPUS, source, f), join(OUT, 'img', name));
    rows.push({ name, source, label: meta.label, family: meta.family });
  }
}

const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>Forensic Lens test corpus</title>
<style>
 body{font:13px system-ui;margin:20px;background:#fff;color:#111}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:18px}
 figure{margin:0}
 img{width:100%;height:190px;object-fit:cover;border-radius:6px;display:block}
 figcaption{font-size:11px;margin-top:4px;color:#444}
 .ai{color:#b91c1c;font-weight:600}.real{color:#047857;font-weight:600}
</style></head><body>
<h1>Forensic Lens test corpus (${rows.length} images)</h1>
<div class="grid">
${rows
  .map(
    (r) => `<figure data-truth="${r.label}" data-family="${r.family}">
  <img src="img/${r.name}" alt="${r.family}">
  <figcaption><span class="${r.label ? 'ai' : 'real'}">${r.label ? 'AI' : 'REAL'}</span> · ${r.family}</figcaption>
</figure>`,
  )
  .join('\n')}
</div></body></html>`;

await writeFile(join(OUT, 'index.html'), html);
await writeFile(join(OUT, 'truth.json'), JSON.stringify(rows, null, 2));
console.log(`test page: ${OUT}/index.html  (${rows.length} images)`);
