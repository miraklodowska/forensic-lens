#!/usr/bin/env node
/**
 * Assembles the browser parity harness into a static directory that can be
 * served over plain HTTP: bundle, ORT runtime assets, the real model weights,
 * and a sample of corpus images with their Python-side reference scores.
 */

import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT = resolve(process.argv[2] ?? '/tmp/fl-parity');
const CORPUS = process.env.CORPUS ?? join(homedir(), 'aidetect-data', 'corpus');
const PER_SOURCE = Number(process.env.PER_SOURCE ?? 4);

await rm(OUT, { recursive: true, force: true });
await mkdir(join(OUT, 'img'), { recursive: true });
await mkdir(join(OUT, 'models'), { recursive: true });
await mkdir(join(OUT, 'vendor', 'ort'), { recursive: true });

await esbuild.build({
  entryPoints: { harness: join(ROOT, 'tools', 'parity', 'harness.ts') },
  outdir: OUT,
  bundle: true,
  format: 'esm',
  target: ['chrome121'],
  platform: 'browser',
  logLevel: 'info',
  external: ['node:fs', 'node:path', 'node:os', 'node:crypto', 'fs', 'path', 'os', 'crypto', 'worker_threads'],
});

const ORT_DIST = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
for (const asset of [
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
]) {
  await cp(join(ORT_DIST, asset), join(OUT, 'vendor', 'ort', asset));
}

const pipeline = JSON.parse(await readFile(join(ROOT, 'models', 'pipeline.json'), 'utf8'));
await cp(join(ROOT, 'models', 'pipeline.json'), join(OUT, 'models', 'pipeline.json'));
for (const model of pipeline.models) {
  await cp(join(ROOT, 'models', 'weights', model.file), join(OUT, 'models', model.file));
}

const labels = JSON.parse(await readFile(join(CORPUS, '_labels.json'), 'utf8'));
const images = [];
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
    images.push({ name, url: `/img/${name}`, source, label: meta.label, family: meta.family, srcPath: join(CORPUS, source, f) });
  }
}
await writeFile(join(OUT, 'images.json'), JSON.stringify(images, null, 2));

await writeFile(
  join(OUT, 'index.html'),
  `<!doctype html><html><head><meta charset="utf-8"><title>Forensic Lens parity</title>
<style>body{font:13px ui-monospace,Menlo,monospace;margin:16px;background:#111;color:#eee}
pre{white-space:pre-wrap}</style></head>
<body><h1>parity harness</h1><pre id="log">starting…</pre>
<script type="module" src="/harness.js"></script></body></html>\n`,
);

console.log(`parity harness: ${OUT} (${images.length} images)`);
