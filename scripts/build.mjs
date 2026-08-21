#!/usr/bin/env node
/**
 * Bundles the extension into dist/ as a loadable unpacked MV3 extension.
 *
 * The build is offline and deterministic: it reads only files already in the
 * tree — weights come from `npm run fetch:models`, the one networked step — so
 * two people building the same commit get the same extension.
 */

import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(ROOT, 'dist');
const WEIGHTS = join(ROOT, 'models', 'weights');
const ORT_DIST = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');

/** WASM/JSEP runtime files ORT loads lazily at run time from `wasmPaths`. */
const ORT_ASSETS = [
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function sha256(p) {
  return createHash('sha256').update(await readFile(p)).digest('hex');
}

async function resolveArtifacts() {
  const registry = JSON.parse(await readFile(join(ROOT, 'models', 'registry.json'), 'utf8'));
  const pipeline = JSON.parse(await readFile(join(ROOT, 'models', 'pipeline.json'), 'utf8'));
  const byFile = new Map([...(registry.artifacts ?? []), ...(registry.bundled ?? [])].map((a) => [a.file, a]));

  // pipeline.json is the source of truth for what actually gets loaded, so a
  // mismatch with the registry is caught here rather than as a 404 at runtime.
  const missing = [];
  const artifacts = [];
  for (const model of pipeline.models) {
    const entry = byFile.get(model.file);
    if (!entry) throw new Error(`pipeline.json loads ${model.file}, which models/registry.json does not describe`);
    artifacts.push(entry);
    const path = join(WEIGHTS, model.file);
    if (!(await exists(path))) {
      missing.push(entry);
      continue;
    }
    const got = await sha256(path);
    if (got !== entry.sha256) {
      throw new Error(
        `models/weights/${model.file} has sha256 ${got}, registry expects ${entry.sha256}.\n` +
          'Delete it and re-run `npm run fetch:models`.',
      );
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `Missing model weights: ${missing.map((m) => m.file).join(', ')}\n` +
        'Run `npm run fetch:models` first (the only step that needs network access).',
    );
  }
  return artifacts;
}

async function bundle() {
  const shared = {
    outdir: DIST,
    bundle: true,
    target: ['chrome121'],
    platform: 'browser',
    sourcemap: false,
    minify: false,
    define: { 'process.env.NODE_ENV': '"production"' },
    // ORT ships node-only branches behind these; drop them from the browser build.
    external: ['node:fs', 'node:path', 'node:os', 'node:crypto', 'fs', 'path', 'os', 'crypto', 'worker_threads'],
  };

  await esbuild.build({
    ...shared,
    entryPoints: {
      'background/service-worker': join(ROOT, 'src', 'background', 'service-worker.ts'),
      'offscreen/offscreen': join(ROOT, 'src', 'offscreen', 'offscreen.ts'),
      'popup/popup': join(ROOT, 'src', 'popup', 'popup.ts'),
    },
    format: 'esm',
    legalComments: 'inline',
    logLevel: 'info',
  });

  // Content scripts are injected as classic scripts, so they cannot be ESM.
  await esbuild.build({
    ...shared,
    entryPoints: { 'content/content': join(ROOT, 'src', 'content', 'content.ts') },
    format: 'iife',
    logLevel: 'info',
  });
}

async function copyStatic(artifacts) {
  await cp(join(ROOT, 'manifest.json'), join(DIST, 'manifest.json'));
  await cp(join(ROOT, 'src', 'offscreen', 'offscreen.html'), join(DIST, 'offscreen', 'offscreen.html'));
  await cp(join(ROOT, 'src', 'popup', 'popup.html'), join(DIST, 'popup', 'popup.html'));
  await cp(join(ROOT, 'icons'), join(DIST, 'icons'), { recursive: true });
  await cp(join(ROOT, 'LICENSE'), join(DIST, 'LICENSE'));

  await mkdir(join(DIST, 'vendor', 'ort'), { recursive: true });
  const available = new Set(await readdir(ORT_DIST));
  for (const asset of ORT_ASSETS) {
    if (!available.has(asset)) {
      throw new Error(`onnxruntime-web is missing ${asset}; run \`npm ci\` to restore node_modules.`);
    }
    await cp(join(ORT_DIST, asset), join(DIST, 'vendor', 'ort', asset));
  }

  await mkdir(join(DIST, 'models'), { recursive: true });
  for (const entry of artifacts) await cp(join(WEIGHTS, entry.file), join(DIST, 'models', entry.file));
  await cp(join(ROOT, 'models', 'pipeline.json'), join(DIST, 'models', 'pipeline.json'));
}

async function dirSize(dir) {
  let total = 0;
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    total += e.isDirectory() ? await dirSize(p) : (await stat(p)).size;
  }
  return total;
}

async function main() {
  const artifacts = await resolveArtifacts();
  await rm(DIST, { recursive: true, force: true });
  await mkdir(DIST, { recursive: true });
  await bundle();
  await copyStatic(artifacts);

  await writeFile(
    join(DIST, 'BUILD.json'),
    `${JSON.stringify({ models: artifacts.map((a) => ({ file: a.file, sha256: a.sha256 })) }, null, 2)}\n`,
  );
  const size = await dirSize(DIST);
  console.log(`\nBuilt dist/ (${(size / 1e6).toFixed(1)} MB). Load it via chrome://extensions → Load unpacked.`);
}

main().catch((error) => {
  console.error(`\nbuild failed: ${error.message}`);
  process.exit(1);
});
