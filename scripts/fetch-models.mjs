#!/usr/bin/env node
/**
 * One-time model download.
 *
 * This is the only step in the whole project that touches the network, and it
 * happens at build time rather than at runtime — the packed extension carries
 * the weights inside it and never opens a connection to fetch a model. Every
 * entry is pinned to a full commit sha and verified by SHA-256, so a rebuild
 * either produces byte-identical weights or fails loudly.
 */

import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, stat } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REGISTRY = join(ROOT, 'models', 'registry.json');
const WEIGHTS_DIR = join(ROOT, 'models', 'weights');

const REVISION_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;

async function sha256File(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

function downloadUrl({ source }) {
  return `https://huggingface.co/${source.repo}/resolve/${source.revision}/${source.path}`;
}

async function fetchOne(entry) {
  const dest = join(WEIGHTS_DIR, entry.file);
  await mkdir(dirname(dest), { recursive: true });

  if (await exists(dest)) {
    if ((await sha256File(dest)) === entry.sha256) {
      console.log(`  ✓ ${entry.file} (cached, sha256 ok)`);
      return;
    }
    console.log(`  ! ${entry.file} present but sha256 mismatch — refetching`);
  }

  const url = downloadUrl(entry);
  console.log(`  ↓ ${entry.file}\n      ${url}`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`${entry.file}: HTTP ${res.status} from ${url}`);

  const tmp = `${dest}.part`;
  await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));

  const got = await sha256File(tmp);
  if (got !== entry.sha256) {
    throw new Error(
      `${entry.file}: sha256 mismatch\n    expected ${entry.sha256}\n    actual   ${got}\n` +
        '  The pinned revision is immutable, so this means the download was corrupted or the ' +
        'registry is wrong. Refusing to use these bytes.',
    );
  }
  const { size } = await stat(tmp);
  if (entry.sizeBytes && size !== entry.sizeBytes) {
    throw new Error(`${entry.file}: expected ${entry.sizeBytes} bytes, got ${size}`);
  }
  await rename(tmp, dest);
  console.log(`  ✓ ${entry.file} (${(size / 1e6).toFixed(1)} MB, sha256 verified)`);
}

function validate(artifacts) {
  for (const entry of artifacts) {
    if (!REVISION_RE.test(entry.source?.revision ?? '')) {
      throw new Error(
        `${entry.file}: source.revision must be a full 40-char commit sha (got ` +
          `${JSON.stringify(entry.source?.revision)}). Mutable refs like "main" are not reproducible.`,
      );
    }
    if (!SHA256_RE.test(entry.sha256 ?? '')) {
      throw new Error(`${entry.file}: sha256 must be 64 lowercase hex characters`);
    }
  }
}

async function main() {
  const registry = JSON.parse(await readFile(REGISTRY, 'utf8'));
  const artifacts = registry.artifacts ?? [];
  validate(artifacts);
  await mkdir(WEIGHTS_DIR, { recursive: true });

  console.log(`Fetching ${artifacts.length} model artifact(s) into models/weights/`);
  for (const entry of artifacts) await fetchOne(entry);

  // Bundled artifacts ship in the repo; verify rather than download them, so a
  // truncated checkout is caught here instead of at first inference.
  for (const entry of registry.bundled ?? []) {
    const path = join(WEIGHTS_DIR, entry.file);
    if (!(await exists(path))) {
      throw new Error(
        `${entry.file} is missing. It ships in the repository (models/weights/); ` +
          're-check out the repo, or regenerate it with scripts/convert-siglip.py.',
      );
    }
    const got = await sha256File(path);
    if (got !== entry.sha256) throw new Error(`${entry.file}: sha256 ${got}, expected ${entry.sha256}`);
    console.log(`  ✓ ${entry.file} (bundled, sha256 verified)`);
  }
  console.log('All model artifacts verified.');
}

main().catch((error) => {
  console.error(`\nfetch-models failed: ${error.message}`);
  process.exit(1);
});
