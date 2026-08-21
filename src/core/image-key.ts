/**
 * Turning whatever an <img> claims as its source into a stable cache key.
 *
 * Keys must be bounded (a 10 MB data: URL cannot become a Map key) and stable
 * across re-renders, because infinite-scroll pages recreate identical nodes
 * constantly and re-running inference on them would waste the whole budget.
 */

const ANALYZABLE_PROTOCOLS = new Set(['http:', 'https:', 'blob:', 'file:']);
const DATA_IMAGE_RE = /^data:image\/[a-z0-9.+-]+[;,]/i;

/** 32-bit FNV-1a; used only to shorten data: URLs, never for integrity. */
function fnv1a(input: string, seed: number): string {
  let hash = seed >>> 0;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

function dataUrlKey(source: string): string {
  const comma = source.indexOf(',');
  const header = comma === -1 ? source.slice(0, 64) : source.slice(0, comma);
  // Two independently-seeded passes make accidental collisions between
  // similarly-shaped payloads far less likely than a single 32-bit hash.
  return `data:${header.slice(5, 40)}#${source.length}#${fnv1a(source, 0x811c9dc5)}${fnv1a(source, 0x9e3779b1)}`;
}

/**
 * @returns a bounded stable key, or null when the source cannot be keyed.
 */
export function imageCacheKey(source: string, base?: string): string | null {
  const trimmed = source.trim();
  if (trimmed.length === 0) return null;
  if (trimmed.toLowerCase().startsWith('data:')) return dataUrlKey(trimmed);
  let url: URL;
  try {
    url = base === undefined ? new URL(trimmed) : new URL(trimmed, base);
  } catch {
    return null;
  }
  url.hash = '';
  return url.href;
}

export function isAnalyzableSource(source: string): boolean {
  const trimmed = source.trim();
  if (trimmed.length === 0) return false;
  if (trimmed.toLowerCase().startsWith('data:')) return DATA_IMAGE_RE.test(trimmed);
  try {
    return ANALYZABLE_PROTOCOLS.has(new URL(trimmed, 'https://placeholder.invalid/').protocol);
  } catch {
    return false;
  }
}
