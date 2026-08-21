/**
 * Page-side entry point: find images, ask the worker to score them, draw badges.
 *
 * This runs on every page the user visits, so the default posture is to do as
 * little as possible — discovery is viewport-gated, requests are queued behind a
 * small concurrency limit, and nothing is sent anywhere except the extension's
 * own service worker.
 */

import { ImageDiscovery } from './discovery.ts';
import { BadgeLayer } from './badges.ts';
import { classify } from '../core/scoring.ts';
import { imageCacheKey } from '../core/image-key.ts';
import { MESSAGE_TYPES, parseResponse } from '../shared/messages.ts';
import { DEFAULT_SETTINGS, normalizeSettings, type Settings } from '../shared/settings.ts';

let settings: Settings = DEFAULT_SETTINGS;
const badges = new BadgeLayer();
const probabilities = new WeakMap<HTMLImageElement, number>();

/** Bounded queue: a gallery page must not fire 200 concurrent inference calls. */
const MAX_INFLIGHT = 3;
let inflight = 0;
const queue: (() => void)[] = [];

function pump(): void {
  while (inflight < MAX_INFLIGHT && queue.length > 0) {
    const job = queue.shift()!;
    inflight += 1;
    job();
  }
}

function enqueue(job: () => Promise<void>): void {
  queue.push(() => {
    void job().finally(() => {
      inflight -= 1;
      pump();
    });
  });
  pump();
}

function paint(img: HTMLImageElement, probability: number): void {
  badges.render(img, classify(probability, settings.threshold), {
    showBadges: settings.showBadges,
    blurFlagged: settings.blurFlagged,
  });
}

async function analyze(img: HTMLImageElement, src: string): Promise<void> {
  if (!settings.enabled) return;
  const key = imageCacheKey(src, document.baseURI);
  if (key === null) return;

  let raw: unknown;
  try {
    raw = await chrome.runtime.sendMessage({
      type: MESSAGE_TYPES.analyze,
      src,
      key,
      width: img.naturalWidth || img.clientWidth,
      height: img.naturalHeight || img.clientHeight,
    });
  } catch {
    // Worker restarting or extension reloaded — drop this image silently.
    return;
  }

  const reply = parseResponse(raw);
  if (reply === null || !reply.ok) return;
  if (!img.isConnected) return;

  probabilities.set(img, reply.probability);
  img.dataset['forensicLens'] = 'done';
  paint(img, reply.probability);
}

const discovery = new ImageDiscovery({
  root: document,
  minImageSize: DEFAULT_SETTINGS.minImageSize,
  onCandidate: (img, src) => enqueue(() => analyze(img, src)),
});

function reposition(): void {
  badges.reposition();
}

async function boot(): Promise<void> {
  try {
    const stored = await chrome.storage.local.get('settings');
    settings = normalizeSettings(stored['settings']);
  } catch {
    settings = DEFAULT_SETTINGS;
  }
  if (!settings.enabled) return;
  discovery.start();
  addEventListener('scroll', reposition, { passive: true });
  addEventListener('resize', reposition, { passive: true });
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local' || !changes['settings']) return;
  const next = normalizeSettings(changes['settings'].newValue);
  const wasEnabled = settings.enabled;
  settings = next;

  if (!next.enabled) {
    discovery.stop();
    badges.destroy();
    return;
  }
  if (!wasEnabled) discovery.start();
  // Re-render existing badges under the new threshold/appearance settings.
  for (const img of document.images) {
    const p = probabilities.get(img);
    if (p !== undefined) paint(img, p);
  }
});

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => void boot(), { once: true });
} else {
  void boot();
}
