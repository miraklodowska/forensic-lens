/**
 * Service worker: lifecycle owner and router.
 *
 * It deliberately does no image work. MV3 kills idle workers after ~30s, so
 * anything holding a warm ONNX session here would be rebuilt constantly; that
 * lives in the offscreen document, which this file starts on demand and
 * forwards to.
 *
 * There is no installer. Model weights are packed into the extension at build
 * time, so by the time this worker first runs there is nothing left to fetch.
 * (The offscreen document does refetch each analysed image URL — see the
 * README's Privacy section — but this worker itself never touches the network.)
 */

import {
  MESSAGE_TYPES,
  parseEngineEvent,
  parseInferResponse,
  parseRequest,
  type AnalyzeRequest,
  type EngineStatus,
  type ScoreResponse,
} from '../shared/messages.ts';
import { DEFAULT_SETTINGS, normalizeSettings } from '../shared/settings.ts';

const OFFSCREEN_URL = 'offscreen/offscreen.html';

let offscreenReady: Promise<void> | null = null;
let status: EngineStatus = {
  ready: false,
  backend: 'unknown',
  modelIds: [],
  analyzed: 0,
  flagged: 0,
  errors: 0,
};

/** Scores keyed by the content script's bounded cache key. */
const CACHE_LIMIT = 600;
const cache = new Map<string, number>();

function cacheGet(key: string): number | undefined {
  const hit = cache.get(key);
  if (hit !== undefined) {
    cache.delete(key);
    cache.set(key, hit); // refresh LRU position
  }
  return hit;
}

function cacheSet(key: string, probability: number): void {
  if (cache.size >= CACHE_LIMIT) {
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  cache.set(key, probability);
}

async function hasOffscreen(): Promise<boolean> {
  // getContexts is the only race-free way to ask; createDocument throws if a
  // document already exists and two tabs can request analysis at the same time.
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT' as chrome.runtime.ContextType],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  return contexts.length > 0;
}

function ensureOffscreen(): Promise<void> {
  offscreenReady ??= (async () => {
    if (await hasOffscreen()) return;
    try {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_URL,
        reasons: ['WORKERS' as chrome.offscreen.Reason],
        justification: 'Runs the bundled ONNX detectors on WebGPU/WASM entirely on-device.',
      });
    } catch (error) {
      // A concurrent call may have won the race; that is success for us.
      if (!(await hasOffscreen())) {
        offscreenReady = null;
        throw error;
      }
    }
  })();
  return offscreenReady;
}

let requestSeq = 0;

async function currentThreshold(): Promise<number> {
  try {
    const stored = await chrome.storage.local.get('settings');
    return normalizeSettings(stored['settings']).threshold;
  } catch {
    return DEFAULT_SETTINGS.threshold;
  }
}

function fallbackBackend(): 'webgpu' | 'wasm' {
  return status.backend === 'unknown' ? 'wasm' : status.backend;
}

async function analyze(request: AnalyzeRequest): Promise<ScoreResponse> {
  const cached = cacheGet(request.key);
  if (cached !== undefined) {
    return { ok: true, probability: cached, backend: fallbackBackend(), elapsedMs: 0 };
  }

  try {
    await ensureOffscreen();
  } catch (error) {
    status = { ...status, errors: status.errors + 1 };
    return { ok: false, error: `engine unavailable: ${String((error as Error)?.message ?? error)}` };
  }

  let raw: unknown;
  try {
    raw = await chrome.runtime.sendMessage({
      type: 'infer',
      requestId: `r${(requestSeq += 1)}`,
      src: request.src,
    });
  } catch (error) {
    // "Receiving end does not exist" means the offscreen document is gone
    // (Chrome can reclaim it). Forget the cached ensure so the next request
    // recreates it instead of failing until this worker is itself torn down.
    offscreenReady = null;
    status = { ...status, errors: status.errors + 1 };
    return { ok: false, error: `engine unavailable: ${String((error as Error)?.message ?? error)}` };
  }

  const reply = parseInferResponse(raw);
  if (reply === null) {
    status = { ...status, errors: status.errors + 1 };
    return { ok: false, error: 'malformed reply from inference document' };
  }
  if (!reply.response.ok) {
    status = { ...status, errors: status.errors + 1 };
    return reply.response;
  }

  cacheSet(request.key, reply.response.probability);
  status = {
    ...status,
    analyzed: status.analyzed + 1,
    flagged: status.flagged + (reply.response.probability >= (await currentThreshold()) ? 1 : 0),
  };
  return reply.response;
}

chrome.runtime.onMessage.addListener((raw, _sender, sendResponse) => {
  // Engine notifications come from our own offscreen document and are not part
  // of the page-facing request vocabulary, so they are matched first.
  const engine = parseEngineEvent(raw);
  if (engine !== null) {
    status =
      engine.type === 'engineReady'
        ? { ...status, ready: true, backend: engine.backend, modelIds: engine.modelIds, error: undefined }
        : { ...status, ready: false, error: engine.error };
    return false;
  }

  const request = parseRequest(raw);
  if (request === null) return undefined;

  switch (request.type) {
    case MESSAGE_TYPES.analyze:
      analyze(request).then(sendResponse, (error: unknown) =>
        sendResponse({ ok: false, error: String((error as Error)?.message ?? error) } satisfies ScoreResponse),
      );
      return true;

    case MESSAGE_TYPES.getState:
      // Touch the offscreen document so the popup shows a real engine state
      // rather than "not started" on a cold worker.
      void ensureOffscreen().catch(() => undefined);
      sendResponse({ status });
      return false;

    case MESSAGE_TYPES.updateSettings:
      void chrome.storage.local.set({ settings: request.settings });
      sendResponse({ ok: true });
      return false;

    default:
      return undefined;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  void ensureOffscreen().catch(() => undefined);
});
