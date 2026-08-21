/**
 * The message vocabulary shared by content script, service worker and the
 * offscreen inference document.
 *
 * Messages arrive from content scripts, which run inside pages the user does not
 * control, so nothing is trusted on arrival: the `parse*` functions validate
 * shape *and* range and return null rather than throwing, because a malformed
 * message must be dropped without taking the worker down.
 */

import { isAnalyzableSource } from '../core/image-key.ts';
import { normalizeSettings } from './settings.ts';
import type { Settings } from './settings.ts';

export const MESSAGE_TYPES = {
  analyze: 'analyze',
  getState: 'getState',
  updateSettings: 'updateSettings',
} as const;

export type MessageType = (typeof MESSAGE_TYPES)[keyof typeof MESSAGE_TYPES];

/** Execution provider a score was produced on. */
export type Backend = 'webgpu' | 'wasm';

const BACKENDS = new Set<string>(['webgpu', 'wasm']);

/**
 * An unbounded key would let a hostile page pin arbitrary strings in the
 * worker's LRU cache; real keys are normalized URLs or short data: digests.
 */
const MAX_KEY_LENGTH = 2048;
/**
 * Large because `src` may be a lossless PNG data: URL carrying a decoded image
 * (see the content script's pixel handoff). A 1500px frame is a few MB.
 */
const MAX_SRC_LENGTH = 12 * 1024 * 1024;
const MAX_REQUEST_ID_LENGTH = 64;

/** content script -> service worker: please score this image. */
export interface AnalyzeRequest {
  readonly type: typeof MESSAGE_TYPES.analyze;
  /** Source the offscreen document should fetch and decode. */
  readonly src: string;
  /** Bounded, stable cache key for `src` (see core/image-key.ts). */
  readonly key: string;
  /** Intrinsic pixel size as measured in the page. */
  readonly width: number;
  readonly height: number;
}

export interface GetStateRequest {
  readonly type: typeof MESSAGE_TYPES.getState;
}

export interface UpdateSettingsRequest {
  readonly type: typeof MESSAGE_TYPES.updateSettings;
  readonly settings: Settings;
}

export type Request = AnalyzeRequest | GetStateRequest | UpdateSettingsRequest;

/** service worker -> content script: the outcome of one analysis. */
export type ScoreResponse =
  | { readonly ok: true; readonly probability: number; readonly backend: Backend; readonly elapsedMs: number }
  | { readonly ok: false; readonly error: string };

function record(raw: unknown): Record<string, unknown> | null {
  return typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : null;
}

function boundedString(value: unknown, maxLength: number): string | null {
  return typeof value === 'string' && value.length > 0 && value.length <= maxLength ? value : null;
}

function positiveNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

function parseAnalyze(source: Record<string, unknown>): AnalyzeRequest | null {
  const src = boundedString(source['src'], MAX_SRC_LENGTH);
  const key = boundedString(source['key'], MAX_KEY_LENGTH);
  const width = positiveNumber(source['width']);
  const height = positiveNumber(source['height']);
  if (src === null || key === null || width === null || height === null) return null;
  // Checked here as well as in the scorer: this is the boundary where
  // page-controlled input first reaches the extension.
  if (!isAnalyzableSource(src)) return null;
  return { type: MESSAGE_TYPES.analyze, src, key, width, height };
}

/** @returns the validated request, or null when it must be ignored. */
export function parseRequest(raw: unknown): Request | null {
  const source = record(raw);
  if (source === null) return null;

  switch (source['type']) {
    case MESSAGE_TYPES.analyze:
      return parseAnalyze(source);
    case MESSAGE_TYPES.getState:
      return { type: MESSAGE_TYPES.getState };
    case MESSAGE_TYPES.updateSettings:
      // Never stored as sent: clamping here keeps an out-of-range threshold from
      // ever reaching storage or the badge logic.
      return { type: MESSAGE_TYPES.updateSettings, settings: normalizeSettings(source['settings']) };
    default:
      return null;
  }
}

/** Validates a score reply, including the ranges the UI relies on. */
export function parseResponse(raw: unknown): ScoreResponse | null {
  const source = record(raw);
  if (source === null) return null;

  if (source['ok'] === true) {
    const probability = source['probability'];
    if (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0 || probability > 1) {
      return null;
    }
    const backend = source['backend'];
    if (typeof backend !== 'string' || !BACKENDS.has(backend)) return null;
    const elapsedMs = source['elapsedMs'];
    if (typeof elapsedMs !== 'number' || !Number.isFinite(elapsedMs) || elapsedMs < 0) return null;
    return { ok: true, probability, backend: backend as Backend, elapsedMs };
  }
  if (source['ok'] === false) {
    const error = source['error'];
    return { ok: false, error: typeof error === 'string' ? error : 'analysis failed' };
  }
  return null;
}

/* ------------------------------------------------------------------------- */
/* Worker <-> offscreen envelopes. These carry a requestId because one          */
/* offscreen document serves every tab, and a reply that does not name the     */
/* request it answers is not trusted.                                          */
/* ------------------------------------------------------------------------- */

export interface InferRequest {
  readonly type: 'infer';
  readonly requestId: string;
  /**
   * An `http(s)` or `data:` URL. Both resolve from the offscreen document;
   * `blob:` URLs do not, because they are scoped to the page's origin, and
   * `isAnalyzableSource` rejects them before they reach here.
   */
  readonly src: string;
}

export interface InferResponse {
  readonly type: 'infer-result';
  readonly requestId: string;
  readonly response: ScoreResponse;
}

export function parseInferRequest(raw: unknown): InferRequest | null {
  const source = record(raw);
  if (source === null || source['type'] !== 'infer') return null;
  const requestId = boundedString(source['requestId'], MAX_REQUEST_ID_LENGTH);
  const src = boundedString(source['src'], MAX_SRC_LENGTH);
  if (requestId === null || src === null || !isAnalyzableSource(src)) return null;
  return { type: 'infer', requestId, src };
}

export function parseInferResponse(raw: unknown): InferResponse | null {
  const source = record(raw);
  if (source === null || source['type'] !== 'infer-result') return null;
  const requestId = boundedString(source['requestId'], MAX_REQUEST_ID_LENGTH);
  const response = parseResponse(source['response']);
  if (requestId === null || response === null) return null;
  return { type: 'infer-result', requestId, response };
}

/* ------------------------------------------------------------------------- */
/* Engine status. Weights ship inside the extension, so there is no install    */
/* step to report — the only thing that varies is which execution provider the */
/* offscreen document managed to build a session on.                           */
/* ------------------------------------------------------------------------- */

export interface EngineStatus {
  readonly ready: boolean;
  readonly backend: Backend | 'unknown';
  readonly modelIds: readonly string[];
  readonly error?: string | undefined;
  readonly analyzed: number;
  readonly flagged: number;
  readonly errors: number;
}

/** service worker -> popup, in reply to `getState`. */
export interface StateResponse {
  readonly status: EngineStatus;
}

export interface EngineReadyEvent {
  readonly type: 'engineReady';
  readonly backend: Backend;
  readonly modelIds: readonly string[];
}

export interface EngineFailedEvent {
  readonly type: 'engineFailed';
  readonly error: string;
}

export function parseEngineEvent(raw: unknown): EngineReadyEvent | EngineFailedEvent | null {
  const source = record(raw);
  if (source === null) return null;
  if (source['type'] === 'engineReady') {
    const backend = source['backend'];
    if (typeof backend !== 'string' || !BACKENDS.has(backend)) return null;
    const ids = source['modelIds'];
    if (!Array.isArray(ids) || ids.some((v) => typeof v !== 'string')) return null;
    return { type: 'engineReady', backend: backend as Backend, modelIds: ids as string[] };
  }
  if (source['type'] === 'engineFailed') {
    const error = source['error'];
    return { type: 'engineFailed', error: typeof error === 'string' ? error.slice(0, 500) : 'engine failed' };
  }
  return null;
}
