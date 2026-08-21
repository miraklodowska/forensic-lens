/**
 * The inference document.
 *
 * MV3 service workers cannot run WebGPU and get torn down mid-task, and content
 * scripts inherit the host page's CSP, so neither can host an ONNX session. An
 * offscreen document is the one extension context with a GPU adapter, a canvas
 * and a lifetime we control — so all decoding and inference happens here and
 * the service worker is only a router.
 */

import * as ort from 'onnxruntime-web';
import { parseInferRequest, type InferResponse, type ScoreResponse } from '../shared/messages.ts';
import { planViews, type ViewPlan, type ViewSpec } from '../core/views.ts';
import { aggregateLogits, fuse, type Aggregation, type FusionSpec } from '../core/fusion.ts';

interface ModelSpec {
  readonly id: string;
  /** Which slot this model fills in the fusion. */
  readonly role: 'cf' | 'sg';
  readonly file: string;
  readonly input: {
    readonly name: string;
    readonly size: number;
    readonly layout: 'nchw' | 'nhwc';
    readonly channelOrder: 'rgb' | 'bgr';
    readonly normalize: { readonly mean: number[]; readonly std: number[] };
  };
  readonly output: {
    readonly name: string;
    readonly kind: 'single-logit-sigmoid' | 'two-logit-softmax';
    /** Index of the AI class for two-logit models. */
    readonly positiveIndex?: number;
  };
  readonly views: readonly ViewSpec[];
  readonly aggregate: Aggregation;
  /**
   * Providers to try for this model, in order.
   *
   * Per-model rather than global because they are not interchangeable: ONNX
   * Runtime's WebGPU backend does not implement this build's INT8 quantized
   * operators and silently returns a *constant* for every input instead of
   * failing, which looks exactly like a working detector that has stopped
   * discriminating. Models carrying quantized weights pin themselves to WASM.
   */
  readonly executionProviders?: readonly ('webgpu' | 'wasm')[];
}

interface PipelineSpec {
  readonly models: readonly ModelSpec[];
  readonly fusion: FusionSpec;
}

let sessions: { spec: ModelSpec; session: ort.InferenceSession }[] = [];
let fusion: FusionSpec | null = null;
let backend: 'webgpu' | 'wasm' = 'wasm';
let readyPromise: Promise<void> | null = null;

function post(message: unknown): void {
  chrome.runtime.sendMessage(message).catch(() => {
    /* worker may be asleep; nothing to do */
  });
}

/**
 * Whether the WebGPU execution provider can actually be used *here*.
 *
 * `'gpu' in navigator` is true inside an offscreen document, but the adapter
 * request does not necessarily resolve: an offscreen document has no
 * compositor, and on this Chrome `InferenceSession.create({webgpu})` spins at
 * 100% CPU forever instead of failing. A hung provider is worse than an absent
 * one — the engine never reports ready, every analysis awaits it, and the
 * extension looks installed but silently scores nothing. So the adapter is
 * probed behind a timeout first, and a null/slow answer means WASM.
 */
async function webgpuUsable(timeoutMs = 3000): Promise<boolean> {
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    const adapter = await Promise.race([
      gpu.requestAdapter(),
      new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs)),
    ]);
    return adapter !== null && adapter !== undefined;
  } catch {
    return false;
  }
}

/** Rejects rather than hanging, so one bad provider cannot wedge startup. */
function withTimeout<T>(work: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    work,
    new Promise<never>((_, reject) => setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)),
  ]);
}

async function buildSessions(): Promise<void> {
  ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/ort/');
  // Extension pages are not cross-origin isolated, so SharedArrayBuffer — and
  // therefore ORT's threaded WASM — is unavailable. Asking for more than one
  // thread only spawns workers that immediately fall back.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'error';

  const res = await fetch(chrome.runtime.getURL('models/pipeline.json'));
  if (!res.ok) throw new Error(`pipeline.json: HTTP ${res.status}`);
  const pipeline = (await res.json()) as PipelineSpec;

  const hasGpu = await webgpuUsable();
  if (!hasGpu) console.info('[forensic-lens] WebGPU unavailable here; using WASM');
  const built: typeof sessions = [];
  const used: ('webgpu' | 'wasm')[] = [];

  for (const spec of pipeline.models) {
    const wanted = spec.executionProviders ?? ['webgpu', 'wasm'];
    const providers = wanted.filter((p) => p !== 'webgpu' || hasGpu);
    const bytes = await (await fetch(chrome.runtime.getURL(`models/${spec.file}`))).arrayBuffer();

    let session: ort.InferenceSession | null = null;
    let lastError: unknown;
    for (const provider of providers.length > 0 ? providers : (['wasm'] as const)) {
      try {
        session = await withTimeout(
          ort.InferenceSession.create(bytes, {
            executionProviders: [provider],
            graphOptimizationLevel: 'all',
          }),
          provider === 'webgpu' ? 20000 : 120000,
          `${spec.id} on ${provider}`,
        );
        used.push(provider);
        break;
      } catch (error) {
        // WebGPU can fail at session-build time on blocklisted or headless
        // GPUs; that is expected, and WASM is a correct (slower) fallback.
        lastError = error;
        console.warn(`[forensic-lens] ${spec.id}: ${provider} unavailable:`, error);
      }
    }
    if (session === null) throw new Error(`${spec.id}: no execution provider could load it: ${String(lastError)}`);
    built.push({ spec, session });
  }

  sessions = built;
  fusion = pipeline.fusion;
  // Report the fastest provider actually in play; the popup shows one label.
  backend = used.includes('webgpu') ? 'webgpu' : 'wasm';
}

function ensureReady(): Promise<void> {
  readyPromise ??= buildSessions().then(
    () => {
      const modelIds = sessions.map((s) => s.spec.id);
      post({ type: 'engineReady', backend, modelIds });
      // Mirrored onto the global so the end-to-end harness can read the real
      // backend without depending on message timing.
      (globalThis as Record<string, unknown>)['__flStatus'] = { ready: true, backend, modelIds };
    },
    (error: unknown) => {
      readyPromise = null; // let a later request retry
      const message = String((error as Error)?.message ?? error);
      post({ type: 'engineFailed', error: message });
      (globalThis as Record<string, unknown>)['__flStatus'] = { ready: false, error: message };
      throw error;
    },
  );
  return readyPromise;
}

async function loadBitmap(src: string): Promise<ImageBitmap> {
  // This is a real network request, not a cache read: Chrome partitions its
  // HTTP cache by top-level site, and a chrome-extension:// document is never
  // the page's site, so the page's cached copy is invisible here and every
  // analysed image costs one outbound GET. It has to be a refetch because the
  // page's pixels are unreachable: a cross-origin image taints any canvas it
  // is drawn into, and extension messaging is JSON — an ImageBitmap cannot be
  // transferred across it. `omit` keeps cookies off the request: we re-read a
  // public image, not act as the user. `force-cache` still helps the rare
  // same-partition case and repeated analyses within this document's lifetime.
  const res = await fetch(src, { credentials: 'omit', cache: 'force-cache' });
  if (!res.ok) throw new Error(`fetch failed: HTTP ${res.status}`);
  return createImageBitmap(await res.blob());
}

const canvases = new Map<number, OffscreenCanvas>();

function canvasFor(size: number): OffscreenCanvas {
  let canvas = canvases.get(size);
  if (!canvas) {
    canvas = new OffscreenCanvas(size, size);
    canvases.set(size, canvas);
  }
  return canvas;
}

/** Renders one view and packs it into the model's expected tensor layout. */
function renderView(bitmap: ImageBitmap, plan: ViewPlan, spec: ModelSpec, out: Float32Array, offset: number): void {
  const size = spec.input.size;
  const ctx = canvasFor(size).getContext('2d', { willReadFrequently: true });
  if (ctx === null) throw new Error('no 2d context');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(bitmap, plan.sx, plan.sy, plan.sw, plan.sh, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);

  const { channelOrder, layout } = spec.input;
  const { mean, std } = spec.input.normalize;
  const order = channelOrder === 'rgb' ? [0, 1, 2] : [2, 1, 0];
  const pixels = size * size;

  if (layout === 'nchw') {
    for (let c = 0; c < 3; c += 1) {
      const channel = order[c]!;
      const m = mean[c]!;
      const s = std[c]!;
      const base = offset + c * pixels;
      for (let i = 0; i < pixels; i += 1) {
        out[base + i] = (data[i * 4 + channel]! / 255 - m) / s;
      }
    }
    return;
  }
  for (let i = 0; i < pixels; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[offset + i * 3 + c] = (data[i * 4 + order[c]!]! / 255 - mean[c]!) / std[c]!;
    }
  }
}

/** Raw model output -> log-odds that view `index` is AI-generated. */
function viewLogit(raw: Float32Array, index: number, spec: ModelSpec): number {
  if (spec.output.kind === 'single-logit-sigmoid') return raw[index]!;
  // Two-logit softmax: the log-odds of a class is just the logit difference.
  const positive = spec.output.positiveIndex ?? 1;
  const a = raw[index * 2]!;
  const b = raw[index * 2 + 1]!;
  return positive === 0 ? a - b : b - a;
}

async function runModel(bitmap: ImageBitmap, entry: { spec: ModelSpec; session: ort.InferenceSession }): Promise<number> {
  const { spec, session } = entry;
  const plans = planViews({ width: bitmap.width, height: bitmap.height }, spec.views);
  if (plans.length === 0) throw new Error('no view applies to this image');

  const size = spec.input.size;
  const perView = 3 * size * size;
  const buffer = new Float32Array(plans.length * perView);
  plans.forEach((plan, i) => renderView(bitmap, plan, spec, buffer, i * perView));

  const dims = spec.input.layout === 'nchw' ? [plans.length, 3, size, size] : [plans.length, size, size, 3];
  const output = await session.run({ [spec.input.name]: new ort.Tensor('float32', buffer, dims) });
  const raw = (output[spec.output.name] ?? Object.values(output)[0])!.data as Float32Array;

  return aggregateLogits(
    plans.map((_, i) => viewLogit(raw, i, spec)),
    spec.aggregate,
  );
}

async function infer(src: string): Promise<ScoreResponse> {
  const started = performance.now();
  await ensureReady();
  if (fusion === null) throw new Error('pipeline not loaded');

  const bitmap = await loadBitmap(src);
  try {
    const logits: { cf: number; sg: number } = { cf: 0, sg: 0 };
    for (const entry of sessions) {
      logits[entry.spec.role] = await runModel(bitmap, entry);
    }
    return {
      ok: true,
      probability: fuse(logits, Math.min(bitmap.width, bitmap.height), fusion),
      backend,
      elapsedMs: Math.round(performance.now() - started),
    };
  } finally {
    bitmap.close();
  }
}

chrome.runtime.onMessage.addListener((raw, _sender, sendResponse) => {
  const request = parseInferRequest(raw);
  if (request === null) return undefined;

  infer(request.src).then(
    (response) =>
      sendResponse({ type: 'infer-result', requestId: request.requestId, response } satisfies InferResponse),
    (error: unknown) =>
      sendResponse({
        type: 'infer-result',
        requestId: request.requestId,
        response: { ok: false, error: String((error as Error)?.message ?? error).slice(0, 200) },
      } satisfies InferResponse),
  );
  return true; // keep the channel open for the async reply
});

void ensureReady().catch(() => {
  /* reported via engineFailed */
});
