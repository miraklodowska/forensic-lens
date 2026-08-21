/**
 * Browser-side parity harness.
 *
 * The offline evaluation decodes and resizes with Pillow; the extension does it
 * with `createImageBitmap` and a canvas. Those are different resamplers, and the
 * signal these detectors read lives in exactly the high frequencies a resampler
 * touches — so an accuracy number measured in Python only transfers if the
 * browser produces the same scores.
 *
 * This runs the real shipped modules (`planViews`, `aggregateLogits`, `fuse`)
 * against the real ONNX weights in a real browser, and publishes the per-image
 * probabilities so they can be diffed against the Python reference.
 */

import * as ort from 'onnxruntime-web';
import { planViews, type ViewPlan, type ViewSpec } from '../../src/core/views.ts';
import { aggregateLogits, fuse, type Aggregation, type FusionSpec } from '../../src/core/fusion.ts';

interface ModelSpec {
  id: string;
  role: 'cf' | 'sg';
  file: string;
  input: {
    name: string;
    size: number;
    layout: 'nchw' | 'nhwc';
    channelOrder: 'rgb' | 'bgr';
    normalize: { mean: number[]; std: number[] };
  };
  output: { name: string; kind: 'single-logit-sigmoid' | 'two-logit-softmax'; positiveIndex?: number };
  views: ViewSpec[];
  executionProviders?: ('webgpu' | 'wasm')[];
  aggregate: Aggregation;
}

interface PipelineSpec {
  models: ModelSpec[];
  fusion: FusionSpec;
}

declare global {
  interface Window {
    __parity?: { done: boolean; backend: string; rows: unknown[]; error?: string };
  }
}

const log = (msg: string) => {
  const el = document.getElementById('log')!;
  el.textContent = `${el.textContent}\n${msg}`;
  console.log(msg);
};

const canvases = new Map<number, HTMLCanvasElement>();
function canvasFor(size: number): HTMLCanvasElement {
  let c = canvases.get(size);
  if (!c) {
    c = document.createElement('canvas');
    c.width = c.height = size;
    canvases.set(size, c);
  }
  return c;
}

function renderView(bitmap: ImageBitmap, plan: ViewPlan, spec: ModelSpec, out: Float32Array, offset: number): void {
  const size = spec.input.size;
  const ctx = canvasFor(size).getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(bitmap, plan.sx, plan.sy, plan.sw, plan.sh, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const { mean, std } = spec.input.normalize;
  const order = spec.input.channelOrder === 'rgb' ? [0, 1, 2] : [2, 1, 0];
  const pixels = size * size;
  for (let c = 0; c < 3; c += 1) {
    const ch = order[c]!;
    const m = mean[c]!;
    const s = std[c]!;
    const base = offset + c * pixels;
    for (let i = 0; i < pixels; i += 1) out[base + i] = (data[i * 4 + ch]! / 255 - m) / s;
  }
}

function viewLogit(raw: Float32Array, index: number, spec: ModelSpec): number {
  if (spec.output.kind === 'single-logit-sigmoid') return raw[index]!;
  const positive = spec.output.positiveIndex ?? 1;
  const a = raw[index * 2]!;
  const b = raw[index * 2 + 1]!;
  return positive === 0 ? a - b : b - a;
}

async function main(): Promise<void> {
  window.__parity = { done: false, backend: 'unknown', rows: [] };
  ort.env.wasm.wasmPaths = '/vendor/ort/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'error';

  const pipeline: PipelineSpec = await (await fetch('/models/pipeline.json')).json();
  const images: { name: string; url: string }[] = await (await fetch('/images.json')).json();
  log(`pipeline: ${pipeline.models.map((m) => m.id).join(', ')}; ${images.length} images`);

  // ?ep=wasm / ?ep=webgpu overrides the per-model choice so the two can be diffed.
  const forced = new URLSearchParams(location.search).get('ep') as 'wasm' | 'webgpu' | null;
  const hasGpu = 'gpu' in navigator;

  const sessions: { spec: ModelSpec; session: ort.InferenceSession }[] = [];
  const used: string[] = [];
  for (const spec of pipeline.models) {
    const wanted = forced ? [forced] : (spec.executionProviders ?? ['webgpu', 'wasm']);
    const providers = wanted.filter((p) => p !== 'webgpu' || hasGpu);
    const bytes = await (await fetch(`/models/${spec.file}`)).arrayBuffer();
    let session: ort.InferenceSession | null = null;
    for (const provider of providers.length ? providers : ['wasm']) {
      try {
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: [provider as 'webgpu' | 'wasm'],
          graphOptimizationLevel: 'all',
        });
        used.push(`${spec.id}:${provider}`);
        log(`  loaded ${spec.id} on ${provider}`);
        break;
      } catch (err) {
        log(`  ${spec.id} ${provider} failed: ${String(err).slice(0, 140)}`);
      }
    }
    if (!session) throw new Error(`${spec.id}: no provider loaded it`);
    sessions.push({ spec, session });
  }
  const backend = used.join(' ');
  window.__parity.backend = backend;
  log(`backend: ${backend}`);

  const rows: unknown[] = [];
  for (const [i, img] of images.entries()) {
    try {
      const blob = await (await fetch(img.url)).blob();
      const bitmap = await createImageBitmap(blob);
      const logits: { cf: number; sg: number } = { cf: 0, sg: 0 };
      const perModel: Record<string, number> = {};

      for (const { spec, session } of sessions) {
        const plans = planViews({ width: bitmap.width, height: bitmap.height }, spec.views);
        const size = spec.input.size;
        const per = 3 * size * size;
        const buf = new Float32Array(plans.length * per);
        plans.forEach((p, k) => renderView(bitmap, p, spec, buf, k * per));
        const out = await session.run({
          [spec.input.name]: new ort.Tensor('float32', buf, [plans.length, 3, size, size]),
        });
        const raw = (out[spec.output.name] ?? Object.values(out)[0])!.data as Float32Array;
        const agg = aggregateLogits(
          plans.map((_, k) => viewLogit(raw, k, spec)),
          spec.aggregate,
        );
        logits[spec.role] = agg;
        perModel[spec.role] = agg;
      }

      const minEdge = Math.min(bitmap.width, bitmap.height);
      rows.push({
        name: img.name,
        width: bitmap.width,
        height: bitmap.height,
        cf: perModel['cf'],
        sg: perModel['sg'],
        probability: fuse(logits, minEdge, pipeline.fusion),
      });
      bitmap.close();
    } catch (err) {
      rows.push({ name: img.name, error: String(err).slice(0, 160) });
    }
    if ((i + 1) % 10 === 0) log(`  ${i + 1}/${images.length}`);
    window.__parity.rows = rows;
  }

  window.__parity = { done: true, backend, rows };
  log(`DONE ${rows.length} images on ${backend}`);
}

main().catch((err) => {
  const message = String(err?.message ?? err);
  log(`FAILED: ${message}`);
  window.__parity = { done: true, backend: 'none', rows: [], error: message };
});
