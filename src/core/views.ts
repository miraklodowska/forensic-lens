/**
 * View geometry: how one source image becomes the fixed-size square crops the
 * detector actually sees.
 *
 * A detector trained on 384x384 crops has two very different things it can be
 * shown for a 4000px photo: the whole frame squeezed down to 384 (composition
 * survives, per-pixel generator fingerprints do not), or a 384px window of the
 * original pixels (fingerprints survive, composition does not). Downscaling is
 * a low-pass filter and the traces these models key on live in exactly the high
 * frequencies it removes — measured on our corpus, switching Community
 * Forensics from the official downscale to a native-resolution crop moved
 * Midjourney's median score from 0.85 to 1.00.
 *
 * Every view is expressed as a source rectangle plus an output size, so the
 * whole thing is one `drawImage` call and stays trivially portable to the
 * Python reference implementation used for offline evaluation.
 */

export type ViewSpec =
  /** Resize so the shortest edge is `shortestEdge`, then take a centre crop. */
  | {
      readonly id: string;
      readonly kind: 'scale';
      readonly shortestEdge: number;
      readonly crop: number;
      readonly maxUpscale?: number;
    }
  /** Crop at the source image's own resolution — no resampling at all. */
  | { readonly id: string; readonly kind: 'native'; readonly position: NativePosition; readonly crop: number }
  /** Squeeze the entire frame into a square, ignoring aspect ratio. */
  | { readonly id: string; readonly kind: 'squash'; readonly size: number };

export type NativePosition = 'center' | 'tl' | 'tr' | 'bl' | 'br';

export interface Size {
  readonly width: number;
  readonly height: number;
}

/** A single `drawImage(src, sx, sy, sw, sh, 0, 0, out, out)` instruction. */
export interface ViewPlan {
  readonly id: string;
  readonly sx: number;
  readonly sy: number;
  readonly sw: number;
  readonly sh: number;
  readonly out: number;
  /** True when sw/sh differ from `out`, i.e. the canvas has to resample. */
  readonly resamples: boolean;
}

function assertPositive(size: Size): void {
  if (!Number.isFinite(size.width) || !Number.isFinite(size.height) || size.width <= 0 || size.height <= 0) {
    throw new Error(`invalid source size ${size.width}x${size.height}`);
  }
}

/**
 * Resolves a view against a concrete source size.
 *
 * Returns null when the view cannot be honoured for this image — currently only
 * when a `scale` view would have to upscale past its `maxUpscale` limit.
 * Upscaling invents smooth interpolated detail that reads as "not generated" to
 * a fingerprint detector, so we drop the view rather than feed the model a
 * confidently wrong input.
 */
export function planView(source: Size, spec: ViewSpec): ViewPlan | null {
  assertPositive(source);
  const { width: w, height: h } = source;

  if (spec.kind === 'squash') {
    return { id: spec.id, sx: 0, sy: 0, sw: w, sh: h, out: spec.size, resamples: true };
  }

  if (spec.kind === 'scale') {
    const scale = spec.shortestEdge / Math.min(w, h);
    if (spec.maxUpscale !== undefined && scale > spec.maxUpscale) return null;
    // Window of source pixels that maps onto exactly `crop` output pixels.
    const win = spec.crop / scale;
    const sw = Math.min(w, win);
    const sh = Math.min(h, win);
    return {
      id: spec.id,
      sx: (w - sw) / 2,
      sy: (h - sh) / 2,
      sw,
      sh,
      out: spec.crop,
      resamples: Math.abs(sw - spec.crop) > 0.5 || Math.abs(sh - spec.crop) > 0.5,
    };
  }

  // Native: crop source pixels 1:1. Only if the image is smaller than the crop
  // do we scale up, and then by the minimum amount needed to fill it.
  const crop = spec.crop;
  const upscale = Math.min(w, h) < crop ? crop / Math.min(w, h) : 1;
  const win = crop / upscale;
  const sw = Math.min(w, win);
  const sh = Math.min(h, win);
  const maxX = w - sw;
  const maxY = h - sh;
  const positions: Record<NativePosition, readonly [number, number]> = {
    center: [maxX / 2, maxY / 2],
    tl: [0, 0],
    tr: [maxX, 0],
    bl: [0, maxY],
    br: [maxX, maxY],
  };
  const [sx, sy] = positions[spec.position];
  return { id: spec.id, sx, sy, sw, sh, out: crop, resamples: upscale !== 1 };
}

/** Resolves a list of views, dropping any that do not apply to this image. */
export function planViews(source: Size, specs: readonly ViewSpec[]): ViewPlan[] {
  const out: ViewPlan[] = [];
  const seen = new Set<string>();
  for (const spec of specs) {
    const plan = planView(source, spec);
    if (plan === null) continue;
    // Distinct specs can collapse onto the same rectangle — on a 384x384 source
    // every native position is identical, and a scale view that cannot upscale
    // lands exactly where the native one does. Running the model twice on the
    // same pixels would silently double that view's weight in the mean.
    const key = [plan.sx, plan.sy, plan.sw, plan.sh].map((n) => n.toFixed(2)).join(':') + `:${plan.out}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(plan);
  }
  return out;
}
