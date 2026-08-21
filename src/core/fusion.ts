/**
 * Turning two detector outputs into the one number a badge can show.
 *
 * Two separate problems live here, and it is worth keeping them apart.
 *
 * 1. Calibration. The detectors rank well but their probabilities do not mean
 *    what they say: Community Forensics puts almost every real image at ~0.000
 *    and spreads generated ones from 0.02 to 0.99, so "flag at 0.65" rejects
 *    most of the AI images it has already ranked above every real one. On our
 *    corpus that model alone scores AUC 0.82 but only 29% recall at 0.65.
 *    Ranking quality and calibration are independent, and a fixed threshold
 *    consumes only the second.
 *
 * 2. Fusion. The two detectors fail in different places, predictably. The
 *    fingerprint reader (CF) needs native-resolution pixels and goes blind on
 *    small images — 1% recall on 256px Flux samples. The semantic one (SigLIP)
 *    does not care about size but fires on real faces and paintings. Image size
 *    is known for free, so it gates how much the size-sensitive model is
 *    trusted.
 *
 * Both are a single affine map in log-odds space, fitted offline against a
 * public corpus (see docs/EVALUATION.md). Because the map is monotone in each
 * model's logit it cannot change either model's ranking — it only decides where
 * 0.65 falls.
 */

export interface FusionWeights {
  /**
   * Maximum weight on the fingerprint detector's logit, reached on large
   * images. The size gate scales this down towards zero and never past it.
   */
  readonly cfMax: number;
  /** Weight on the semantic detector's aggregated logit. */
  readonly sg: number;
}

/**
 * Multiplicative gate on the fingerprint detector, in [0, 1].
 *
 * This replaces an earlier additive formulation (`w_cf + cfBig*big +
 * cfSmall*small`) that was wrong in a way worth recording, because the mistake
 * is easy to make again. The intent was "trust CF less on small images", and
 * additive interaction terms express that only while CF's logit is informative.
 * Below ~256px CF saturates at roughly -10.5 for everything, so
 * `cfSmall * cf * small` stopped being an interaction and became a constant
 * +1.78-per-octave push towards "AI" — exactly the standalone size prior the
 * design was meant to exclude. Worse, the effective weight crossed zero at
 * ~160px and went negative, so a confident "real" from CF actively voted AI:
 * a 128px image both detectors called real scored 0.77 and was flagged.
 *
 * A multiplicative gate cannot degenerate that way. The weight is
 * `cfMax * sigmoid(a*(octaves - t))`, bounded in [0, cfMax] by construction, so
 * losing confidence in CF can only ever mean discounting it towards silence.
 */
export interface SizeGate {
  /** Steepness of the transition, in inverse octaves. */
  readonly a: number;
  /** Octave offset at which CF is trusted half as much as its maximum. */
  readonly t: number;
}

export interface FusionSpec {
  readonly kind: 'size-gated-linear';
  readonly weights: FusionWeights;
  readonly gate: SizeGate;
  readonly intercept: number;
  /**
   * Shift that places the fitted decision boundary at the reported threshold.
   * The offline fit is class-balanced, so its natural boundary is p=0.5; adding
   * logit(0.65) moves that to 0.65 without touching the ranking.
   */
  readonly thresholdOffset: number;
  /** Crop size the size gate is measured against. */
  readonly referenceEdge: number;
}

export function sigmoid(x: number): number {
  // Branch on sign so exp() never overflows for large-magnitude logits.
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/**
 * How much native resolution the fingerprint detector had to work with, in
 * octaves relative to its crop size. Clamped so a favicon or a gigapixel scan
 * cannot swing the gate arbitrarily far.
 */
export function resolutionOctaves(minEdge: number, referenceEdge: number): number {
  const clamped = Math.min(8192, Math.max(32, minEdge));
  return Math.log2(clamped / referenceEdge);
}

export interface ModelLogits {
  readonly cf: number;
  readonly sg: number;
}

/**
 * @param minEdge shortest edge of the *source* image in pixels, not of the crop.
 * @returns calibrated probability that the image is AI-generated.
 */
export function sizeGate(minEdge: number, spec: FusionSpec): number {
  const octaves = resolutionOctaves(minEdge, spec.referenceEdge);
  return sigmoid(spec.gate.a * (octaves - spec.gate.t));
}

/** Weight actually applied to the fingerprint detector. Never negative. */
export function effectiveCfWeight(minEdge: number, spec: FusionSpec): number {
  return spec.weights.cfMax * sizeGate(minEdge, spec);
}

export function fuse(logits: ModelLogits, minEdge: number, spec: FusionSpec): number {
  // There is deliberately no additive size term. Given one, the offline fit
  // learns "large images are real" — true of any corpus whose AI samples skew
  // small, false of the world. Size enters only as a multiplicative gate on how
  // far one model is trusted, so it can discount a vote but never cast one.
  const z =
    effectiveCfWeight(minEdge, spec) * logits.cf +
    spec.weights.sg * logits.sg +
    spec.intercept +
    spec.thresholdOffset;

  return sigmoid(z);
}

export type Aggregation = 'mean' | 'max' | 'trimmed' | 'median';

/**
 * Combines per-view logits into one.
 *
 * `mean` is the default and what the shipped calibration was fitted against.
 * `trimmed` drops the extremes, which helps when a crop lands on flat sky or a
 * blown-out highlight — no forensic signal there, but it still votes.
 */
export function aggregateLogits(logits: readonly number[], how: Aggregation): number {
  if (logits.length === 0) throw new Error('aggregateLogits: no views');
  if (logits.length === 1) return logits[0]!;
  switch (how) {
    case 'max':
      return Math.max(...logits);
    case 'mean':
      return logits.reduce((a, b) => a + b, 0) / logits.length;
    case 'median': {
      const s = [...logits].sort((a, b) => a - b);
      const mid = s.length >> 1;
      return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
    }
    case 'trimmed': {
      if (logits.length <= 2) return logits.reduce((a, b) => a + b, 0) / logits.length;
      const s = [...logits].sort((a, b) => a - b).slice(1, -1);
      return s.reduce((a, b) => a + b, 0) / s.length;
    }
  }
}
