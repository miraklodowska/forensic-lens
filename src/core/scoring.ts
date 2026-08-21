/**
 * Turning raw model output into the number shown on a badge.
 *
 * The probability is always "probability that this image is AI-generated";
 * `outputKind` says how to get there from the raw tensor.
 */

/** How a detector reports its verdict, mirroring `models/pipeline.json`. */
export interface ModelIoSpec {
  readonly outputKind: 'single-logit-sigmoid' | 'two-logit-softmax';
}

export type Verdict = 'ai-generated' | 'not-flagged';
export type ConfidenceBand = 'none' | 'low' | 'medium' | 'high';

export interface Classification {
  readonly probability: number;
  readonly threshold: number;
  readonly verdict: Verdict;
  readonly band: ConfidenceBand;
}

export function sigmoid(logit: number): number {
  // Branch on the sign so exp() never overflows for large-magnitude logits.
  if (logit >= 0) {
    const z = Math.exp(-logit);
    return 1 / (1 + z);
  }
  const z = Math.exp(logit);
  return z / (1 + z);
}

/** Probability of class 1 under a 2-logit softmax, computed without overflow. */
export function softmax2(logit0: number, logit1: number): number {
  return sigmoid(logit1 - logit0);
}

export function probabilityFromOutput(
  output: ArrayLike<number>,
  io: ModelIoSpec,
): number {
  const expected = io.outputKind === 'single-logit-sigmoid' ? 1 : 2;
  if (output.length !== expected) {
    throw new Error(
      `model output for ${io.outputKind} must contain exactly ${expected} value(s), got ${output.length}`,
    );
  }
  for (let i = 0; i < output.length; i += 1) {
    if (!Number.isFinite(output[i]!)) {
      throw new Error(`model output contains a non-finite logit at index ${i}`);
    }
  }
  return io.outputKind === 'single-logit-sigmoid'
    ? sigmoid(output[0]!)
    : softmax2(output[0]!, output[1]!);
}

function bandFor(probability: number, threshold: number): ConfidenceBand {
  if (probability < threshold) return 'none';
  if (probability >= 0.95) return 'high';
  if (probability >= 0.8) return 'medium';
  return 'low';
}

export function classify(probability: number, threshold: number): Classification {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error(`probability must be within [0,1], got ${probability}`);
  }
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error(`threshold must be within [0,1], got ${threshold}`);
  }
  return {
    probability,
    threshold,
    // Inclusive at the boundary: a score of exactly the threshold is flagged.
    verdict: probability >= threshold ? 'ai-generated' : 'not-flagged',
    band: bandFor(probability, threshold),
  };
}

/** Truncates rather than rounds, so a 64.9% score never reads as "65%". */
export function formatPercent(probability: number): string {
  return `${Math.floor(probability * 100)}%`;
}
