import { describe, expect, it } from 'vitest';
import {
  classify,
  formatPercent,
  probabilityFromOutput,
  sigmoid,
  softmax2,
} from '../../src/core/scoring.ts';
import type { ModelIoSpec } from '../../src/core/scoring.ts';

const sigmoidIo: ModelIoSpec = { outputKind: 'single-logit-sigmoid' };

describe('sigmoid', () => {
  it('maps 0 to 0.5 and is monotonic', () => {
    expect(sigmoid(0)).toBe(0.5);
    expect(sigmoid(2)).toBeGreaterThan(sigmoid(1));
    expect(sigmoid(-2)).toBeLessThan(sigmoid(-1));
  });

  it('stays finite and in [0,1] for extreme logits', () => {
    for (const x of [-1e6, -800, 800, 1e6]) {
      const p = sigmoid(x);
      expect(Number.isFinite(p)).toBe(true);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
    expect(sigmoid(-1000)).toBe(0);
    expect(sigmoid(1000)).toBe(1);
  });

  it('matches known values', () => {
    expect(sigmoid(1)).toBeCloseTo(0.7310585786, 10);
    expect(sigmoid(-3)).toBeCloseTo(0.0474258732, 10);
  });
});

describe('softmax2', () => {
  it('returns the positive-class probability and is shift invariant', () => {
    expect(softmax2(0, 0)).toBeCloseTo(0.5, 12);
    expect(softmax2(1, 3)).toBeCloseTo(softmax2(101, 103), 12);
    // softmax2(a, b) is P(class 1), so swapping the logits mirrors the score.
    expect(softmax2(3, 1)).toBeCloseTo(sigmoid(-2), 12);
    expect(softmax2(3, 1) + softmax2(1, 3)).toBeCloseTo(1, 12);
  });

  it('does not overflow on large logits', () => {
    expect(softmax2(1000, -1000)).toBeCloseTo(0, 12);
    expect(softmax2(-1000, 1000)).toBeCloseTo(1, 12);
  });
});

describe('probabilityFromOutput', () => {
  // Float32Array narrows the literals, so compare at float32-meaningful precision.
  it('applies sigmoid to a single logit', () => {
    expect(probabilityFromOutput(new Float32Array([0.4]), sigmoidIo)).toBeCloseTo(sigmoid(0.4), 6);
  });

  it('applies softmax when the model emits two logits', () => {
    const io: ModelIoSpec = { ...sigmoidIo, outputKind: 'two-logit-softmax' };
    expect(probabilityFromOutput(new Float32Array([1, 3]), io)).toBeCloseTo(softmax2(1, 3), 6);
  });

  it('rejects an output whose length does not match the declared kind', () => {
    expect(() => probabilityFromOutput(new Float32Array([1, 2]), sigmoidIo)).toThrow(/1 value/i);
    expect(() =>
      probabilityFromOutput(new Float32Array([1]), { ...sigmoidIo, outputKind: 'two-logit-softmax' }),
    ).toThrow(/2 value/i);
  });

  it('rejects a non-finite logit instead of reporting a bogus score', () => {
    expect(() => probabilityFromOutput(new Float32Array([NaN]), sigmoidIo)).toThrow(/finite/i);
  });
});

describe('classify', () => {
  it('uses the configured threshold inclusively at the boundary', () => {
    expect(classify(0.65, 0.65).verdict).toBe('ai-generated');
    expect(classify(0.6499, 0.65).verdict).toBe('not-flagged');
    expect(classify(1, 0.65).verdict).toBe('ai-generated');
    expect(classify(0, 0.65).verdict).toBe('not-flagged');
  });

  it('honours a non-default threshold', () => {
    expect(classify(0.5, 0.9).verdict).toBe('not-flagged');
    expect(classify(0.95, 0.9).verdict).toBe('ai-generated');
  });

  it('reports confidence bands used for badge colour', () => {
    expect(classify(0.98, 0.65).band).toBe('high');
    expect(classify(0.8, 0.65).band).toBe('medium');
    expect(classify(0.66, 0.65).band).toBe('low');
    expect(classify(0.2, 0.65).band).toBe('none');
  });

  it('rejects an out-of-range probability or threshold', () => {
    expect(() => classify(1.2, 0.65)).toThrow(/probability/i);
    expect(() => classify(0.5, 1.5)).toThrow(/threshold/i);
  });
});

describe('formatPercent', () => {
  it('renders whole percentages with no decimals', () => {
    expect(formatPercent(0.6512)).toBe('65%');
    expect(formatPercent(0)).toBe('0%');
    expect(formatPercent(1)).toBe('100%');
  });

  it('never rounds a sub-threshold score up to a threshold-looking label', () => {
    expect(formatPercent(0.999)).toBe('99%');
    expect(formatPercent(0.6499)).toBe('64%');
  });
});
