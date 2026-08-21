import { describe, expect, it } from 'vitest';
import {
  aggregateLogits,
  effectiveCfWeight,
  fuse,
  resolutionOctaves,
  sizeGate,
  sigmoid,
  type FusionSpec,
} from '../../src/core/fusion.ts';
import pipeline from '../../models/pipeline.json' with { type: 'json' };

const SHIPPED = pipeline.fusion as unknown as FusionSpec;
const SIZES = [32, 48, 64, 96, 128, 160, 224, 256, 320, 384, 512, 768, 1024, 2048, 4096, 8192];

describe('size gate', () => {
  it('never applies a negative weight to the fingerprint detector', () => {
    // The regression this guards: an additive gate drove the effective weight
    // through zero at ~160px, so a confident "real" from a saturated detector
    // started voting "AI". A multiplicative gate cannot do that, and this test
    // fails loudly if anyone reintroduces the additive form.
    for (const size of SIZES) {
      expect(effectiveCfWeight(size, SHIPPED)).toBeGreaterThanOrEqual(0);
    }
  });

  it('is bounded above by cfMax and increases with resolution', () => {
    let previous = -Infinity;
    for (const size of SIZES) {
      const w = effectiveCfWeight(size, SHIPPED);
      expect(w).toBeLessThanOrEqual(SHIPPED.weights.cfMax + 1e-12);
      expect(w).toBeGreaterThanOrEqual(previous - 1e-12);
      previous = w;
    }
  });

  it('keeps the gate itself inside [0,1]', () => {
    for (const size of SIZES) {
      const g = sizeGate(size, SHIPPED);
      expect(g).toBeGreaterThan(0);
      expect(g).toBeLessThan(1);
    }
  });
});

describe('fuse', () => {
  it('never flags an image both detectors call real, at any size', () => {
    // The concrete failure from review: at 128px this scored 0.766 and was
    // flagged, because size alone was pushing towards "AI".
    for (const size of SIZES) {
      const p = fuse({ cf: -10.5, sg: -10 }, size, SHIPPED);
      expect(p).toBeLessThan(0.65);
    }
  });

  it('flags an image both detectors call generated, at any size', () => {
    for (const size of SIZES) {
      expect(fuse({ cf: 10, sg: 10 }, size, SHIPPED)).toBeGreaterThanOrEqual(0.65);
    }
  });

  it('is monotonic in each detector logit', () => {
    const at = (cf: number, sg: number) => fuse({ cf, sg }, 512, SHIPPED);
    expect(at(-5, 0)).toBeLessThan(at(5, 0));
    expect(at(0, -5)).toBeLessThan(at(0, 5));
  });

  it('returns a probability for degenerate sizes rather than NaN', () => {
    for (const size of [1, 2, 1e6]) {
      const p = fuse({ cf: 0, sg: 0 }, size, SHIPPED);
      expect(Number.isFinite(p)).toBe(true);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
});

describe('resolutionOctaves', () => {
  it('is zero at the reference edge and clamps the extremes', () => {
    expect(resolutionOctaves(384, 384)).toBeCloseTo(0, 12);
    expect(resolutionOctaves(1, 384)).toBe(resolutionOctaves(32, 384));
    expect(resolutionOctaves(99999, 384)).toBe(resolutionOctaves(8192, 384));
  });
});

describe('aggregateLogits', () => {
  it('averages, and trims the extremes when asked', () => {
    expect(aggregateLogits([1, 2, 3], 'mean')).toBeCloseTo(2, 12);
    expect(aggregateLogits([-100, 1, 2, 3, 100], 'trimmed')).toBeCloseTo(2, 12);
    expect(aggregateLogits([1, 5, 3], 'max')).toBe(5);
    expect(aggregateLogits([1, 5, 3], 'median')).toBe(3);
  });

  it('rejects an empty view list rather than inventing a score', () => {
    expect(() => aggregateLogits([], 'mean')).toThrow();
  });
});

describe('sigmoid', () => {
  it('does not overflow on large-magnitude logits', () => {
    expect(sigmoid(1000)).toBe(1);
    expect(sigmoid(-1000)).toBe(0);
    expect(sigmoid(0)).toBe(0.5);
  });
});
