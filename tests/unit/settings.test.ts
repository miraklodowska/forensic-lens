import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, normalizeSettings } from '../../src/shared/settings.ts';

describe('DEFAULT_SETTINGS', () => {
  it('defaults the AI-confidence threshold to 65%', () => {
    expect(DEFAULT_SETTINGS.threshold).toBe(0.65);
  });

  it('is analysis-on, blur-off, with a minimum image size gate', () => {
    expect(DEFAULT_SETTINGS.enabled).toBe(true);
    expect(DEFAULT_SETTINGS.blurFlagged).toBe(false);
    expect(DEFAULT_SETTINGS.showBadges).toBe(true);
    expect(DEFAULT_SETTINGS.minImageSize).toBe(128);
  });
});

describe('normalizeSettings', () => {
  it('fills defaults for missing keys', () => {
    expect(normalizeSettings({})).toEqual(DEFAULT_SETTINGS);
    expect(normalizeSettings(undefined)).toEqual(DEFAULT_SETTINGS);
    expect(normalizeSettings(null)).toEqual(DEFAULT_SETTINGS);
  });

  it('keeps valid overrides', () => {
    const s = normalizeSettings({ threshold: 0.9, blurFlagged: true, minImageSize: 200 });
    expect(s.threshold).toBe(0.9);
    expect(s.blurFlagged).toBe(true);
    expect(s.minImageSize).toBe(200);
    expect(s.enabled).toBe(true);
  });

  it('clamps an out-of-range threshold into [0.01, 0.99]', () => {
    expect(normalizeSettings({ threshold: 5 }).threshold).toBe(0.99);
    expect(normalizeSettings({ threshold: -1 }).threshold).toBe(0.01);
  });

  it('clamps the minimum image size to a sane pixel range', () => {
    expect(normalizeSettings({ minImageSize: 1 }).minImageSize).toBe(32);
    expect(normalizeSettings({ minImageSize: 99999 }).minImageSize).toBe(2048);
  });

  it('falls back to defaults for wrongly-typed values instead of throwing', () => {
    const s = normalizeSettings({ threshold: 'high', enabled: 'yes', minImageSize: NaN });
    expect(s.threshold).toBe(DEFAULT_SETTINGS.threshold);
    expect(s.enabled).toBe(DEFAULT_SETTINGS.enabled);
    expect(s.minImageSize).toBe(DEFAULT_SETTINGS.minImageSize);
  });

  it('drops unknown keys so stored junk cannot reach the rest of the extension', () => {
    expect(Object.keys(normalizeSettings({ evil: 1 })).sort()).toEqual(
      Object.keys(DEFAULT_SETTINGS).sort(),
    );
  });

  it('rounds the threshold to whole percent steps for stable display', () => {
    expect(normalizeSettings({ threshold: 0.65432 }).threshold).toBe(0.65);
  });
});
