/**
 * User settings. Everything here lives in chrome.storage.local and never
 * leaves the browser.
 */

export interface Settings {
  /** Master switch for automatic analysis. */
  readonly enabled: boolean;
  /** Probability at or above which an image is reported as AI-generated. */
  readonly threshold: number;
  /** Draw the score badge over analysed images. */
  readonly showBadges: boolean;
  /** Blur images that cross the threshold until the user hovers them. */
  readonly blurFlagged: boolean;
  /** Images narrower or shorter than this (CSS px) are skipped. */
  readonly minImageSize: number;
}

export const DEFAULT_SETTINGS: Settings = {
  enabled: true,
  threshold: 0.65,
  showBadges: true,
  blurFlagged: false,
  minImageSize: 128,
};

export const THRESHOLD_RANGE = { min: 0.01, max: 0.99 } as const;
export const MIN_IMAGE_SIZE_RANGE = { min: 32, max: 2048 } as const;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function boolOr(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

/**
 * Coerces arbitrary stored data into a valid Settings object. Invalid values
 * fall back to defaults and out-of-range values clamp; this never throws,
 * because a corrupt storage entry must not brick the extension.
 */
export function normalizeSettings(raw: unknown): Settings {
  const source = (typeof raw === 'object' && raw !== null ? raw : {}) as Record<string, unknown>;

  const rawThreshold = source['threshold'];
  const threshold =
    typeof rawThreshold === 'number' && Number.isFinite(rawThreshold)
      ? // Snap to whole percent so the slider, the badge and the stored value agree.
        Math.round(clamp(rawThreshold, THRESHOLD_RANGE.min, THRESHOLD_RANGE.max) * 100) / 100
      : DEFAULT_SETTINGS.threshold;

  const rawMin = source['minImageSize'];
  const minImageSize =
    typeof rawMin === 'number' && Number.isFinite(rawMin)
      ? Math.round(clamp(rawMin, MIN_IMAGE_SIZE_RANGE.min, MIN_IMAGE_SIZE_RANGE.max))
      : DEFAULT_SETTINGS.minImageSize;

  return {
    enabled: boolOr(source['enabled'], DEFAULT_SETTINGS.enabled),
    threshold,
    showBadges: boolOr(source['showBadges'], DEFAULT_SETTINGS.showBadges),
    blurFlagged: boolOr(source['blurFlagged'], DEFAULT_SETTINGS.blurFlagged),
    minImageSize,
  };
}

export const SETTINGS_STORAGE_KEY = 'settings';
