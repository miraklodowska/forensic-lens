/**
 * Visible score badges, drawn in a shadow root so the host page's CSS cannot
 * restyle, cover or hide a forensic result.
 */

import { formatPercent } from '../core/scoring.ts';
import type { Classification } from '../core/scoring.ts';

export const BADGE_HOST_ID = 'forensic-lens-badge-host';

export interface BadgeDisplayOptions {
  readonly showBadges: boolean;
  readonly blurFlagged: boolean;
}

const HOST_STYLE = `
:host { all: initial; }
.badge {
  position: absolute;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin: 6px;
  padding: 2px 6px;
  border-radius: 6px;
  font: 600 11px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color: #fff;
  background: rgba(31, 41, 55, 0.88);
  border: 1px solid rgba(255, 255, 255, 0.25);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
  pointer-events: auto;
  white-space: nowrap;
  z-index: 2147483647;
}
.badge[data-verdict="ai-generated"] { background: rgba(153, 27, 27, 0.92); }
.badge[data-band="high"] { background: rgba(127, 29, 29, 0.95); }
.badge[data-band="medium"] { background: rgba(154, 52, 18, 0.92); }
.badge[data-band="low"] { background: rgba(146, 64, 14, 0.92); }
.badge[data-verdict="not-flagged"] { background: rgba(31, 41, 55, 0.82); font-weight: 500; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: 0.9; }
`;

const BLUR_FILTER = 'blur(16px)';

export class BadgeLayer {
  #host: HTMLElement | null = null;
  #shadow: ShadowRoot | null = null;
  readonly #badges = new Map<HTMLImageElement, HTMLElement>();
  readonly #blurred = new Map<HTMLImageElement, string>();

  get count(): number {
    return this.#badges.size;
  }

  badgeElementFor(img: HTMLImageElement): HTMLElement | null {
    return this.#badges.get(img) ?? null;
  }

  render(img: HTMLImageElement, result: Classification, options: BadgeDisplayOptions): void {
    this.#applyBlur(img, result, options);
    if (!options.showBadges) {
      this.#badges.get(img)?.remove();
      this.#badges.delete(img);
      return;
    }

    const shadow = this.#ensureHost();
    let badge = this.#badges.get(img);
    if (!badge) {
      badge = img.ownerDocument.createElement('div');
      badge.className = 'badge';
      const dot = img.ownerDocument.createElement('span');
      dot.className = 'dot';
      const label = img.ownerDocument.createElement('span');
      label.className = 'label';
      badge.append(dot, label);
      shadow.append(badge);
      this.#badges.set(img, badge);
    }

    const label = badge.querySelector('.label')!;
    const percent = formatPercent(result.probability);
    label.textContent =
      result.verdict === 'ai-generated' ? `AI ${percent}` : `AI ${percent} · below ${formatPercent(result.threshold)}`;
    badge.dataset['verdict'] = result.verdict;
    badge.dataset['band'] = result.band;
    badge.title = `Forensic Lens: ${percent} likelihood this image is AI-generated (threshold ${formatPercent(result.threshold)}). Analysed on this device.`;
    this.#position(img, badge);
  }

  /** Re-anchors every badge; call on scroll, resize and layout changes. */
  reposition(): void {
    for (const [img, badge] of [...this.#badges]) {
      if (!img.isConnected) {
        badge.remove();
        this.#badges.delete(img);
        continue;
      }
      this.#position(img, badge);
    }
  }

  remove(img: HTMLImageElement): void {
    this.#badges.get(img)?.remove();
    this.#badges.delete(img);
    this.#unblur(img);
  }

  destroy(): void {
    for (const badge of this.#badges.values()) badge.remove();
    this.#badges.clear();
    for (const img of [...this.#blurred.keys()]) this.#unblur(img);
    this.#host?.remove();
    this.#host = null;
    this.#shadow = null;
  }

  #ensureHost(): ShadowRoot {
    if (this.#shadow) return this.#shadow;
    const host = document.createElement('div');
    host.id = BADGE_HOST_ID;
    host.style.cssText =
      'position:absolute;top:0;left:0;width:0;height:0;margin:0;padding:0;border:0;pointer-events:none;z-index:2147483647;';
    const shadow = host.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = HOST_STYLE;
    shadow.append(style);
    document.documentElement.append(host);
    this.#host = host;
    this.#shadow = shadow;
    return shadow;
  }

  #position(img: HTMLImageElement, badge: HTMLElement): void {
    const rect = img.getBoundingClientRect();
    badge.style.top = `${Math.round(rect.top + window.scrollY)}px`;
    badge.style.left = `${Math.round(rect.left + window.scrollX)}px`;
  }

  #applyBlur(img: HTMLImageElement, result: Classification, options: BadgeDisplayOptions): void {
    const shouldBlur = options.blurFlagged && result.verdict === 'ai-generated';
    if (shouldBlur) {
      if (this.#blurred.has(img)) return;
      this.#blurred.set(img, img.style.filter);
      img.style.filter = img.style.filter ? `${img.style.filter} ${BLUR_FILTER}` : BLUR_FILTER;
      img.dataset['forensicLensBlur'] = 'on';
      return;
    }
    this.#unblur(img);
  }

  #unblur(img: HTMLImageElement): void {
    if (!this.#blurred.has(img)) return;
    img.style.filter = this.#blurred.get(img)!;
    this.#blurred.delete(img);
    delete img.dataset['forensicLensBlur'];
  }
}
