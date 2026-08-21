import { beforeEach, describe, expect, it } from 'vitest';
import { BADGE_HOST_ID, BadgeLayer } from '../../../src/content/badges.ts';

function makeImage(): HTMLImageElement {
  const img = document.createElement('img');
  img.src = 'https://e.com/a.jpg';
  img.getBoundingClientRect = () =>
    ({ x: 10, y: 20, top: 20, left: 10, width: 300, height: 200, right: 310, bottom: 220 }) as DOMRect;
  document.body.append(img);
  return img;
}

function badgeFor(layer: BadgeLayer, img: HTMLImageElement): HTMLElement | null {
  return layer.badgeElementFor(img);
}

describe('BadgeLayer', () => {
  let layer: BadgeLayer;

  beforeEach(() => {
    document.body.innerHTML = '';
    document.documentElement.querySelector(`#${BADGE_HOST_ID}`)?.remove();
    layer = new BadgeLayer();
  });

  it('renders into a shadow root so page CSS cannot restyle or hide badges', () => {
    layer.render(makeImage(), { probability: 0.9, threshold: 0.65, verdict: 'ai-generated', band: 'medium' }, {
      showBadges: true,
      blurFlagged: false,
    });
    const host = document.getElementById(BADGE_HOST_ID)!;
    expect(host).not.toBeNull();
    expect(host.shadowRoot).not.toBeNull();
    expect(host.shadowRoot!.querySelector('style')).not.toBeNull();
  });

  it('shows the score on a flagged image', () => {
    const img = makeImage();
    layer.render(img, { probability: 0.873, threshold: 0.65, verdict: 'ai-generated', band: 'medium' }, {
      showBadges: true,
      blurFlagged: false,
    });
    const badge = badgeFor(layer, img)!;
    expect(badge.textContent).toContain('87%');
    expect(badge.dataset['verdict']).toBe('ai-generated');
    expect(badge.dataset['band']).toBe('medium');
  });

  it('shows a score badge for analysed images below the threshold too', () => {
    const img = makeImage();
    layer.render(img, { probability: 0.12, threshold: 0.65, verdict: 'not-flagged', band: 'none' }, {
      showBadges: true,
      blurFlagged: false,
    });
    const badge = badgeFor(layer, img)!;
    expect(badge.textContent).toContain('12%');
    expect(badge.dataset['verdict']).toBe('not-flagged');
  });

  it('positions the badge over the image using its viewport rect and scroll offset', () => {
    const img = makeImage();
    Object.defineProperty(window, 'scrollX', { value: 5, configurable: true });
    Object.defineProperty(window, 'scrollY', { value: 100, configurable: true });
    layer.render(img, { probability: 0.9, threshold: 0.65, verdict: 'ai-generated', band: 'medium' }, {
      showBadges: true,
      blurFlagged: false,
    });
    const badge = badgeFor(layer, img)!;
    expect(badge.style.top).toBe('120px'); // rect.top 20 + scrollY 100
    expect(badge.style.left).toBe('15px'); // rect.left 10 + scrollX 5
  });

  it('updates an existing badge in place instead of stacking duplicates', () => {
    const img = makeImage();
    const settings = { showBadges: true, blurFlagged: false };
    layer.render(img, { probability: 0.2, threshold: 0.65, verdict: 'not-flagged', band: 'none' }, settings);
    layer.render(img, { probability: 0.99, threshold: 0.65, verdict: 'ai-generated', band: 'high' }, settings);
    expect(layer.count).toBe(1);
    expect(badgeFor(layer, img)!.textContent).toContain('99%');
  });

  it('blurs a flagged image only when blur is enabled, and restores it when disabled', () => {
    const img = makeImage();
    const flagged = { probability: 0.9, threshold: 0.65, verdict: 'ai-generated', band: 'medium' } as const;
    layer.render(img, flagged, { showBadges: true, blurFlagged: true });
    expect(img.dataset['forensicLensBlur']).toBe('on');
    expect(img.style.filter).toContain('blur');

    layer.render(img, flagged, { showBadges: true, blurFlagged: false });
    expect(img.dataset['forensicLensBlur']).toBeUndefined();
    expect(img.style.filter).not.toContain('blur');
  });

  it('never blurs an image below the threshold', () => {
    const img = makeImage();
    layer.render(img, { probability: 0.3, threshold: 0.65, verdict: 'not-flagged', band: 'none' }, {
      showBadges: true,
      blurFlagged: true,
    });
    expect(img.style.filter).not.toContain('blur');
  });

  it('hides badges when the user turns them off but keeps the record', () => {
    const img = makeImage();
    layer.render(img, { probability: 0.9, threshold: 0.65, verdict: 'ai-generated', band: 'high' }, {
      showBadges: false,
      blurFlagged: false,
    });
    expect(badgeFor(layer, img)).toBeNull();
    expect(layer.count).toBe(0);
  });

  it('removes badges and restores every image on destroy', () => {
    const img = makeImage();
    layer.render(img, { probability: 0.9, threshold: 0.65, verdict: 'ai-generated', band: 'high' }, {
      showBadges: true,
      blurFlagged: true,
    });
    layer.destroy();
    expect(document.getElementById(BADGE_HOST_ID)).toBeNull();
    expect(img.style.filter).not.toContain('blur');
    expect(layer.count).toBe(0);
  });

  it('drops badges for images detached from the document on reposition', () => {
    const img = makeImage();
    layer.render(img, { probability: 0.9, threshold: 0.65, verdict: 'ai-generated', band: 'high' }, {
      showBadges: true,
      blurFlagged: false,
    });
    img.remove();
    layer.reposition();
    expect(layer.count).toBe(0);
  });
});
