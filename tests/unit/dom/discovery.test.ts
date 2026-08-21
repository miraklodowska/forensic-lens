import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ImageDiscovery, evaluateImage } from '../../../src/content/discovery.ts';

/** Minimal stand-in for IntersectionObserver, which jsdom does not implement. */
class FakeIntersectionObserver {
  static instances: FakeIntersectionObserver[] = [];
  readonly observed = new Set<Element>();
  disconnected = false;
  constructor(private readonly callback: (entries: { target: Element; isIntersecting: boolean }[]) => void) {
    FakeIntersectionObserver.instances.push(this);
  }
  observe(el: Element): void {
    this.observed.add(el);
  }
  unobserve(el: Element): void {
    this.observed.delete(el);
  }
  disconnect(): void {
    this.disconnected = true;
    this.observed.clear();
  }
  enter(el: Element): void {
    this.callback([{ target: el, isIntersecting: true }]);
  }
}

function makeImage(src: string, width: number, height: number): HTMLImageElement {
  const img = document.createElement('img');
  img.setAttribute('src', src);
  // jsdom performs no layout or loading, so the intrinsic size is stubbed.
  Object.defineProperty(img, 'naturalWidth', { value: width, configurable: true });
  Object.defineProperty(img, 'naturalHeight', { value: height, configurable: true });
  Object.defineProperty(img, 'complete', { value: width > 0, configurable: true });
  return img;
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('evaluateImage', () => {
  it('accepts an image at or above the minimum size', () => {
    expect(evaluateImage(makeImage('https://e.com/a.jpg', 128, 128), 128)).toMatchObject({
      status: 'eligible',
      src: 'https://e.com/a.jpg',
    });
  });

  it('rejects images below the minimum size in either dimension', () => {
    expect(evaluateImage(makeImage('https://e.com/a.jpg', 127, 400), 128).status).toBe('too-small');
    expect(evaluateImage(makeImage('https://e.com/a.jpg', 400, 16), 128).status).toBe('too-small');
  });

  it('treats a not-yet-loaded image as pending rather than rejecting it', () => {
    expect(evaluateImage(makeImage('https://e.com/a.jpg', 0, 0), 128).status).toBe('pending');
  });

  it('rejects sources that cannot be analysed', () => {
    expect(evaluateImage(makeImage('javascript:0', 300, 300), 128).status).toBe('unsupported');
    const noSrc = makeImage('', 300, 300);
    noSrc.removeAttribute('src');
    expect(evaluateImage(noSrc, 128).status).toBe('unsupported');
  });

  it('analyses the variant the browser actually loaded, not the largest srcset candidate', () => {
    // The page laid out at 320px and the browser picked the small variant;
    // fetching big.jpg would download bytes the page never loaded.
    const img = makeImage('https://e.com/small.jpg', 320, 240);
    img.setAttribute('srcset', 'https://e.com/small.jpg 320w, https://e.com/big.jpg 1600w');
    Object.defineProperty(img, 'currentSrc', { value: 'https://e.com/small.jpg', configurable: true });
    expect(evaluateImage(img, 128)).toMatchObject({ src: 'https://e.com/small.jpg' });
  });

  it('falls back to the src attribute when currentSrc is not yet populated', () => {
    const img = makeImage('https://e.com/a.jpg', 300, 300);
    // jsdom never loads resources, so currentSrc is empty here by default.
    expect(evaluateImage(img, 128)).toMatchObject({ src: 'https://e.com/a.jpg' });
  });

  it('resolves relative sources against the document', () => {
    const img = makeImage('pic.jpg', 300, 300);
    expect(evaluateImage(img, 128).src).toBe(new URL('pic.jpg', document.baseURI).href);
  });

  it('skips images the extension has already marked', () => {
    const img = makeImage('https://e.com/a.jpg', 300, 300);
    img.dataset['forensicLens'] = 'done';
    expect(evaluateImage(img, 128).status).toBe('already-handled');
  });
});

describe('ImageDiscovery', () => {
  let onCandidate: ReturnType<typeof vi.fn<(img: HTMLImageElement, src: string) => void>>;

  beforeEach(() => {
    document.body.innerHTML = '';
    FakeIntersectionObserver.instances = [];
    onCandidate = vi.fn();
  });

  const start = (minImageSize = 128) => {
    const discovery = new ImageDiscovery({
      root: document,
      minImageSize,
      onCandidate,
      observerFactory: (cb) => new FakeIntersectionObserver(cb) as unknown as IntersectionObserver,
    });
    discovery.start();
    return { discovery, observer: () => FakeIntersectionObserver.instances[0]! };
  };

  it('finds images present at start-up but only reports them once visible', async () => {
    const img = makeImage('https://e.com/a.jpg', 300, 300);
    document.body.append(img);
    const { observer } = start();
    await flush();

    expect(onCandidate).not.toHaveBeenCalled();
    expect(observer().observed.has(img)).toBe(true);

    observer().enter(img);
    await flush();
    expect(onCandidate).toHaveBeenCalledWith(img, 'https://e.com/a.jpg');
  });

  it('picks up images appended later by infinite scroll, including nested ones', async () => {
    const { observer } = start();
    const container = document.createElement('div');
    const img = makeImage('https://e.com/lazy.jpg', 300, 300);
    container.append(img);
    document.body.append(container);
    await flush();

    expect(observer().observed.has(img)).toBe(true);
    observer().enter(img);
    await flush();
    expect(onCandidate).toHaveBeenCalledWith(img, 'https://e.com/lazy.jpg');
  });

  it('re-analyses an image whose src is swapped by a lazy loader', async () => {
    const img = makeImage('https://e.com/placeholder.jpg', 300, 300);
    document.body.append(img);
    const { observer } = start();
    await flush();
    observer().enter(img);
    await flush();
    onCandidate.mockClear();

    img.setAttribute('src', 'https://e.com/real.jpg');
    await flush();
    observer().enter(img);
    await flush();
    expect(onCandidate).toHaveBeenCalledWith(img, 'https://e.com/real.jpg');
  });

  it('does not report the same image and src twice', async () => {
    const img = makeImage('https://e.com/a.jpg', 300, 300);
    document.body.append(img);
    const { observer } = start();
    await flush();
    observer().enter(img);
    observer().enter(img);
    await flush();
    expect(onCandidate).toHaveBeenCalledTimes(1);
  });

  it('ignores images below the size gate entirely', async () => {
    document.body.append(makeImage('https://e.com/icon.png', 16, 16));
    const { observer } = start();
    await flush();
    expect(observer().observed.size).toBe(0);
    expect(onCandidate).not.toHaveBeenCalled();
  });

  it('waits for load before judging the size of a pending image', async () => {
    const img = makeImage('https://e.com/a.jpg', 0, 0);
    document.body.append(img);
    const { observer } = start();
    await flush();
    expect(observer().observed.size).toBe(0);

    Object.defineProperty(img, 'naturalWidth', { value: 300, configurable: true });
    Object.defineProperty(img, 'naturalHeight', { value: 300, configurable: true });
    img.dispatchEvent(new Event('load'));
    await flush();
    expect(observer().observed.has(img)).toBe(true);
  });

  it('stop() disconnects observers and stops reporting', async () => {
    const { discovery, observer } = start();
    discovery.stop();
    document.body.append(makeImage('https://e.com/late.jpg', 300, 300));
    await flush();
    expect(observer().disconnected).toBe(true);
    expect(onCandidate).not.toHaveBeenCalled();
  });

  it('reports how many images it has seen and skipped', async () => {
    document.body.append(makeImage('https://e.com/a.jpg', 300, 300));
    document.body.append(makeImage('https://e.com/icon.png', 10, 10));
    const { discovery } = start();
    await flush();
    expect(discovery.stats()).toMatchObject({ seen: 2, tooSmall: 1, observing: 1 });
  });
});
