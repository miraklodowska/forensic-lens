/**
 * Finds images worth analysing: those on the page at load, those inserted by
 * infinite scroll, and those whose src is swapped in by a lazy loader.
 *
 * Nothing is reported until the image is actually near the viewport, so a page
 * with 5,000 thumbnails costs 5,000 cheap DOM checks rather than 5,000
 * inference runs.
 */

import { imageCacheKey, isAnalyzableSource } from '../core/image-key.ts';

export type EvaluationStatus =
  | 'eligible'
  | 'pending'
  | 'too-small'
  | 'unsupported'
  | 'already-handled';

export interface Evaluation {
  readonly status: EvaluationStatus;
  readonly src: string | null;
  readonly key: string | null;
  readonly width: number;
  readonly height: number;
}

const NOT_ELIGIBLE = (status: EvaluationStatus, width = 0, height = 0): Evaluation => ({
  status,
  src: null,
  key: null,
  width,
  height,
});

function rawSource(img: HTMLImageElement): string | null {
  // currentSrc is the URL the browser actually loaded for this element. Every
  // analysed image costs one outbound GET (the offscreen document's refetch is
  // a guaranteed cache miss under Chrome's partitioned HTTP cache), so
  // analysing anything other than what the page displayed — e.g. the largest
  // srcset candidate — would download variants the page never loaded.
  if (img.currentSrc) return img.currentSrc;
  // Not selected yet (or a DOM without loading, as in tests): the src
  // attribute is the best available answer.
  return img.getAttribute('src');
}

function measure(img: HTMLImageElement): { width: number; height: number } {
  const attrWidth = Number(img.getAttribute('width') ?? 0);
  const attrHeight = Number(img.getAttribute('height') ?? 0);
  return {
    width: img.naturalWidth || img.clientWidth || (Number.isFinite(attrWidth) ? attrWidth : 0),
    height: img.naturalHeight || img.clientHeight || (Number.isFinite(attrHeight) ? attrHeight : 0),
  };
}

/** Pure decision about a single image; no side effects, so it is easy to test. */
export function evaluateImage(img: HTMLImageElement, minImageSize: number): Evaluation {
  if (img.dataset['forensicLens'] === 'done') return NOT_ELIGIBLE('already-handled');

  const raw = rawSource(img);
  if (!raw || !isAnalyzableSource(raw)) return NOT_ELIGIBLE('unsupported');

  const { width, height } = measure(img);
  // A zero measurement means "not loaded yet", not "tiny": re-check on load.
  if (width === 0 || height === 0) return NOT_ELIGIBLE('pending');
  if (width < minImageSize || height < minImageSize) return NOT_ELIGIBLE('too-small', width, height);

  const isData = raw.trim().toLowerCase().startsWith('data:');
  const src = isData ? raw.trim() : (imageCacheKey(raw, img.ownerDocument.baseURI) ?? raw);
  const key = imageCacheKey(src, img.ownerDocument.baseURI);
  if (!key) return NOT_ELIGIBLE('unsupported');
  return { status: 'eligible', src, key, width, height };
}

export interface DiscoveryOptions {
  readonly root: Document;
  readonly minImageSize: number;
  readonly onCandidate: (img: HTMLImageElement, src: string) => void;
  readonly observerFactory?: (
    callback: (entries: { target: Element; isIntersecting: boolean }[]) => void,
  ) => IntersectionObserver;
  /** How far outside the viewport to start analysing, in CSS pixels. */
  readonly rootMargin?: string;
}

export interface DiscoveryStats {
  readonly seen: number;
  readonly tooSmall: number;
  readonly unsupported: number;
  readonly observing: number;
  readonly reported: number;
}

interface ImageState {
  observedSrc: string | null;
  reportedSrc: string | null;
  loadListener: (() => void) | null;
}

export class ImageDiscovery {
  readonly #options: DiscoveryOptions;
  readonly #states = new WeakMap<HTMLImageElement, ImageState>();
  readonly #observed = new Set<HTMLImageElement>();
  #intersection: IntersectionObserver | null = null;
  #mutation: MutationObserver | null = null;
  #running = false;
  #seen = 0;
  #tooSmall = 0;
  #unsupported = 0;
  #reported = 0;

  constructor(options: DiscoveryOptions) {
    this.#options = options;
  }

  start(): void {
    if (this.#running) return;
    this.#running = true;

    const factory =
      this.#options.observerFactory ??
      ((cb: (entries: { target: Element; isIntersecting: boolean }[]) => void) =>
        new IntersectionObserver(cb as IntersectionObserverCallback, {
          rootMargin: this.#options.rootMargin ?? '200px',
        }));

    this.#intersection = factory((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) this.#report(entry.target as HTMLImageElement);
      }
    });

    this.#mutation = new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === 'attributes') {
          const img = record.target as HTMLImageElement;
          this.#consider(img);
          // A src/srcset swap updates currentSrc only once the new resource
          // has loaded, which is after this mutation fires — so look again on
          // 'load' or the swapped-in image would be missed.
          this.#waitForLoad(img, this.#stateFor(img));
          continue;
        }
        for (const node of record.addedNodes) this.#considerSubtree(node);
      }
    });
    this.#mutation.observe(this.#options.root.documentElement ?? this.#options.root, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset'],
    });

    this.scanNow();
  }

  stop(): void {
    this.#running = false;
    this.#intersection?.disconnect();
    this.#mutation?.disconnect();
    this.#observed.clear();
  }

  /** Re-scan the whole document, e.g. after a single-page-app navigation. */
  scanNow(): void {
    for (const img of this.#options.root.images ?? []) this.#consider(img);
  }

  stats(): DiscoveryStats {
    return {
      seen: this.#seen,
      tooSmall: this.#tooSmall,
      unsupported: this.#unsupported,
      observing: this.#observed.size,
      reported: this.#reported,
    };
  }

  #considerSubtree(node: Node): void {
    if (!(node instanceof Element)) return;
    if (node instanceof HTMLImageElement) this.#consider(node);
    for (const img of node.querySelectorAll('img')) this.#consider(img);
  }

  #stateFor(img: HTMLImageElement): ImageState {
    let state = this.#states.get(img);
    if (!state) {
      state = { observedSrc: null, reportedSrc: null, loadListener: null };
      this.#states.set(img, state);
      this.#seen += 1;
    }
    return state;
  }

  #consider(img: HTMLImageElement): void {
    if (!this.#running) return;
    const state = this.#stateFor(img);
    const evaluation = evaluateImage(img, this.#options.minImageSize);

    switch (evaluation.status) {
      case 'pending':
        this.#waitForLoad(img, state);
        return;
      case 'too-small':
        this.#tooSmall += 1;
        return;
      case 'unsupported':
        this.#unsupported += 1;
        return;
      case 'already-handled':
        return;
      case 'eligible':
        break;
    }

    const src = evaluation.src!;
    if (state.observedSrc === src && state.reportedSrc !== src) return; // already queued
    if (state.reportedSrc === src) return; // already delivered
    state.observedSrc = src;
    state.reportedSrc = null;
    this.#observed.add(img);
    this.#intersection?.observe(img);
  }

  #waitForLoad(img: HTMLImageElement, state: ImageState): void {
    if (state.loadListener) return;
    const listener = () => {
      img.removeEventListener('load', listener);
      state.loadListener = null;
      this.#consider(img);
    };
    state.loadListener = listener;
    img.addEventListener('load', listener);
  }

  #report(img: HTMLImageElement): void {
    if (!this.#running) return;
    const state = this.#states.get(img);
    if (!state?.observedSrc) return;
    if (state.reportedSrc === state.observedSrc) return;
    state.reportedSrc = state.observedSrc;
    this.#observed.delete(img);
    this.#intersection?.unobserve(img);
    this.#reported += 1;
    this.#options.onCandidate(img, state.observedSrc);
  }
}
