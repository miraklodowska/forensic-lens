import { describe, expect, it } from 'vitest';
import { imageCacheKey, isAnalyzableSource } from '../../src/core/image-key.ts';

describe('imageCacheKey', () => {
  it('ignores the fragment but keeps the query, which can select a different image', () => {
    expect(imageCacheKey('https://e.com/a.jpg#x')).toBe(imageCacheKey('https://e.com/a.jpg'));
    expect(imageCacheKey('https://e.com/a.jpg?w=100')).not.toBe(imageCacheKey('https://e.com/a.jpg?w=200'));
  });

  it('is stable across equivalent absolute forms', () => {
    expect(imageCacheKey('https://E.com/a.jpg')).toBe(imageCacheKey('https://e.com/a.jpg'));
    expect(imageCacheKey('https://e.com:443/a.jpg')).toBe(imageCacheKey('https://e.com/a.jpg'));
  });

  it('resolves relative URLs against the page', () => {
    expect(imageCacheKey('/a.jpg', 'https://e.com/dir/page.html')).toBe('https://e.com/a.jpg');
    expect(imageCacheKey('b.jpg', 'https://e.com/dir/page.html')).toBe('https://e.com/dir/b.jpg');
  });

  it('collapses long data URLs to a bounded, content-derived key', () => {
    const long = `data:image/png;base64,${'A'.repeat(10_000)}`;
    const key = imageCacheKey(long)!;
    expect(key.length).toBeLessThan(128);
    expect(key).toBe(imageCacheKey(long));
    expect(key).not.toBe(imageCacheKey(`data:image/png;base64,${'B'.repeat(10_000)}`));
    expect(key.startsWith('data:')).toBe(true);
  });

  it('returns null for sources that cannot be keyed stably', () => {
    expect(imageCacheKey('')).toBeNull();
    expect(imageCacheKey('   ')).toBeNull();
    expect(imageCacheKey('not a url')).toBeNull();
  });
});

describe('isAnalyzableSource', () => {
  it('accepts http(s), data and blob images', () => {
    expect(isAnalyzableSource('https://e.com/a.jpg')).toBe(true);
    expect(isAnalyzableSource('http://e.com/a.jpg')).toBe(true);
    expect(isAnalyzableSource('data:image/png;base64,AAAA')).toBe(true);
    expect(isAnalyzableSource('blob:https://e.com/1234')).toBe(true);
  });

  it('rejects sources that are not fetchable image bytes', () => {
    expect(isAnalyzableSource('')).toBe(false);
    expect(isAnalyzableSource('javascript:alert(1)')).toBe(false);
    expect(isAnalyzableSource('about:blank')).toBe(false);
    expect(isAnalyzableSource('chrome-extension://abc/icon.png')).toBe(false);
    expect(isAnalyzableSource('data:text/html,<b>hi</b>')).toBe(false);
  });
});
