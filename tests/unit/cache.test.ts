import { describe, expect, it } from 'vitest';
import { LruCache } from '../../src/core/cache.ts';

describe('LruCache', () => {
  it('stores and retrieves values', () => {
    const cache = new LruCache<string, number>(3);
    cache.set('a', 1);
    expect(cache.get('a')).toBe(1);
    expect(cache.get('missing')).toBeUndefined();
  });

  it('never exceeds its capacity', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.set('b', 2);
    cache.set('c', 3);
    expect(cache.size).toBe(2);
  });

  it('evicts the least recently used entry', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.set('b', 2);
    cache.get('a'); // 'a' is now the most recent, so 'b' is next out.
    cache.set('c', 3);
    expect(cache.has('b')).toBe(false);
    expect(cache.get('a')).toBe(1);
    expect(cache.get('c')).toBe(3);
  });

  it('counts hits and misses for the popup stats', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.get('a');
    cache.get('a');
    cache.get('b');
    expect(cache.stats()).toMatchObject({ hits: 2, misses: 1, size: 1, capacity: 2 });
  });

  it('refreshes recency on overwrite without growing', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.set('b', 2);
    cache.set('a', 10);
    cache.set('c', 3);
    expect(cache.get('a')).toBe(10);
    expect(cache.has('b')).toBe(false);
    expect(cache.size).toBe(2);
  });

  it('clears everything including counters', () => {
    const cache = new LruCache<string, number>(2);
    cache.set('a', 1);
    cache.get('a');
    cache.clear();
    expect(cache.size).toBe(0);
    expect(cache.stats()).toMatchObject({ hits: 0, misses: 0 });
  });

  it('rejects a non-positive capacity', () => {
    expect(() => new LruCache<string, number>(0)).toThrow(/capacity/i);
  });
});
