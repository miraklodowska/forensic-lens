/**
 * Fixed-capacity LRU cache. Memory use is bounded by construction, which
 * matters in a content script that may live on an infinite-scroll page for
 * hours.
 */

export interface CacheStats {
  readonly size: number;
  readonly capacity: number;
  readonly hits: number;
  readonly misses: number;
  readonly evictions: number;
}

export class LruCache<K, V> {
  readonly #capacity: number;
  // Map preserves insertion order, so the first key is the least recent.
  readonly #entries = new Map<K, V>();
  #hits = 0;
  #misses = 0;
  #evictions = 0;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`cache capacity must be a positive integer, got ${capacity}`);
    }
    this.#capacity = capacity;
  }

  get size(): number {
    return this.#entries.size;
  }

  has(key: K): boolean {
    return this.#entries.has(key);
  }

  get(key: K): V | undefined {
    if (!this.#entries.has(key)) {
      this.#misses += 1;
      return undefined;
    }
    const value = this.#entries.get(key)!;
    this.#entries.delete(key);
    this.#entries.set(key, value);
    this.#hits += 1;
    return value;
  }

  set(key: K, value: V): void {
    if (this.#entries.has(key)) this.#entries.delete(key);
    this.#entries.set(key, value);
    while (this.#entries.size > this.#capacity) {
      const oldest = this.#entries.keys().next();
      if (oldest.done) break;
      this.#entries.delete(oldest.value);
      this.#evictions += 1;
    }
  }

  delete(key: K): boolean {
    return this.#entries.delete(key);
  }

  clear(): void {
    this.#entries.clear();
    this.#hits = 0;
    this.#misses = 0;
    this.#evictions = 0;
  }

  stats(): CacheStats {
    return {
      size: this.#entries.size,
      capacity: this.#capacity,
      hits: this.#hits,
      misses: this.#misses,
      evictions: this.#evictions,
    };
  }
}
