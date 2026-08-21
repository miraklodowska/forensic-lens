/**
 * Bounded, deduplicating work queue.
 *
 * A page with thousands of images must never translate into thousands of
 * pending inference jobs: submissions past `capacity` are rejected outright
 * (and counted) rather than buffered, and identical keys share one run.
 */

export interface QueueOptions {
  /** Maximum running + pending tasks. */
  readonly capacity: number;
  /** Maximum simultaneously running tasks. */
  readonly concurrency: number;
}

export interface QueueStats {
  readonly running: number;
  readonly pending: number;
  readonly capacity: number;
  readonly completed: number;
  readonly failed: number;
  readonly dropped: number;
}

export class QueueFullError extends Error {
  constructor(capacity: number) {
    super(`queue is at capacity (${capacity}); submission rejected`);
    this.name = 'QueueFullError';
  }
}

export class QueueClearedError extends Error {
  constructor() {
    super('queued task was cleared before it started');
    this.name = 'QueueClearedError';
  }
}

interface PendingEntry<T> {
  readonly key: string;
  readonly run: () => Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (error: unknown) => void;
}

export class BoundedTaskQueue<T> {
  readonly #capacity: number;
  readonly #concurrency: number;
  readonly #pending: PendingEntry<T>[] = [];
  readonly #inFlight = new Map<string, Promise<T>>();
  #running = 0;
  #completed = 0;
  #failed = 0;
  #dropped = 0;

  constructor({ capacity, concurrency }: QueueOptions) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`queue capacity must be a positive integer, got ${capacity}`);
    }
    if (!Number.isInteger(concurrency) || concurrency <= 0) {
      throw new Error(`queue concurrency must be a positive integer, got ${concurrency}`);
    }
    this.#capacity = capacity;
    this.#concurrency = concurrency;
  }

  get size(): number {
    return this.#running + this.#pending.length;
  }

  stats(): QueueStats {
    return {
      running: this.#running,
      pending: this.#pending.length,
      capacity: this.#capacity,
      completed: this.#completed,
      failed: this.#failed,
      dropped: this.#dropped,
    };
  }

  has(key: string): boolean {
    return this.#inFlight.has(key);
  }

  submit(key: string, run: () => Promise<T>): Promise<T> {
    const existing = this.#inFlight.get(key);
    if (existing) return existing;

    if (this.size >= this.#capacity) {
      this.#dropped += 1;
      return Promise.reject(new QueueFullError(this.#capacity));
    }

    const promise = new Promise<T>((resolve, reject) => {
      this.#pending.push({ key, run, resolve, reject });
    });
    this.#inFlight.set(key, promise);
    // Swallow rejection on the stored copy: callers own error handling, and an
    // unobserved shared promise must not raise an unhandled rejection.
    promise.catch(() => {});
    this.#drain();
    return promise;
  }

  clear(): void {
    const dropped = this.#pending.splice(0, this.#pending.length);
    for (const entry of dropped) {
      this.#inFlight.delete(entry.key);
      this.#dropped += 1;
      entry.reject(new QueueClearedError());
    }
  }

  #drain(): void {
    while (this.#running < this.#concurrency && this.#pending.length > 0) {
      const entry = this.#pending.shift()!;
      this.#running += 1;
      void (async () => {
        try {
          const value = await entry.run();
          this.#completed += 1;
          entry.resolve(value);
        } catch (error) {
          this.#failed += 1;
          entry.reject(error);
        } finally {
          this.#running -= 1;
          this.#inFlight.delete(entry.key);
          this.#drain();
        }
      })();
    }
  }
}
