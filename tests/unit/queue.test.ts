import { describe, expect, it, vi } from 'vitest';
import { BoundedTaskQueue } from '../../src/core/queue.ts';

const deferred = () => {
  let resolve!: (v: number) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<number>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe('BoundedTaskQueue', () => {
  it('runs at most `concurrency` tasks at a time', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 10, concurrency: 2 });
    let active = 0;
    let peak = 0;
    const task = async () => {
      active += 1;
      peak = Math.max(peak, active);
      await Promise.resolve();
      active -= 1;
      return 1;
    };
    await Promise.all(['a', 'b', 'c', 'd', 'e'].map((k) => queue.submit(k, task)));
    expect(peak).toBe(2);
  });

  it('deduplicates by key and shares the single in-flight result', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 10, concurrency: 1 });
    const run = vi.fn(async () => 42);
    const [a, b] = await Promise.all([queue.submit('same', run), queue.submit('same', run)]);
    expect(run).toHaveBeenCalledTimes(1);
    expect(a).toBe(42);
    expect(b).toBe(42);
  });

  it('rejects submissions once the queue is at capacity instead of growing', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 2, concurrency: 1 });
    const gate = deferred();
    const running = queue.submit('running', () => gate.promise);
    const queued = queue.submit('queued', async () => 1);
    await expect(queue.submit('overflow', async () => 1)).rejects.toThrow(/capacity/i);
    expect(queue.stats().dropped).toBe(1);
    gate.resolve(1);
    await Promise.all([running, queued]);
  });

  it('reports pending and running counts', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 4, concurrency: 1 });
    const gate = deferred();
    const running = queue.submit('a', () => gate.promise);
    const pending = queue.submit('b', async () => 2);
    expect(queue.stats()).toMatchObject({ running: 1, pending: 1 });
    gate.resolve(1);
    await Promise.all([running, pending]);
    expect(queue.stats()).toMatchObject({ running: 0, pending: 0 });
  });

  it('keeps draining after a task throws', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 4, concurrency: 1 });
    const failed = queue.submit('bad', async () => {
      throw new Error('boom');
    });
    await expect(failed).rejects.toThrow('boom');
    await expect(queue.submit('good', async () => 7)).resolves.toBe(7);
    expect(queue.stats().running).toBe(0);
  });

  it('clear() drops queued work and rejects its waiters but leaves running work alone', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 4, concurrency: 1 });
    const gate = deferred();
    const running = queue.submit('running', () => gate.promise);
    const queued = queue.submit('queued', async () => 1);
    queue.clear();
    await expect(queued).rejects.toThrow(/cleared/i);
    gate.resolve(5);
    await expect(running).resolves.toBe(5);
  });

  it('frees the key after completion so the same image can be re-analysed later', async () => {
    const queue = new BoundedTaskQueue<number>({ capacity: 2, concurrency: 1 });
    const run = vi.fn(async () => 1);
    await queue.submit('k', run);
    await queue.submit('k', run);
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('rejects a non-positive capacity or concurrency at construction', () => {
    expect(() => new BoundedTaskQueue({ capacity: 0, concurrency: 1 })).toThrow(/capacity/i);
    expect(() => new BoundedTaskQueue({ capacity: 1, concurrency: 0 })).toThrow(/concurrency/i);
  });
});
