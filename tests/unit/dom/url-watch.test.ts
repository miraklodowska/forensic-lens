import { afterEach, describe, expect, it, vi } from 'vitest';
import { watchUrlChanges } from '../../../src/content/url-watch.ts';

const flush = () => new Promise((r) => setTimeout(r, 0));

describe('watchUrlChanges', () => {
  const stops: Array<() => void> = [];
  const watch = (cb: (url: string) => void) => {
    const stop = watchUrlChanges(cb);
    stops.push(stop);
    return stop;
  };

  afterEach(() => {
    while (stops.length) stops.pop()!();
    history.replaceState(null, '', '/');
  });

  it('fires on a history.pushState navigation', async () => {
    const onChange = vi.fn();
    watch(onChange);
    history.pushState(null, '', '/next');
    await flush();
    expect(onChange).toHaveBeenCalledWith(expect.stringContaining('/next'));
  });

  it('fires on history.replaceState', async () => {
    const onChange = vi.fn();
    watch(onChange);
    history.replaceState(null, '', '/replaced');
    await flush();
    expect(onChange).toHaveBeenCalledWith(expect.stringContaining('/replaced'));
  });

  it('fires on popstate (back/forward)', async () => {
    const onChange = vi.fn();
    watch(onChange);
    history.pushState(null, '', '/a');
    await flush();
    onChange.mockClear();
    history.replaceState(null, '', '/b');
    window.dispatchEvent(new PopStateEvent('popstate'));
    await flush();
    expect(onChange).toHaveBeenCalledWith(expect.stringContaining('/b'));
  });

  it('fires on hashchange', async () => {
    const onChange = vi.fn();
    watch(onChange);
    history.replaceState(null, '', '/p#one');
    await flush();
    onChange.mockClear();
    history.replaceState(null, '', '/p#two');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    await flush();
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it('does not fire when the URL did not actually change', async () => {
    const onChange = vi.fn();
    watch(onChange);
    history.replaceState(null, '', location.href);
    window.dispatchEvent(new PopStateEvent('popstate'));
    await flush();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('restores the original history methods when stopped', () => {
    const push = history.pushState;
    const replace = history.replaceState;
    const stop = watch(vi.fn());
    expect(history.pushState).not.toBe(push);
    stop();
    expect(history.pushState).toBe(push);
    expect(history.replaceState).toBe(replace);
  });

  it('stops firing after the watcher is stopped', async () => {
    const onChange = vi.fn();
    const stop = watch(onChange);
    stop();
    history.pushState(null, '', '/after-stop');
    await flush();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('nested watchers each restore correctly', () => {
    const original = history.pushState;
    const stopA = watch(vi.fn());
    const stopB = watch(vi.fn());
    stopB();
    stopA();
    expect(history.pushState).toBe(original);
  });
});
