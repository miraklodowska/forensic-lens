/**
 * Detects navigations that do not reload the document.
 *
 * Single-page apps swap the whole gallery under a pushState, so badges anchored
 * to the old DOM must be dropped and discovery restarted. History methods are
 * patched (and faithfully restored) because they emit no event of their own.
 */

export type UrlChangeListener = (url: string) => void;

export function watchUrlChanges(onChange: UrlChangeListener): () => void {
  let lastUrl = location.href;
  let stopped = false;
  let scheduled = false;

  const check = (): void => {
    scheduled = false;
    if (stopped) return;
    const current = location.href;
    if (current === lastUrl) return;
    lastUrl = current;
    onChange(current);
  };

  // Coalesce bursts: a router may call replaceState several times per click.
  const schedule = (): void => {
    if (stopped || scheduled) return;
    scheduled = true;
    setTimeout(check, 0);
  };

  const originalPush = history.pushState;
  const originalReplace = history.replaceState;

  history.pushState = function patchedPushState(this: History, ...args: Parameters<History['pushState']>) {
    const result = originalPush.apply(this, args);
    schedule();
    return result;
  };
  history.replaceState = function patchedReplaceState(this: History, ...args: Parameters<History['replaceState']>) {
    const result = originalReplace.apply(this, args);
    schedule();
    return result;
  };

  window.addEventListener('popstate', schedule);
  window.addEventListener('hashchange', schedule);

  return () => {
    if (stopped) return;
    stopped = true;
    // Restore whatever was installed before us, so nested watchers unwind cleanly.
    history.pushState = originalPush;
    history.replaceState = originalReplace;
    window.removeEventListener('popstate', schedule);
    window.removeEventListener('hashchange', schedule);
  };
}
