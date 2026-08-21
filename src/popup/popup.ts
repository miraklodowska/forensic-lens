/**
 * Popup: engine status and the handful of settings worth exposing.
 */

import { DEFAULT_SETTINGS, normalizeSettings, type Settings } from '../shared/settings.ts';
import { MESSAGE_TYPES, type StateResponse } from '../shared/messages.ts';

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node === null) throw new Error(`missing #${id}`);
  return node as T;
}

const ui = {
  dot: el<HTMLSpanElement>('dot'),
  engine: el<HTMLSpanElement>('engine'),
  backend: el<HTMLSpanElement>('backend'),
  enabled: el<HTMLInputElement>('enabled'),
  threshold: el<HTMLInputElement>('threshold'),
  thresholdVal: el<HTMLSpanElement>('thresholdVal'),
  showBadges: el<HTMLInputElement>('showBadges'),
  blurFlagged: el<HTMLInputElement>('blurFlagged'),
  analyzed: el<HTMLElement>('analyzed'),
  flagged: el<HTMLElement>('flagged'),
  errors: el<HTMLElement>('errors'),
  error: el<HTMLParagraphElement>('error'),
};

let settings: Settings = DEFAULT_SETTINGS;

function paintSettings(): void {
  const percent = Math.round(settings.threshold * 100);
  ui.enabled.checked = settings.enabled;
  ui.threshold.value = String(percent);
  ui.thresholdVal.textContent = `${percent}%`;
  ui.showBadges.checked = settings.showBadges;
  ui.blurFlagged.checked = settings.blurFlagged;
}

async function save(patch: Partial<Settings>): Promise<void> {
  settings = normalizeSettings({ ...settings, ...patch });
  await chrome.storage.local.set({ settings });
  paintSettings();
}

async function refreshStatus(): Promise<void> {
  let reply: StateResponse | undefined;
  try {
    reply = (await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.getState })) as StateResponse | undefined;
  } catch {
    reply = undefined;
  }

  const status = reply?.status;
  if (status === undefined) {
    ui.engine.textContent = 'worker unavailable';
    ui.dot.className = 'dot error';
    return;
  }

  ui.analyzed.textContent = String(status.analyzed);
  ui.flagged.textContent = String(status.flagged);
  ui.errors.textContent = String(status.errors);

  if (status.error !== undefined) {
    ui.dot.className = 'dot error';
    ui.engine.textContent = 'engine error';
    ui.error.hidden = false;
    ui.error.textContent = status.error;
    return;
  }
  ui.error.hidden = true;

  if (status.ready) {
    ui.dot.className = 'dot ready';
    const n = status.modelIds.length;
    ui.engine.textContent = `ready · ${n} model${n === 1 ? '' : 's'}`;
    ui.backend.textContent = status.backend.toUpperCase();
  } else {
    ui.dot.className = 'dot';
    ui.engine.textContent = 'loading models…';
    ui.backend.textContent = '—';
  }
}

ui.enabled.addEventListener('change', () => void save({ enabled: ui.enabled.checked }));
ui.showBadges.addEventListener('change', () => void save({ showBadges: ui.showBadges.checked }));
ui.blurFlagged.addEventListener('change', () => void save({ blurFlagged: ui.blurFlagged.checked }));
ui.threshold.addEventListener('input', () => {
  ui.thresholdVal.textContent = `${ui.threshold.value}%`;
});
ui.threshold.addEventListener('change', () => void save({ threshold: Number(ui.threshold.value) / 100 }));

async function boot(): Promise<void> {
  const stored = await chrome.storage.local.get('settings');
  settings = normalizeSettings(stored['settings']);
  paintSettings();
  await refreshStatus();
  // Sessions take a moment to build, so keep polling while the popup is open.
  const timer = setInterval(() => void refreshStatus(), 700);
  addEventListener('unload', () => clearInterval(timer));
}

void boot();
