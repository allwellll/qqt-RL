'use strict';

(function controlsFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTControls = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildControls() {
  // Physical codes stay stable when NumLock changes key values (5/Clear,
  // 1/End, 2/ArrowDown, 3/PageDown). Keep aliases separate in held state so
  // releasing one key never releases another key with the same direction.
  const MOVEMENT_DIRECTIONS = Object.freeze({
    ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
    Numpad5: 0, Numpad1: 2, Numpad2: 1, Numpad3: 3,
  });
  const MOVEMENT_KEYS = Object.keys(MOVEMENT_DIRECTIONS);
  // 原版用 Ctrl 放道具，但浏览器里 Ctrl+W/R 会关闭/刷新页面，改用 E / Shift。
  const ITEM_KEYS = ['KeyE', 'ShiftLeft', 'ShiftRight'];
  const BOMB_KEYS = ['Space', 'KeyW'];
  // 数字键 1-7 对应道具栏格位；E / Shift 使用第 1 格。
  const ITEM_SLOT_KEYS = ['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7'];

  function moveForHeld(held) {
    let move = 4;
    for (const key of held) {
      if (Object.hasOwn(MOVEMENT_DIRECTIONS, key)) move = MOVEMENT_DIRECTIONS[key];
    }
    return move;
  }

  function isTypingTarget(target) {
    return !!(target && target.closest && target.closest('input, textarea, [contenteditable]:not([contenteditable="false"])'));
  }

  function isRestartKey(event, held) {
    return event.code === 'KeyR' && !event.repeat && !held.has('KeyR') &&
      !event.ctrlKey && !event.metaKey && !event.altKey && !isTypingTarget(event.target);
  }

  // Coalesce keyboard/button requests while the existing reset (including map load)
  // is in progress. Completed-match buttons cannot restart the replacement match.
  function createRestartGate(reset, isFinished) {
    let pending = null;
    return function restart(finishedOnly = false) {
      if (pending) return pending;
      if (finishedOnly && !isFinished()) return Promise.resolve(false);
      pending = Promise.resolve().then(reset).then(() => true).finally(() => { pending = null; });
      return pending;
    };
  }

  // Pagination changes only visible shortcuts, never the authoritative inventory.
  function touchItemPage(slots, requestedPage = 0) {
    const pages = Math.max(1, Math.ceil(Math.min(slots.length, ITEM_SLOT_KEYS.length) / 3));
    const page = Math.max(0, Math.min(pages - 1, Number.isInteger(requestedPage) ? requestedPage : 0));
    return {page, pages, buttons: Array.from({length: 3}, (_, i) => {
      const index = page * 3 + i;
      return {index, number: index < ITEM_SLOT_KEYS.length ? index + 1 : null,
        key: ITEM_SLOT_KEYS[index] || '', slot: slots[index] || null};
    })};
  }

  function bindTouchControls(root, held, onPress) {
    const active = new Map();
    const release = event => {
      const key = active.get(event.pointerId);
      active.delete(event.pointerId);
      if (key && ![...active.values()].includes(key)) held.delete(key);
    };
    // Restrict browser gesture suppression to the game controls.
    for (const name of ['contextmenu', 'selectstart', 'dragstart'])
      root.addEventListener?.(name, event => event.preventDefault());
    for (const button of root.querySelectorAll('[data-control-key]')) {
      button.addEventListener('pointerdown', event => {
        if (event.button != null && event.button !== 0) return;
        event.preventDefault();
        const key = button.dataset.controlKey;
        if (!key || active.has(event.pointerId)) return;
        const alreadyHeld = [...active.values()].includes(key);
        active.set(event.pointerId, key); held.add(key);
        button.setPointerCapture(event.pointerId);
        if (!alreadyHeld) onPress(key);
      });
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) button.addEventListener(name, release);
    }
    return () => { for (const key of active.values()) held.delete(key); active.clear(); };
  }

  return { MOVEMENT_KEYS, MOVEMENT_DIRECTIONS, ITEM_KEYS, ITEM_SLOT_KEYS, BOMB_KEYS, moveForHeld,
    isTypingTarget, isRestartKey, createRestartGate, touchItemPage, bindTouchControls };
});
