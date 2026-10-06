'use strict';

(function controlsFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTControls = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildControls() {
  const MOVEMENT_KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
  // 原版用 Ctrl 放道具，但浏览器里 Ctrl+W/R 会关闭/刷新页面，改用 E / Shift。
  const ITEM_KEYS = ['KeyE', 'ShiftLeft', 'ShiftRight'];
  const BOMB_KEYS = ['Space', 'KeyW'];
  // 数字键 1-7 对应道具栏格位；E / Shift 使用第 1 格。
  const ITEM_SLOT_KEYS = ['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7'];

  function moveForHeld(held) {
    let move = 4;
    for (const key of held) {
      const direction = MOVEMENT_KEYS.indexOf(key);
      if (direction >= 0) move = direction;
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

  return { MOVEMENT_KEYS, ITEM_KEYS, ITEM_SLOT_KEYS, BOMB_KEYS, moveForHeld,
    isTypingTarget, isRestartKey, createRestartGate };
});
