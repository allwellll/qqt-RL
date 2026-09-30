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

  function moveForHeld(held) {
    if (held.has('ArrowUp')) return 0;
    if (held.has('ArrowDown')) return 1;
    if (held.has('ArrowLeft')) return 2;
    if (held.has('ArrowRight')) return 3;
    return 4;
  }

  return { MOVEMENT_KEYS, ITEM_KEYS, BOMB_KEYS, moveForHeld };
});
