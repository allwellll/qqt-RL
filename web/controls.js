'use strict';

(function controlsFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTControls = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildControls() {
  const MOVEMENT_KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
  // 原版用 Ctrl 放道具，但浏览器里 Ctrl+W/R 会关闭/刷新页面，改用 E / Shift。
  const ITEM_KEYS = ['KeyE', 'ShiftLeft', 'ShiftRight'];

  function moveForHeld(held) {
    if (held.has('ArrowUp') || held.has('KeyW')) return 0;
    if (held.has('ArrowDown') || held.has('KeyS')) return 1;
    if (held.has('ArrowLeft') || held.has('KeyA')) return 2;
    if (held.has('ArrowRight') || held.has('KeyD')) return 3;
    return 4;
  }

  return { MOVEMENT_KEYS, ITEM_KEYS, moveForHeld };
});
