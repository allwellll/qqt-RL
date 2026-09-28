'use strict';

(function controlsFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTControls = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildControls() {
  const MOVEMENT_KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];

  function moveForHeld(held) {
    if (held.has('ArrowUp') || held.has('KeyW')) return 0;
    if (held.has('ArrowDown') || held.has('KeyS')) return 1;
    if (held.has('ArrowLeft') || held.has('KeyA')) return 2;
    if (held.has('ArrowRight') || held.has('KeyD')) return 3;
    return 4;
  }

  return { MOVEMENT_KEYS, moveForHeld };
});
