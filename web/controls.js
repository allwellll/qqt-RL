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
      const code = key.startsWith('touch:') ? key.slice(6) : key;
      if (Object.hasOwn(MOVEMENT_DIRECTIONS, code)) move = MOVEMENT_DIRECTIONS[code];
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

  const STICK_KEYS = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp'];
  // A short stroke expresses intent, independent of the displayed dial center.
  // Forget completed travel so a small turn/reversal works anywhere on screen.
  function createFloatingStick({radius = 48, deadzone = 4, turnDistance = 3, hysteresis = 4} = {}) {
    let center = null, habit = null, direction = null, last = null, stroke = null;
    const snapshot = () => ({center: center && {...center}, habit: habit && {...habit}, direction});
    const heading = (dx, dy) => {
      const angle = Math.atan2(dy, dx), old = STICK_KEYS.indexOf(direction);
      const delta = old < 0 ? Infinity : Math.abs(Math.atan2(
        Math.sin(angle - old * Math.PI / 2), Math.cos(angle - old * Math.PI / 2)));
      return delta <= (45 + hysteresis) * Math.PI / 180 ? direction :
        STICK_KEYS[(Math.round(angle / (Math.PI / 2)) + 4) % 4];
    };
    return {
      start(x, y) {
        center = {x, y}; habit = {x, y}; direction = null; stroke = null;
        last = {x, y};
        return snapshot();
      },
      move(x, y) {
        if (!last) return snapshot();
        const dx = x - last.x, dy = y - last.y;
        if (dx || dy) {
          const next = heading(dx, dy);
          if (next === direction) stroke = null;
          else {
            // Only coherent displacement accumulates. Reversing jitter resets it;
            // a slow deliberate stroke still works at any pointer event rate.
            if (!stroke || stroke.direction !== next)
              stroke = {x: last.x, y: last.y, direction: next};
            const sx = x - stroke.x, sy = y - stroke.y;
            if (Math.hypot(sx, sy) >= (direction ? turnDistance : deadzone) && heading(sx, sy) === next) {
              direction = next; stroke = null;
            }
          }
          last = {x, y};
        }
        // Feedback follows the finger, slightly behind the accepted direction.
        // Its position never feeds back into input or character movement.
        const offset = Math.min(radius * .35, 16);
        center = {x: x - (direction === 'ArrowRight' ? offset : direction === 'ArrowLeft' ? -offset : 0),
          y: y - (direction === 'ArrowDown' ? offset : direction === 'ArrowUp' ? -offset : 0)};
        return snapshot();
      },
      end() {
        habit = center && {...center}; center = null; direction = null; last = stroke = null;
        return snapshot();
      },
      reset() { center = habit = direction = last = stroke = null; return snapshot(); },
    };
  }

  function bindFloatingDirection(zone, held) {
    let pointer = null, stick = null, lastDirection = null;
    const dial = zone.querySelector('.touch-stick');
    const paint = state => {
      for (const key of STICK_KEYS) held.delete('touch:' + key);
      if (state.direction) held.add('touch:' + state.direction);
      const rect = zone.getBoundingClientRect(), p = state.center || state.habit;
      if (p) { dial.style.left = (p.x - rect.left) + 'px'; dial.style.top = (p.y - rect.top) + 'px'; }
      else { dial.style.left = ''; dial.style.top = ''; }
      zone.dataset.direction = state.direction || '';
      zone.dataset.active = pointer == null ? 'false' : 'true';
      if (state.direction && state.direction !== lastDirection) {
        try { globalThis.navigator?.vibrate?.(8); } catch (_) { /* Optional feedback. */ }
      }
      lastDirection = state.direction;
    };
    zone.addEventListener('pointerdown', event => {
      if (event.button !== 0 || pointer != null) return;
      event.preventDefault(); pointer = event.pointerId;
      if (!stick) {
        const radius = dial.getBoundingClientRect().width / 2 || 48;
        stick = createFloatingStick({radius});
      }
      zone.setPointerCapture(pointer);
      paint(stick.start(event.clientX, event.clientY));
    });
    zone.addEventListener('pointermove', event => {
      if (event.pointerId !== pointer) return;
      event.preventDefault();
      // Preserve short turns that the browser bundled into one pointer event.
      const samples = event.getCoalescedEvents?.();
      let state;
      for (const sample of samples?.length ? samples : [event])
        state = stick.move(sample.clientX, sample.clientY);
      paint(state);
    });
    const release = event => {
      if (event.pointerId !== pointer) return;
      pointer = null; paint(stick.end());
    };
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) zone.addEventListener(name, release);
    const clear = () => {
      const id = pointer; pointer = null;
      if (stick) paint(stick.reset());
      stick = null; // Re-read the scaled radius after rotation or viewport resize.
      if (id != null && zone.hasPointerCapture?.(id)) zone.releasePointerCapture(id);
    };
    globalThis.addEventListener?.('resize', clear);
    return clear;
  }

  function bindTouchControls(root, held, onPress) {
    const active = new Map();
    const zone = root.querySelector?.('[data-floating-direction]');
    const clearDirection = zone ? bindFloatingDirection(zone, held) : () => {};
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
        if (!key || button.getAttribute?.('aria-disabled') === 'true' || active.has(event.pointerId)) return;
        const alreadyHeld = [...active.values()].includes(key);
        active.set(event.pointerId, key); held.add(key);
        button.setPointerCapture(event.pointerId);
        if (!alreadyHeld) onPress(key);
      });
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) button.addEventListener(name, release);
    }
    return () => { clearDirection(); for (const key of active.values()) held.delete(key); active.clear(); };
  }

  return { MOVEMENT_KEYS, MOVEMENT_DIRECTIONS, ITEM_KEYS, ITEM_SLOT_KEYS, BOMB_KEYS, moveForHeld,
    isTypingTarget, isRestartKey, createRestartGate, touchItemPage, createFloatingStick, bindTouchControls };
});
