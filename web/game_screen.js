'use strict';

// Screen controls are independent of the touch pad, which disappears at game over.
(function () {
  function mount(stage, button, status) {
    const doc = stage.ownerDocument;
    let busy = false, orientationLocked = false, scrollPosition = null;
    const touches = new Map();
    const fullscreenElement = () => doc.fullscreenElement || doc.webkitFullscreenElement;
    const active = () => stage.classList.contains('game-immersive');
    function setActive(value) {
      if (value && !active()) {
        scrollPosition = {x: globalThis.scrollX, y: globalThis.scrollY};
        doc.body.style.setProperty('--game-scroll-top', `${-scrollPosition.y}px`);
      }
      stage.classList.toggle('game-immersive', value);
      doc.body.classList.toggle('game-immersive', value);
      doc.documentElement.classList.toggle('game-immersive', value);
      if (!value && scrollPosition) {
        doc.body.style.removeProperty('--game-scroll-top');
        globalThis.scrollTo(scrollPosition.x, scrollPosition.y);
        scrollPosition = null;
      }
      button.textContent = value ? '退出全屏' : '全屏';
      button.setAttribute('aria-pressed', String(value));
      if (!value && orientationLocked) {
        globalThis.screen?.orientation?.unlock?.();
        orientationLocked = false;
      }
    }
    async function toggle() {
      if (busy) return;
      busy = true;
      try {
        if (active()) {
          if (fullscreenElement() === stage) {
            const exit = doc.exitFullscreen || doc.webkitExitFullscreen;
            await exit.call(doc);
          }
          setActive(false);
          status.textContent = '已退出全屏';
          return;
        }
        // CSS immersion remains usable if native fullscreen is unavailable or denied.
        setActive(true);
        const request = stage.requestFullscreen || stage.webkitRequestFullscreen;
        try { if (request) await request.call(stage); } catch (_) { /* Use immersion. */ }
        if (!active()) return; // The user may have exited through browser controls.
        const native = fullscreenElement() === stage;
        status.textContent = native ? '已进入全屏，横屏操作更方便' : '已进入沉浸模式，横屏操作更方便；浏览器可能保留地址栏';
        if (native && globalThis.screen?.orientation?.lock) {
          try {
            await globalThis.screen.orientation.lock('landscape');
            orientationLocked = true;
            if (!active()) { globalThis.screen.orientation.unlock(); orientationLocked = false; }
          } catch (_) { /* Rotation remains manual on unsupported browsers. */ }
        }
      } catch (_) {
        status.textContent = '请使用浏览器的退出全屏操作';
      } finally { busy = false; }
    }
    button.addEventListener('click', toggle);
    for (const name of ['fullscreenchange', 'webkitfullscreenchange']) doc.addEventListener(name, () => {
      if (fullscreenElement() === stage) setActive(true);
      else if (active()) {
        // Browser/OS escape gestures cannot be disabled. Keep the game locked
        // in the viewport until the player explicitly uses our exit button.
        if (orientationLocked) { globalThis.screen?.orientation?.unlock?.(); orientationLocked = false; }
        status.textContent = '原生全屏已退出，继续使用沉浸模式；点击退出全屏可返回网页';
      }
    });
    doc.addEventListener('keydown', event => {
      if (event.key === 'Escape' && active() && fullscreenElement() !== stage) setActive(false);
    });
    // Keep form fields editable and zoomable; suppress double-tap only in play UI.
    stage.addEventListener('dblclick', event => {
      if (!QQTControls.isTypingTarget(event.target)) event.preventDefault();
    });
    stage.querySelector('#game').addEventListener('contextmenu', event => event.preventDefault());
    const inPlayArea = target => stage.contains(target) && !target.closest?.('.settlement, dialog, input, textarea, select, [contenteditable]:not([contenteditable="false"])');
    // Safari may still recognize tap zoom and rubber-band scrolling despite
    // touch-action. Cancel native Touch Events as well as using Pointer Events
    // for movement/actions. Explicit clicks preserve toolbar/page/canvas use.
    doc.addEventListener('touchstart', event => {
      let block = false;
      for (const touch of event.changedTouches) {
        if (!inPlayArea(touch.target)) continue;
        const action = touch.target.closest?.('button, #game');
        touches.set(touch.identifier, {x: touch.clientX, y: touch.clientY, action,
          click: !!action && !action.closest('[data-control-key]') && event.touches.length === 1});
        block = true;
      }
      if (event.touches.length > 1) for (const touch of touches.values()) touch.click = false;
      if (block && event.cancelable) event.preventDefault();
    }, {capture: true, passive: false});
    doc.addEventListener('touchmove', event => {
      let block = inPlayArea(event.target);
      for (const touch of event.changedTouches) {
        const start = touches.get(touch.identifier);
        if (!start) continue;
        block = true;
        if (Math.hypot(touch.clientX - start.x, touch.clientY - start.y) > 12) start.click = false;
      }
      if (block && event.cancelable) event.preventDefault();
    }, {capture: true, passive: false});
    const endTouch = event => {
      let block = inPlayArea(event.target);
      const clicks = [];
      for (const touch of event.changedTouches) {
        const start = touches.get(touch.identifier);
        touches.delete(touch.identifier);
        if (!start) continue;
        block = true;
        if (event.type === 'touchend' && start.click && Math.hypot(touch.clientX - start.x, touch.clientY - start.y) <= 12) {
          const underFinger = doc.elementFromPoint(touch.clientX, touch.clientY);
          if (underFinger && start.action.contains(underFinger)) clicks.push(start.action);
        }
      }
      if (block && event.cancelable) event.preventDefault();
      for (const action of clicks) action.click();
    };
    for (const name of ['touchend', 'touchcancel']) doc.addEventListener(name, endTouch, {capture: true, passive: false});
    for (const name of ['gesturestart', 'gesturechange', 'gestureend']) doc.addEventListener(name, event => {
      if (inPlayArea(event.target) && event.cancelable) event.preventDefault();
    }, {capture: true, passive: false});
    globalThis.addEventListener('blur', () => touches.clear());
    doc.addEventListener('visibilitychange', () => { if (doc.hidden) touches.clear(); });
  }
  globalThis.QQTGameScreen = {mount};
})();
