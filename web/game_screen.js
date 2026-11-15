'use strict';

// Screen controls are independent of the touch pad, which disappears at game over.
(function () {
  function mount(stage, button, status) {
    const doc = stage.ownerDocument;
    let busy = false, orientationLocked = false;
    const fullscreenElement = () => doc.fullscreenElement || doc.webkitFullscreenElement;
    const active = () => stage.classList.contains('game-immersive');
    function setActive(value) {
      stage.classList.toggle('game-immersive', value);
      doc.body.classList.toggle('game-immersive', value);
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
      setActive(fullscreenElement() === stage);
    });
    doc.addEventListener('keydown', event => {
      if (event.key === 'Escape' && active() && fullscreenElement() !== stage) setActive(false);
    });
    // Keep form fields editable and zoomable; suppress double-tap only in play UI.
    stage.addEventListener('dblclick', event => {
      if (!QQTControls.isTypingTarget(event.target)) event.preventDefault();
    });
    stage.querySelector('#game').addEventListener('contextmenu', event => event.preventDefault());
  }
  globalThis.QQTGameScreen = {mount};
})();
