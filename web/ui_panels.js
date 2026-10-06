'use strict';
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTPanels = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  function mount(document) {
    const el = id => document.getElementById(id);
    const settings = el('advanced-settings'), opener = el('advanced-open');
    const password = el('password-dialog'), input = el('settings-password');
    const announcement = el('announcement-dialog');
    function lock() {
      settings.hidden = true; opener.setAttribute('aria-expanded', 'false');
    }
    lock();
    opener.addEventListener('click', () => {
      if (!settings.hidden) { lock(); return; }
      input.value = ''; el('password-status').textContent = '每次打开都需验证，刷新后锁定。';
      password.showModal(); input.focus();
    });
    el('password-form').addEventListener('submit', event => {
      event.preventDefault();
      if (input.value !== 'demaxiya') {
        input.value = ''; el('password-status').textContent = '口令不对，再试一次吧～'; input.focus(); return;
      }
      input.value = ''; password.close(); settings.hidden = false;
      opener.setAttribute('aria-expanded', 'true'); el('advanced-close').focus();
    });
    el('advanced-close').addEventListener('click', () => { lock(); opener.focus(); });
    settings.addEventListener('keydown', event => {
      if (event.key === 'Escape') { event.preventDefault(); lock(); opener.focus(); }
    });
    el('announcement-open').addEventListener('click', () => announcement.showModal());
    for (const dialog of [password, announcement]) {
      dialog.addEventListener('keydown', event => {
        if (event.key !== 'Tab') return;
        const targets = Array.from(dialog.querySelectorAll('button, input, a[href]')).filter(target => !target.disabled);
        const first = targets[0], last = targets[targets.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      });
      dialog.querySelector('[data-dialog-close]').addEventListener('click', () => dialog.close());
      dialog.addEventListener('click', event => {
        const rect = dialog.getBoundingClientRect();
        if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right ||
            event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
      });
      dialog.addEventListener('close', () => {
        if (dialog === password) input.value = '';
        el(dialog === password ? settings.hidden ? 'advanced-open' : 'advanced-close' : 'announcement-open').focus();
      });
    }
    return { lock };
  }
  return { mount };
});
