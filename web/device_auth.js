'use strict';
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QQTDeviceAuth = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  const KEY = 'qqt.advanced.authorization';
  const VERSION = 'qqt.advanced/v1';
  const VERIFIER = 'c48032f5a5457a9d09be15fa90618ca9bb498061bda1aa961636793e30dbcae3';
  // Static Pages UI gate, not a server session. Bind grants to the validation rule.
  function create(storage, crypto, version = VERSION, verifier = VERIFIER) {
    const marker = JSON.stringify({ schema: version, rule: verifier, granted: true });
    let memory = false;
    function forget() { memory = false; try { storage.removeItem(KEY); } catch (_) {} }
    function authorized() {
      try {
        const saved = storage.getItem(KEY);
        if (saved && saved !== marker) { forget(); return false; }
        return saved === marker || memory;
      } catch (_) { return memory; }
    }
    async function verify(password) {
      const bytes = new TextEncoder().encode(`${version}:${password}`);
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
      if (digest !== verifier) { forget(); return false; }
      memory = true;
      try { storage.setItem(KEY, marker); memory = false; } catch (_) {}
      return true;
    }
    return { authorized, verify, forget };
  }
  return { create, KEY, VERSION };
});
