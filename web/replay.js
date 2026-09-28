'use strict';

(function replayFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTReplay = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildReplay() {
  function validateReplay(document) {
    if (!document || document.schema !== 'qqt.replay/v1') throw new Error('invalid replay schema');
    if (!document.meta || !Number.isInteger(document.meta.seed)) throw new Error('invalid replay meta');
    if (!Array.isArray(document.actions) || !document.actions.length) throw new Error('invalid replay actions');
    for (const row of document.actions) {
      if (!Array.isArray(row) || row.length !== 6 || row.some((value) => !Number.isInteger(value))) {
        throw new Error('invalid replay action row');
      }
    }
    if (!document.summary || document.summary.ticks !== document.actions.length) throw new Error('invalid replay summary');
    return document;
  }

  function frameIndexAt(elapsedMs, frameCount, tickMs = 100) {
    return Math.min(frameCount - 1, Math.max(0, Math.floor(elapsedMs / tickMs)));
  }

  return { validateReplay, frameIndexAt };
});
