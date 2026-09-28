'use strict';

(function modelCatalogFactory(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.QQTModelCatalog = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildCatalog() {
  const SHA256 = /^[0-9a-f]{64}$/;
  const FILE = /^[a-z0-9][a-z0-9._-]*\.json$/;

  function validateManifest(document) {
    if (!document || document.schema !== 'qqt.web-models/v1') throw new Error('invalid model manifest schema');
    if (!Array.isArray(document.models)) throw new Error('invalid model list');
    return document.models.map((row) => {
      if (!row || typeof row.id !== 'string' || !row.id) throw new Error('invalid model id');
      if (!FILE.test(row.file || '')) throw new Error('invalid model file');
      if (!SHA256.test(row.sha256 || '')) throw new Error('invalid model sha256');
      if (!Number.isInteger(row.bytes) || row.bytes <= 0) throw new Error('invalid model bytes');
      if (!Number.isInteger(row.cycle) || row.cycle < 0) throw new Error('invalid model cycle');
      return Object.freeze({ ...row });
    });
  }

  function modelUrl(row) { return `models/${row.file}`; }
  function matchLabel(mode) { return mode === 'model-vs-rule' ? '模型 vs 规则Bot' : '真人 vs 对手'; }
  return { validateManifest, modelUrl, matchLabel };
});
