'use strict';

const assert = require('assert');
const catalog = require('./model_catalog.js');

const manifest = {
  schema: 'qqt.web-models/v1',
  models: [
    { id: 'safe-high-c16', display_name: 'Safe High · 周期16（综合最佳）', file: 'safe-high-c16.json', sha256: 'a'.repeat(64), bytes: 123, candidate: 'safe_high', cycle: 16, score: 0.0390625 },
  ],
};
const rows = catalog.validateManifest(manifest);
assert.strictEqual(rows.length, 1);
assert.strictEqual(rows[0].cycle, 16);
assert.throws(() => catalog.validateManifest({ schema: 'bad', models: [] }), /schema/);
assert.throws(() => catalog.validateManifest({ schema: 'qqt.web-models/v1', models: [{ ...manifest.models[0], file: '../x' }] }), /file/);
assert.strictEqual(catalog.modelUrl(rows[0]), 'models/safe-high-c16.json');
assert.strictEqual(catalog.matchLabel('model-vs-rule'), '模型 vs 规则Bot');
console.log('网页模型目录与观战模式契约通过');
