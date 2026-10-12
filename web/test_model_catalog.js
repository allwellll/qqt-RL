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
const production = require('./models.json');
const bun392 = production.models.find(row => row.id === 'bun392-selfplay3072');
assert(bun392, 'g3072 must be selectable without replacing existing models');
assert.strictEqual(bun392.display_name, 'Bun392动态自对弈3072');
assert.strictEqual(bun392.cycle, 3072);
assert.strictEqual(bun392.source_sha256, 'dc3919a5e7a2f415f951e23f20069d14c3c1df332acf5dfa7b1bb21314ca4f70');
assert(bun392.url.includes('/releases/download/web-bun392-selfplay3072-v1/'));
for (const id of ['bun-tf-it4000','bun-kill-mlp4','safe-high-c16','safe-medium-c16','safe-high-c24']) {
  assert(production.models.some(row => row.id === id), `existing model retained: ${id}`);
}
for (const row of catalog.validateManifest(production)) {
  assert.strictEqual(catalog.modelUrl(row), `models/${row.file}`, '浏览器必须同源加载模型，避免GitHub Release跨域失败');
}
assert.strictEqual(catalog.matchLabel('model-vs-rule'), '模型 vs 规则Bot');
console.log('网页模型目录与观战模式契约通过');
