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
const current=production.models[0];
assert.strictEqual(current.id,'bun392-a3-u800');
assert.strictEqual(current.source_sha256,'8153dafb2b15b7e4be17225bc95cc482417ce0afd8455081234f85f2406b16b6');
assert.strictEqual(current.onnx.meta.rule_version,'stock-window-v1');
assert(current.description.includes('P1') && current.description.includes('零泡') && current.description.includes('振荡'));
assert.strictEqual(current.score,42/256);
const latest=production.models.find(row=>row.id==='bun392-pbt-slot05-u720');
assert.strictEqual(latest.id,'bun392-pbt-slot05-u720');
assert.strictEqual(latest.source_sha256,'ed5a2b1860e8256fa34ac5a5830ff9b5c81e67a0b33d5c8a73101b3149796663');
assert.strictEqual(latest.display_name,'Bun392 PBT u720（D）');
assert.strictEqual(latest.cycle,720);
assert.strictEqual(latest.onnx.meta.bun_state_encoding,'bun24-trap-timer-v2');
for (const asset of [latest,latest.onnx]) {
  assert(asset.file.includes(asset.sha256));
  assert(asset.url.endsWith('/'+asset.file));
}
const bridge = production.models.find(row=>row.id==='bun392-je09-stage-c-g8');
assert.strictEqual(bridge.id, 'bun392-je09-stage-c-g8');
assert.strictEqual(bridge.display_name, 'Bun392 JE09运包桥接g8');
assert.strictEqual(bridge.cycle, 8);
assert.strictEqual(bridge.strategy, 'raw');
assert.strictEqual(bridge.source_sha256, '53dff88d00391bfb99b04a1d15c4da9fcebba9ad9d70e118903a6eeff93497fe');
assert.strictEqual(bridge.onnx.meta.bun_state_encoding, 'bun24-trap-timer-v2');
assert.deepStrictEqual(bridge.onnx.meta.obs_shape, [24,13,15]);
assert.strictEqual(bridge.onnx.meta.state_dim, 24);
assert.strictEqual(bridge.onnx.meta.embed, 392);
assert.strictEqual(bridge.onnx.meta.ability_actions, 3);
assert(bridge.description.includes('fixed32') && bridge.description.includes('full'));
for (const asset of [bridge, bridge.onnx]) {
  assert(asset.file.includes(asset.sha256));
  assert(asset.url.endsWith('/' + asset.file));
  assert(asset.url.includes('/web-bun392-je09-stage-c-g8-v1/'));
}
const transport = production.models.find(row => row.id === 'bun392-transport1536');
assert.strictEqual(transport.id, 'bun392-transport1536', 'Previous transport policy stays selectable');
assert.strictEqual(transport.cycle, 1536);
assert.strictEqual(transport.strategy, 'raw');
assert.strictEqual(transport.source_sha256, '16ba817d53ef1f3ed4cf6039ff58f33f1f0de4adc69b3578047c5c3cdfbda719');
assert.strictEqual(transport.onnx.meta.bun_state_encoding, 'bun24-trap-timer-v2');
assert(transport.onnx.file.includes(transport.onnx.sha256));
assert(transport.file.includes(transport.sha256));
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
