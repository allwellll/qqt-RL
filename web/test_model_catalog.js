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
const specialist = {...manifest.models[0], rule_version: 'bun-native-v3-items', specialist: {
  label: '对接测试样例', strengths: ['测试长处'], weaknesses: ['测试短板'],
  evidence: {ruleset: 'bun-native-v3-items', games: 16, summary: '测试范围，不是候选成绩', url: 'specialists/integration.json#sample'},
}};
assert.equal(catalog.optionLabel(rows[0]), rows[0].display_name);
assert.deepEqual(catalog.specialistLines(rows[0]), []);
assert.deepEqual(catalog.specialistLines(null), []);
assert.equal(catalog.validateManifest({...manifest, models: [specialist]}).length, 1);
assert(catalog.optionLabel(specialist).includes('专项：对接测试样例'));
assert(catalog.specialistLines(specialist).join('\n').includes('短板：测试短板'));
for (const patch of [{weaknesses: []}, {strengths: []}, {evidence: {...specialist.specialist.evidence, ruleset: 'bun-native-v2'}},
  {evidence: {...specialist.specialist.evidence, games: 0}}, {evidence: {...specialist.specialist.evidence, url: 'https://example.org/unverified'}}]) {
  assert.throws(() => catalog.validateManifest({...manifest, models: [{...specialist, specialist: {...specialist.specialist, ...patch}}]}), /specialist/);
}
const scoped={...specialist,specialist:{...specialist.specialist,compatibility:{ruleset:'bun-native-v4-facing',games:8,summary:'仅兼容性快筛'}}};
assert(catalog.specialistLines(catalog.validateManifest({...manifest,models:[scoped]})[0]).join('\n').includes('当前规则快筛：bun-native-v4-facing'));
assert.throws(()=>catalog.validateManifest({...manifest,models:[{...scoped,specialist:{...scoped.specialist,compatibility:{...scoped.specialist.compatibility,games:0}}}]}),/compatibility/);
assert.strictEqual(rows.length, 1);
assert.strictEqual(rows[0].cycle, 16);
assert.throws(() => catalog.validateManifest({ schema: 'bad', models: [] }), /schema/);
assert.throws(() => catalog.validateManifest({ schema: 'qqt.web-models/v1', models: [{ ...manifest.models[0], file: '../x' }] }), /file/);
assert.strictEqual(catalog.modelUrl(rows[0]), 'models/safe-high-c16.json');
const production = require('./models.json');
const defaults=require('./model_inference_defaults.json');
const retired=require('./retired-models-20261107.json');
assert.strictEqual(production.models[0].id,defaults.default_model_id);
assert.strictEqual(defaults.ruleset,'bun-native-v2');
assert.strictEqual(defaults.max_ticks,2400);
for(const [id,inference]of Object.entries(defaults.policies)){
  assert(production.models.some(row=>row.id===id));
  require('./model_sampling_v1').profile({inference});
}
const current=retired.models.find(row=>row.id==='bun392-v2-risk-u9809');
assert.strictEqual(current.id,'bun392-v2-risk-u9809');
assert.strictEqual(current.source_sha256,require('./promotion-review.json').policy_sha256);
assert.strictEqual(current.sha256,require('./promotion-review.json').model_sha256);
assert.strictEqual(current.onnx.meta.rule_version,'bun-native-v2');
assert.strictEqual(current.onnx.meta.ability_actions,7);
assert.strictEqual(current.onnx.meta.state_dim,52);
assert(current.description.includes('P1') && current.description.includes('零泡') && current.description.includes('振荡'));
assert(current.description.includes('holdout') && current.description.includes('P0') && current.description.includes('退步'));
assert.strictEqual(current.score,23/24);
assert(production.models.some(r=>r.id==='bun392-v2-risk-u9969'));
assert(retired.models.some(r=>r.id==='bun392-v2-r4-k1-u5282'));
const broken={...current,onnx:{...current.onnx,meta:{...current.onnx.meta,ability_encoding:['none','bomb','use_item']}}};
assert.throws(()=>catalog.validateManifest({schema:'qqt.web-models/v1',models:[broken]}),/five-slot/);
assert(retired.models.some(r=>r.source_sha256==='f12ab745ca746ae364bcb172fb425399cbf47038f60f7129617036a4051d6a30'));
const latest=retired.models.find(row=>row.id==='bun392-pbt-slot05-u720');
assert.strictEqual(latest.id,'bun392-pbt-slot05-u720');
assert.strictEqual(latest.source_sha256,'ed5a2b1860e8256fa34ac5a5830ff9b5c81e67a0b33d5c8a73101b3149796663');
assert.strictEqual(latest.display_name,'Bun392 PBT u720（D）');
assert.strictEqual(latest.cycle,720);
assert.strictEqual(latest.onnx.meta.bun_state_encoding,'bun24-trap-timer-v2');
for (const asset of [latest,latest.onnx]) {
  assert(asset.file.includes(asset.sha256));
  assert(asset.url.endsWith('/'+asset.file));
}
const bridge = retired.models.find(row=>row.id==='bun392-je09-stage-c-g8');
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
const transport = retired.models.find(row => row.id === 'bun392-transport1536');
assert.strictEqual(transport.id, 'bun392-transport1536', 'Previous transport identity stays archived');
assert.strictEqual(transport.cycle, 1536);
assert.strictEqual(transport.strategy, 'raw');
assert.strictEqual(transport.source_sha256, '16ba817d53ef1f3ed4cf6039ff58f33f1f0de4adc69b3578047c5c3cdfbda719');
assert.strictEqual(transport.onnx.meta.bun_state_encoding, 'bun24-trap-timer-v2');
assert(transport.onnx.file.includes(transport.onnx.sha256));
assert(transport.file.includes(transport.sha256));
const bun392 = retired.models.find(row => row.id === 'bun392-selfplay3072');
assert(bun392, 'g3072 metadata remains archived');
assert.strictEqual(bun392.display_name, 'Bun392动态自对弈3072');
assert.strictEqual(bun392.cycle, 3072);
assert.strictEqual(bun392.source_sha256, 'dc3919a5e7a2f415f951e23f20069d14c3c1df332acf5dfa7b1bb21314ca4f70');
assert(bun392.url.includes('/releases/download/web-bun392-selfplay3072-v1/'));
for (const id of ['bun-tf-it4000','bun-kill-mlp4','safe-high-c16','safe-medium-c16','safe-high-c24']) {
  assert(retired.models.some(row => row.id === id), `historical metadata retained: ${id}`);
}
for (const row of [...catalog.validateManifest(production),...catalog.validateManifest(retired)]) {
  assert.strictEqual(catalog.modelUrl(row), row.oss_url, 'OSS is preferred, Release and same-origin mirror remain fallbacks');
  for(const asset of [row,...(row.onnx?[row.onnx]:[])]){assert(asset.oss_url.startsWith('https://qqt-model.oss-cn-beijing.aliyuncs.com/web-models/'));assert(asset.oss_url.includes(asset.sha256));assert(asset.url.startsWith('https://github.com/'));}
  assert.throws(()=>catalog.validateManifest({...production,models:[{...row,oss_url:'https://example.org/'+row.file}]}),/sources/);
}
assert.strictEqual(catalog.matchLabel('model-vs-rule'), '模型 vs 规则Bot');
console.log('网页模型目录与观战模式契约通过');

assert.deepEqual(production.models.map(r=>r.id),['items-l2-sp-u023302-tau03','bun392-g-frozen','items-l2-duel-u019902-tau03','bun392-v2-risk-u9969']);
assert.equal(retired.models.length,18);
for(const row of production.models)assert.equal(catalog.optionLabel(row),row.menu_label);
for(const value of ['model:bun392-v2-risk-u9809','model:missing',''])assert.equal(catalog.resolveSelection(value,production.models,defaults.default_model_id),'model:items-l2-sp-u023302-tau03');
assert.equal(catalog.resolveSelection('model:bun392-g-frozen',production.models,defaults.default_model_id),'model:bun392-g-frozen');
assert.equal(catalog.resolveSelection('bun.hunter@hard',production.models,defaults.default_model_id),'bun.hunter@hard');
