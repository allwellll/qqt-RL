'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const crypto=require('node:crypto');

const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json','utf8'));
const row=catalog.replays.find(x=>x.id==='h8-route-v36-g32-2026101736');
assert(row,'v36 review entry exists');
const doc=JSON.parse(fs.readFileSync('web/review-replays/'+row.file,'utf8'));
assert.equal(crypto.createHash('sha256').update(fs.readFileSync('web/review-replays/'+row.file)).digest('hex'),row.sha256);
assert.equal(doc.meta.checkpoint_sha256,'e2dd2cb9890abf5753b34adcdb14abc93b3f977229c29326fd2033e584a73945');
assert.equal(doc.meta.advantage_audit?.status,'blocked',
  'when strict same-checkpoint A1 cannot be recovered, publish an explicit technical blocker');
assert.match(doc.meta.advantage_audit?.reason||'',/raw.task critic|原始任务 Critic/i);
assert.match(doc.meta.advantage_audit?.reason||'',/reward ledger|奖励账本/i);
assert.match(doc.meta.advantage_audit?.training_state_sha256||'',/^[a-f0-9]{64}$/);
assert.equal(doc.meta.advantage_audit?.raw_task_critic_sha256,null,
  'do not claim an independent raw-task Critic that the checkpoint did not preserve');
assert.match(doc.meta.advantage_audit?.reason||'',/PBRS Critic/);
assert.equal(Object.hasOwn(doc,'advantages'),false,'unproved Q/value data must not be published as A1');
assert.equal(doc.meta.advantage_status,'无法严格复算');

const legacy=JSON.parse(fs.readFileSync('web/replays.json','utf8'));
assert.equal(legacy.deprecated,true,'legacy replay catalog is read-only compatibility');
assert.equal(legacy.replacement,'review-catalog.json');
assert(legacy.replays.length>0,'legacy URLs remain available');
const app=fs.readFileSync('web/app.js','utf8');
assert.match(app,/只读兼容/);
const review=fs.readFileSync('web/review.js','utf8');
assert.match(review,/无法严格复算/);
const legacyGenerator=require('node:child_process').spawnSync(process.execPath,['scripts/generate_web_replays.js'],{encoding:'utf8'});
assert.notEqual(legacyGenerator.status,0);
assert.match(legacyGenerator.stderr,/Deprecated: generate diagnostic recordings with the Review pipeline/);
const legacyVerifier=require('node:child_process').spawnSync(process.execPath,['scripts/verify_web_replays.js'],{encoding:'utf8'});
assert.notEqual(legacyVerifier.status,0);
assert.match(legacyVerifier.stderr,/Deprecated: Review is the only diagnostic replay acceptance path/);
const checkpointGenerator=require('node:child_process').spawnSync(process.execPath,[
  'scripts/generate_je09_checkpoint_replay.js',
  'runs/je09_bun_20261017/stage_h8_route_pbrs_continuous_v36_attempt02/authority_je09_v1/bun392_bridge_pfsp8_g000032/model/model.json',
  'web/replays/forbidden-review-migration-test.json'],{encoding:'utf8'});
assert.notEqual(checkpointGenerator.status,0);
assert.match(checkpointGenerator.stderr,/legacy replay targets are read-only/);
assert.equal(fs.existsSync('web/replays/forbidden-review-migration-test.json'),false);
console.log('v36 A1 evidence and legacy replay deprecation contract passed');
