'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const crypto=require('node:crypto');
const P=require('./review_protocol');

async function main(){
  const catalog=JSON.parse(fs.readFileSync('web/review-catalog.json','utf8'));
  const row=catalog.replays.find(r=>r.id==='h8-route-v36-g32-2026101736');
  assert(row,'v36 must be listed in the review catalog');
  assert.equal(catalog.replays[0].id,row.id,'the current v36 replay must be the first visible activity');
  assert(Array.isArray(row.summary.failures),'v36 summary.failures must be an array for Review list rendering');
  assert.equal(row.name,'H8 v36 连续路线 PBRS g32');
  const bytes=fs.readFileSync('web/review-replays/'+row.file);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),row.sha256);
  const d=JSON.parse(bytes);
  P.validate(d);
  assert.equal(d.schema,'qqt.diagnostic-replay/v1');
  assert.equal(d.meta.model,'H8 v36 连续路线 PBRS g32');
  assert.equal(d.meta.outcome_status,'未胜、未交付；片段结束时仍未终局');
  assert.equal(d.meta.advantage_status,'无法严格复算');
  assert.equal(d.summary.winner,null);
  assert.deepEqual(d.summary.score,[0,0]);
  assert.equal(d.meta.advantages_status,'blocked');
  assert.equal(d.meta.advantage_audit.status,'blocked');
  assert.equal(Object.hasOwn(d,'route'),false,'no unsupported route measurements');
  assert.equal(Object.hasOwn(d,'advantages'),false,'no fabricated advantage measurements');
  const source=JSON.parse(fs.readFileSync('web/replays/bun392-je09-route-pbrs-continuous-v36-g32-seed-2026101736.json'));
  assert.equal(d.meta.source_replay_sha256,crypto.createHash('sha256').update(fs.readFileSync('web/replays/bun392-je09-route-pbrs-continuous-v36-g32-seed-2026101736.json')).digest('hex'));
  assert.deepEqual(d.actions,source.actions.map(a=>[a[0],a[1],a[3],a[4]]));
  assert.equal(d.checks.length,7);
  assert.equal(d.meta.truncated,true);
  assert.deepEqual(d.summary.stored[0],[1,0],'player 0 never delivered the enemy-source bun');
  const level=require('../scripts/eval_web_hunter').loadLevel();
  assert.equal(d.meta.arena_map_sha256,await P.hash(require('./tactical_arena').buildTacticalArena(level)));
  const player=new P.Player(d,level);
  assert.equal(await player.verify(),true);
  const app=fs.readFileSync('web/review.js','utf8');
  assert.match(app,/fetch\(['"]review-catalog\.json['"],\s*\{\s*cache:\s*['"]no-store['"]\s*\}\)/,
    'catalog fetch must bypass stale browser and intermediary cache entries');
  assert.match(app,/const p=new URLSearchParams\(location\.hash\.slice\(1\)\);await load\(p\.get\(['"]replay['"]\)\|\|catalog\.replays\[0\]\.id/,
    'deep links must continue loading by replay ID after catalog reordering');
  console.log('v36 review replay RED/GREEN contract passed');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
