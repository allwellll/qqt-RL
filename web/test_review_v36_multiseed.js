'use strict';
// Historical compatibility fixtures remain verified independently of public admission.
// The current catalog is exhaustively gated by test_review_admission.js.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const crypto=require('node:crypto');
const P=require('./review_protocol');
const {verifyBatch,MATRIX}=require('../scripts/generate_v36_multiseed_review');
(async()=>{
  assert.deepEqual(MATRIX.map(x=>x.seed),Array.from({length:8},(_,i)=>2026101780+i));
  const catalog=JSON.parse(fs.readFileSync('reports/review_catalog_before_gate_20261018.json'));
  await verifyBatch(catalog);
  const docs=catalog.replays.filter(r=>r.batch_id==='v36-fixed-eight-20261017').map(r=>JSON.parse(fs.readFileSync('web/review-replays/'+r.file)));
  for(const doc of docs){
    assert.equal(doc.meta.reward_status,'offline_recomputed_current_reward');
    assert.equal(doc.meta.offline_recomputed_current_reward.measured_count,doc.actions.length);
    assert.equal(doc.meta.route.measured_count,doc.actions.length);
    assert(doc.diagnostic_ticks.every(t=>t.offline_recomputed_current_reward&&t.raw_a1===null&&t.shaped_a1===null));
    const bad=structuredClone(doc);bad.diagnostic_ticks[0].offline_recomputed_current_reward.components[0].lesson_task+=1;
    assert.throws(()=>P.validate(bad),/offline reward/);
    const forged=structuredClone(doc);forged.diagnostic_ticks[0].raw_a1=1;assert.throws(()=>P.validate(forged),/offline reward tick/);
    const forgedDelta=structuredClone(doc);forgedDelta.diagnostic_ticks[0].cross_critic_delta_a=1;assert.throws(()=>P.validate(forgedDelta),/offline reward tick/);
    const falseScope=structuredClone(doc);falseScope.meta.training_reward_status='recorded';assert.throws(()=>P.validate(falseScope),/offline reward scope/);
    const route=structuredClone(doc);route.diagnostic_ticks[0].offline_recomputed_current_reward.route[0].route_phi_after+=1;assert.throws(()=>P.validate(route),/offline route formula/);
  }
  const server=require('./server').createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
  try{
    const url='http://127.0.0.1:'+server.address().port;
    const response=await fetch(url+'/review-catalog.json');assert.equal(response.status,200);assert.deepEqual(await response.json(),require('./review-catalog.json'));
    for(const row of catalog.replays.filter(r=>r.batch_id==='v36-fixed-eight-20261017')){
      const response=await fetch(url+'/review-replays/'+row.file);assert.equal(response.status,200);const bytes=Buffer.from(await response.arrayBuffer());assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),row.sha256);
      assert.equal((await fetch(url+'/review.html#replay='+row.id+'&tick=1')).status,200);
    }
  }finally{await new Promise(r=>server.close(r));}
  console.log('v36 fixed eight seeds, no selection, per-action evidence and deterministic replay GREEN');
})().catch(e=>{console.error(e);process.exitCode=1;});
