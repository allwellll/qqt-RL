'use strict';
// Historical compatibility fixtures remain verified independently of public admission.
// The current catalog is exhaustively gated by test_review_admission.js.
const assert=require('assert'),fs=require('fs'),P=require('./review_protocol');
(async()=>{
 const catalog=JSON.parse(fs.readFileSync('reports/review_catalog_before_gate_20261018.json'));
 const level=require('../scripts/eval_web_hunter').loadLevel();let checks=0;
 for(const row of [...catalog.replays,...(catalog.history_replays||[])]){
  const bytes=fs.readFileSync(`web/review-replays/${row.file}`),d=JSON.parse(bytes);
  assert.equal(await P.sha256(bytes),row.sha256);assert.equal(bytes.length,row.bytes);P.validate(d);
  const player=new P.Player(d,level);await player.verify();
  for(const tick of [0,1,199,200,201,Math.floor(d.actions.length/2),d.actions.length,17]){
   if(tick>d.actions.length)continue;player.seek(tick);const fresh=new P.Player(d,level);fresh.seek(tick);assert.equal(await P.hash(P.state(player.sim)),await P.hash(P.state(fresh.sim)));checks++;
  }
  for(const mutate of [x=>x.schema='bad',x=>x.actions[0][0]=5,x=>x.actions[0][1]=3,x=>x.actions[0][0]=1.5,x=>x.actions.pop(),x=>x.summary.reward[0]+=1,x=>x.checks[0][1]='broken',x=>x.checks[1][0]=0,x=>{x.summary.done=false;delete x.meta.truncated;},x=>x.rewards.push([1,{invalid:NaN},{}])]){
   const bad=structuredClone(d);mutate(bad);assert.throws(()=>P.validate(bad));checks++;
  }
  const bad=structuredClone(d);bad.checks[1][1]='0'.repeat(64);await assert.rejects(new P.Player(bad,level).verify(),/state hash/);checks++;
  const tamper=structuredClone(d);tamper.meta.replay_identity='0'.repeat(64);await assert.rejects(new P.Player(tamper,level).verify(),/identity/);checks++;
  const a={id:'test-annotation',tick:17,category:'return',severity:'high',note:'test <script>'},exported=P.annotationDocument(d,[a]);assert.deepEqual(P.validateAnnotations(exported,d),[a]);
  for(const mutate of [x=>x.replay_id='other',x=>x.replay_hash='0'.repeat(64),x=>x.schema='bad',x=>x.annotations[0].tick=-1,x=>x.annotations[0].severity='invalid',x=>x.annotations.push(x.annotations[0])]){const bad=structuredClone(exported);mutate(bad);assert.throws(()=>P.validateAnnotations(bad,d));checks++;}
  assert.throws(()=>player.seek(-1));assert.throws(()=>player.seek(d.actions.length+1));assert.throws(()=>player.seek(1.5));
 }
 assert.equal(P.time(123), '00:12.300');assert.equal(P.time(2400),'04:00.000');
 console.log(`Diagnostic protocol GREEN: ${catalog.replays.length} replays, ${checks} schema RED/GREEN/hash/seek/annotation checks`);
})().catch(e=>{console.error(e);process.exitCode=1;});
