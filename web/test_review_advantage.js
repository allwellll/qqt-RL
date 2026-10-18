'use strict';
// Historical compatibility fixtures remain verified independently of public admission.
// The current catalog is exhaustively gated by test_review_admission.js.
const assert=require('assert'),fs=require('fs'),A=require('./review_advantage'),P=require('./review_protocol');
(async()=>{
  const candidates=[
    {action:[0,0],legal:true,probability:.2,reward:1,nextValue:4},
    {action:[1,0],legal:true,probability:.3,reward:-1,nextValue:2},
    {action:[2,0],legal:false,probability:.5,reward:999,nextValue:999},
  ];
  // Q=[3,0], legal probabilities=[.4,.6], mean=1.2, A=1.8.
  const result=A.calculate(candidates,[0,0],.5);
  assert.equal(result.taken_q,3);assert(Math.abs(result.mean_q-1.2)<1e-12);assert(Math.abs(result.advantage-1.8)<1e-12);
  assert.equal(A.calculate([{...candidates[0],terminal:true}],[0,0],.5).taken_q,1);
  assert.throws(()=>A.calculate(candidates,[2,0],.5));assert.throws(()=>A.calculate(candidates,[0,0],NaN));
  assert.throws(()=>A.calculate(candidates.map(c=>({...c,probability:0})),[0,0],.5));
  const catalog=JSON.parse(fs.readFileSync('reports/review_catalog_before_gate_20261018.json'));
  const doc=JSON.parse(fs.readFileSync('web/review-replays/'+catalog.replays[0].file));
  doc.meta.advantage=A.missing({...doc.meta,gamma:.995});doc.advantages=A.empty();A.validate(doc);
  assert.deepEqual(A.atTick(doc,10),{taken_q:null,mean_q:null,advantage:null,delta:null});
  doc.meta.advantage={...doc.meta.advantage,status:'recorded',reason:'fixture only',critic_sha256:'a'.repeat(64),counterfactual_sha256:'b'.repeat(64)};
  doc.advantages={schema:A.SCHEMA,scale:1e-6,ticks:[1,2,4],taken_q:[3000000,4000000,3000000],mean_q:[1200000,1000000,1000000],advantage:[1800000,3000000,2000000],delta:[null,1200000,null]};
  P.validate(doc);assert.equal(A.atTick(doc,2).delta,1.2);assert.equal(A.atTick(doc,3).advantage,null);
  for(const mutate of [d=>d.advantages.ticks[1]=1,d=>d.advantages.delta[0]=0,d=>d.advantages.delta[2]=0,
    d=>d.advantages.advantage[0]=4,d=>d.advantages.delta[1]=1,d=>d.advantages.mean_q.pop(),
    d=>d.advantages.ticks[0]=0,d=>d.advantages.scale=1,d=>d.advantages.q_table=[],
    d=>d.meta.advantage.critic_sha256=null,d=>d.meta.advantage.policy_sha256='c'.repeat(64),
    d=>d.meta.advantage.action_scope='sum_heads',d=>d.meta.advantage.status='not_recorded']){
    const bad=structuredClone(doc);mutate(bad);assert.throws(()=>P.validate(bad),/advantage/);
  }
  const old=await P.hash(P.identity(doc));doc.advantages.taken_q[0]++;assert.notEqual(old,await P.hash(P.identity(doc)));
  for(const row of catalog.replays){const real=JSON.parse(fs.readFileSync('web/review-replays/'+row.file));
    if(row.id.startsWith('h8-')&&!row.diagnostic){assert.equal(real.meta.advantage.status,'recorded');assert.match(real.meta.advantage.critic_sha256,/^[a-f0-9]{64}$/);assert.equal(real.advantages.ticks.length,real.actions.length);assert.equal(real.advantages.delta[0],null);assert(A.legalCount(real,1)>0);A.validate(real);}
  }
  console.log('A1 GREEN: weighted joint fixture, illegal exclusion, terminal, first/gap delta null, sparse columns RED/GREEN, hash, real H8 critic measurements');
})().catch(e=>{console.error(e);process.exitCode=1;});
