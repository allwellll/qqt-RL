'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {probabilities,abilityRank,install}=require('../scripts/bomb_threat_iter2_actions');
test('temperature masks, legal ranks and independent ability semantics',()=>{
 assert.deepEqual(probabilities([0,0,100],[1,1,0],.5),[.5,.5,0]);
 assert.ok(probabilities([0,1],[1,1],.3)[1]>probabilities([0,1],[1,1],1)[1]);
 assert.equal(abilityRank([3,2,1],[1,1,1]).rank,2);
 assert.equal(abilityRank([3,2,1],[1,0,1]).rank,null);
 assert.equal(abilityRank([1,1,0],[1,1,1]).rank,2);
 assert.throws(()=>probabilities([1],[1],0));
});
test('zero and unit temperature preserve original sampler/RNG exactly',()=>{
 for(const tau of [0,1]){
  let calls=0;const model={_sampleMasked(z,m,r){calls++;return z[0]+r();},forward:async()=>({}),act:async()=>[0,0,0]};
  install(model,{tau});let draws=0;
  assert.equal(model._sampleMasked([4,3],[1,1],()=>{draws++;return .25;}),4.25);
  assert.equal(calls,1);assert.equal(draws,1);
 }
});
