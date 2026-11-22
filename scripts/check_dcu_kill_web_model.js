'use strict';
// Actual kill-mode native web states, both seats; parity checks, no training.
const fs=require('node:fs'),assert=require('node:assert/strict');
const {create}=require('./kill_mode_bridge'),Q=require('../web/sim-bun-native-v4-facing'),R=require('../web/r3_model_inputs');
const [file,output]=process.argv.slice(2),doc=JSON.parse(fs.readFileSync(file));
assert.equal(doc.meta.rule_version,'bun-native-v4-facing');assert.equal(doc.meta.fixed_mode,0);
const model=R.attach(new Q.TransformerModel(doc));
const plain=x=>({move:Array.from(x.move),bomb:Array.from(x.bomb),value:x.value});
const reference_zero=plain(model.forward(new Float32Array(25*195),new Float32Array(37))),samples=[];
for(const seed of [2026111501,2026111601]){
 const s=create(seed),rng=Q.mulberry32(seed);
 for(let t=0;t<=60;t++){
  if([0,12,36,60].includes(t))for(const seat of [0,1]){
   const x=R.inputs(s,seat),masks=s.legalMask();
   samples.push({map:'kill806',seat,tick:t,seed,obs:Array.from(x.obs),state:Array.from(x.state),
     masks:{move:Array.from(masks.mm[seat]),bomb:Array.from(model._abilityMask(s,masks,seat))},js:plain(model.forward(x.obs,x.state))});
  }
  const masks=s.legalMask();s.step([0,1].map(p=>{const legal=Array.from(masks.mm[p]).flatMap((v,i)=>v?[i]:[]);return[legal[Math.floor(rng()*legal.length)],+(masks.bm[p][1]&&rng()<.2),0];}));
 }
}
fs.writeFileSync(output,JSON.stringify({ruleset:'bun-native-v4-facing',reference_zero,samples})+'\n');
console.log(JSON.stringify({status:'PASS',kill_state_samples:samples.length}));
