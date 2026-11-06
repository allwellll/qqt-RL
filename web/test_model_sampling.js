'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const S=require('./model_sampling_v1'),Q=require('./sim-bun-native-v2'),I=require('./r3_model_inputs');
const logits={move:[0,1,2,3,4],bomb:[1,2,3],value:7};
assert.equal(S.scale(logits,.3,true),logits);assert.equal(S.scale(logits,1,false),logits);
assert.equal(S.scale(logits,.5,false).move[4],8);assert.equal(S.scale(logits,.5,false).value,7);
for(const t of [0,-1,NaN,Infinity])assert.throws(()=>S.profile({inference:{temperature:t,greedy:false,seed:1}}));
assert.throws(()=>S.profile({inference:{temperature:1,greedy:false,seed:-1}}));
const proto=Q.TransformerModel.prototype;
const policy={meta:{},forward:()=>logits,_sampleMasked:proto._sampleMasked};
S.attach(policy,{inference:{temperature:.3,greedy:false,seed:123}});
const run=()=>{const r=Q.mulberry32(42),v=policy.forward();return Array.from({length:300},()=>policy._sampleMasked(v.move,[1,1,0,0,0],r));};
assert.deepEqual(run(),run());assert(run().every(a=>a===0||a===1));
assert(run().filter(a=>a===1).length>270,'low temperature concentrates probability without violating masks');
policy.greedy=true;assert.equal(policy._sampleMasked(policy.forward().move,[1,1,0,0,0],Q.mulberry32(1)),1);
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
I.installSim(Q);const sim=new Q.Sim(42);sim.reset(level,{nativeItems:true,nativeTrap:true});S.watch(sim);
for(let t=0;t<140&&!sim.done;t++){
 sim.step([[t%5,Number(t%11===0),0],[(t*3)%5,Number(t%13===0),0]]);
 for(const seat of [0,1])assert.deepEqual(S.inputs(sim,seat),I.inputs(sim,seat));
}
sim.trapped[0]=23;assert.deepEqual(S.inputs(sim,0).state,I.inputs(sim,0).state);
const act=()=>7;
const seven={meta:{schema_version:'five-item-slots-v1',state_dim:52,ability_actions:7},act,forward:async()=>logits};
S.attach(seven,{inference:{temperature:.5,greedy:false,seed:8}});assert.equal(seven.act,act);
(async()=>{assert.equal((await seven.forward()).bomb[2],6);console.log('Sampling PASS: seeded masked two-head temperature, argmax, R3 visible parity, trap timers, preserved seven-action consumer');})().catch(e=>{console.error(e);process.exitCode=1;});
