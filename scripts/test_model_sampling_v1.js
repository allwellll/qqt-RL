'use strict';
const assert=require('node:assert/strict'),Q=require('../web/sim-bun-native-v2'),A=require('./model_sampling_v1');
const R=require('./r3_items_inference'),R3=require('./r3_items_metrics');
const fs=require('node:fs');
const Legacy=require('../web/sim');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
async function main(){
 for(const x of [0,-1,NaN,Infinity])assert.throws(()=>A.profile({inference:{greedy:false,temperature:x,seed:1}}));
 const v={move:[0,1,2,3,4],bomb:[2,1,0],value:7};assert.equal(A.scale(v,.3,true),v);assert.equal(A.scale(v,1,false),v);
 assert.deepEqual(Array.from(A.scale(v,.5,false).move),[0,2,4,6,8]);assert.deepEqual(v.move,[0,1,2,3,4]);
 R3.install(Q);
 const s=new Q.Sim(4),plain=new Q.Sim(4);s.reset(level,{nativeItems:true,nativeTrap:true});plain.reset(level,{nativeItems:true,nativeTrap:true});
 A.watch(s);s.__btIdentity=true;R3.exportMetrics(s);
 for(let t=0;t<80;t++){
  for(const p of [0,1]){const a=A.inputs(s,p),b=R.inputs(s,p);assert.deepEqual(a.obs,b.obs);assert.deepEqual(a.state,b.state);}
  const acts=[[t%8===0?4:3,t%8===0?1:0,0],[t%6,0,0]];acts[1][0]%=5;s.step(acts);plain.step(acts);
  assert.deepEqual(s.snapshotReplay(),plain.snapshotReplay());
 }
 const model=Object.create(Q.TransformerModel.prototype);model.meta={r3_input_encoding:'r3-items-route-chain-visible-v1',obs_shape:[25,13,15],state_dim:37,ability_actions:3};model.abilityActions=3;model.isBunModel=true;
 model.forward=async(o,g)=>{assert.equal(o.length,4875);assert.equal(g.length,37);return v;};
 A.attach(model,{inference:{greedy:false,temperature:.5,seed:123}});
 const play=async()=>{const rng=Q.mulberry32(92),a=[];for(let i=0;i<40;i++)a.push(await model.act(s,1,rng));return a;};
 assert.deepEqual(await play(),await play());
 const browser=Object.create(Legacy.TransformerModel.prototype);browser.meta=model.meta;browser.abilityActions=3;browser.isBunModel=true;
 assert.deepEqual(browser._abilityMask(s,s.legalMask(),1),model._abilityMask(s,s.legalMask(),1));
 for(const ability of [0,1,2])assert.deepEqual(browser._decodeAbility(3,ability),model._decodeAbility(3,ability));
 const ar=Q.mulberry32(615),br=Q.mulberry32(615);browser.greedy=model.greedy=false;
 for(let k=0;k<100;k++)assert.equal(browser._sampleMasked([0,1,2,3,4],[1,0,1,1,1],ar),model._sampleMasked([0,1,2,3,4],[1,0,1,1,1],br));
 const sampler={greedy:false},rng=Q.mulberry32(17),fn=Q.MLPModel.prototype._sampleMasked;
 const samples=Array.from({length:200},()=>fn.call(sampler,[0,1,100],[1,1,0],rng));assert(samples.every(x=>x!==2));assert(new Set(samples).size>1);
 let calls=0;sampler.greedy=true;assert.equal(fn.call(sampler,[1,1,99],[1,1,0],()=>{calls++;return .1;}),0);assert.equal(calls,0);
 assert.throws(()=>A.attach({meta:{state_dim:38},forward(){return v;}}),/Unknown/);
 console.log('PASS temperature/masks/seed, exact R3 visible encoder, observer physics, legacy greedy, ABI refusal');
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
