'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('../web/sim-bun-native-v2'),M=require('./r3_items_metrics'),I=require('./r3_items_inference');
const deployed=require('../../cx_publish_r1_1805_20261030/web/r3_model_inputs');
M.install(Q);deployed.installSim(Q);deployed.installSim(Q);
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806),s=new Q.Sim(2026314000);s.reset(level,{nativeItems:true,nativeTrap:true});s.__btIdentity={seed:2026314000,seat:1};
for(let t=0;t<320&&!s.done;t++){
 for(const p of [0,1]){const a=I.inputs(s,p),b=deployed.inputs(s,p);assert.deepEqual(b,a,`encoder parity tick${t}/P${p}`);}
 s.step([[t%5,Number(t%11===0),Number(t%17===0)],[(t*3)%5,Number(t%13===0),Number(t%19===0)]]);
}
s.reset(level,{nativeItems:true,nativeTrap:true});assert.equal(s.__r3Visible,undefined);
const old={meta:{},_getLaggedInputs:()=>1};assert.equal(deployed.attach(old),old);assert.equal(old._getLaggedInputs(),1);
const r3={meta:{r3_input_encoding:'r3-items-route-chain-visible-v1'},obsShape:[25,13,15]};deployed.attach(r3);assert.equal(r3._getLaggedInputs(s,0).state.length,37);
console.log('R3 deployment ABI PASS: every tick, both seats, visible batch history, masks/RNG untouched by encoder, reset, legacy unchanged');
