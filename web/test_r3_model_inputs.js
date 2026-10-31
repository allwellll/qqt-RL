'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const I=require('./r3_model_inputs'),Q=require('./sim-bun-native-v2'),old=require('./sim-stock-window-v1');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const s=new Q.Sim(731),plain=new Q.Sim(731);s.reset(level,{nativeItems:true,nativeTrap:true});plain.reset(level,{nativeItems:true,nativeTrap:true});
I.installSim(Q);I.installSim(Q);
for(let t=0;t<320&&!s.done;t++){
 const a=[[t%5,Number(t%13===0),0],[(t*3)%5,Number(t%17===0),0]];
 assert.deepEqual(s.step(a),plain.step(a));assert.deepEqual(s.snapshotReplay(),plain.snapshotReplay());
 for(let p=0;p<2;p++){const r=I.inputs(s,p);assert.equal(r.obs.length,25*195);assert.equal(r.state.length,37);assert.deepEqual(r.obs.slice(0,24*195),Float32Array.from(s.encodeObsJAX(p,24)));assert.deepEqual(r.state.slice(0,24),Float32Array.from(s.encodeStateJAX(p,'bun24-trap-timer-v2')));assert([...r.obs,...r.state].every(Number.isFinite));}
}
s.reset(level,{nativeItems:true,nativeTrap:true});assert.equal(s.__r3Visible,undefined);
const legacy={meta:{},_getLaggedInputs:()=>7};I.attach(legacy);assert.equal(legacy._getLaggedInputs(),7);
const model={meta:{r3_input_encoding:'r3-items-route-chain-visible-v1'},obsShape:[25,13,15]};I.attach(model);assert.equal(model._getLaggedInputs(s,1).state.length,37);
const browser={QQT:Q,QQTStockWindowV1:old,QQTR3Inputs:I};browser.globalThis=browser;vm.runInNewContext(fs.readFileSync('web/ruleset_selector.js','utf8'),browser);assert.equal(browser.QQTRulesets.current,'bun-native-v2');browser.QQTRulesets.select('stock-window-v1');assert.equal(browser.QQT,old);browser.QQTRulesets.select('bun-native-v2');assert.equal(browser.QQT,Q);assert.throws(()=>browser.QQTRulesets.select('bad'));
console.log('R3 deployment visible inputs, passive physics, reset, legacy and rule selection PASS');
