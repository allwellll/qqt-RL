'use strict';
const assert=require('node:assert/strict'),T=require('./cx_temperature_inference_v1'),Q=require('../web/sim-bun-native-v2');
const m=Object.create(Q.TransformerModel.prototype); // production sampler only; no fake model/forward
const v={move:[1000,1001,-2000],bomb:[-1,2],value:3};
for(const t of [.3,.5,.7]){const x=T.scale(v,t,false);assert.equal(x.value,3);assert.equal(x.move[1],1001/t);assert.equal(x.bomb[0],-1/t);assert.equal(v.move[1],1001);}
assert.strictEqual(T.scale(v,.3,true),v);assert.strictEqual(T.scale(v,1,false),v);
for(const t of [0,-1,NaN,Infinity])assert.throws(()=>T.scale(v,t,false));
for(const t of [.3,.5,.7]){m.greedy=false;let n=0;const a=m._sampleMasked([0,10000/t,1/t],[true,false,true],()=>{n++;return .99;});assert.equal(a,2);assert.equal(n,1);}
m.greedy=true;assert.equal(m._sampleMasked([1,2,2],[true,true,true],()=>{throw Error('argmax consumes no RNG');}),1);
const p=t=>{const x=Math.exp(1/t);return x/(1+x);};assert(p(.3)>p(.5)&&p(.5)>p(.7));
console.log('PASS temperature scales both heads, stable masked sampler, native argmax ties/RNG unchanged, invalid temperatures rejected');
