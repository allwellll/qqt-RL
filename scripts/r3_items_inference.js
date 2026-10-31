'use strict';
// Explicit evaluation-only R3 encoder. Production defaults are untouched.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const Q=require('../web/sim-bun-native-v2');
const encoding=require('./je05_bun_encoding');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
function inputs(s,p){
 const obs=new Float32Array(25*195);obs.set(s.encodeObsJAX(p,24));
 const live=[];for(let i=0;i<195;i++)if(s.fuse[i]>0&&s.owner[i]===p)live.push(i);
 const batches=live.map(i=>s.__r3?.bombs.get(i)?.batch??-1),valid=batches.filter(v=>v>=0),minimum=valid.length?Math.min(...valid):0;
 for(let k=0;k<live.length;k++)if(batches[k]>=0)obs[24*195+live[k]]=(batches[k]-minimum+1)/8;
 const state=new Float32Array(37);state.set(encoding.encodeState(s,p));const enemy=1-p;
 let k=24;state[k++]=Math.min(1,s.itemSlots[p].reduce((a,b)=>a+b.count,0)/50);
 for(const type of [1,2])state[k++]=Number(s.itemSlots[enemy].some(x=>x.item===type&&x.count>0));
 for(const player of [p,enemy])for(let type=0;type<4;type++)state[k++]=Number(s.movementStatus[player]===type);
 for(const player of [p,enemy])state[k++]=s.movementStatusTicks[player]/100;
 assert.equal(k,37);return {obs,state};
}
async function loadModel(file){
 const bytes=fs.readFileSync(file),doc=JSON.parse(bytes);assert.equal(doc.meta.r3_input_encoding,'r3-items-route-chain-visible-v1');
 assert.deepEqual(doc.meta.obs_shape,[25,13,15]);assert.equal(doc.meta.state_dim,37);
 const model=new Q.TransformerModel(doc),reference=model.forward.bind(model);
 const ort=require(path.resolve('runs/je03_original_nonbun_20261011/baseline_v1/deps/node_modules/onnxruntime-node'));
 assert.equal(hash(fs.readFileSync(doc.meta.onnx_path)),doc.meta.onnx_sha256);
 const session=await ort.InferenceSession.create(doc.meta.onnx_path,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
 let verified=false;
 model.forward=async(obs,state)=>{
  const r=await session.run({obs:new ort.Tensor('float32',obs,[1,25,13,15]),state:new ort.Tensor('float32',Float32Array.from(state),[1,37])});
  const result={move:r.move.data,bomb:r.bomb.data,value:r.value.data[0]};
  if(!verified){const expected=reference(obs,state);model.forwardMaxError=Math.max(...['move','bomb'].flatMap(k=>Array.from(result[k],(x,i)=>Math.abs(x-expected[k][i]))));assert(model.forwardMaxError<.002);verified=true;}
  return result;
 };
 model.act=async(s,p,rng)=>{const input=inputs(s,p),mask=s.legalMask(),v=await model.forward(input.obs,input.state);
  assert([...v.move,...v.bomb].every(Number.isFinite));return model._decodeAbility(model._sampleMasked(v.move,mask.mm[p],rng),model._sampleMasked(v.bomb,model._abilityMask(s,mask,p),rng));};
 model.greedy=true;model.inferEvery=1;model.backendSha=doc.meta.onnx_sha256;
 return {model,sha:hash(bytes),meta:doc.meta};
}
module.exports={inputs,loadModel};
