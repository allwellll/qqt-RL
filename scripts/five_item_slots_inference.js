'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const Q=require('../web/sim-bun-native-v2'),A=require('../web/five-item-slots-model-v1');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
async function loadModel(file){
 const bytes=fs.readFileSync(file),doc=JSON.parse(bytes);A.validate(doc.meta);
 const model=A.create(Q,doc),reference=model.forward.bind(model);
 const ort=require(path.resolve('runs/je03_original_nonbun_20261011/baseline_v1/deps/node_modules/onnxruntime-node'));
 assert.equal(hash(fs.readFileSync(doc.meta.onnx_path)),doc.meta.onnx_sha256);
 const session=await ort.InferenceSession.create(doc.meta.onnx_path,{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});let verified=0;
 model.forward=async(obs,state)=>{
  const out=await session.run({obs:new ort.Tensor('float32',Float32Array.from(obs),[1,25,13,15]),state:new ort.Tensor('float32',Float32Array.from(state),[1,52])});
  const result={move:Float32Array.from(out.move.data),bomb:Float32Array.from(out.bomb.data),value:out.value.data[0]};
  assert.equal(result.move.length,5);assert.equal(result.bomb.length,7);assert([...result.move,...result.bomb,result.value].every(Number.isFinite));
  // Compare a fixed first32 inputs, then retain the exact same verified ONNX
  // weights. Full JS arithmetic at every tick is a diagnostic CPU bottleneck.
  if(verified<32){const expected=reference(obs,state),error=Math.max(...['move','bomb'].flatMap(k=>Array.from(result[k],(x,i)=>Math.abs(x-expected[k][i]))),Math.abs(result.value-expected.value));
   assert(error<.002,`JSON/ONNX mismatch ${error}`);model.forwardMaxError=Math.max(model.forwardMaxError||0,error);model.lastReference=expected;verified++;model.verifiedInputs=verified;
  }else model.lastReference=null;
  return result;
 };
 model.greedy=true;model.inferEvery=1;return {model,meta:doc.meta,sha:hash(bytes),session};
}
module.exports={loadModel,inputs:A.inputs};
