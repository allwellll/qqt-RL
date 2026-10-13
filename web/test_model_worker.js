'use strict';
require('./test_bun_model_encoding.js');
const assert=require('node:assert/strict');
const {Client}=require('./model_worker.js');
const {verifyBuffer}=require('./model_loader.js');
const crypto=require('node:crypto');
(async()=>{
  const sent=[];let terminated=false;
  const fake={postMessage(message,transfer){sent.push({message,transfer});},terminate(){terminated=true;}};
  const progress=[];const client=new Client(fake,p=>progress.push(p));
  const first=client.request('infer',{obs:new Float32Array([1]),state:new Float32Array([2])});
  const second=client.request('infer',{obs:new Float32Array([3]),state:new Float32Array([4])});
  assert.equal(sent[0].transfer.length,2);assert.notEqual(sent[0].message.id,sent[1].message.id);
  fake.onmessage({data:{progress:{loaded:2,total:4}}});assert.equal(progress.length,1);
  fake.onmessage({data:{id:sent[1].message.id,result:{value:2},ms:7}});
  fake.onmessage({data:{id:sent[0].message.id,result:{value:1},ms:8}});
  assert.equal((await first).value,1);assert.equal((await second).value,2);
  assert.deepEqual(client.times,[7,8]);assert.equal(client.pending.size,0);
  const failure=client.request('infer',{obs:new Float32Array(1),state:new Float32Array(1)});
  fake.onmessage({data:{id:sent[2].message.id,error:'nonfinite'}});await assert.rejects(failure,/nonfinite/);
  const pending=client.request('init');client.close();await assert.rejects(pending,/closed/);assert(terminated);
  const bytes=Uint8Array.from([1,2,3]).buffer;const sha=crypto.createHash('sha256').update(Buffer.from(bytes)).digest('hex');
  await verifyBuffer(bytes,{bytes:3,sha256:sha},crypto.webcrypto);
  await assert.rejects(verifyBuffer(bytes,{bytes:4,sha256:sha},crypto.webcrypto),/大小/);
  await assert.rejects(verifyBuffer(bytes,{bytes:3,sha256:'0'.repeat(64)},crypto.webcrypto),/SHA-256/);
  const catalog=require('./model_catalog.js'),manifest=require('./models.json');
  const row=manifest.models.find(r=>r.id==='bun392-selfplay3072');assert(row.onnx);
  assert.throws(()=>catalog.validateManifest({...manifest,models:[{...row,onnx:{...row.onnx,file:'../x.onnx'}}]}),/ONNX/);
  assert.throws(()=>catalog.validateManifest({...manifest,models:[{...row,onnx:{...row.onnx,meta:{...row.onnx.meta,checkpoint_sha256:'0'.repeat(64)}}}]}),/metadata/);
  assert.throws(()=>catalog.validateManifest({...manifest,models:[{...row,onnx:{...row.onnx,reference_zero:{...row.onnx.reference_zero,value:NaN}}}]}),/metadata/);
  const app=require('fs').readFileSync(require.resolve('./app.js'),'utf8');
  assert(app.includes('await QQTModelWorker.load(row'));
  assert(app.includes('setInterval(tick, 100)'),'Keep physical 10Hz tick');
  const workerSource=require('fs').readFileSync(require.resolve('./model_worker.js'),'utf8');
  assert(workerSource.includes("client.request('init',{row},20*60*1000)"));
  // Exercise the real worker script with controlled runtime/download errors.
  const vm=require('node:vm'),source=require('fs').readFileSync(require.resolve('./model_inference_worker.js'),'utf8');
  async function initialize({downloadError,parityError,runtimeError,metadataOnly}={}) {
    let resolve;const answer=new Promise(r=>{resolve=r;});let downloads=0;
    const context={importScripts(){},performance,Float32Array,TextDecoder,URL,
      self:{location:{href:'https://example.org/model_inference_worker.js'}},
      fetch:async()=>{if(++downloads===(metadataOnly?1:2)&&downloadError)throw Error(downloadError);return {};},
      QQTModelLoader:{downloadVerifiedAsset:async()=>{await context.fetch();return new TextEncoder().encode(JSON.stringify({meta:{arch:'transformer',obs_shape:[24,13,15],global_dim:24}})).buffer;}},
      QQT:{TransformerModel:class {forward(){return {move:[0],bomb:[0],value:0};}}},
      ort:{env:{wasm:{}},Tensor:class{},InferenceSession:{create:async()=>{
        if(runtimeError)throw Error(runtimeError);
        return {run:async()=>Object.fromEntries(['move','bomb','value'].map(k=>[k,{data:[parityError?1:0],dispose(){}}])),release:async()=>{}};
      }}},postMessage:message=>{if(message.id)resolve(message);}};
    vm.createContext(context);vm.runInContext(source,context);
    context.onmessage({data:{id:1,type:'init',row:{onnx:metadataOnly?{meta:{arch:'transformer',obs_shape:[24,13,15],state_dim:24},reference_zero:{move:[0],bomb:[0],value:0}}:{}}}});
    const result=await answer;result.downloads=downloads;return result;
  }
  assert.equal((await initialize()).result.backend,'worker-wasm');
  const lean=await initialize({metadataOnly:true});assert.equal(lean.result.backend,'worker-wasm');assert.equal(lean.downloads,1);
  assert.match((await initialize({metadataOnly:true,downloadError:'HTTP 500'})).error,/HTTP 500/);
  const leanFallback=await initialize({metadataOnly:true,runtimeError:'WASM unsupported'});assert.equal(leanFallback.result.backend,'worker-js');assert.equal(leanFallback.downloads,2);
  assert.match((await initialize({downloadError:'HTTP 500'})).error,/HTTP 500/);
  assert.match((await initialize({downloadError:'SHA-256 校验失败'})).error,/校验失败/);
  assert.match((await initialize({parityError:true})).error,/numerical parity/);
  const fallback=await initialize({runtimeError:'WASM unsupported'});
  assert.equal(fallback.result.backend,'worker-js');assert.match(fallback.result.backendReason,/unsupported/);
  console.log('Worker request lifecycle, verified ONNX catalog/download, unchanged 10Hz contract passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
