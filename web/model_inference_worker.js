'use strict';
// All neural arithmetic runs off the rendering thread. No simulator/RNG lives
// here: native encoding, masks and greedy action decoding remain in sim.js.
importScripts('sim.js', 'model_loader.js');
let session, meta, reference, runtime, backend;
async function verifiedDownload(asset) {
  return QQTModelLoader.downloadVerifiedAsset(`models/${asset.file}`,asset,
    (loaded,total)=>postMessage({progress:{loaded,total,asset:asset.file}}));
}
async function initialize(row) {
  async function loadReference() {
    const bytes=await verifiedDownload(row);
    const doc=JSON.parse(new TextDecoder().decode(bytes));
    meta=doc.meta;reference=new QQT.TransformerModel(doc);
  }
  // Published metadata/reference logits come from the exact SHA-verified JSON
  // in the reproducible parity benchmark. Healthy WASM needs only ONNX bytes;
  // avoid downloading and unpacking a second 40MB copy of the same policy.
  if(row.onnx && row.onnx.meta && row.onnx.reference_zero)meta=row.onnx.meta;
  else await loadReference();
  if(meta.arch!=='transformer')throw Error('Worker requires native Transformer');
  backend='worker-js';
  if(row.onnx) {
    // Required assets must download and verify before considering a runtime
    // fallback. HTTP/integrity failures never turn into a skipped model.
    const onnx=await verifiedDownload(row.onnx);
    try {
      importScripts('vendor/ort/ort.wasm.min.js');runtime=ort;
      runtime.env.wasm.numThreads=1; // Pages has no COOP/COEP / SharedArrayBuffer.
      runtime.env.wasm.proxy=false; // Already in our dedicated worker.
      runtime.env.wasm.wasmPaths=new URL('vendor/ort/',self.location.href).href;
      session=await runtime.InferenceSession.create(onnx,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
      const obs=new Float32Array(meta.obs_shape.reduce((a,b)=>a*b,1));
      const state=new Float32Array(meta.global_dim || 24);
      const expected=row.onnx.reference_zero || reference.forward(obs,state),actual=await infer(obs,state);
      const error=Math.max(...['move','bomb'].flatMap(k=>Array.from(actual[k],(v,i)=>Math.abs(v-expected[k][i]))),Math.abs(actual.value-expected.value));
      if(!Number.isFinite(error)||error>1e-3)throw Error('WASM/JS numerical parity failed');
      backend='worker-wasm';
      return {meta,backend,parityMaxError:error};
    } catch(error) {
      // Corrupt ONNX must never silently bypass integrity checking. Runtime
      // availability errors may use the SAME verified policy in worker JS.
      if(/numerical parity|Nonfinite/.test(error.message))throw error;
      if(session){await session.release();session=null;}
      if(!reference)await loadReference();
      return {meta,backend,backendReason:String(error.message)};
    }
  }
  return {meta,backend};
}
async function infer(obs,state) {
  let out;
  if(session) {
    const tensors=await session.run({obs:new runtime.Tensor('float32',obs,[1,...meta.obs_shape]),
      state:new runtime.Tensor('float32',Float32Array.from(state),[1,state.length])});
    out={move:Float32Array.from(tensors.move.data),bomb:Float32Array.from(tensors.bomb.data),value:Number(tensors.value.data[0])};
    for(const tensor of Object.values(tensors))tensor.dispose();
  } else out=reference.forward(obs,state);
  if(![...out.move,...out.bomb,out.value].every(Number.isFinite))throw Error('Nonfinite model inference');
  return out;
}
// Serialize even if callers submit multiple requests. Fixed-batch ONNX exports
// cannot combine independent seats into batch2; preserve seat/RNG ordering.
let queue=Promise.resolve();
onmessage=({data})=>{queue=queue.then(async()=>{
  try {
    const start=performance.now();
    const result=data.type==='init'?await initialize(data.row):await infer(data.obs,data.state);
    postMessage({id:data.id,result,ms:performance.now()-start});
  } catch(error){postMessage({id:data.id,error:String(error.message)});}
});};
