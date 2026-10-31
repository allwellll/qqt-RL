'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),vm=require('node:vm'),path=require('path');
const QQT=require('./sim.js');
const Phase=require('./retention_phase_conditioning.js');
(async()=>{
  // Real shared worker lifecycle via the R5-only entry; reference loading
  // receives its add-on before init, including the runtime fallback path.
  const messages=[],imports=[];let resolve;
  const context={performance,Float32Array,Float64Array,TextDecoder,URL,Promise,
    QQT,self:{location:{href:'https://example.org/retention_phase_inference_worker.js'}},
    QQTModelLoader:{downloadVerifiedAsset:async()=>new TextEncoder().encode(JSON.stringify({
      meta:{arch:'transformer',embed:8,obs_shape:[24,13,15],phase_conditioning:Phase.encoding}})).buffer},
    postMessage:m=>{messages.push(m);if(resolve)resolve(m);},
    importScripts(...names){for(const name of names){imports.push(name);
      if(name==='sim.js'||name==='model_loader.js')continue;
      vm.runInContext(fs.readFileSync(path.join(__dirname,name),'utf8'),context);
    }}
  };
  // Keep network arithmetic covered by the real ONNX/JS Python regression;
  // this fixture verifies the actual worker init/fallback/callback sequence.
  const original=QQT.TransformerModel;
  QQT.TransformerModel=class{constructor(doc){this.meta=doc.meta;} _run(){return [{move:[0],bomb:[0],value:0}];} forward(){return this._run()[0];}};
  try{
    vm.createContext(context);
    vm.runInContext(fs.readFileSync(path.join(__dirname,'retention_phase_inference_worker.js'),'utf8'),context);
    const response=new Promise(r=>{resolve=r;});
    context.onmessage({data:{id:1,type:'init',row:{}}});
    assert.equal((await response).result.backend,'worker-js');
    assert(imports.includes('model_inference_worker.js'));
    assert(imports.includes('retention_phase_conditioning.js'));
    assert.equal(context.QQTPhaseConditioning.encoding,Phase.encoding);
  }finally{QQT.TransformerModel=original;}
  console.log('R5-only worker entry installs visible phase reference before initialization');
})().catch(error=>{console.error(error);process.exitCode=1;});
