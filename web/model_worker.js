'use strict';
(function(root,factory){
  const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.QQTModelWorker=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  class Client {
    constructor(worker,onProgress=()=>{}) {
      this.worker=worker;this.pending=new Map();this.sequence=0;this.times=[];
      worker.onmessage=({data})=>{
        if(data.progress){onProgress(data.progress);return;}
        const entry=this.pending.get(data.id);if(!entry)return;
        this.pending.delete(data.id);clearTimeout(entry.timer);
        if(data.error)entry.reject(Error(data.error));
        else {if(entry.infer)this.times.push(data.ms);entry.resolve(data.result);}
      };
      worker.onerror=event=>this.fail(Error(event.message||'Model worker failed'));
    }
    request(type,fields={},timeout=30000) {
      const id=++this.sequence;
      return new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>{this.pending.delete(id);reject(Error('Model worker timeout'));},timeout);
        this.pending.set(id,{resolve,reject,timer,infer:type==='infer'});
        const transfer=type==='infer'?[fields.obs.buffer,fields.state.buffer]:[];
        this.worker.postMessage({id,type,...fields},transfer);
      });
    }
    fail(error){for(const entry of this.pending.values()){clearTimeout(entry.timer);entry.reject(error);}this.pending.clear();}
    close(){this.fail(Error('Model closed'));this.worker.terminate();}
  }
  async function load(row,onProgress) {
    const client=new Client(new Worker('model_inference_worker.js'),onProgress);
    try {
      // Large verified assets can download slowly. Bound initialization to
      // 20 minutes; inference requests retain their separate 30s timeout.
      const initialized=await client.request('init',{row},20*60*1000);
      // Reuse the existing asynchronous act/caching/greedy/mask implementation.
      const model=new QQT.ORTTransformerModel({meta:initialized.meta},null);
      model.backend=initialized.backend;model.backendReason=initialized.backendReason;
      model.parityMaxError=initialized.parityMaxError;model.greedy=true;model.inferEvery=1;
      model.forward=(obs,state)=>client.request('infer',{obs:Float32Array.from(obs),state:Float32Array.from(state)});
      model.forward2=async(o0,s0,o1,s1)=>[await model.forward(o0,s0),await model.forward(o1,s1)];
      if(typeof QQTR3Inputs!=='undefined')QQTR3Inputs.attach(model);
      model.close=()=>client.close();model.inferenceTimes=client.times;
      return model;
    } catch(error){client.close();throw error;}
  }
  return {Client,load};
});
