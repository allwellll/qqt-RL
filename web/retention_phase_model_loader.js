'use strict';
(function(root){
  async function load(row,onProgress){
    const client=new root.QQTModelWorker.Client(new Worker('retention_phase_inference_worker.js'),onProgress);
    try{
      const initialized=await client.request('init',{row},20*60*1000);
      const model=new root.QQT.ORTTransformerModel({meta:initialized.meta},null);
      model.backend=initialized.backend;model.backendReason=initialized.backendReason;
      model.parityMaxError=initialized.parityMaxError;model.greedy=true;model.inferEvery=1;
      model.forward=(obs,state)=>client.request('infer',{obs:Float32Array.from(obs),state:Float32Array.from(state)});
      model.forward2=async(o0,s0,o1,s1)=>[await model.forward(o0,s0),await model.forward(o1,s1)];
      model.close=()=>client.close();model.inferenceTimes=client.times;
      return model;
    }catch(error){client.close();throw error;}
  }
  const api={load};root.QQTPhaseModelLoader=api;
  if(typeof module==='object'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
