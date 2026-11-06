'use strict';
// Evaluation-only temperature, applied before both native masked samplers.
const assert=require('node:assert/strict'),base=require('./interceptor_primary_inference_v1');
function scale(v,t,greedy){
 assert(Number.isFinite(t)&&t>0,'temperature must be finite positive');
 if(greedy||t===1)return v;
 return {...v,move:Float64Array.from(v.move,x=>x/t),bomb:Float64Array.from(v.bomb,x=>x/t)};
}
async function loadModel(file){const out=await base.loadModel(file),m=out.model,forward=m.forward.bind(m);
 m.temperature=1;m.forward=async(...args)=>scale(await forward(...args),m.temperature,m.greedy);
 return out;
}
module.exports={loadModel,scale};
