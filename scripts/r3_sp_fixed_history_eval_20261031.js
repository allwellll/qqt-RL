'use strict';
// Evaluation-only policy opponent around the existing production JS evaluator.
const assert=require('node:assert/strict');
function transform(source){
  function once(from,to){assert.equal(source.split(from).length,2,`fixed-history source anchor: ${from}`);source=source.replace(from,to);}
  once('const loaded=await inference.loadModel(modelFile)',
    "const fixedOpponent=await inference.loadModel(process.env.R3_SP_FIXED_OPPONENT),loaded=await inference.loadModel(modelFile)");
  once("hunter=authority.createHunter('hard')",
    "hunter={adapter:{reset(){},async act(obs,pid,rng){const a=await fixedOpponent.model.act(obs.metadata.sim,pid,rng);return {move:a[0],ability:a[2]?2:a[1]?1:0};}}}");
  once('bot=hunter.adapter.act(', 'bot=await hunter.adapter.act(');
  return source;
}
function main(){
  assert(process.env.R3_SP_FIXED_OPPONENT,'explicit frozen opponent model required');
  assert.equal(process.env.BUN_V2_TIER,'hard');
  const Module=require('node:module'),path=require('node:path');
  const original=Module.prototype._compile,route=path.resolve('scripts/dagger_rl_route.js');
  let patched=0;
  Module.prototype._compile=function(source,filename){
    if(path.resolve(filename)===route){source=transform(source);patched++;}
    return original.call(this,source,filename);
  };
  try{require('./r3_items_candidate_eval');assert.equal(patched,1);}
  finally{Module.prototype._compile=original;}
}
if(require.main===module)main();
module.exports={transform};
