'use strict';
// Isolated CPU diagnostic. No production decoder, teacher or simulator edit.
const fs=require('node:fs'),crypto=require('node:crypto');
function probabilities(logits,mask,tau=1){
 if(!(tau>0&&Number.isFinite(tau)))throw Error('positive finite temperature required');
 const max=Math.max(...logits.filter((_,i)=>mask[i]));
 const w=Array.from(logits,(v,i)=>mask[i]?Math.exp((v-max)/tau):0),sum=w.reduce((a,b)=>a+b,0);
 if(!(sum>0))throw Error('empty legal action set');return w.map(v=>v/sum);
}
function abilityRank(logits,mask){
 const order=Array.from(logits,(_,i)=>i).filter(i=>mask[i]).sort((a,b)=>logits[b]-logits[a]||a-b);
 const p=probabilities(logits,mask);return {rank:mask[1]?order.indexOf(1)+1:null,top:order[0],
  probability:p[1],gap:order.length>1?p[order[0]]-p[order[1]]:1,probabilities:p};
}
function install(model,{tau,record=false}){
 if(!Number.isFinite(tau)||tau<0)throw Error('invalid temperature');
 const sample=model._sampleMasked.bind(model);
 model._sampleMasked=function(logits,mask,rng){
  // tau=0 and tau=1 delegate exactly, preserving RNG calls/tie rules.
  if(tau===0||tau===1)return sample(logits,mask,rng);
  return sample(Array.from(logits,v=>v/tau),mask,rng);
 };
 let last=null;const forward=model.forward.bind(model);
 model.forward=async(...args)=>{last=await forward(...args);return last;};
 const rows=new Map(),examples=[];const act=model.act.bind(model);
 model.act=async(s,pid,rng)=>{
  const result=await act(s,pid,rng);
  if(!record)return result;
  const mask=model._abilityMask(s,s.legalMask(),pid),rank=abilityRank(Array.from(last.bomb),mask);
  const phase=!s.alive[pid]||s.trapped[pid]>0?'uncontrolled':s.bunCarried[pid]>=0?'carrier':
    !s.alive[1-pid]||s.trapped[1-pid]>0?'window':'combat';
  const key=JSON.stringify([s.__btIdentity?.seed,pid,phase]);
  if(!rows.has(key))rows.set(key,{seed:s.__btIdentity?.seed,seat:pid,phase,ticks:0,bomb_legal_ticks:0,
    rank_counts:[0,0,0],bomb_top1_gap_sum:0,bomb_top1_count:0,bomb_probability_sum:0,
    top_ability_counts:[0,0,0],movement_counts:[0,0,0,0,0]});
  const r=rows.get(key);r.ticks++;r.top_ability_counts[rank.top]++;r.movement_counts[result[0]]++;
  if(rank.rank!==null){r.bomb_legal_ticks++;r.rank_counts[rank.rank-1]++;r.bomb_probability_sum+=rank.probability;
    if(rank.rank===1){r.bomb_top1_count++;r.bomb_top1_gap_sum+=rank.gap;}}
  // Fixed first three distinct seed/phase examples, never outcome selected.
  if(rank.rank===2&&examples.length<3&&!examples.some(e=>e.seed===s.__btIdentity?.seed&&e.phase===phase)){
   const state=require('../web/review_protocol').state(s),hash=crypto.createHash('sha256').update(JSON.stringify(state)).digest('hex');
   examples.push({seed:s.__btIdentity?.seed,seat:pid,tick:s.t,phase,ability_rank:rank.rank,
    ability_logits:Array.from(last.bomb),legal:Array.from(mask),probabilities:rank.probabilities,
    top_ability:rank.top,top_ability_name:['none','bomb','held_item'][rank.top],
    movement_action:result[0],movement_name:['up','down','left','right','stay'][result[0]],
    movement_logits:Array.from(last.move),joint_action:result,state,state_sha256:hash});
  }
  return result;
 };
 return ()=>({tau,record,rows:[...rows.values()],examples,
  denominator:'all learner decision ticks, bomb-legal decision ticks separately; rank excludes masked actions; equal-logit ties use lower index',
  heads:'ability none/bomb/held_item is independent of movement; none never means stay'});
}
module.exports={probabilities,abilityRank,install};

if(require.main===module){
 const inference=require('./je05_web_hunter'),original=inference.loadModel;
 const tau=Number(process.env.ITER2_TAU),record=process.env.ITER2_RANK==='1';let collect;
 inference.loadModel=async(...args)=>{const loaded=await original(...args);collect=install(loaded.model,{tau,record});return loaded;};
 process.on('exit',code=>{
  if(code===0&&collect){const file=process.argv[4]+'.actions.json';fs.writeFileSync(file+'.tmp',JSON.stringify(collect()));fs.renameSync(file+'.tmp',file);}
 });
 require('./bomb_threat_eval');
}
