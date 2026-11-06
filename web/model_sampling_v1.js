'use strict';
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.QQTModelSampling=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 const history=new WeakMap();
 function check(ok,message){if(!ok)throw Error(message);}
 function profile(row={}){
  const p=row.inference||{greedy:true,temperature:1,seed:0x515154};
  check(typeof p.greedy==='boolean'&&Number.isFinite(p.temperature)&&p.temperature>0,'Invalid sampling configuration');
  check(Number.isInteger(p.seed)&&p.seed>=0&&p.seed<=0xffffffff,'Invalid sampling seed');
  return Object.freeze({...p});
 }
 function scale(v,t,greedy){check(Number.isFinite(t)&&t>0,'Invalid temperature');if(greedy||t===1)return v;
  return {...v,move:Float64Array.from(v.move,x=>x/t),bomb:Float64Array.from(v.bomb,x=>x/t)};}
 function watch(sim){
  if(history.has(sim))return history.get(sim);
  const h={bombs:new Map(),last_placement:[-1000,-1000],batch:[0,0]};history.set(sim,h);
  const step=sim.step;
  sim.step=function(...args){const before=Int16Array.from(this.fuse),r=step.apply(this,args);
   for(const key of h.bombs.keys())if(this.fuse[key]<=0)h.bombs.delete(key);
   for(let p=0;p<2;p++)if(r.placed[p])for(let i=0;i<195;i++)if(this.fuse[i]>0&&this.owner[i]===p&&before[i]===0){
    if(this.t-h.last_placement[p]>=8)h.batch[p]++;h.last_placement[p]=this.t;h.bombs.set(i,{batch:h.batch[p],tick:this.t,owner:p});}
   return r;};return h;
 }
 function inputs(s,p){
  const obs=new Float32Array(25*195);obs.set(s.encodeObsJAX(p,24));
  const h=s.__r3||watch(s),live=[];for(let i=0;i<195;i++)if(s.fuse[i]>0&&s.owner[i]===p)live.push(i);
  const batches=live.map(i=>h.bombs.get(i)?.batch??-1),valid=batches.filter(x=>x>=0),minimum=valid.length?Math.min(...valid):0;
  for(let k=0;k<live.length;k++)if(batches[k]>=0)obs[24*195+live[k]]=(batches[k]-minimum+1)/8;
  const state=new Float32Array(37),base=s.encodeStateJAX(p);base[9]+=s.trapped[p]/60;base[10]+=s.trapped[1-p]/60;state.set(base);
  let k=24;state[k++]=Math.min(1,s.itemSlots[p].reduce((a,b)=>a+b.count,0)/50);
  for(const type of [1,2])state[k++]=Number(s.itemSlots[1-p].some(x=>x.item===type&&x.count>0));
  for(const q of [p,1-p])for(let type=0;type<4;type++)state[k++]=Number(s.movementStatus[q]===type);
  for(const q of [p,1-p])state[k++]=s.movementStatusTicks[q]/100;
  check(k===37,'R3 encoder ABI');return {obs,state};
 }
 function attach(model,row={}){
  const p=profile(row),forward=model.forward.bind(model);model.greedy=p.greedy;model.temperature=p.temperature;model.samplingProfile=p;
  model.forward=(...args)=>{const v=forward(...args);return v&&typeof v.then==='function'?v.then(x=>scale(x,model.temperature,model.greedy)):scale(v,model.temperature,model.greedy);};
  const meta=model.meta||{};
  if(meta.schema_version==='five-item-slots-v1'){
   check(meta.state_dim===52&&meta.ability_actions===7,'Unsupported five-slot schema');
  } else if(meta.r3_input_encoding){
   check(meta.r3_input_encoding==='r3-items-route-chain-visible-v1'&&meta.state_dim===37&&meta.ability_actions===3&&JSON.stringify(meta.obs_shape)==='[25,13,15]','Unsupported R3 model schema');
   model.act=async(s,seat,rng)=>{const x=inputs(s,seat),mask=s.legalMask(),v=await model.forward(x.obs,x.state),am=model._abilityMask(s,mask,seat);
    check([...v.move,...v.bomb].every(Number.isFinite),'Nonfinite logits');
    const move=model._sampleMasked(v.move,mask.mm[seat],rng),ability=model._sampleMasked(v.bomb,am,rng);
    model.lastDecision={move,ability,move_mask:Array.from(mask.mm[seat]),ability_mask:Array.from(am)};
    return model._decodeAbility(move,ability);};
  } else check((meta.state_dim||24)<=24,'Unknown model state schema');
  return model;
 }
 return {profile,scale,watch,inputs,attach};
});
