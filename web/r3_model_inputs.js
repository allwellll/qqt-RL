'use strict';
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.QQTR3Inputs=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 const installed=new WeakSet();
 function installSim(Q){
  if(installed.has(Q.Sim.prototype))return;installed.add(Q.Sim.prototype);
  const reset=Q.Sim.prototype.reset;Q.Sim.prototype.reset=function(...args){delete this.__r3Visible;return reset.apply(this,args);};
  const step=Q.Sim.prototype.step;
  Q.Sim.prototype.step=function(actions){
   const x=this.__r3Visible||(this.__r3Visible={bombs:new Map(),batch:[0,0],last:[-1000,-1000]}),before=Array.from(this.fuse),info=step.call(this,actions);
   for(const [cell]of x.bombs)if(this.fuse[cell]<=0)x.bombs.delete(cell);
   for(let p=0;p<2;p++)if(info.placed[p])for(let i=0;i<195;i++)if(this.fuse[i]>0&&this.owner[i]===p&&before[i]===0){if(this.t-x.last[p]>=8)x.batch[p]++;x.last[p]=this.t;x.bombs.set(i,{batch:x.batch[p]});}
   return info;
  };
 }
 function inputs(s,p){
  const obs=new Float32Array(25*195);obs.set(s.encodeObsJAX(p,24));const live=[];
  for(let i=0;i<195;i++)if(s.fuse[i]>0&&s.owner[i]===p)live.push(i);
  const ids=live.map(i=>s.__r3Visible?.bombs.get(i)?.batch??-1),valid=ids.filter(v=>v>=0),minimum=valid.length?Math.min(...valid):0;
  for(let k=0;k<live.length;k++)if(ids[k]>=0)obs[24*195+live[k]]=(ids[k]-minimum+1)/8;
  const state=new Float32Array(37);state.set(s.encodeStateJAX(p,'bun24-trap-timer-v2'));const enemy=1-p;let k=24;
  state[k++]=Math.min(1,s.itemSlots[p].reduce((a,b)=>a+b.count,0)/50);
  for(const type of [1,2])state[k++]=Number(s.itemSlots[enemy].some(x=>x.item===type&&x.count>0));
  for(const player of [p,enemy])for(let type=0;type<4;type++)state[k++]=Number(s.movementStatus[player]===type);
  for(const player of [p,enemy])state[k++]=s.movementStatusTicks[player]/100;
  if(k!==37)throw Error('R3 input ABI');return {obs,state};
 }
 function attach(model){
  if(!model.meta.r3_input_encoding)return model;
  if(model.meta.r3_input_encoding!=='r3-items-route-chain-visible-v1'||model.obsShape[0]!==25)throw Error('Unsupported R3 input ABI');
  model._getLaggedInputs=inputs;return model;
 }
 return {inputs,installSim,attach};
});
