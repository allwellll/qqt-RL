'use strict';
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.QQTBoundaryRecovery=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 const VERSION='post-banana-boundary-recovery-v1';
 const LIMITS=Object.freeze({idleTicks:10,windowTicks:50,maxOverrides:4,releaseTravel:.5});
 const epsilon=2e-4;
 function onBoundary(sim,p){const y=sim.pos[p*2],x=sim.pos[p*2+1];return y<=.3602||y>=12.6398||x<=.3602||x>=14.6398;}
 class Guard{
  constructor(){this.sim=null;this.generation=null;this.states=new Map();this.events=[];}
  apply(sim,p,raw,decision,enabled=true){
   if(this.sim!==sim||this.generation!==sim._gen){this.sim=sim;this.generation=sim._gen;this.states.clear();this.events=[];}
   let state=this.states.get(p);
   if(!enabled||sim.nPlayers!==2||!sim.nativeTrap||!sim.nativeItems||!sim.alive[p]||sim.trapped[p]>0||sim.done){this.states.delete(p);return raw;}
   if(state?.tick===sim.t)return state.output;
   const position=Array.from(sim.pos.slice(p*2,p*2+2));
   if(!state){state={seenSlide:false,armed:false,prev:position,tick:sim.t-1,idle:0,overrides:0};this.states.set(p,state);}
   if(sim.movementStatus[p]===2){state.seenSlide=true;state.armed=false;state.slideDir=sim.slideDir[p];state.prev=position;state.tick=sim.t;state.output=raw;return raw;}
   if(state.seenSlide){state.seenSlide=false;state.armed=onBoundary(sim,p);state.anchor=position;state.start=sim.t;state.overrides=0;state.idle=0;}
   const moved=Math.abs(position[0]-state.prev[0])+Math.abs(position[1]-state.prev[1]),elapsed=sim.t-state.tick;
   state.prev=position;state.tick=sim.t;state.output=raw;
   if(!state.armed)return raw;
   if(sim.t-state.start>LIMITS.windowTicks||state.overrides>=LIMITS.maxOverrides||Math.abs(position[0]-state.anchor[0])+Math.abs(position[1]-state.anchor[1])>=LIMITS.releaseTravel){state.armed=false;return raw;}
   state.idle=moved<=epsilon&&raw[0]===4&&!raw[1]&&!raw[2]?state.idle+Math.min(1,elapsed):0;
   if(state.idle<LIMITS.idleTicks||!onBoundary(sim,p)||sim.spawnProtection[p]>0)return raw;
   const masks=sim.legalMask().mm[p],blocked=new Uint8Array(195),danger=sim.dangerMap();
   for(let i=0;i<195;i++)blocked[i]=sim.wall[i]||sim.brick[i]||sim.fuse[i]>0?1:0;
   const cell=Math.floor(position[0])*15+Math.floor(position[1]);
   if(danger[cell]>0||sim.blastLinger[cell]>0)return raw;
   const inward=[];
   if(position[0]<=.3602)inward.push(1);if(position[0]>=12.6398)inward.push(0);
   if(position[1]<=.3602)inward.push(3);if(position[1]>=14.6398)inward.push(2);
   const candidates=inward.filter(move=>{
    if(!masks[move])return false;
    const dist=.3*sim.spdG[p]*sim.playerMoveScale(p);
    if(dist>.5)return false;
    const next=sim._steer(...position,move,blocked,dist,p);
    if(Math.abs(next[0]-position[0])+Math.abs(next[1]-position[1])<=epsilon)return false;
    for(let part=0;part<=8;part++){
     const y=position[0]+(next[0]-position[0])*part/8,x=position[1]+(next[1]-position[1])*part/8;
     for(let r=Math.floor(y-.36+1e-6);r<=Math.floor(y+.36-1e-6);r++)for(let c=Math.floor(x-.36+1e-6);c<=Math.floor(x+.36-1e-6);c++){
      const i=r*15+c;
      if(r<0||r>=13||c<0||c>=15||danger[i]>0||sim.blastLinger[i]>0||sim.fuse[i]>0||sim.fieldItem[i]>0||sim.wall[i]||sim.brick[i]||sim.crate[i])return false;
     }
    }return true;
   }).sort((a,b)=>(decision?.move_probability?.[b]||0)-(decision?.move_probability?.[a]||0)||a-b);
   if(!candidates.length)return raw;
   const output=raw.slice();output[0]=candidates[0];state.output=output;state.overrides++;state.idle=0;
   const event={version:VERSION,tick:sim.t,seat:p,original_action:Array.from(raw),effective_action:Array.from(output),position,reason:'idle_after_banana_boundary_release',override:state.overrides};
   this.events.push(event);if(this.events.length>128)this.events.shift();return output;
  }
 }
 function attach(model,options={}){
  if(model.boundaryRecovery)return model;const guard=new Guard(),act=model.act;model.boundaryRecovery=guard;
  model.act=async function(sim,p,rng){const raw=await act.call(this,sim,p,rng),effective=guard.apply(sim,p,raw,this.lastDecision,options.enabled?options.enabled():true);
   this.lastDeploymentDecision={version:VERSION,tick:sim.t,seat:p,original_action:Array.from(raw),effective_action:Array.from(effective),intervened:effective!==raw};return effective;};
  return model;
 }
 return {VERSION,LIMITS,Guard,attach};
});
