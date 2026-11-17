'use strict';
const assert=require('node:assert/strict'),C=require('./controls'),Q=require('./sim-bun-native-v4-facing');
function fixture(){
 const held=new Set(),handlers={},dial={style:{},getBoundingClientRect:()=>({width:96})};
 const rect={left:10,top:20,width:200,height:240,right:210,bottom:260};
 const zone={dataset:{},querySelector:()=>dial,getBoundingClientRect:()=>rect,setPointerCapture(){},hasPointerCapture:()=>false,addEventListener:(name,fn)=>handlers[name]=fn};
 const input=C.bindTouchControls({querySelector:()=>zone,querySelectorAll:()=>[],addEventListener(){}},held,()=>{});
 const event=(type,x=150,y=140,id=1)=>handlers[type]({type,clientX:x,clientY:y,pointerId:id,button:0,preventDefault(){}});
 const sim=new Q.Sim(9);sim.reset('open',{nativeItems:true,nativeTrap:true,killMode:true});
 for(const name of ['wall','brick','fuse','cover','crate','fieldItem','blastLinger','pushable'])sim[name].fill(0);
 sim.pos.set([6.5,6.5,11.5,12.5]);sim.nativeMove=null;
 return {held,zone,dial,event,input,sim};
}
for(const [x,y,mv]of [[110,100,0],[110,180,1],[70,140,2],[150,140,3]]){
 const {held,event,input,sim,dial}=fixture();event('pointerdown',x,y);
 assert.equal(C.moveForHeld(held),mv,'down chooses the visible sector synchronously');
 assert.equal(input.moveFor(sim,0,performance.now()),mv);sim.frameStep(0,mv,input.frameSeconds(.016));
 event('pointerup',x,y);assert.equal(C.moveForHeld(held),4);assert.equal(input.moveFor(sim,0,performance.now()),4,'sampled holds do not queue release movement');
 assert.equal(dial.style.left,'100px');assert.equal(dial.style.top,'120px','released dial retains the visible reference');
}
{
 const f=fixture();for(const [x,y]of [[110,140],[112,142],[12,140],[110,22],[208,140],[110,258]]){
  f.event('pointerdown',x,y);assert.equal(C.moveForHeld(f.held),4,'centre/edge contacts do not start movement');f.event('pointerup',x,y);assert.equal(f.input.moveFor(f.sim,0,performance.now()),4);f.input();
 }
 f.event('pointerdown',110,140);assert.equal(C.moveForHeld(f.held),4,'centre touch starts a neutral floating drag');
 f.event('pointermove',140,140);assert.equal(C.moveForHeld(f.held),3);f.event('pointermove',200,140);assert.equal(f.dial.style.left,'142px','drag centre follows beyond radius');f.input();
}
{
 const f=fixture();f.event('pointerdown');f.event('pointerup');const t=f.sim.t,x=f.sim.pos[1];
 const mv=f.input.moveFor(f.sim,0,performance.now());assert.equal(mv,3,'same-frame down/up survives as one intent');
 const dt=f.input.frameSeconds(.1);assert.equal(dt,1/60,'queued tap has a fixed bounded duration');f.sim.frameStep(0,mv,dt);
 assert(f.sim.pos[1]>x&&f.sim.pos[1]-x<.15);assert.equal(f.sim.t,t,'no whole-simulator advancement');
 assert.equal(f.input.moveFor(f.sim,0,performance.now()),4,'intent consumed only once');assert.equal(f.input.frameSeconds(.016),.016);
 f.event('pointerdown',150,140);f.event('pointerup',150,140);f.event('pointerdown',70,140);f.event('pointerup',70,140);
 assert.equal(f.input.moveFor(f.sim,0,performance.now()),2,'rapid unsampled opposite taps keep only the latest');
 assert.equal(f.input.moveFor(f.sim,0,performance.now()),4);
 f.event('pointerdown');f.event('pointerup');assert.equal(f.input.moveFor(f.sim,0,performance.now()+101),4,'100ms queue expiry prevents delayed movement');
}
for(const cleanup of ['pointercancel','lostpointercapture','clear']){
 const f=fixture();f.event('pointerdown');if(cleanup==='clear')f.input();else f.event(cleanup);
 assert.equal(f.input.moveFor(f.sim,0,performance.now()),4,'cancel/lost/blur reset never preserve short tap');
 f.event('pointerdown');f.event('pointerup');f.input();assert.equal(f.input.moveFor(f.sim,0,performance.now()),4,'global cleanup drops already queued tap');
}
{
 const f=fixture();f.event('pointerdown');f.input.moveFor(f.sim,0,performance.now());f.event('pointerup');f.event('pointerdown',70,140);
 assert.equal(C.moveForHeld(f.held),2,'explicit opposite sector overrides previous direction immediately');f.event('pointerup',70,140);
 f.held.add('Numpad5');assert.equal(f.input.moveFor(f.sim,0,performance.now()),0,'keyboard wins over a released queued tap');f.held.clear();assert.equal(f.input.moveFor(f.sim,0,performance.now()),4);
}
for(const [dx,dy,key,second]of [[20,20,'ArrowRight','ArrowDown'],[-20,20,'ArrowLeft','ArrowDown'],[20,-20,'ArrowRight','ArrowUp'],[-20,-20,'ArrowLeft','ArrowUp']]){
 const f=fixture();f.event('pointerdown',110+dx,140+dy);
 assert.equal(f.zone.dataset.direction,key,'exact diagonal ties use horizontal consistently');assert.equal(f.zone.dataset.secondary,second);
 f.event('pointermove',110+dx,140+dy+1);assert.equal(f.zone.dataset.direction,key,'boundary micro-jitter keeps the initial direction');f.input();
}
{
 const f=fixture();f.sim.fuse[6*15+7]=25;f.event('pointerdown');const mv=f.input.moveFor(f.sim,0,performance.now());assert.equal(mv,3,'blocked cardinal bubble input is not swallowed');
 f.sim.frameStep(0,mv,f.input.frameSeconds(.016));assert.equal(f.sim.lastMoveDir[0],3);assert(f.sim.nativeMove[0].touchValid,'facing bubble still charges');f.input();
 f.sim.movementStatus[0]=2;f.sim.slideDir[0]=3;f.event('pointerdown',70,140);f.sim.frameStep(0,f.input.moveFor(f.sim,0,performance.now()),f.input.frameSeconds(.016));assert.equal(f.sim.lastMoveDir[0],3,'banana keeps forced direction');
}
console.log('Tap move PASS: synchronous sectors, stable centre/guard, radial drag, diagonal ties, reverse, one-frame tap, latest-only queue, expiry and cleanup, bubble facing and forced banana');
