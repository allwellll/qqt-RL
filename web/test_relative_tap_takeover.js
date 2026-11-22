'use strict';
const assert=require('node:assert/strict'),C=require('./controls'),Q=require('./sim-bun-native-v4-facing');
function fixture(){
 const held=new Set(),handlers={},captured=new Set(),dial={style:{},getBoundingClientRect:()=>({width:96})};
 const rect={left:0,top:0,width:240,height:240,right:240,bottom:240};
 const zone={dataset:{},querySelector:()=>dial,getBoundingClientRect:()=>rect,
  setPointerCapture:id=>captured.add(id),hasPointerCapture:id=>captured.has(id),releasePointerCapture(id){captured.delete(id);handlers.lostpointercapture({type:'lostpointercapture',pointerId:id});},addEventListener:(name,fn)=>handlers[name]=fn};
 const input=C.bindTouchControls({querySelector:()=>zone,querySelectorAll:()=>[],addEventListener(){}},held,()=>{});
 let time=1;const event=(type,id,x=120,y=120,stamp)=>handlers[type]({type,pointerId:id,clientX:x,clientY:y,timeStamp:stamp??time++,button:0,preventDefault(){}});
 const sim=new Q.Sim(21);sim.reset('open',{nativeItems:true,nativeTrap:true,killMode:true});for(const k of ['wall','brick','fuse','cover','crate','fieldItem','blastLinger','pushable'])sim[k].fill(0);sim.pos.set([6.5,6.5,11.5,12.5]);sim.nativeMove=null;
 const step=()=>{const mv=input.moveFor(sim,0,performance.now());sim.frameStep(0,mv,input.frameSeconds(1/60));return mv;};
 return {event,input,held,zone,dial,sim,step,captured};
}
{
 const f=fixture();f.event('pointerdown',1,155,120);f.step();
 f.event('pointerdown',2,120,85);assert.equal(C.moveForHeld(f.held),0,'RED: second down must take over right hold immediately');
 const y=f.sim.pos[0],x=f.sim.pos[1];assert.equal(f.step(),0);assert(f.sim.pos[0]<y&&f.sim.pos[1]===x,'native player moves up, not only feedback');
 f.event('pointerup',2,120,85);assert.equal(f.step(),4,'owner up stops without restoring old right');
 f.event('pointermove',1,157,121);assert.equal(f.step(),4,'old finger micro-jitter cannot steal back');
 f.event('pointermove',1,155,140);assert.equal(f.step(),1,'old finger deliberately drags to retake with rebased direction');
 f.event('pointerup',1,155,140);assert.equal(f.step(),4);
}
{
 const f=fixture();f.event('pointerdown',1,155,120);f.event('pointermove',1,210,120);f.step();
 const cx=parseFloat(f.dial.style.left),cy=parseFloat(f.dial.style.top);assert.equal(cx,162);
 f.event('pointerdown',2,cx,cy-70);assert.equal(C.moveForHeld(f.held),0,'outside small disc but inside zone: relative up immediately');f.step();
 f.event('lostpointercapture',1);assert.equal(C.moveForHeld(f.held),0,'old lost capture cannot clear new owner');
 f.event('pointermove',1,20,20);assert.equal(C.moveForHeld(f.held),0,'stale old move ignored');
 f.event('pointerup',2);f.step();const shownX=f.dial.style.left,shownY=f.dial.style.top;
 f.event('pointerdown',2,parseFloat(shownX),parseFloat(shownY)+35);assert.equal(f.step(),1,'ID reused, single finger retap relative to visible shifted centre');
 f.event('pointerup',2,0,0,1);assert.equal(C.moveForHeld(f.held),1,'timestamp-stale up cannot clear reused ID');f.input();
}
{
 const f=fixture();f.event('pointerdown',1,155,120);f.step();
 f.event('pointerdown',2,122,122);assert.equal(C.moveForHeld(f.held),3,'centre touch does not steal');
 f.event('pointerdown',3,2,120);assert.equal(C.moveForHeld(f.held),3,'guard does not steal');
 f.event('pointerup',2);f.event('pointerup',3);assert.equal(f.step(),3);f.input();
}
{
 const f=fixture();f.event('pointerdown',1,155,120);f.step();
 for(const [dx,dy,mv]of [[0,-35,0],[0,35,1],[-35,0,2],[35,0,3]]){
  f.event('pointerdown',2,120+dx,120+dy);assert.equal(C.moveForHeld(f.held),mv,'clear relative cardinal overrides old hysteresis');
  const old=Array.from(f.sim.pos);f.event('pointerup',2);assert.equal(f.step(),mv,'down/up before next frame commits one bounded takeover');
  assert(Math.hypot(f.sim.pos[0]-old[0],f.sim.pos[1]-old[1])>0);assert.equal(f.step(),4,'no old-finger revival after short tap');
 }
 f.event('pointerdown',2,120,85);f.event('pointerup',2);f.step();f.event('pointerdown',2,85,120);f.event('pointerup',2);assert.equal(f.step(),2,'separate inter-frame taps both survive');f.event('pointerup',1);assert.equal(f.step(),4);
}
{
 const f=fixture();f.event('pointerdown',1,120,75);f.step();f.event('pointerdown',2,145,95);
 assert.equal(C.moveForHeld(f.held),0,'ambiguous diagonal uses current compatible vertical rather than forced horizontal');
 f.event('pointermove',2,146,96);assert.equal(C.moveForHeld(f.held),0,'small diagonal jitter retains ownership and direction');f.input();
 const g=fixture();g.event('pointerdown',1,155,120);g.step();g.event('pointerdown',2,144,96);
 assert.equal(C.moveForHeld(g.held),0,'clear upward inter-touch displacement breaks a diagonal tie');g.input();
}
{
 const f=fixture();f.event('pointerdown',1,155,120);f.step();f.event('pointerdown',2,120,85);f.step();
 f.event('pointerup',1);assert.equal(f.step(),0,'old up cannot stop active new owner');f.event('pointerup',2);assert.equal(f.step(),4);
 const g=fixture();g.event('pointerdown',1,155,120);g.step();g.event('pointerdown',2,120,85);g.event('pointerup',2);g.event('pointerup',1);assert.equal(g.step(),0,'old up cannot discard pending new short tap');assert.equal(g.step(),4);
}
for(const cleanup of ['pointercancel','lostpointercapture','clear']){
 const f=fixture();f.event('pointerdown',1,155,120);f.step();f.event('pointerdown',2,120,85);
 if(cleanup==='clear')f.input();else f.event(cleanup,2);
 assert.equal(f.step(),4);f.event('pointermove',1,155,150);assert.equal(f.step(),4,'cleanup removes every pointer');
 f.event('pointerup',1);f.event('pointerup',2);assert.equal(f.step(),4,'stale releases cannot queue');
 assert.equal(f.dial.style.left,'','global cleanup restores neutral visible centre');
}
console.log('Relative takeover PASS: second-down native movement, shifted centre, old jitter/explicit retake, short taps, owner release, guard/deadzone, diagonal ties, capture/stale/ID cleanup');
