'use strict';
const assert=require('node:assert/strict'),C=require('./controls'),Q=require('./sim-bun-native-v4-facing');
// Regression: old recent-stroke input turned down after 4px jitter at (130,104),
// and turned left when returning to its original centre. Radial input must not.
{
 const s=C.createFloatingStick({radius:48});s.start(100,100,0);s.move(130,100,10);
 assert.equal(s.move(130,104,20).direction,'ArrowRight');
 assert.equal(s.move(100,100,30).direction,null);
 assert.equal(s.move(109,100,40).direction,'ArrowRight');
 assert.equal(s.move(108,100,50).direction,null,'18% radial deadzone');
 s.move(200,100,60);assert.equal(s.move(200,100,70).center.x,152,'centre follows overflow, keeps radius');
 assert.equal(s.move(140,100,80).direction,'ArrowLeft','opposite radial input responds in the same event');
 assert.equal(s.end(90).direction,null);assert.equal(s.move(300,300,100).direction,null);
 assert.equal(s.start(20,20,110).direction,null);assert.equal(s.reset(120).habit,null);
}
for(const samples of [1,4,16,120])for(const [dx,dy,name] of [[1,0,'ArrowRight'],[-1,0,'ArrowLeft'],[0,1,'ArrowDown'],[0,-1,'ArrowUp']]){
 const s=C.createFloatingStick();s.start(0,0,0);
 for(let i=1;i<=samples;i++)s.move(dx*32*i/samples,dy*32*i/samples,160*i/samples);
 assert.equal(s.move(dx*32,dy*32,180).direction,name,'slow/fast sample-rate invariant radial heading');
}
{
 const s=C.createFloatingStick();s.start(0,0,0);s.move(30,0,1);
 for(let i=0;i<20;i++)for(const degrees of [43,47,44,46]){
  const angle=degrees*Math.PI/180,out=s.move(Math.cos(angle)*30,Math.sin(angle)*30,2+i);
  assert.equal(out.direction,'ArrowRight','45 degree noise stays in 58 degree hysteresis');assert.equal(out.secondary,'ArrowDown');
 }
 assert.equal(s.move(10,30,30).direction,'ArrowDown');
 const out=s.move(-30,0,31);assert.equal(out.direction,'ArrowLeft');assert.equal(out.secondary,null);
}
{
 const t=C.createTouchSteering(),s=(direction,secondary,changedAt)=>({direction,secondary,changedAt});
 assert.equal(t.select(s('ArrowRight',null,0),0,()=>true),'ArrowRight');
 assert.equal(t.select(s('ArrowRight','ArrowUp',10),10,mv=>mv===3),'ArrowRight','early perpendicular turn continues along corridor');
 assert.equal(t.select(s('ArrowRight','ArrowUp',10),100,()=>true),'ArrowUp','turn wins as soon as junction opens');
 assert.equal(t.select(s('ArrowRight','ArrowUp',10),110,()=>true),'ArrowUp','held diagonal does not alternate directions');
 t.clear();t.select(s('ArrowRight',null,0),0,()=>true);
 assert.equal(t.select(s('ArrowRight','ArrowUp',10),10,mv=>mv===3),'ArrowRight');
 assert.equal(t.select(s('ArrowRight','ArrowUp',10),171,()=>true),'ArrowRight','expired diagonal turn is not resurrected at the next junction');
 t.clear();t.select(s('ArrowRight',null,0),0,()=>true);
 assert.equal(t.select(s('ArrowUp',null,10),10,mv=>mv===3),'ArrowRight','cardinal early turn is buffered');
 assert.equal(t.select(s('ArrowUp',null,10),171,()=>false),'ArrowUp','expired buffer cannot keep moving on stale old heading');
 assert.equal(t.select(null,172,()=>true),null);assert.equal(t.select(s('ArrowDown',null,180),180,()=>true),'ArrowDown','release clears buffer');
 t.select(s('ArrowDown','ArrowLeft',190),190,mv=>mv===1);
 t.clear();assert.equal(t.select(s('ArrowRight',null,200),200,()=>true),'ArrowRight','cancel/blur reset has no queued turn');
 const blocked=C.createTouchSteering();assert.equal(blocked.select(s('ArrowRight','ArrowUp',0),0,mv=>mv===0),'ArrowUp','blocked main route uses secondary');
}
function scene(){const sim=new Q.Sim(17);sim.reset('open',{nativeItems:true,nativeTrap:true,killMode:true});
 for(const name of ['wall','brick','fuse','cover','crate','fieldItem','blastLinger','pushable'])sim[name].fill(0);
 sim.pos.set([6.5,6.2,11.5,12.5]);sim.nativeMove=null;return sim;}
{
 const sim=scene();for(let c=0;c<15;c++)if(c!==6)sim.wall[5*15+c]=1;
 const t=C.createTouchSteering();t.select({direction:'ArrowRight',changedAt:0},0,()=>true);
 const state={direction:'ArrowRight',secondary:'ArrowUp',changedAt:10};let turned=false;
 for(let now=10;now<=160;now+=10){const chosen=t.select(state,now,mv=>C.nativeDirectionReachable(sim,0,mv));sim.frameStep(0,C.MOVEMENT_DIRECTIONS[chosen],.01);if(chosen==='ArrowUp'&&sim.pos[0]<6.5){turned=true;break;}}
 assert(turned,'real native corridor allows buffered turn before expiry');assert(sim.pos[0]<6.5,'actually moved upwards');
 assert.equal(sim.wall[5*15+5],1,'turn did not remove the corner wall');
}
{
 const sim=scene();sim.pos.set([6.5,6.5,11.5,12.5]);sim.fuse[6*15+7]=25;
 sim.frameStep(0,Q.MOVE_IDLE,.02);sim.nativeMove[0].wallExit={from:1,to:2,move:3,cross:260};
 const before=JSON.stringify(sim);for(let i=0;i<20;i++)for(let mv=0;mv<4;mv++)C.nativeDirectionReachable(sim,0,mv);
 assert.equal(JSON.stringify(sim),before,'queries do not mutate clocks, touch/pass, wall exit, position, inventory or RNG state');
 sim.nativeMove[0].wallExit=null;sim.pos[1]=6.5;
 assert(!C.nativeDirectionReachable(sim,0,Q.MOVE_RIGHT),'bubble remains solid without actual pass permission');
 sim.wall[5*15+6]=1;assert(!C.nativeDirectionReachable(sim,0,Q.MOVE_UP));
 sim.movementStatus[0]=2;sim.slideDir[0]=Q.MOVE_LEFT;sim.frameStep(0,Q.MOVE_RIGHT,.025);
 assert(sim.pos[1]<6.5,'forced banana direction remains owned by physics');
}
console.log('Radial mobile stick PASS: slow/fast samples, deadzone, jitter, reverse, diagonal buffer, native corridor, read-only collisions, bubble and banana');
