'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('../web/sim-bun-native-v2'),H=require('../web/bun_hunter_bot');
const I=require('../web/bun_interceptor_bot');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
function fixture(kind){
 const s=new Q.Sim(2026335000+kind);s.reset(level,{nativeItems:true,nativeTrap:true,playerModes:['old','old']});
 s.wall.fill(1);s.brick.fill(0);s.pushable.fill(0);s.pushBoxes=[];s.pushBoxAt.fill(-1);s.crate.fill(0);s.bunLoose.fill(0);
 for(let c=1;c<=13;c++)s.wall[6*15+c]=0;
 // A real escape alcove outside the blast radius, not an invulnerability cheat.
 for(let r=2;r<=6;r++)s.wall[r*15+8]=0;
 if(kind===1)for(let c=5;c<=9;c++)s.wall[5*15+c]=0;
 if(kind===2){for(let c=2;c<=10;c++)s.wall[8*15+c]=0;for(const c of [2,10])s.wall[7*15+c]=0;}
 s.bunBases=[[6,11],[1,1]];s.bunStored=[[1,0],[0,0]];s.bunCarried=[1,-1];
 s.pos.set([6.5,2.5,5.5,8.5]);s.bunSpawnPos=[[6.5,2.5],[5.5,8.5]];
 s.invuln=[0,0];s.spawnProtection=[0,0];s.blastCap=[2,3];s.bombsCap=[2,2];
 s.maxSteps=300;s.bunInitial=[Infinity,Infinity];s._gen++;return s;
}
const results=[];
for(let scene=0;scene<3;scene++){
 const s=fixture(scene),b=new I.BunInterceptorBot({seed:17});let placed=0,trapped=false,blocked=0;
 let initialPlan;
 for(let t=0;t<110&&!s.done;t++){
  const observation=H.hunterStateFromSim(s),g=b.geometry(observation),f=observation.players[0];
  if(!initialPlan)initialPlan=b.routePlan(observation,g,1);
  const goals=b.baseCells(observation,g,0),obstacles=Uint8Array.from(s.fuse,x=>Number(x>0));
  const d=I.distances(g,goals,obstacles);let move=4,best=d[f.cell];
  for(let a=0;a<4;a++){const n=g.nb[f.cell*4+a];if(n>=0&&!obstacles[n]&&d[n]<best){move=a;best=d[n];}}
  const action=b.act(s,1),r=s.step([[move,0,0],action]);
  placed+=Number(r.placed[1]);trapped ||= s.trapped[0]>0||!s.alive[0];
  if(move===4&&s.alive[0]&&s.trapped[0]===0&&s.bunCarried[0]===1)blocked++;
 }
 assert(initialPlan?.best,`scene${scene}: no interception plan`);
 assert(placed>0,`scene${scene}: no actual bomb placement`);
 assert(trapped||blocked>=4,`scene${scene}: neither sugar/death nor route blockage`);
 results.push({scene,placements:placed,carrier_trapped_or_dead:trapped,blocked_ticks:blocked,common:initialPlan.common.length});
}
assert.deepEqual(I.DIFFICULTIES.hard_intercept,H.DIFFICULTIES.hard);
const s=fixture(0),b=new I.BunInterceptorBot(),state=H.hunterStateFromSim(s),g=b.geometry(state);
const fast=b.routePlan(state,g,1);state.players[1].speed/=2;const slow=b.routePlan(state,g,1);
const sameTarget=slow.candidates.find(x=>x.plant===fast.best.plant);
assert(!sameTarget||sameTarget.arrival>=fast.best.arrival,'glue slowdown must not improve same-target arrival');
state.players[0].movementStatus=2;state.players[0].slideDir=3;state.players[0].postSlideSpeed=.6;
const projected=b.projectedCarrier(g,state.players[0]);assert(projected.cell>state.players[0].cell&&projected.slideTicks>0);
console.log(JSON.stringify({status:'PASS',fixture_only:true,results}));
