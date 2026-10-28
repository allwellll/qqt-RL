'use strict';
const assert=require('node:assert/strict');
const bridge=require('../scripts/je09_web_bridge.js'),previous=require('../scripts/je08_web_diagnostics.js');
const QQT=require('../web/sim.js'),authority=require('../scripts/eval_web_hunter.js');
const s=new QQT.Sim(10);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
s.brick.fill(0);s.crate.fill(0);s.pos=[5.5,4.5,5.5,6.5];s.bunSpawnPos=[[5.5,4.5],[5.5,6.5]];
s.hp=[1,1];s.invuln=[0,0];s.initialHp=1;s.maxSteps=600;
s.bunCarried=[-1,1];s.bunStored=[[1,0],[0,0]];s.trapped=[0,1];
const tracker=new bridge.Tracker(0);tracker.trapCause[1]=[true,false];
function tick(move){const before=previous.snapshot(s),info=s.step([[move,0,0],[4,0,0]]);tracker.observe(s,before,info);}
tick(4);assert.equal(tracker.result().eligible_drop,true);assert.equal(tracker.result().first_pickup,false);
for(let i=0;i<20&&s.bunCarried[0]<0;i++)tick(3);
assert.equal(tracker.result().first_pickup,true);assert.equal(tracker.result().pickup_return,false);
for(let i=0;i<15;i++)tick(2);
assert.equal(tracker.result().pickup_return,true);assert.equal(tracker.result().progress25,true);
assert.equal(tracker.result().capture,false);
const summary=bridge.summarize([tracker.result(),{first_pickup:false}]);
assert.equal(summary.games,2);assert.equal(summary.first_pickup,1);
assert.equal(bridge.manifest.assist,false);
assert.notEqual(bridge.manifest.seed,bridge.manifest.holdout_seed);
const reachable=bridge.setup('reachable_near',123,0,null,1200),goal=[];
for(let r=reachable.bunBases[0][0];r<reachable.bunBases[0][0]+3;r++)
  for(let c=reachable.bunBases[0][1];c<reachable.bunBases[0][1]+3;c++)goal.push(r*QQT.W+c);
const legal=previous.distance(Array.from(reachable.wall).map((x,i)=>x||reachable.brick[i]),Array(QQT.H*QQT.W).fill(0),goal);
const cage=reachable.bunBases[1];
assert.ok(legal[(cage[0]+1)*QQT.W+cage[1]+1]<32767);
assert.equal(reachable.bunCarried[0],-1);
console.log('JE09 continuous native drop/pickup/switch/fixed-denominator tests passed');
