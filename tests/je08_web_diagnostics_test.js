'use strict';
const assert=require('node:assert/strict'),fs=require('fs');
const p=require('../scripts/je08_web_diagnostics.js');
const native=require('../web/sim.js'),authority=require('../scripts/eval_web_hunter.js');
const w=6,h=5,wall=Array(w*h).fill(0),brick=Array(w*h).fill(0);
for(let r=0;r<4;r++)wall[r*w+2]=1;brick[4*w+2]=1;
const d=p.distance(wall,brick,[4],h,w);
assert.ok(d[1*w+1]<d[1]);assert.equal(d[2],32767);
assert.equal(d[1],p.distance(wall,Array(w*h).fill(0),[4],h,w)[1]+3);
function fixture(team){
  const s=new native.Sim(10);s.reset(authority.loadLevel(),{nativeTrap:true,nativeItems:true});
  s.brick.fill(0);s.crate.fill(0);s.pos=[5.5,4.5,5.5,6.5];s.bunSpawnPos=[[5.5,4.5],[5.5,6.5]];
  s.hp=[1,1];s.invuln=[0,0];s.initialHp=1;s.maxSteps=600;
  s.bunCarried=[-1,1];s.bunStored=[[1,0],[0,0]];s.trapped=[0,1];
  const t=new p.Tracker(team);t.trapCause[1]=[true,false];
  let b=p.snapshot(s),info=s.step([[4,0,0],[4,0,0]]);t.observe(s,b,info);
  assert.equal(t.nonce[1],1);assert.equal(t.credit[1],0);assert.equal(t.eligibleDrops,1);
  for(let i=0;i<20&&s.bunCarried[0]<0;i++){b=p.snapshot(s);info=s.step([[3,0,0],[4,0,0]]);t.observe(s,b,info);}
  assert.equal(s.bunCarried[0],1);assert.equal(t.result().first_pickup,true);assert.ok(t.result().pickup_ticks>0);
  assert.equal(s.bunStored[0][1],0);assert.equal(t.result().capture,false);
  return t.result();
}
assert.deepEqual(fixture(0),fixture(0));
assert.equal(p.manifest.assist,false);
console.log('JE08 web topology/drop/pickup diagnostic tests passed');
