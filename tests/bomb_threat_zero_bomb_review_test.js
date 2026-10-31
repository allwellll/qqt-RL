'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),P=require('../web/review_protocol');
const A=require('../scripts/eval_web_hunter'),Q=require('../web/sim-stock-window-v1');
(async()=>{
 const level=A.loadLevel(),seed=2026122932,s=new Q.Sim(seed);s.reset(level,{nativeTrap:true,nativeItems:true});
 s.maxSteps=3;s.pos.set([.5,2.5,5.5,3.5]);s.bunSpawnPos=[[.5,2.5],[5.5,3.5]];s._gen++;
 const hash=require('crypto').createHash('sha256').update(fs.readFileSync('web/sim-stock-window-v1.js')).digest('hex');
 const d={schema:P.SCHEMA,meta:{id:'frozen-birth-test',seed,tick_ms:100,max_ticks:3,
  simulator_sha256:hash,checkpoint_sha256:'0'.repeat(64),model_sha256:'0'.repeat(64),opponent_sha256:'0'.repeat(64),map_sha256:await P.hash(level),
  frozen_birth:{schema:'qqt.frozen-birth/v1',cells:[[0,2],[5,3]]},reward_status:'unrecorded'},
  actions:[],rewards:[],events:[],checks:[[0,await P.hash(P.state(s))]]};
 while(!s.done){d.actions.push([4,0,4,0]);s.step([[4,0],[4,0]]);}
 d.checks.push([s.t,await P.hash(P.state(s))]);d.summary={ticks:s.t,done:true,winner:s.winner,reward:[null,null]};
 d.meta.replay_identity=await P.hash(P.identity(d));await new P.Player(d,level).verify();
 const broken=structuredClone(d);broken.meta.frozen_birth.cells[0][0]=-1;
 assert.throws(()=>new P.Player(broken,level),/frozen birth/);
 const legacy=structuredClone(d);delete legacy.meta.frozen_birth;
 assert.notDeepEqual(Array.from(new P.Player(legacy,level).sim.pos),Array.from(s.pos));
 console.log('frozen birth exact Review state/RNG/terminal parity, bounds, legacy isolation GREEN');
})().catch(e=>{console.error(e);process.exitCode=1;});
