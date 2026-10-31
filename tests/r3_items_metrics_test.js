'use strict';
const assert=require('node:assert/strict');
const {field,route,install,exportMetrics}=require('../scripts/r3_items_metrics');
const Q=require('../web/sim-bun-native-v2');
const s={wall:new Uint8Array(195),brick:new Uint8Array(195),pushable:new Uint8Array(195),bunBases:[[1,1],[9,11]]};
assert.equal(field(s,[0]).distance[194],26);
for(let r=0;r<13;r++)s.brick[r*15+7]=1;
assert.equal(route(s).bfs_distance,null);assert.equal(route(s).minimum_bricks,1);
assert.equal(route(s).weighted_distance,22);
const levels=require('../web/assets/maps/levels.json'),level=levels.find(x=>x.qqt_id===806);
// Zero-intervention parity: identical fixed actions, RNG state and all public
// physical state across an instrumented and an uninstrumented full game.
const baseline=new Q.Sim(2026208100),actions=[];baseline.reset(level,{nativeItems:true,nativeTrap:true});
for(let t=0;t<160;t++)actions.push([[t%5,Number(t%13===0)],[(t+2)%5,Number(t%17===0)]]);
for(const a of actions)baseline.step(a);
const expected=baseline.snapshotReplay();
install(Q);
const measured=new Q.Sim(2026208100);measured.reset(level,{nativeItems:true,nativeTrap:true});measured.__btIdentity={seed:2026208100,seat:0};measured.__bt={kills:[]};
for(const a of actions)measured.step(a);
assert.deepEqual(measured.snapshotReplay(),expected);
const m=exportMetrics(measured);assert.equal(m.live_hist[0].length,m.alive_ticks[0]);
assert(m.zero_bomb_ticks[0]<=m.alive_ticks[0]);
console.log(JSON.stringify({status:'PASS',fixed_actions:160,passive_state_parity:true,weighted_route:22}));
