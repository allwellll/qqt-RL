'use strict';
const fs=require('fs'),assert=require('node:assert/strict'),crypto=require('crypto'),P=require('../web/review_protocol');
const cases=[['h8-v36-g32-carry_home-2026101785',107,'22f0359884838fff2a5d57c8fcbd12952b2547d44d0042e80f6f5499e1309114'],['h8-v36-g32-open-2026101782',436,'04a572ad739b70acff36cba744817bc185b854f1617f571b30b439e2a17ec5cb']];
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function audit(){return {schema:'qqt.route-target-audit/v1',cases:cases.map(([id,tick,expected])=>{const b=fs.readFileSync('web/review-replays/'+id+'.json');assert.equal(sha(b),expected);const d=JSON.parse(b),p=new P.Player(d,require('./eval_web_hunter').loadLevel()),rows=[];
 for(let t=Math.max(1,tick-30);t<=Math.min(d.actions.length,tick+30);t++){
  p.seek(t-1);const before=P.state(p.sim);p.step();const after=P.state(p.sim),r=d.diagnostic_ticks[t-1];
  rows.push({tick:t,action:d.actions[t-1],before:{pos:before.pos,alive:before.alive,trapped:before.trapped,carried:before.bunCarried,stock:before.bunStored,bases:before.bunBases},after:{pos:after.pos,alive:after.alive,trapped:after.trapped,carried:after.bunCarried,stock:after.bunStored,done:after.done,live_bomb_fuses:after.fuse.filter(v=>v>0)},decision:r.decision,recorded_reward:r.offline_recomputed_current_reward,raw_a1:r.raw_a1,shaped_a1:r.shaped_a1});
 }
 return {id,sha256:expected,tick,source_status:d.meta.reward_status,rows};})};}
if(require.main===module){const out=process.argv[2];fs.writeFileSync(out,JSON.stringify(audit()));console.log('Two immutable replay hashes and +/-30 tick raw evidence verified');}
module.exports={audit};
