'use strict';
const assert=require('node:assert/strict'),P=require('./review_protocol'),V=require('./review_presentation');
const d=require('./review-replays/'+require('./review-catalog.json').replays.find(r=>r.diagnostic).file),level=require('../scripts/eval_web_hunter').loadLevel();
function pair(tick){const before=new P.Player(d,level),after=new P.Player(d,level);before.seek(tick);after.seek(tick+1);return [before.sim,after.sim];}
for(const [pid,tick]of [[0,2],[0,20],[0,36],[0,60],[1,104]]){
  const [before,after]=pair(tick),a=d.actions[tick],intents=[a[0],a[2]],authority=JSON.stringify(P.state(before));
  const frames=[0,.13,.27,.41,.59,.73,.87,1].map(alpha=>V.motion(before,after,alpha,true,d.events,intents));
  const axis=Math.abs(after.pos[pid*2]-before.pos[pid*2])>1e-8?pid*2:pid*2+1,values=frames.map(f=>f.visualPos[axis]*60),direction=Math.sign(values.at(-1)-values[0]);
  assert(direction&&values.every((v,i)=>!i||(v-values[i-1])*direction>0),`RED: safe ${pid===0?'model':'opponent'} movement must not freeze then snap @${tick}`);
  assert.equal(JSON.stringify(P.state(before)),authority);assert.deepEqual(V.motion(before,after,0,false,d.events,intents).visualPos,Array.from(before.pos));
}
for(const [pid,tick,reason]of [[0,9,'collision_side_steering'],[0,17,'collision_side_steering'],[0,79,'state_change:bunCarried'],[0,115,'collision_side_steering'],[0,1199,'terminal'],[1,0,'death_or_respawn'],[1,100,'death_or_respawn'],[1,114,'collision_clamp']]){
  const [before,after]=pair(tick),a=d.actions[tick],intents=[a[0],a[2]];
  assert.equal(V.interpolationReason(before,after,pid,d.events,intents[pid]),reason);
  for(const alpha of [.1,.5,.9])assert.deepEqual(V.motion(before,after,alpha,true,d.events,intents).visualPos.slice(pid*2,pid*2+2),Array.from(before.pos.slice(pid*2,pid*2+2)));
}
for(const tick of [2,20,36,60]){const [before,after]=pair(tick),motion=V.motion(before,after,.5,true,d.events,[d.actions[tick][0],d.actions[tick][2]]),origin=Math.floor(before.pos[0])*15+Math.floor(before.pos[1]);assert.equal(motion.sourceTick,tick);assert.equal(motion.nextTick,tick+1);assert.equal(before.fuse[origin],0);assert(after.fuse[origin]>0);assert.equal(after.owner[origin],0);}
console.log('Motion regression GREEN: real g32 bomb/movement separation for both players, monotonic coordinates, authoritative state unchanged, discrete boundary snaps');
