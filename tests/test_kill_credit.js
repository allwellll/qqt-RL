'use strict';
const assert=require('node:assert/strict');
const Q=require('../web/sim-bun-native-v4-facing');
const idle=[[4,0,0],[4,0,0]];
function scene(killMode=true) {
  const s=new Q.Sim(123);s.reset('open',{nativeItems:true,nativeTrap:true,killMode,teams:[0,1]});
  for(const name of ['wall','brick','crate','fuse','pushable'])s[name].fill(0);
  s.pos.set([3.5,3.5,9.5,9.5]);s.invuln.fill(0);s.spawnProtection.fill(0);
  s.maxSteps=1800;return s;
}
function trap(s,victim,owner) {
  const [r,c]=s.centerCell(victim),cell=r*15+c;
  s.fuse[cell]=1;s.owner[cell]=owner;s.bombBlast[cell]=1;
  const info=s.step(idle);assert(s.trapped[victim]>0);assert(info.physicalDamageSource[victim][owner]);
  return info;
}
let s=scene();trap(s,1,0);
assert.deepEqual(s.killCount,[0,0],'trapping alone is not a kill');
const replay=scene();replay.restoreReplay(s.snapshotReplay());
for(let t=0;t<60;t++) {
  const a=s.step(idle),b=replay.step(idle);
  assert.deepEqual(s.snapshotReplay(),replay.snapshotReplay(),'delayed provenance survives replay');
  if(a.died[1]) {assert(a.creditedKill[0]);assert(b.creditedKill[0]);assert(!a.ownBombDefeat[1]);}
}
assert.deepEqual(s.killCount,[1,0],'timer death credits original attacker once');
assert.equal(s.alive[1],false);
s=scene();trap(s,1,1);s.trapped[1]=1;
let info=s.step(idle);assert(info.ownBombDefeat[1]);assert.deepEqual(s.killCount,[0,0]);
s=scene();trap(s,1,1);s.pos.set([9.5,8.5],0);
info=s.step(idle);assert(info.creditedKill[0],'enemy contact finisher credits a kill');assert(!info.ownBombDefeat[1]);
s=scene();trap(s,1,0);s.trapped[1]=1;s.t=1799;
info=s.step(idle);assert(info.creditedKill[0]);assert(s.done);assert.equal(s.winner,0,'last tick kill decides winner');
s=scene();trap(s,1,0);trap(s,0,1);s.trapped.fill(1);
info=s.step(idle);assert(info.mutualDeath);assert.deepEqual(info.creditedKill,[true,true]);assert.deepEqual(s.killCount,[1,1]);
s=scene(false);trap(s,1,0);s.trapped[1]=1;
info=s.step(idle);assert.deepEqual(info.creditedKill,[false,false],'legacy non-kill attribution unchanged');
console.log('PASS delayed kill credit: real blast, timeout, self defeat, contact, last tick, trade, replay, legacy');
