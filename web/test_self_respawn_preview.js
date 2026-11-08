'use strict';
const assert = require('node:assert/strict');
const Q = require('./sim-bun-native-v4-facing');
const preview = require('./self_respawn_preview');
const visual = require('./visual_renderer');
const level = require('./assets/maps/levels.json').find(x => x.qqt_id === 806);
function game(seed=73,killMode=true) {
  const s = new Q.Sim(seed); s.reset(level,{nativeItems:true,nativeTrap:true,killMode});
  for(const key of ['wall','brick','cover','pushable','crate','fieldItem','fuse','blastLinger','brickLinger'])s[key].fill(0);
  s.pendingCrateType.fill(-1);s.bunLoose.fill(0);s.airdropFalls=[];
  return preview.attach(s);
}
function cell(point) { return Math.floor(point[0])*15+Math.floor(point[1]); }
const a=game(),b=game();a._killPlayer(0);b._killPlayer(0);
assert.deepEqual(a.killRespawnPos,b.killRespawnPos,'same match seed yields same selected point');
assert.equal(a.killRespawn[0],30);assert.deepEqual(visual.respawnMarkers(a,0).map(m=>[m.row,m.col]),[a.killRespawnPos[0]]);
assert.deepEqual(visual.respawnMarkers(a,1),[]);assert.deepEqual(visual.respawnMarkers(a,-1),[]);
const initial=a.killRespawnPos[0].slice();
for(let i=0;i<29;i++)a._killRespawnStep();
assert.deepEqual(a.killRespawnPos[0],initial,'stable safe marker does not resample each tick');assert(!a.alive[0]);
a._killRespawnStep();assert(a.alive[0]);assert.deepEqual(Array.from(a.pos.slice(0,2)),initial);assert.deepEqual(visual.respawnMarkers(a,0),[]);assert.equal(a.invuln[0],10);
for(const threat of ['blast','occupied','ray']) {
  const s=game();s._killPlayer(0);const old=s.killRespawnPos[0].slice(),c=cell(old);
  if(threat==='blast')s.blastLinger[c]=3;
  if(threat==='occupied')s.pos.set(old,2);
  if(threat==='ray'){
    // Full open board: an adjacent bubble whose native ray covers the target.
    const bomb=c%15>0?c-1:c+1;s.fuse[bomb]=10;s.bombBlast[bomb]=2;
  }
  s._killRespawnStep();assert.notDeepEqual(s.killRespawnPos[0],old,threat+' reroutes preview');
  assert(s._dropCellAvailable(cell(s.killRespawnPos[0])));
  const final=s.killRespawnPos[0].slice();s.killRespawn[0]=1;s._killRespawnStep();assert.deepEqual(Array.from(s.pos.slice(0,2)),final);
}
const blocked=game();blocked._killPlayer(0);blocked.wall.fill(1);blocked.killRespawn[0]=1;blocked._killRespawnStep();
assert(!blocked.alive[0]);assert.equal(blocked.killRespawn[0],1);assert.deepEqual(visual.respawnMarkers(blocked,0),[],'no safe cell hides marker and postpones respawn');
blocked.wall[80]=0;blocked._killRespawnStep();assert(blocked.alive[0]);assert.equal(cell(Array.from(blocked.pos.slice(0,2))),80);
const obs=game();obs._killPlayer(0);const before=Array.from(obs.encodeObs()),state=Array.from(obs.encodeStateJAX(0,'bun24-trap-timer-v2'));obs.killRespawnPos[0]=[1.5,1.5];assert.deepEqual(Array.from(obs.encodeObs()),before);assert.deepEqual(Array.from(obs.encodeStateJAX(0,'bun24-trap-timer-v2')),state,'future spawn is not fed to model');
for(const encoder of [require('./r3_model_inputs').inputs,require('./five-item-slots-model-v1').inputs]){
  for(const p of [0,1]){const original=encoder(obs,p);obs.killRespawnPos[0]=obs.killRespawnPos[0][0]>2?[1.5,1.5]:[8.5,8.5];const changed=encoder(obs,p);assert.deepEqual(changed,original,'published-model input ABI excludes preview');}
}
const legacy=new Q.Sim(73);legacy.reset(level,{nativeItems:true,nativeTrap:true,killMode:true});legacy._killPlayer(0);assert(!legacy.selfRespawnPreview);assert.deepEqual(visual.respawnMarkers(legacy,0),[],'unmodified frozen simulation cannot opt into preview');
const bun=game(73,false);bun._killPlayer(0);bun.lastDied.fill(false);const target=bun.bunSpawnPos[0].slice();bun.bunRespawn[0]=1;bun._bunRespawnStep();assert(bun.alive[0]);assert.deepEqual(Array.from(bun.pos.slice(0,2)),target);
console.log('Self-only live respawn: seeded selection, exact arrival, rays/blasts/occupancy reroute, shortage, unchanged model observations and frozen Review PASS');
