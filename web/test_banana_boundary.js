'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const Current=require('./sim-bun-native-v2'),Base=require('./sim'),Old=require('./review-sim/bun-native-v2-20261104'),P=require('./review_protocol');
const level=require('./assets/maps/levels.json').find(x=>x.qqt_id===806),b='runs/banana_physics_20261105';
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
assert.equal(sha('web/review-sim/bun-native-v2-20261104.js'),'f6e346f3022640556efc2a988eb3b9b8e90abd5776af6ca1b4b1b0ee2acd6abc');
assert.equal(sha('web/review-sim/stock-legacy-20261028.js'),'7d2047b7db2025608b38afec0f8283625b6f98a3d09ac49693f2a0a44d1d485b');
assert.equal(P.simulator({simulator_sha256:sha('web/review-sim/bun-native-v2-20261104.js')}),Old);
assert.equal(P.simulator({simulator_sha256:sha('web/sim-bun-native-v2.js')}),Current);
function scene(Q,p=1){const s=new Q.Sim(42);s.reset(level,{nativeItems:true,nativeTrap:true,bananaSlideSpeedPx:480});for(const k of ['wall','brick','crate','fuse','pushable','fieldItem'])s[k].fill(0);s.pos.set([3.5,3.5,9.5,9.5]);s.invuln.fill(0);s.spawnProtection.fill(0);return s;}
const rows=[],parity=[];
for(const Q of [Current,Base])for(let p=0;p<2;p++)for(const speed of [268,480,600])for(let move=0;move<4;move++)for(const corner of [false,true]){
 const s=scene(Q,p),opposite=[1,0,3,2][move],v=[[-1,0],[1,0],[0,-1],[0,1]][move],axis=move<2?0:1;
 s.bananaSlideSpeedPx=speed;s.pos[2*p]=move===0?1.5:move===1?11.5:corner?12.5:6.5;s.pos[2*p+1]=move===2?1.5:move===3?13.5:corner?14.5:7.5;
 const [r,c]=s.centerCell(p),cell=r*15+c;s.fieldItem[cell]=1;s.fieldArmed[cell]=1;s.fieldOwner[cell]=1-p;
 const initial=s.snapshotReplay(null),actions=[],trace=[];
 for(let t=0;t<20;t++){const a=[[4,0],[4,0]];a[p][0]=t===0?move:opposite;actions.push(a);s.step(a);trace.push({t:s.t,pos:Array.from(s.pos.slice(p*2,p*2+2)),status:s.movementStatus[p]});}
 assert.equal(s.movementStatus[p],0,'forced slide releases '+move);assert(trace.some(x=>x.status===2));
 const final=s.pos[p*2+axis],bound=v[axis]<0?.36:(axis?15:13)-.36;
 assert(Math.abs(final-bound)>.5,'opposite input must leave the boundary');
 const replay=scene(Q,p);replay.restoreReplay(initial);for(const a of actions)replay.step(a);assert.deepEqual(replay.snapshotReplay(null),s.snapshotReplay(null),'effective actions replay exactly');
 rows.push({api:Q===Current?'native':'base',p,speed,move,corner,trace});
}
for(const kind of ['wall','brick','fuse'])for(let move=0;move<4;move++){
 const s=scene(Current),[dy,dx]=Current.DIRS[move];s.pos[2]=6.5;s.pos[3]=7.5;const start=6*15+7,target=(6+dy*2)*15+7+dx*2;s[kind][target]=kind==='fuse'?100:1;
 s.fieldItem[start]=1;s.fieldArmed[start]=1;s.fieldOwner[start]=0;
 for(let i=0;i<6;i++)s.step([[4,0],[move,0]]);
 assert.equal(s.movementStatus[1],0,'obstacle clears forced slide');
 assert.notDeepEqual(s.centerCell(1),[6+dy*2,7+dx*2]);
}
// Native human pixel path must also release, retain 480 speed, and accept reverse.
for(let move=0;move<4;move++){
 const s=scene(Current,0),axis=move<2?0:1;s.pos[0]=move===0?1.5:move===1?11.5:12.5;s.pos[1]=move===2?1.5:move===3?13.5:14.5;
 s._setMovementStatus(0,2,0);s.slideDir[0]=move;for(let i=0;i<100&&s.movementStatus[0]===2;i++)s.frameStep(0,[1,0,3,2][move],.025);
 assert.equal(s.movementStatus[0],0);const before=s.pos[axis];for(let i=0;i<12;i++)s.frameStep(0,[1,0,3,2][move],.025);assert(Math.abs(s.pos[axis]-before)>.1);
}
{
 const s=scene(Current);s._setMovementStatus(1,2,0);s.slideDir[1]=3;s._killPlayer(1);assert.equal(s.movementStatus[1],0);
 s.bunRespawn[1]=1;s.step([[4,0],[4,0]]);assert(s.alive[1]);assert.equal(s.movementStatus[1],0);assert.equal(s.bananaSlideSpeedPx,480);
}
// Controlled no-slide traces remain byte-identical away from world boundaries.
for(let seed=1;seed<=12;seed++){
 const a=scene(Current),z=scene(Old);a.pos.set([5.5,5.5,7.5,9.5]);z.pos.set(a.pos);
 for(let t=0;t<24;t++){const x=[[t%4,0],[(t+2)%4,0]];a.step(x);z.step(x);assert.deepEqual(a.snapshotReplay(null),z.snapshotReplay(null));}
}
// Movement parity fixtures, shared speeds and obstacles. This does not assert
// complete JAX/JS engine equivalence or alter either training environment.
for(const speed of [.7,1.3,2.4,268/120,4,5])for(let move=0;move<4;move++)for(const gap of [0,.0001,.1,.7,1.2])for(const obstacle of [false,true]){
 const s=scene(Current),[dy,dx]=Current.DIRS[move],axis=move<2?0:1,negative=move===0||move===2,pos=[6.5,7.5];pos[axis]=negative?.36+gap:(axis?15:13)-.36-gap;
 const blocked=new Uint8Array(195);if(obstacle){const r=Math.floor(pos[0])+dy,c=Math.floor(pos[1])+dx;if(r>=0&&r<13&&c>=0&&c<15)blocked[r*15+c]=1;}
 parity.push({pos,move,speed,blocked:Array.from(blocked),js:s._tryMove(...pos,move,blocked,.3*speed),old:scene(Old)._tryMove(...pos,move,blocked,.3*speed)});
}
if(fs.existsSync(b)){fs.writeFileSync(b+'/BOUNDARY.json',JSON.stringify({status:'PASS',cases:rows.length,physics_revision:Current.PHYSICS_REVISION,old_native_sha256:sha('web/review-sim/bun-native-v2-20261104.js'),native_sha256:sha('web/sim-bun-native-v2.js'),base_sha256:sha('web/sim.js'),rows},null,2));fs.writeFileSync(b+'/PARITY_FIXTURES.json',JSON.stringify(parity));}
console.log('Banana boundary PASS: '+rows.length+' both-seat/direction/corner/speed cases, reverse input, native frames, wall/brick/bubble, death/respawn, exact action replay and normal movement baseline');
