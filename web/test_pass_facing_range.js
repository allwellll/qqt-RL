'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const Q=require('./sim-bun-native-v4-facing'),Old=require('./sim-bun-native-v3-items'),V=require('./visual_renderer');
const front=81;
function scene(Api=Q){const s=new Api.Sim(77);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','cover','crate','fuse','pushable','fieldItem'])s[k].fill(0);s.pos.set([5.5,4.5,10.5,10.5]);s.spawnProtection.fill(0);s.invuln.fill(100);s.spdG[0]=80/120;s.lastMoveDir[0]=Q.MOVE_RIGHT;s.fuse[front]=30;s.owner[front]=1;s.bombsCap[0]=5;return s;}
function charge(s,time=550){s.frameStep(0,Q.MOVE_IDLE,.016);const start=s.nativeMove[0].touchStart;while(s.nativeMove[0].clock<start+time-1e-6)s.frameStep(0,Q.MOVE_IDLE,Math.min(.016,(start+time-s.nativeMove[0].clock)/1000));}
const place=s=>s.step([[Q.MOVE_IDLE,1,0,1],[Q.MOVE_IDLE,0,0,1]]);
for(const x of [5,4.5,3.5]){
 const s=scene();s.pos[1]=x;const before=Array.from(s.pos);let collisionCalls=0;const segment=s._nativeSegmentCollision,collisions=s._nativeCollisions;
 s._nativeSegmentCollision=s._nativeCollisions=()=>{collisionCalls++;throw Error('idle facing must not probe movement collision');};
 charge(s);assert.equal(collisionCalls,0);assert.deepEqual(Array.from(s.pos),before);assert.equal(s.nativeMove[0].touchCell,front);assert(V.bubblePassageHints(s,0)[0].ready);assert(s._nativePassReady(0));
 s._nativeSegmentCollision=segment;s._nativeCollisions=collisions;assert(place(s).placed[0]);assert(s.nativeMove[0].passActive);assert.equal(s.nativeMove[0].passCell,front);
 for(let i=0;i<40;i++)s.frameStep(0,Q.MOVE_RIGHT,.05);assert(s.pos[1]>=7,'half/full/two-cell gap grants real target traversal');
}
{
 const old=scene(Old);charge(old);assert(!old.nativeMove[0].touchValid,'reproduce old v3 failure at one-cell visible gap');
}
for(const [direction,position,target] of [[0,[7.5,6.5],81],[1,[3.5,6.5],81],[2,[5.5,8.5],81],[3,[5.5,4.5],81]]){
 const s=scene();s.pos[0]=position[0];s.pos[1]=position[1];s.lastMoveDir[0]=direction;assert.equal(s._nativeFacingCell(0),target);charge(s);assert(s._nativePassReady(0));
}
{
 const s=scene();s.fuse[front+1]=30;assert.equal(s._nativeFacingCell(0),front,'nearest bubble wins');s.fuse[front]=0;assert.equal(s._nativeFacingCell(0),front+1);s.pos[1]=3.49;assert.equal(s._nativeFacingCell(0),-1,'beyond 3 centre-to-centre cells excluded');s.pos[1]=3.5;s.fuse[front]=30;assert.equal(s._nativeFacingCell(0),front,'exact 3 cells included');
 for(const key of ['wall','brick','pushable','cover']){const t=scene();t[key][80]=1;assert.equal(t._nativeFacingCell(0),-1,key+' occludes vision');charge(t);assert(!t.nativeMove[0].touchValid);}
 const diagonal=scene();diagonal.fuse[front]=0;diagonal.fuse[front+15]=30;assert.equal(diagonal._nativeFacingCell(0),-1);diagonal.lastMoveDir[0]=Q.MOVE_LEFT;diagonal.fuse[front]=30;assert.equal(diagonal._nativeFacingCell(0),-1,'behind bubble excluded');
}
{
 const s=scene();charge(s,200);const start=s.nativeMove[0].touchStart;s.lastMoveDir[0]=Q.MOVE_UP;s.frameStep(0,Q.MOVE_IDLE,.1);assert(!V.bubblePassageHints(s,0)[0].contact);s.lastMoveDir[0]=Q.MOVE_RIGHT;s.frameStep(0,Q.MOVE_IDLE,.1);assert.equal(s.nativeMove[0].touchStart,start);s.fuse[78]=30;s.lastMoveDir[0]=Q.MOVE_LEFT;s.frameStep(0,Q.MOVE_IDLE,.016);assert.equal(s.nativeMove[0].touchCell,78);assert.equal(s.nativeMove[0].touchStart,s.nativeMove[0].clock);
}
for(const ms of [500,600,650]){const s=scene();charge(s,ms);assert(!V.bubblePassageHints(s,0)[0]?.ready);place(s);assert(!s.nativeMove[0].passActive);}
{
 const s=scene();charge(s);s.lastMoveDir[0]=Q.MOVE_LEFT;assert(!s._nativePassReady(0));assert(!V.bubblePassageHints(s,0)[0].ready);place(s);assert(!s.nativeMove[0].passActive);
 const block=scene();charge(block);block.wall[80]=1;assert(!block._nativePassReady(0));assert(!V.bubblePassageHints(block,0)[0].ready);place(block);assert(!block.nativeMove[0].passActive);
}
{
 const s=scene();for(let i=0;i<6;i++)s.step([[4,0],[4,0]]);assert(s._nativePassReady(0));assert(place(s).placed[0]);assert(s.nativeMove[0].passActive);for(let i=0;i<14;i++)s.step([[3,0],[4,0]]);assert(s.pos[1]>6,'JS models/bots share range rule and collision permission');
 const snap=s.snapshotReplay(),r=scene();r.restoreReplay(snap);assert.deepEqual(r.snapshotReplay(),snap);
}
{
 const s=scene();charge(s);place(s);s.fuse[79]=30;const st=s.nativeMove[0];assert.equal(s._nativePassBlocked(new Uint8Array(195).fill(1),0,3)[front],0);assert.equal(s._nativePassBlocked(new Uint8Array(195).fill(1),0,2)[79],1,'grant is target-only');
 s.wall[front+1]=1;for(let i=0;i<16;i++)s.frameStep(0,Q.MOVE_RIGHT,.05);assert(s.pos[1]<6,'target far side wall still blocks traversal');
}
{
 for(const key of ['_applyItemSupply','_dropDeathInventory','_releaseDeathDrops','_dropCellAvailable','_nativeAirdropStep'])assert.equal(Q.Sim.prototype[key].toString(),Old.Sim.prototype[key].toString(),'item rule unchanged: '+key);
 const level=require('./assets/maps/levels.json').find(x=>x.qqt_id===806);for(const map of [level,require('./tactical_arena').buildTacticalArena(level)]){
  const a=new Q.Sim(7),b=new Old.Sim(7);for(const s of [a,b])s.reset(map,{nativeItems:true,nativeTrap:true});assert.deepEqual(a.itemSupply,b.itemSupply);assert.deepEqual(Array.from(a.crate),Array.from(b.crate));assert.deepEqual(Array.from(a.crateType),Array.from(b.crateType));assert.deepEqual([a.bombsMax,a.blastMax,a.speedMax],[10,8,2.4]);
 }
 assert.equal(crypto.createHash('sha256').update(fs.readFileSync('web/sim-bun-native-v3-items.js')).digest('hex'),'24c5bcc0f12088a11f24bc0f581d4b3cfd25094148a144afa573bf7af88a7c40');
}
console.log('Facing range PASS: separated idle charge without collision, all axes, nearest/occluded/range/back rules, real traversal, shared ring readiness, model clocks, bounded target permission, item preservation and frozen v3');
