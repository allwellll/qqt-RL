'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const Q=require('./sim-bun-native-v2-face'),Old=require('./sim-bun-native-v2'),V=require('./visual_renderer');
const cell=80,front=81;
function scene(Api=Q){const s=new Api.Sim(77);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','crate','fuse','pushable'])s[k].fill(0);s.pos.set([5.5,5.5,10.5,10.5]);s.spawnProtection.fill(0);s.invuln.fill(0);s.spdG[0]=80/120;s.lastMoveDir[0]=Q.MOVE_RIGHT;s.fuse[front]=30;s.owner[front]=1;return s;}
const idle=s=>s.frameStep(0,Q.MOVE_IDLE,.02);
function charge(s,time=550){if(!s.nativeMove?.[0]?.touchValid)idle(s);const st=s.nativeMove[0],target=st.touchStart+time;while(st.clock<target-1e-6)s.frameStep(0,Q.MOVE_IDLE,Math.min(.02,(target-st.clock)/1000));}
const place=s=>s.step([[Q.MOVE_IDLE,1,0,1],[Q.MOVE_IDLE,0,0,1]]);
{
 const s=scene();idle(s);assert.equal(s.nativeMove[0].touchCell,front);assert.equal(V.bubblePassageHints(s,0)[0].progress,0);
 charge(s);assert(V.bubblePassageHints(s,0)[0].ready);assert(place(s).placed[0]);assert(s.nativeMove[0].passActive);
 for(let i=0;i<8;i++)s.frameStep(0,Q.MOVE_RIGHT,.1);assert(s.pos[1]>7,'idle charge then real placement/traversal');
}
{
 const s=scene();charge(s);s.lastMoveDir[0]=Q.MOVE_LEFT;assert(place(s).placed[0]);assert(!s.nativeMove[0].passActive,'back-facing placement cannot grant pass, even within freshness grace');
}
{
 const s=scene();charge(s,200);const start=s.nativeMove[0].touchStart;s.lastMoveDir[0]=Q.MOVE_UP;charge(s,400);assert.equal(s.nativeMove[0].touchStart,start);assert(!V.bubblePassageHints(s,0)[0].contact);s.lastMoveDir[0]=Q.MOVE_RIGHT;idle(s);assert.equal(s.nativeMove[0].touchStart,start);assert.equal(s.nativeMove[0].touchLast,s.nativeMove[0].clock);
 s.fuse[cell-1]=30;s.lastMoveDir[0]=Q.MOVE_LEFT;idle(s);assert.equal(s.nativeMove[0].touchCell,cell-1);assert.equal(s.nativeMove[0].touchStart,s.nativeMove[0].clock);
}
for(const time of [500,600,650]){const s=scene();charge(s,time);place(s);assert(!s.nativeMove[0].passActive,'strict original charge boundaries');}
{
 const s=scene();s.pos[0]=5.99;s.fuse[front]=0;s.fuse[front+15]=30;idle(s);assert(!s.nativeMove[0].touchValid,'diagonal corner is not the primary row');s.fuse[front]=30;idle(s);charge(s,200);const start=s.nativeMove[0].touchStart;idle(s);assert.equal(s.nativeMove[0].touchCell,front);assert.equal(s.nativeMove[0].touchStart,start,'split corners keep primary bubble');
 s.pos[1]=5.4;assert.equal(s._nativeFacingCell(0),-1,'beyond 3px leading-edge gap is not nearby');s.pos[1]=5.475;assert.equal(s._nativeFacingCell(0),front,'3px gap boundary is included');
}
{
 const s=scene();for(let i=0;i<6;i++)s.step([[4,0],[4,0]]);assert.equal(s.nativeMove[0].clock-s.nativeMove[0].touchStart,575);const out=s.step([[4,1],[4,0]]);assert(out.placed[0]&&s.nativeMove[0].passActive,'logical model/bot clocks share charge and placement');assert(s.legalMask().mm[0][3]);for(let i=0;i<3;i++)s.step([[3,0],[4,0]]);assert(s.pos[1]>6,'logical model/bot collision permits traversal');
 s.nativeMove[0].remainder=.375;const snapshot=s.snapshotReplay();const restored=scene();restored.restoreReplay(snapshot);assert.deepEqual(restored.nativeMove,s.nativeMove,'versioned replay preserves charge and permissions');
}
{
 const baseline='593b50e8278b5e544132f86dc4011c4dc06d0de1166c8ff8e751d9ac4f553154';assert.equal(crypto.createHash('sha256').update(fs.readFileSync('web/sim-bun-native-v2.js')).digest('hex'),baseline,'old engine byte-for-byte frozen');
 const R=require('./review_protocol');assert.equal(R.simulatorFile(baseline),'sim-bun-native-v2.js');
 const a=scene(Old),b=scene(R.simulator({simulator_sha256:baseline}));for(let i=0;i<120;i++){const m=i<30?3:i<60?4:i<90?2:0;for(const s of [a,b]){s.frameStep(0,m,.02);if(i%5===0)s.step([[m,Number(i===25),0,1],[4,0,0,1]]);}assert.deepEqual(a.snapshotReplay(),b.snapshotReplay());}
}
{
 const a=scene(Old),b=scene();a.fuse.fill(0);b.fuse.fill(0);
 const physical=sim=>{const frame=sim.snapshotReplay();delete frame.nativeFacing;delete frame.ruleset_version;return frame;};
 for(let tick=0;tick<160;tick++){const actions=[[tick%5,0,0],[(tick*3)%5,0,0]];assert.deepEqual(b.step(actions),a.step(actions));assert.deepEqual(physical(b),physical(a),'unrelated physics remains identical without passage');assert.deepEqual(b.legalMask(),a.legalMask());}
}
{
 const s=scene();charge(s);place(s);s.wall[front+1]=1;const x=s.pos[1];for(let i=0;i<3;i++)s.step([[3,0,0],[4,0,0]]);assert(s.pos[1]<6,'pass cannot enter bubble when its far side is blocked');
}
console.log('Facing pass PASS: idle facing/placement/traversal, wrong facing, resume/transfer, charge boundaries, corner/nearby geometry, policy clocks/masks, snapshot restore and frozen old engine');
