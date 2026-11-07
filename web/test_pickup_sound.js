'use strict';
const assert=require('node:assert/strict'),Q=require('./sim-bun-native-v2-face'),S=require('./sound');
function scene(pid=0){const s=new Q.Sim(7);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','crate','fuse','pushable'])s[k].fill(0);s.pos.set([5.5,5.5,9.5,9.5]);s.spawnProtection.fill(0);s.invuln.fill(100);s.trapped.fill(0);s.isBun=false;const cell=pid===0?80:144;s.crate[cell]=1;s.crateType[cell]=Q.CRATE_BANANA;s.crateCount[cell]=1;return {s,cell};}
for(const pid of [0,1]){
 const {s,cell}=scene(pid),before=S.snapshot(s),info=s.step([[4,0,0,1],[4,0,0,1]]);
 assert.equal(s.crate[cell],0);assert.equal(s.itemSlots[pid][0].count,1);
 assert.deepEqual(S.detectEvents(before,s,info,pid).filter(x=>x==='pickup'),['pickup']);
 assert(!S.detectEvents(before,s,info,1-pid).includes('pickup'),'opponent pickup stays silent');
 const after=S.snapshot(s);assert(!S.detectEvents(after,s,s.step([[4,0,0,1],[4,0,0,1]]),pid).includes('pickup'),'next tick cannot repeat');
 assert.equal(S.detectEvents(before,s,{...info,itemReleased:[true,true]},pid).filter(x=>x==='pickup').length,1,'same-tick release and pickup share one sound');
}
{
 const {s,cell}=scene();s.itemSlots[0]=Array.from({length:Q.ITEM_SLOT_COUNT},()=>({item:Q.ITEM_SLOW_GLUE,count:1}));
 const before=S.snapshot(s),info=s.step([[4,0,0,1],[4,0,0,1]]);assert.equal(s.crate[cell],1);assert(!S.detectEvents(before,s,info,0).includes('pickup'),'full inventory rejected pickup is silent');
 s.fuse[cell]=1;s.owner[cell]=1;s.bombBlast[cell]=1;const pre=S.snapshot(s),blast=s.step([[4,0,0,1],[4,0,0,1]]);assert(!S.detectEvents(pre,s,blast,0).includes('pickup'),'blast removal is not pickup');
}
(async()=>{let starts=0,resumes=0,gain;class Context{constructor(){this.state='suspended';this.destination={};Context.instance=this;}async decodeAudioData(){return {};}async resume(){resumes++;this.state='running';}createBufferSource(){return {connect(){return this;},start(){starts++;}};}createGain(){return gain={gain:{},connect(){return this;}};}}
 const player=S.createPlayer({AudioContext:Context,fetch:async()=>({arrayBuffer:async()=>new ArrayBuffer(1)})});await player.load();player.play('pickup');assert.equal(starts,0,'locked context does not queue stale sounds');player.unlock();await Promise.resolve();assert.equal(resumes,1);player.play('pickup');assert.equal(starts,1);assert.equal(gain.gain.value,.6);player.setEnabled(false);player.play('pickup');assert.equal(starts,1);Context.instance.state='interrupted';player.unlock();await Promise.resolve();assert.equal(resumes,2);assert.equal(player.enabled,false);player.play('pickup');assert.equal(starts,1);player.setEnabled(true);player.play('pickup');assert.equal(starts,2);console.log('Pickup sound PASS: real collections by both seats, no repeat/full-inventory/blast false positives, mute/gain and gesture resume');})().catch(e=>{console.error(e);process.exitCode=1;});
