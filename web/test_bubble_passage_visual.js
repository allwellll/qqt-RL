'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('./sim-bun-native-v2'),V=require('./visual_renderer');
function scene(){const s=new Q.Sim(77);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','crate','fuse','pushable'])s[k].fill(0);s.pos.set([5.5,5.5,10.5,10.5]);s.spawnProtection.fill(0);s.invuln.fill(0);s.spdG[0]=80/120;return s;}
const cell=5*15+5,front=cell+1;
function frontHint(s,p=0){return V.bubblePassageHints(s,p).find(x=>x.cell===front);}
{
 const s=scene();s.fuse[cell]=30;s.owner[cell]=0;
 const before=JSON.stringify(s),hint=V.bubblePassageHints(s,0).find(x=>x.cell===cell);
 assert(hint&&hint.kind==='departure'&&hint.directions.includes(Q.MOVE_RIGHT));
 assert.equal(JSON.stringify(s),before,'drawing must not initialize/alter native state');
 for(let t=0;t<16;t++)s.frameStep(0,Q.MOVE_RIGHT,.02);
 assert(s.pos[1]>6,'leave actual own bubble center cell');
 const x=s.pos[1];for(let t=0;t<20;t++)s.frameStep(0,Q.MOVE_LEFT,.02);
 assert.equal(s.pos[1],x,'leave then re-enter remains blocked');
 assert(!V.bubblePassageHints(s,0).some(x=>x.cell===cell),'no stale departure permission');
}
function activated(){const s=scene();s.fuse[front]=30;s.owner[front]=1;
 assert(!frontHint(s));for(let t=0;t<28;t++)s.frameStep(0,Q.MOVE_RIGHT,.02);
 const before=s.pos[1],info=s.step([[Q.MOVE_IDLE,1,0,1],[Q.MOVE_IDLE,0,0,1]]);
 assert(info.placed[0]&&s.nativeMove[0].passActive,'actual placement activates native pass');assert.equal(s.pos[1],before);
 return s;
}
{
 const s=activated(),state=JSON.stringify(s),hint=frontHint(s);
 assert(hint&&hint.kind==='timed'&&hint.directions.includes(Q.MOVE_RIGHT));
 assert.equal(V.bubblePassageHints(s,1).length,0,'other player does not inherit permission');
 for(let t=0;t<100;t++)V.bubblePassageHints(s,0);
 assert.equal(JSON.stringify(s),state,'queries must not charge, expire, move or mutate replay state');
 s.frameStep(0,Q.MOVE_RIGHT,.1);assert(s.pos[1]>5.5,'actual frame movement enters bubble');
}
for(const obstacle of ['wall','brick','fuse']){
 const s=activated();s[obstacle][front+1]=obstacle==='fuse'?30:1;
 assert(!frontHint(s),'blocked beyond bubble rejects '+obstacle);
 const x=s.pos[1];s.frameStep(0,Q.MOVE_RIGHT,.025);assert.equal(s.pos[1],x);
}
{
 const s=activated(),st=s.nativeMove[0];st.clock=st.passStart+st.passDuration;
 const before=JSON.stringify(st);assert(!frontHint(s),'exact native clock expiry removes timed arrow');assert.equal(JSON.stringify(st),before);
 s.frameStep(0,Q.MOVE_IDLE,0);assert.equal(st.passActive,false);
}
for(const pid of [-1,undefined,2])assert.deepEqual(V.bubblePassageHints(activated(),pid),[],'no implicit spectator permission');
for(const state of ['dead','trapped','done']){const s=activated();if(state==='dead')s.alive[0]=false;else if(state==='trapped')s.trapped[0]=3;else s.done=true;assert.deepEqual(V.bubblePassageHints(s,0),[]);}
{
 const s=scene();s.fuse[front]=30;s.owner[front]=0;assert(!frontHint(s),'ownership alone never grants passage');
 const before=JSON.stringify(s);for(let t=0;t<100;t++)V.bubblePassageHints(s,0);assert.equal(JSON.stringify(s),before);
}
assert(fs.readFileSync('web/index.html','utf8').includes('id="bubble-passage-legend"'));
console.log('Bubble passage visual PASS: real activation, own departure/re-entry, per-player permissions, exact expiry, blocked beyond, spectator/dead/trap and render immutability');
