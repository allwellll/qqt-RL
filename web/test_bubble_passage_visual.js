'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('./sim-bun-native-v2'),V=require('./visual_renderer');
function scene(){const s=new Q.Sim(77);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','crate','fuse','pushable'])s[k].fill(0);s.pos.set([5.5,5.5,10.5,10.5]);s.spawnProtection.fill(0);s.invuln.fill(0);s.spdG[0]=80/120;return s;}
const cell=5*15+5,front=cell+1;
const hints=(s,p=0,move=Q.MOVE_RIGHT)=>V.bubblePassageHints(s,p,move);
function touching(){const s=scene();s.fuse[front]=30;s.owner[front]=1;s.frameStep(0,Q.MOVE_RIGHT,.02);assert(s.nativeMove[0].touchValid);return s;}
function advanceTo(s,charged){const target=s.nativeMove[0].touchStart+charged;while(s.nativeMove[0].clock<target-1e-6)s.frameStep(0,Q.MOVE_RIGHT,Math.min(.02,(target-s.nativeMove[0].clock)/1000));}
function place(s){return s.step([[Q.MOVE_IDLE,1,0,1],[Q.MOVE_IDLE,0,0,1]]);}
function unchanged(s){const before=JSON.stringify(s);for(let i=0;i<1000;i++)hints(s);assert.equal(JSON.stringify(s),before,'1000 queries must preserve entire simulation/replay JSON');}
{
 const s=touching();assert.equal(hints(s).length,1,'one native touchCell, one countdown');
 assert.equal(hints(s)[0].remainingMs,500);let last=501;
 for(const t of [0,8,16,33,71,199,350,499,500]){
  advanceTo(s,t);const h=hints(s)[0];assert.equal(h.kind,'charging');assert.equal(h.chargedMs,t);assert.equal(h.remainingMs,500-t);assert(h.remainingMs<last);last=h.remainingMs;unchanged(s);
 }
 assert.equal(s.t,0,'countdown updates between logic ticks');
 advanceTo(s,500.5);const h=hints(s)[0];assert.equal(h.kind,'window');assert.equal(h.label,'现在放泡');assert.equal(h.remainingMs,99.5);unchanged(s);
 assert(place(s).placed[0]);assert(s.nativeMove[0].passActive,'real native placement grants passage');
 assert.equal(hints(s)[0].kind,'active');assert.equal(hints(s)[0].label,'已获穿泡');unchanged(s);
 for(let i=0;i<8;i++)s.frameStep(0,Q.MOVE_RIGHT,.1);
 assert(s.pos[1]>7,'actual native movement passes all the way through the target bubble');
 assert.equal(hints(s).length,0,'no lingering active hint at real expiry');
}
for(const charged of [500,600,650]){
 const s=touching();advanceTo(s,charged);
 if(charged>=600)assert.deepEqual(hints(s),[],'MAX is excluded and missed window disappears');
 assert(place(s).placed[0]);assert(!s.nativeMove[0].passActive,'boundary/late placement cannot grant passage');
 const x=s.pos[1];s.frameStep(0,Q.MOVE_RIGHT,.02);assert.equal(s.pos[1],x,'no false ability to pass');
}
{
 const s=touching();advanceTo(s,550);assert.equal(hints(s)[0].kind,'window');
 assert.deepEqual(hints(s,0,Q.MOVE_IDLE),[],'release immediately hides countdown within freshness grace');
 assert.deepEqual(hints(s,0,Q.MOVE_LEFT),[],'turn away immediately hides countdown');
 s.frameStep(0,Q.MOVE_IDLE,.1);assert.deepEqual(hints(s),[],'stale contact cannot show a window');unchanged(s);
 // Native physics retains touchStart through idle; the view must not invent a reset.
 const start=s.nativeMove[0].touchStart;s.frameStep(0,Q.MOVE_RIGHT,.02);assert.equal(s.nativeMove[0].touchStart,start);assert.deepEqual(hints(s),[]);
 s.fuse[front]=0;s.fuse[cell-1]=30;
 for(let i=0;i<20&&s.nativeMove[0].touchCell!==cell-1;i++)s.frameStep(0,Q.MOVE_LEFT,.02);
 const h=hints(s,0,Q.MOVE_LEFT)[0];assert(h&&h.cell===cell-1);assert.equal(h.kind,'charging');assert.equal(h.remainingMs,500,'new target resets native charge');
}
{
 const s=touching();advanceTo(s,200);s.fuse[front]=0;assert.deepEqual(hints(s),[],'removed target cannot leave a stale badge');
}
{
 const s=scene();s.fuse[cell]=30;s.owner[cell]=0;
 assert.deepEqual(hints(s),[],'own bubble departure is not an activated pass point');unchanged(s);
 for(let t=0;t<16;t++)s.frameStep(0,Q.MOVE_RIGHT,.02);
 assert(s.pos[1]>6);const x=s.pos[1];for(let t=0;t<20;t++)s.frameStep(0,Q.MOVE_LEFT,.02);assert.equal(s.pos[1],x,'departure/re-entry physics unchanged');
}
for(const state of ['charging','window','active']){
 const s=touching();if(state!=='charging')advanceTo(s,550);if(state==='active')place(s);unchanged(s);
 assert.deepEqual(hints(s,1),[],'another player never inherits the human countdown');
 for(const pid of [-1,undefined,2])assert.deepEqual(V.bubblePassageHints(s,pid,Q.MOVE_RIGHT),[],'spectator/Review has no implicit pid0');
 for(const condition of ['dead','trapped','done']){
  const copy=touching();if(state!=='charging')advanceTo(copy,550);if(state==='active')place(copy);if(condition==='dead')copy.alive[0]=false;else if(condition==='trapped')copy.trapped[0]=3;else copy.done=true;
  assert.deepEqual(hints(copy),[],condition+' hides countdown');
 }
}
{
 const s=scene();assert.deepEqual(hints(s),[]);unchanged(s);
 s.fuse[front]=30;s.owner[front]=0;assert.deepEqual(hints(s),[],'proximity/ownership alone never activates');
 // Leading corners in different cells keep native charge resetting: never show a false window.
 s.pos[0]=5;for(let i=0;i<40;i++){s.frameStep(0,Q.MOVE_RIGHT,.02);assert.deepEqual(hints(s),[]);}
}

for(const move of [Q.MOVE_UP,Q.MOVE_DOWN,Q.MOVE_LEFT,Q.MOVE_RIGHT]){
 const s=scene(),offset=[-15,15,-1,1][move];s.fuse[cell+offset]=30;
 for(let i=0;i<4&&!hints(s,0,move).length;i++)s.frameStep(0,move,.02);
 const h=hints(s,0,move)[0];assert(h&&h.cell===cell+offset,'all four native contact directions render');
}
const html=fs.readFileSync('web/index.html','utf8');assert(html.includes('穿点倒计时')&&html.includes('现在放泡')&&!html.includes('泡上虚线箭头'));
console.log('Pass-point countdown PASS: per-frame charge, strict native boundaries, real placement/traversal, missed/released/changed target, privacy of player view, and 1000-query immutability');
