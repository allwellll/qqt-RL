'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs');
const Q=require('./sim-bun-native-v2'),V=require('./visual_renderer');
function scene(){const s=new Q.Sim(77);s.reset('open',{nativeItems:true,nativeTrap:true});for(const k of ['wall','brick','crate','fuse','pushable'])s[k].fill(0);s.pos.set([5.5,5.5,10.5,10.5]);s.spawnProtection.fill(0);s.invuln.fill(0);s.spdG[0]=80/120;return s;}
const cell=5*15+5,front=cell+1;
const hints=(s,p=0)=>V.bubblePassageHints(s,p);
function touching(){const s=scene();s.fuse[front]=30;s.owner[front]=1;s.frameStep(0,Q.MOVE_RIGHT,.02);assert(s.nativeMove[0].touchValid);return s;}
function advanceTo(s,charged,move=Q.MOVE_RIGHT){const target=s.nativeMove[0].touchStart+charged;while(s.nativeMove[0].clock<target-1e-6)s.frameStep(0,move,Math.min(.02,(target-s.nativeMove[0].clock)/1000));}
function place(s){return s.step([[Q.MOVE_IDLE,1,0,1],[Q.MOVE_IDLE,0,0,1]]);}
function unchanged(s){const before=JSON.stringify(s);for(let i=0;i<1000;i++)hints(s);assert.equal(JSON.stringify(s),before,'1000 queries preserve entire simulation/replay JSON');}
{
 const s=touching();assert.equal(hints(s).length,1);assert.equal(hints(s)[0].progress,0);
 for(const t of [8,16,33,199,350,499,500]){advanceTo(s,t);const h=hints(s)[0];assert.equal(h.kind,'charging');assert.equal(h.progress,t/550);unchanged(s);}
 assert.equal(s.t,0,'ring updates between 10Hz ticks');
 advanceTo(s,500.5);const h=hints(s)[0];assert.equal(h.kind,'window');assert.equal(h.progress,500.5/550);assert(h.ready&&h.contact);unchanged(s);
 assert(place(s).placed[0]);assert(s.nativeMove[0].passActive,'real native placement grants passage');assert.equal(hints(s)[0].kind,'active');unchanged(s);
 for(let i=0;i<8;i++)s.frameStep(0,Q.MOVE_RIGHT,.1);assert(s.pos[1]>7,'real movement through target bubble');assert.deepEqual(hints(s),[],'no stale active mark');
}
for(const move of [Q.MOVE_IDLE,Q.MOVE_LEFT,Q.MOVE_UP,Q.MOVE_DOWN]){
 const s=touching();advanceTo(s,100);const start=s.nativeMove[0].touchStart;
 advanceTo(s,300,move);assert.equal(s.nativeMove[0].touchStart,start);assert.equal(hints(s)[0].progress,300/550,'release/turn/walk keeps growing');
 advanceTo(s,550,move);const h=hints(s)[0];assert.equal(h.progress,1);assert.equal(h.kind,'window');assert(!h.contact&&!h.ready,'full ring is faded/dashed after freshness expires');unchanged(s);
 assert(place(s).placed[0]);assert(!s.nativeMove[0].passActive,'full but stale ring cannot grant real passage');
}
{
 const s=touching();advanceTo(s,450);advanceTo(s,510,Q.MOVE_IDLE);assert(hints(s)[0].ready,'native freshness grace remains valid through short release');
 advanceTo(s,550,Q.MOVE_IDLE);assert(!hints(s)[0].ready,'exactly 100ms freshness boundary is excluded');
 s.frameStep(0,Q.MOVE_RIGHT,.02);assert(hints(s)[0].ready,'return before MAX restores native readiness without resetting progress');assert(place(s).placed[0]);assert(s.nativeMove[0].passActive);
}
for(const ms of [500,500.5,549,550,575,599]){
 const s=touching();advanceTo(s,ms);const h=hints(s)[0];
 assert.equal(h.progress,Math.min(1,ms/550));assert.equal(h.ready,ms>500);
}
for(const charged of [500,600,650]){
 const s=touching();advanceTo(s,charged);if(charged>=600)assert.deepEqual(hints(s),[],'600ms expiration');
 assert(place(s).placed[0]);assert(!s.nativeMove[0].passActive,'strict native open boundaries');
}
{
 const s=touching();advanceTo(s,200);s.fuse[cell-1]=30;
 for(let i=0;i<20&&s.nativeMove[0].touchCell!==cell-1;i++)s.frameStep(0,Q.MOVE_LEFT,.02);
 const h=hints(s)[0];assert(h&&h.cell===cell-1);assert.equal(h.progress,0,'ring transfers to new native target');assert(!hints(s).some(x=>x.cell===front));unchanged(s);
}
for(const state of ['charging','window','active']){
 const s=touching();if(state!=='charging')advanceTo(s,550);if(state==='active')place(s);unchanged(s);
 assert.deepEqual(hints(s,1),[],'other player does not inherit ring');for(const p of [-1,undefined,2])assert.deepEqual(V.bubblePassageHints(s,p),[],'spectator/Review hidden');
 for(const condition of ['dead','trapped','done','removed']){const copy=touching();if(state!=='charging')advanceTo(copy,550);if(state==='active')place(copy);if(condition==='dead')copy.alive[0]=false;else if(condition==='trapped')copy.trapped[0]=3;else if(condition==='removed')copy.fuse[front]=0;else copy.done=true;assert.deepEqual(hints(copy),[],condition+' hides every phase');unchanged(copy);}
}
{
 const s=scene();assert.deepEqual(hints(s),[]);unchanged(s);s.fuse[cell]=30;s.owner[cell]=0;assert.deepEqual(hints(s),[],'own departure/proximity is not activation');
}
for(const move of [Q.MOVE_UP,Q.MOVE_DOWN,Q.MOVE_LEFT,Q.MOVE_RIGHT]){
 const s=scene(),offset=[-15,15,-1,1][move];s.fuse[cell+offset]=30;for(let i=0;i<4&&!hints(s).length;i++)s.frameStep(0,move,.02);assert.equal(hints(s)[0].cell,cell+offset);
}
const html=fs.readFileSync('web/index.html','utf8');assert(html.includes('圆环')&&html.includes('淡绿虚圈')&&!html.includes('穿点倒计时'));
console.log('Pass ring PASS: native target lifecycle across release/turn/walk, transfer, strict charge/freshness bounds, real placement/traversal, hidden perspectives and 1000-query immutability');
