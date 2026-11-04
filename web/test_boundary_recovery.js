'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),crypto=require('crypto');
const Q=require('./sim-bun-native-v2'),R=require('./boundary-recovery-v1');
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const idle=[4,0,0,0,-1,-1,-1],prob={move_probability:[.15,.1,.37,.1,.48]};
function scene(p=1,mv=3,corner=false){const s=new Q.Sim(42);s.reset(level,{nativeItems:true,nativeTrap:true,bananaSlideSpeedPx:480});for(const k of ['wall','brick','crate','fuse','pushable','fieldItem'])s[k].fill(0);s.pos.set([3.5,3.5,9.5,9.5]);s.pos.set((corner?[[1.5,14.5],[11.5,14.5],[12.5,1.5],[12.5,13.5]]:[[1.5,5.5],[11.5,5.5],[5.5,1.5],[5.5,13.5]])[mv],p*2);s.invuln.fill(0);s.spawnProtection.fill(0);const i=Math.floor(s.pos[p*2])*15+Math.floor(s.pos[p*2+1]);s.fieldItem[i]=1;s.fieldArmed[i]=1;s.fieldOwner[i]=1-p;const a=[idle.slice(),idle.slice()];a[p][0]=mv;s.step(a);assert.equal(s.movementStatus[p],2);return s;}
function tick(s,p,a){const actions=[idle.slice(),idle.slice()];actions[p]=a;s.step(actions);}
function arm(s,p,g){for(let n=0;n<5&&s.movementStatus[p]===2;n++){g.apply(s,p,idle,prob);tick(s,p,idle);}assert.equal(s.movementStatus[p],0);g.apply(s,p,idle,prob);}
(async()=>{
const evidence=[];
for(let p=0;p<2;p++)for(let mv=0;mv<4;mv++){
 const s=scene(p,mv),g=new R.Guard(),initial=s.snapshotReplay(null),actions=[],trace=[];
 for(let i=0;i<65;i++){const before=s.snapshotReplay(null),a=g.apply(s,p,idle,prob);assert.deepEqual(s.snapshotReplay(null),before,'guard must not mutate physics');assert.deepEqual(g.apply(s,p,idle,prob),a);trace.push({tick:s.t,raw:idle,effective:a,pos:Array.from(s.pos.slice(p*2,p*2+2))});const pair=[idle.slice(),idle.slice()];pair[p]=a;actions.push(pair);s.step(pair);}
 assert(g.events.length>0&&g.events.length<=4);assert(trace.find(x=>x.raw[0]!==x.effective[0]).tick-trace.find(x=>x.tick>initial.t&&x.pos.some((v,i)=>v!==trace[0].pos[i])).tick>=8);
 const replay=new Q.Sim(42);replay.reset(level,{nativeItems:true,nativeTrap:true});replay.restoreReplay(initial);for(const pair of actions)replay.step(pair);assert.deepEqual(replay.snapshotReplay(null),s.snapshotReplay(null));
 assert(g.events.every(e=>e.effective_action[0]===[1,0,3,2][mv]));evidence.push({seat:p,direction:mv,events:g.events,final:Array.from(s.pos.slice(p*2,p*2+2)),effective_replay_exact:true});
}
// With no observed banana, even boundary idle is never changed.
{const s=scene(),g=new R.Guard();while(s.movementStatus[1]===2)tick(s,1,idle);for(let i=0;i<70;i++){assert.strictEqual(g.apply(s,1,idle,prob),idle);tick(s,1,idle);}assert.equal(g.events.length,0);}
// Normal legal moves and bomb/item decisions are never overridden.
for(const raw of [[2,0,0],[4,1,0],[4,0,1]]){const s=scene(),g=new R.Guard();arm(s,1,g);for(let i=0;i<20;i++){s.t++;assert.strictEqual(g.apply(s,1,raw,prob),raw);}assert.equal(g.events.length,0);}
const negatives=[];
for(const type of ['disabled','dead','trapped','protection','danger','body_danger','bubble','banana','wall','brick','crate','fast','timeout','reset','non_native','team']){
 const s=scene(),g=new R.Guard();arm(s,1,g);const r=Math.floor(s.pos[2]),c=Math.floor(s.pos[3]),cell=r*15+c;
 if(type==='dead')s.alive[1]=false;if(type==='trapped')s.trapped[1]=20;if(type==='protection')s.spawnProtection[1]=20;
 if(type==='danger')s.blastLinger[cell]=2;
 if(type==='body_danger'){s.pos[2]=5.1;s.blastLinger[(r-1)*15+c]=2;}
 if(type==='bubble')s.fuse[cell-1]=30;if(type==='banana')s.fieldItem[cell]=1;if(type==='wall')s.wall[cell]=1;if(type==='brick')s.brick[cell]=1;if(type==='crate')s.crate[cell]=1;
 if(type==='fast')s.spdG[1]=10;if(type==='timeout')s.t+=51;if(type==='reset')s._gen++;if(type==='non_native')s.nativeTrap=false;if(type==='team')s.nPlayers=3;
 for(let i=0;i<25;i++){s.t++;assert.strictEqual(g.apply(s,1,idle,prob,type!=='disabled'),idle,type);}assert.equal(g.events.length,0,type);negatives.push(type);
}
// Hard bound remains valid even if physics does not advance the requested move.
{const s=scene(),g=new R.Guard();arm(s,1,g);for(let i=0;i<80;i++){s.t++;g.apply(s,1,idle,prob);}assert.equal(g.events.length,4);}
// Normal seeded gameplay is byte-for-byte identical when recovery is attached.
const normal=[];for(let seed=1;seed<=12;seed++){
 const a=new Q.Sim(seed),b=new Q.Sim(seed);for(const s of [a,b])s.reset(level,{nativeItems:true,nativeTrap:true});const g=new R.Guard(),rng=Q.mulberry32(seed);for(let t=0;t<80;t++){const pair=[[Math.floor(rng()*5),Number(rng()<.08),0],[Math.floor(rng()*5),Number(rng()<.08),0]],effective=[g.apply(b,0,pair[0],prob),g.apply(b,1,pair[1],prob)];assert.deepEqual(effective,pair);a.step(pair);b.step(effective);assert.deepEqual(a.snapshotReplay(null),b.snapshotReplay(null));}assert.equal(g.events.length,0);normal.push(seed);
}
// The wrapper keeps the raw decision object and supports immediate opt-out.
{
 const s=scene(),decision={...prob},model={lastDecision:decision,act:async()=>idle};let enabled=true;
 assert.strictEqual(R.attach(model,{enabled:()=>enabled}),model);assert.strictEqual(R.attach(model),model);
 for(let i=0;i<16;i++){const action=await model.act(s,1);assert.strictEqual(model.lastDecision,decision);assert.deepEqual(model.lastDeploymentDecision.original_action,idle);tick(s,1,action);}
 assert(model.boundaryRecovery.events.length>0);enabled=false;s.t++;
 assert.strictEqual(await model.act(s,1),idle);assert.equal(model.lastDeploymentDecision.intervened,false);
}
console.log('PASS: 8 boundaries, exact effective replay, physics unchanged by guard, opt-out, raw decision retained, 17 negative cases, bounded overrides, 960 unchanged normal ticks');
})().catch(e=>{console.error(e);process.exitCode=1;});
