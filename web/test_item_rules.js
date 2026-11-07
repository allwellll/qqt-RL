'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const Q=require('./sim-bun-native-v3-items'),Old=require('./sim-bun-native-v2-face'),A=require('./tactical_arena');
const level=require('./assets/maps/levels.json').find(x=>x.qqt_id===806),opts={nativeItems:true,nativeTrap:true};
function open(seed=7){const s=new Q.Sim(seed);s.reset('open',opts);for(const k of ['wall','brick','cover','crate','fieldItem','fuse','blastLinger','brickLinger','pushable','bunLoose'])s[k].fill(0);s.pendingCrateType.fill(-1);s.airdropFalls=[];s.alive.fill(false);s._pendingDeathDrops=[];s.graveyard=[];return s;}
function ledger(s){const counts={};const add=(t,n)=>counts[t]=(counts[t]||0)+n;for(let i=0;i<Q.N;i++)if(s.crate[i])add(s.crateType[i],s.crateCount[i]||1);for(const d of s._pendingDeathDrops)add(d.type,d.count);for(const d of s.graveyard)add(d.type,d.count||1);for(const d of s._airdropQueue)add(d.type,d.count||1);for(const d of s.airdropFalls)add(d.type,d.count||1);return counts;}
{
 const s=new Q.Sim(31);s.reset(level,opts);assert.equal(s.itemSupply.added,24);assert.equal(s.itemSupply.initialActual,24);assert.equal(s.itemSupply.baselineBricks,81);assert(Math.abs(s.itemSupply.expectedTotal/s.itemSupply.expectedBaseline-1.5)<.02);assert.deepEqual([s.bombsMax,s.blastMax,s.speedMax],[10,8,2.4]);
 const first=s.snapshotReplay();s._pendingDeathDrops.push({type:0,count:99});s.reset(level,opts);assert.deepEqual(s.snapshotReplay(),first,'reset re-seeds and clears supply/death state');
 const twin=new Q.Sim(31);twin.reset(level,opts);assert.deepEqual(twin.snapshotReplay(),first,'identical seeds/config reproduce exact supply and positions');
 s.reset(A.buildTacticalArena(level),opts);assert.equal(s.itemSupply.expectedBaseline,0);assert.equal(s.itemSupply.added,15);assert(s.itemsEnabled);assert.equal(s.crateRate,0);assert.equal(s.itemSupply.zeroSupplyReference,10);
 const old=new Old.Sim(31);old.reset(A.buildTacticalArena(level),opts);assert.equal(old.crate.reduce((a,b)=>a+b),0);assert(!old.itemsEnabled,'old duel physics remains unchanged');
 const disabled={...level,no_items:true};s.reset(disabled,opts);assert.equal(s.itemSupply.added,0);
}
{
 const s=open();s.alive[0]=true;s.pos[0]=5.5;s.pos[1]=5.5;s.bombsCap[0]=s.loBombs[0]+3;s.blastCap[0]=s.loBlast[0]+2;s.spdG[0]=s.loSpeed[0]+2*s.speedStep;s.itemSlots[0]=[{item:Q.ITEM_BANANA,count:4},{item:Q.ITEM_SLOW_GLUE,count:6}];s._killPlayer(0);assert.deepEqual(ledger(s),{0:3,1:2,2:2,3:4,4:6});assert.deepEqual([s.bombsCap[0],s.blastCap[0],s.spdG[0]],[s.loBombs[0],s.loBlast[0],s.loSpeed[0]]);assert.deepEqual(s.itemSlots[0],[]);s._releaseDeathDrops();assert.deepEqual(ledger(s),{0:3,1:2,2:2,3:4,4:6});assert.equal(s.crate.reduce((a,b)=>a+b),5);s._releaseDeathDrops();assert.equal(s.crate.reduce((a,b)=>a+b),5,'retries never duplicate');
}
{
 const legal=[0,14,90,104,180,194,97];
 for(let k=0;k<legal.length;k++){const s=open();s.wall.fill(1);for(const c of legal)s.wall[c]=0;const sorted=legal.slice().sort((a,b)=>a-b);s.rng=()=> (k+.5)/legal.length;s._pendingDeathDrops=[{type:0,count:3,row:0,col:0}];s._releaseDeathDrops();assert.equal(s.crate.findIndex(Boolean),sorted[k],'every whole-map region is equally selectable; no nearest-cell preference');}
 const placed=[];for(const [r,c] of [[0,0],[0,14],[12,0],[12,14]]){const s=open(55);s.pos.set([r+.5,c+.5,6.5,7.5]);s.spawnCells=[[r,c],[12-r,14-c]];s._pendingDeathDrops=[{type:0,count:4,row:r,col:c},{type:1,count:2,row:r,col:c}];s._releaseDeathDrops();placed.push(Array.from(s.crate));}for(const x of placed)assert.deepEqual(x,placed[0],'corpse and spawn-side metadata cannot affect placement');
 const bins=[0,0,0,0];for(let seed=0;seed<1024;seed++){const s=open(seed);s.rng=Q.mulberry32(seed);s._pendingDeathDrops=[{type:0,count:1}];s._releaseDeathDrops();const c=s.crate.findIndex(Boolean);bins[(Math.floor(c/15)>=6?2:0)+(c%15>=7?1:0)]++;}assert(bins.every(n=>n>180&&n<340),'fixed seed sweep covers all quadrants');console.log('global drop quadrant counts',bins);
}
{
 const s=open();for(const [n,k] of ['wall','brick','pushable','cover','crate','fieldItem','fuse','blastLinger','brickLinger'].entries())s[k][n]=1;s.pendingCrateType[10]=0;s.airdropFalls=[{cell:11}];s.bunLoose[24]=1;s.alive[0]=true;s.trapped[0]=20;s.pos[0]=.5;s.pos[1]=13.5;for(const c of [0,1,2,3,4,5,6,7,8,10,11,12,13,-1,195])assert(!s._dropCellAvailable(c),'illegal drop cell '+c);const forbidden=Array.from(s.crate);s._pendingDeathDrops=Array.from({length:8},(_,i)=>({type:i%5,count:i+1}));s._releaseDeathDrops();for(let c=0;c<14;c++)if(c!==9)assert.equal(s.crate[c],forbidden[c],'cannot overwrite forbidden cell '+c);
 const full=open();full.wall.fill(1);full._pendingDeathDrops=[{type:0,count:7},{type:3,count:4}];full._releaseDeathDrops();assert.equal(full._pendingDeathDrops.length,2);assert.deepEqual(ledger(full),{0:7,3:4});full.wall[194]=0;full._releaseDeathDrops();assert.equal(full.crate[194],1);assert.equal(full._pendingDeathDrops.length,1);assert.deepEqual(ledger(full),{0:7,3:4});full.crate[194]=0;full._releaseDeathDrops();assert.equal(full.crateType[194],3);assert.equal(full.crateCount[194],4);
 const tiny={...level,wall:Array(195).fill(1),brick:Array(195).fill(0),push_boxes:[],initial_crates:[]};const z=new Q.Sim(8);z.reset(tiny,opts);assert(z.itemSupply.unplaced>0);assert.equal(z.crate.reduce((a,b)=>a+b),z.itemSupply.added);
}
{
 const s=new Q.Sim(9);s.reset(level,opts);s.alive.fill(true);s.pos.set([5.5,5.5,9.5,9.5]);for(let type=0;type<3;type++){const before=[s.bombsCap[0],s.blastCap[0],s.spdG[0]];s.crate[80]=1;s.crateType[80]=type;assert(s._collectCrate(0,80));assert([s.bombsCap[0],s.blastCap[0],s.spdG[0]][type]>before[type]);for(let n=0;n<40;n++){s.crate[80]=1;s.crateType[80]=type;s.superCrate[80]=1;s._collectCrate(0,80);}assert(s.bombsCap[0]<=s.bombsMax&&s.blastCap[0]<=s.blastMax&&s.spdG[0]<=s.speedMax);}
}
{
 const s=open(90);s._pendingDeathDrops=[{type:0,count:6},{type:4,count:9}];const frame=s.snapshotReplay(),r=open(1);r.restoreReplay(frame);s._releaseDeathDrops();r._releaseDeathDrops();assert.deepEqual(r.snapshotReplay(),s.snapshotReplay(),'restored RNG continues global sampling exactly');
 // Existing bomb destruction transfers the whole stack into the graveyard.
 const b=open();b.alive.fill(true);b.pos.set([1.5,1.5,11.5,13.5]);b.invuln.fill(100);b.crate[80]=1;b.crateType[80]=3;b.crateCount[80]=6;b.fuse[81]=1;b.owner[81]=0;b.bombBlast[81]=2;b.step([[4,0,0,1],[4,0,0,1]]);assert.equal(b.crate[80],0);assert.equal(ledger(b)[3],6,'bombed drop quantity remains in graveyard');
}
{
 const s=open();s.alive.fill(true);s.pos.set([1.5,1.5,11.5,13.5]);s.invuln.fill(100);s.brick[81]=1;s.crateRate=1;s.fuse[80]=1;s.owner[80]=0;s.bombBlast[80]=2;s.step([[4,0,0,1],[4,0,0,1]]);assert.equal(s.itemSupply.brickDropsActual,1,'actual brick supply is counted when generated');assert(s.pendingCrateType[81]>=0);const saved=s.itemSupply.brickDropsActual;s.step([[4,0,0,1],[4,0,0,1]]);assert.equal(s.itemSupply.brickDropsActual,saved,'lingering flames cannot count the same brick twice');
 const R=require('./review_protocol'),h=crypto.createHash('sha256').update(fs.readFileSync('web/sim-bun-native-v3-items.js')).digest('hex');assert.equal(R.simulatorFile(h),'sim-bun-native-v3-items.js');assert.equal(R.simulator({simulator_sha256:h}).RULE_VERSION,Q.RULE_VERSION);
 const Eval=require('../scripts/native_eval_simulator_v1');assert.equal(Eval.RULE_VERSION,'bun-native-v4-facing');assert.equal(Eval.SIMULATOR_SHA256,crypto.createHash('sha256').update(fs.readFileSync('web/sim-bun-native-v4-facing.js')).digest('hex'));
}
{
 const s=open();s.graveyard=[{type:3,count:6,isSuper:false}];for(let t=270;t<=300;t++){s.t=t;s._nativeAirdropStep();assert.equal(ledger(s)[3],6,'bird recycling never loses or duplicates the bombed death stack');}assert.equal(s.crate.reduce((a,b)=>a+b),1);
 const b=open();b.t=10;const drop={cell:80,tick:7,type:3,count:6,isSuper:false};b.airdropFalls=[drop];b.blastLinger[80]=1;b._nativeAirdropStep();assert.equal(b.crate[80],0);assert.equal(ledger(b)[3],6,'new burning landing returns whole stack to graveyard');
 const c=open();c.t=10;c.alive[0]=true;c.trapped[0]=20;c.pos[0]=5.5;c.pos[1]=5.5;c.airdropFalls=[{...drop}];c._nativeAirdropStep();assert.equal(c.crate[80],0);assert.equal(ledger(c)[3],6,'landing into a newly occupied cage preserves stack in graveyard');
}
for(const [name,hash] of Object.entries({'sim-bun-native-v2-face.js':'6a9d3c07fad3de828e8bbb550e469c2e7b2a76cb057d1a9a58c6fbcdc03e48c9','sim-bun-native-v2.js':'593b50e8278b5e544132f86dc4011c4dc06d0de1166c8ff8e751d9ac4f553154'}))assert.equal(crypto.createHash('sha256').update(fs.readFileSync('web/'+name)).digest('hex'),hash);
console.log('Item rules PASS: supply/caps/pickup, global unbiased legal drops, conservation/shortage, seed/reset/RNG restore, bombed stacks and frozen old rules');
