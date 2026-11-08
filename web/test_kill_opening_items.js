'use strict';
const assert=require('node:assert/strict');
const Q=require('./sim-bun-native-v4-facing'),O=require('./kill_opening_items');
const level=require('./assets/maps/levels.json').find(x=>x.qqt_id===806);
function scene(seed=7,killMode=true){const s=new Q.Sim(seed);s.reset(level,{nativeItems:true,nativeTrap:true,killMode,teams:[0,1]});return s;}
function distances(s,start){const d=new Array(195).fill(Infinity),queue=[start[0]*15+start[1]];d[queue[0]]=0;for(const i of queue)for(const [dr,dc]of [[-1,0],[1,0],[0,-1],[0,1]]){const y=Math.floor(i/15)+dr,x=i%15+dc,j=y*15+x;if(y>=0&&y<13&&x>=0&&x<15&&!s.wall[j]&&!s.brick[j]&&d[j]>d[i]+1){d[j]=d[i]+1;queue.push(j);}}return d;}
for(let seed=0;seed<32;seed++){
 const s=scene(seed),report=O.apply(s),spawns=[s.centerCell(0),s.centerCell(1)];
 assert.deepEqual(spawns[1],[spawns[0][0],14-spawns[0][1]]);
 const copyLegal=scene(seed);copyLegal.pos.set(s.pos);const legal=new Set(copyLegal._itemSupplyCells());
 assert.equal(report.actual,O.OPENING_ITEM_COUNT);assert.equal(s.crate.reduce((n,v)=>n+v,0),24);
 const counts=[0,0,0];for(const {cell,type}of report.placements){assert(legal.has(cell));assert(spawns.every(([r,c])=>Math.max(Math.abs(Math.floor(cell/15)-r),Math.abs(cell%15-c))>1));const mirror=Math.floor(cell/15)*15+14-cell%15;assert(s.crate[mirror]);assert.equal(s.crateType[mirror],type);counts[type]++;}
 assert.deepEqual(counts,[8,8,8]);const ds=spawns.map(p=>distances(s,p));for(const type of [-1,0,1,2]){const cells=report.placements.filter(x=>type<0||x.type===type).map(x=>x.cell);assert.equal(Math.min(...cells.map(c=>ds[0][c])),Math.min(...cells.map(c=>ds[1][c])));}
 const copy=scene(seed);assert.deepEqual(O.apply(copy),report);assert.equal(O.apply(s),report,'idempotent');
}
for(const type of [0,1,2]){
 const s=scene();const c=O.apply(s).placements.find(x=>x.type===type).cell;
 const field=['bombsCap','blastCap','spdG'][type],before=s[field][0];s.pos[0]=Math.floor(c/15)+.5;s.pos[1]=c%15+.5;
 s.step([[4,0,0,1],[4,0,0,1]]);assert(s[field][0]>before,'native pickup enhances '+field);assert.equal(s.crate[c],0);assert(s[field][0]<=[s.bombsMax,s.blastMax,s.speedMax][type]);
}
const carry=scene(7,false),before=JSON.stringify(carry.snapshotReplay());assert.equal(O.apply(carry),null);assert.equal(JSON.stringify(carry.snapshotReplay()),before,'carry is byte-identical');
const full=scene();full.wall.fill(1);assert.equal(O.apply(full).actual,0);assert.equal(full.killOpeningItems.shortage,24);
const drop=scene();O.apply(drop);drop._killPlayer(0);assert.equal(drop._pendingDeathDrops.reduce((n,x)=>n+x.count,0),8,'original eight death drops unchanged');
console.log('Kill opening items PASS: 24 mirrored, eight per type, spawn exclusion, equal actual path distance, seeded/idempotent, native growth/caps, carry unchanged and eight death drops');
