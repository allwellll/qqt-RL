'use strict';
// Replay every action, verify frozen state hashes, export sparse drawing frames.
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const Q=require('../web/sim-bun-native-v2'),S=require('../web/five-item-slots-v1');
const cfg=JSON.parse(fs.readFileSync(process.argv[2]));
const level=JSON.parse(fs.readFileSync('web/assets/maps/levels.json')).find(x=>x.qqt_id===806);
const hash=s=>crypto.createHash('sha256').update(JSON.stringify(s.snapshotReplay())).digest('hex');
const rows=[];
for(const result of JSON.parse(fs.readFileSync(cfg.output+'/COMPLETE.json')).results){
 const id=`five-slots-${result.seed}-${result.seat}-${result.mode}`,file=cfg.output+'/'+id+'/review.json';
 const d=JSON.parse(fs.readFileSync(file)),m=d.meta,s=new Q.Sim(m.seed);s.reset(level,{nativeTrap:true,nativeItems:true});s.maxSteps=m.max_ticks;
 for(let p=0;p<2;p++){s.pos[2*p]=m.frozen_birth.cells[p][0]+.5;s.pos[2*p+1]=m.frozen_birth.cells[p][1]+.5;}
 s.bunSpawnPos=(m.region_native_respawn_cells||m.frozen_birth.cells).map(x=>x.map(v=>v+.5));s.bombsCap[m.defender_player]=m.defender_initial_bomb_capacity;s._gen++;
 const checks=new Map(d.checks),frames=[];
 function frame(){frames.push({t:s.t,pos:Array.from(s.pos),alive:Array.from(s.alive),trapped:Array.from(s.trapped),
  carried:Array.from(s.bunCarried),wall:Array.from(s.wall),brick:Array.from(s.brick),fuse:Array.from(s.fuse),
  fire:Array.from(s.lastReplayCovered||new Uint8Array(195)),stored:s.bunStored.map(x=>Array.from(x)),done:s.done,winner:s.winner});}
 assert.equal(hash(s),checks.get(0));frame();
 for(const a of d.actions){s.step([S.engineAction(a[0],a[1]),S.engineAction(a[2],a[3])]);
  if(checks.has(s.t))assert.equal(hash(s),checks.get(s.t),`replay mismatch @${s.t}`);
  if(s.t%5===0||s.done)frame();}
 assert.equal(s.done,true);assert.equal(s.winner,d.summary.winner);
 const out=cfg.output+'/'+id+'/DRAWING_FRAMES.json';fs.writeFileSync(out,JSON.stringify({meta:m,frames}));
 rows.push({region:m.start_region,source_review:file,frames:out,verified:true,ticks:s.t,winner:s.winner});
}
fs.writeFileSync(cfg.output+'/DRAWING_MANIFEST.json',JSON.stringify(rows,null,2));console.log(JSON.stringify(rows));
